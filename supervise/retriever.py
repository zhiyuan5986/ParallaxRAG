import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class DE(nn.Module):
    def __init__(
        self,
        num_rounds,
        num_reverse_rounds
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())

        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())

    def forward(
        self,
        topic_entity_one_hot,
        edge_index,
        reverse_edge_index,
        num_heads: int = None,
        vendi_beta: float = 0.0,
    ):
        """
        Args:
            topic_entity_one_hot: [N, C_pe]
            edge_index: [2, E]
            reverse_edge_index: [2, E]
            num_heads: optional, enable multi-head if >1
            vendi_beta: suppression strength, >0 reduces head redundancy
        Returns:
            - If single-head: List[Tensor [N, C_pe]] (legacy compatible)
            - If multi-head:  List[Tensor [H, N, C_pe]] (per-layer)
        """
        use_multi_head = (num_heads is not None) and (num_heads > 1)
        if not use_multi_head:
            # Single-head propagation (legacy behavior).
            result_list = []
            h_pe = topic_entity_one_hot
            for layer in self.layers:
                h_pe = layer(edge_index, h_pe)
                result_list.append(h_pe)

            h_pe_rev = topic_entity_one_hot
            for layer in self.reverse_layers:
                h_pe_rev = layer(reverse_edge_index, h_pe_rev)
                result_list.append(h_pe_rev)
            return result_list

        # Multi-head propagation with suppression.
        H = int(num_heads)
        result_list_heads = []  # Per-layer [H, N, C_pe]
        # Init per-head PE state.
        h_pe_heads = [topic_entity_one_hot for _ in range(H)]

        # Forward direction.
        for layer in self.layers:
            new_heads = [layer(edge_index, h) for h in h_pe_heads]  # List[Tensor [N, C_pe]]
            if vendi_beta is not None and vendi_beta > 0:
                with torch.no_grad():
                    # Head similarity via node intensity vectors.
                    S = []
                    for h in new_heads:
                        s = h.sum(dim=-1)  # [N]
                        s = F.normalize(s, p=2, dim=-1)
                        S.append(s)
                    S = torch.stack(S, dim=0)  # [H, N]
                    G = S @ S.t()              # [H, H]
                    r = (G - torch.diag_embed(G.diag())).sum(dim=-1)  # [H]
                    scale = torch.exp(-float(vendi_beta) * r).clamp(0.5, 1.5)  # [H]
                h_pe_heads = [scale[i] * new_heads[i] for i in range(H)]
            else:
                h_pe_heads = new_heads
            # Record layer output.
            result_list_heads.append(torch.stack(h_pe_heads, dim=0))  # [H, N, C_pe]

        # Reverse direction.
        h_pe_heads = [topic_entity_one_hot for _ in range(H)]
        for layer in self.reverse_layers:
            new_heads = [layer(reverse_edge_index, h) for h in h_pe_heads]
            if vendi_beta is not None and vendi_beta > 0:
                with torch.no_grad():
                    S = []
                    for h in new_heads:
                        s = h.sum(dim=-1)  # [N]
                        s = F.normalize(s, p=2, dim=-1)
                        S.append(s)
                    S = torch.stack(S, dim=0)
                    G = S @ S.t()
                    r = (G - torch.diag_embed(G.diag())).sum(dim=-1)
                    scale = torch.exp(-float(vendi_beta) * r).clamp(0.5, 1.5)
                h_pe_heads = [scale[i] * new_heads[i] for i in range(H)]
            else:
                h_pe_heads = new_heads
            result_list_heads.append(torch.stack(h_pe_heads, dim=0))

        return result_list_heads


class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe,
        DE_kwargs,
        num_heads=None,
        head_dim=None
    ):
        super().__init__()

        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        # Drop args not accepted by DE (vendi_beta is forward-only).
        _de_cfg = dict(DE_kwargs)
        _de_cfg.pop('vendi_beta', None)
        self.de = DE(**_de_cfg)

        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (DE_kwargs['num_rounds'] + DE_kwargs['num_reverse_rounds'])

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )
        # Multi-head alignment branch (train-time only).
        self.num_heads = num_heads
        self.head_dim = head_dim
        if (self.num_heads is not None) and (self.head_dim is not None):
            self.head_proj = nn.Linear(pred_in_size, self.head_dim, bias=False)
        else:
            self.head_proj = None
        # Map head_dim -> emb_size for head-averaged entity/relation embeddings.
        self.head_to_emb = None
        if (self.num_heads is not None) and (self.head_dim is not None):
            self.head_to_emb = nn.Linear(self.head_dim, emb_size, bias=False)
        # Gate: produce head weights from q_emb for aggregation.
        self.gate = nn.Linear(emb_size, self.num_heads) if self.num_heads else None
        # Vendi suppression strength.
        self.vendi_beta = float(DE_kwargs.get('vendi_beta', 0.0))

    def gate_heads(self, q_emb, topk=None):
        """
        Generate head weights from q_emb with optional Top-K sparsity.
        Args:
            q_emb: [D]
            topk: int or None
        Returns:
            alpha: [H]
        """
        if (self.gate is None) or (self.num_heads is None):
            raise RuntimeError("Strict mode: gate is None or num_heads is None. Ensure retriever.num_heads/head_dim and q_head_embs are provided.")
        # Ensure dtype matches gate's weight dtype to avoid Half/Float matmul error
        q_in = q_emb
        if isinstance(q_in, torch.Tensor) and self.gate is not None:
            w_dtype = self.gate.weight.dtype
            if q_in.dtype != w_dtype:
                q_in = q_in.to(w_dtype)
        logits = self.gate(q_in)
        alpha = torch.softmax(logits, dim=-1)
        if topk is not None and topk > 0:
            k = min(int(topk), alpha.numel())
            vals, idx = torch.topk(alpha, k)
            mask = torch.zeros_like(alpha)
            mask[idx] = 1
            alpha = alpha * mask
            alpha = alpha / (alpha.sum().clamp_min(1e-12))
        return alpha

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        entity_head_embs=None,
        relation_head_embs=None,
        q_head_embs=None
    ):
        device = entity_embs.device

        # Ensure indices are integer type for graph ops and indexing
        h_id_tensor = h_id_tensor.long()
        r_id_tensor = r_id_tensor.long()
        t_id_tensor = t_id_tensor.long()

        # Graph indices.
        edge_index = torch.stack([h_id_tensor, t_id_tensor], dim=0)
        reverse_edge_index = torch.stack([t_id_tensor, h_id_tensor], dim=0)

        use_heads = (self.num_heads is not None) and (self.num_heads > 1)
        have_eh = (entity_head_embs is not None) and (hasattr(entity_head_embs, 'numel') and entity_head_embs.numel() > 0)
        have_rh = (relation_head_embs is not None) and (hasattr(relation_head_embs, 'numel') and relation_head_embs.numel() > 0)
        have_qh = (q_head_embs is not None) and (hasattr(q_head_embs, 'numel') and q_head_embs.numel() > 0)

        # Require multi-head embeddings; no single-head fallback.
        if not (use_heads and have_eh and have_rh and have_qh and (self.head_to_emb is not None)):
            raise RuntimeError(
                f"Strict mode: Multi-head embeddings required but not available. "
                f"use_heads={use_heads}, have_eh={have_eh}, have_rh={have_rh}, have_qh={have_qh}, "
                f"head_to_emb={self.head_to_emb is not None}. "
                f"Please ensure preprocessing generates multi-head embeddings (q_head_embs, entity_head_embs, relation_head_embs)."
            )

        # DE multi-head propagation over topic PE.
        # Always run multi-head DE (falls back to single-head if use_heads=False).
        de_out = self.de(topic_entity_one_hot, edge_index, reverse_edge_index,
                           num_heads=self.num_heads if use_heads else None,
                           vendi_beta=getattr(self, 'vendi_beta', 0.0))

        # Non-text entity embedding.
        non_text = self.non_text_entity_emb(torch.LongTensor([0]).to(device))  # [1,D]
        non_text = non_text.expand(num_non_text_entities, -1)  # [N_non_text, D]
        # Align non_text dtype to entity_embs for safe concatenation
        if isinstance(entity_embs, torch.Tensor) and hasattr(entity_embs, 'dtype'):
            non_text = non_text.to(entity_embs.dtype)

        if use_heads and have_eh and have_rh and have_qh and (self.head_to_emb is not None):
            # Per-head entity/relation + per-head scoring + gated fusion.
            H = int(self.num_heads)
            # Precompute per-head entity embeddings (project to D).
            e_heads = []  # List[[N_text_ent, D]]
            for i in range(H):
                e_i = entity_head_embs[:, i, :]  # [N_text_ent, d_h]
                e_i = e_i.to(dtype=self.head_to_emb.weight.dtype)
                e_i = self.head_to_emb(e_i)      # [N_text_ent, D]
                # Align dtype to entity_embs.
                if entity_embs is not None and hasattr(entity_embs, 'dtype') and e_i.dtype != entity_embs.dtype:
                    e_i = e_i.to(entity_embs.dtype)
                e_heads.append(e_i)

            # Precompute per-head relation embeddings (project to D).
            r_heads = []  # List[[E, D]]
            for i in range(H):
                r_i = relation_head_embs[r_id_tensor, i, :]  # [E, d_h]
                r_i = r_i.to(dtype=self.head_to_emb.weight.dtype)
                r_i = self.head_to_emb(r_i)  # [E, D]
                if relation_embs is not None and hasattr(relation_embs, 'dtype') and r_i.dtype != relation_embs.dtype:
                    r_i = r_i.to(relation_embs.dtype)
                r_heads.append(r_i)

            # Precompute per-head query embeddings (project to D).
            q_heads_D = []  # List[[D]]
            for i in range(H):
                qi = q_head_embs[i, :]  # [d_h]
                qi = qi.to(dtype=self.head_to_emb.weight.dtype)
                qi = self.head_to_emb(qi.unsqueeze(0)).squeeze(0)  # [D]
                if qi.dtype != q_emb.dtype:
                    qi = qi.to(q_emb.dtype)
                q_heads_D.append(qi)

            # Split DE outputs into per-head [N, C] lists.
            # de_out: List of layers; each layer_h: [H,N,C] in multi-head mode
            de_layers_per_head = None
            if use_heads:
                de_layers_per_head = [[] for _ in range(H)]  # List[List[[N,C]]]
                for layer_h in de_out:  # [H,N,C]
                    for i in range(H):
                        de_layers_per_head[i].append(layer_h[i])  # [N,C]
            else:
                # Single-head: each layer [N, C].
                de_layers_per_head = [[layer_h for layer_h in de_out]]

            # Build h_e_cat_i and score per head.
            head_logits_list = []  # will become [E,H]

            for i in range(H):
                # Base entity vectors (concat non-text).
                h_e_i = torch.cat([e_heads[i], non_text], dim=0)  # [N_total_text+non_text, D]

                # Align node counts when they differ.
                # Target N from DE first layer or topic_entity_one_hot.
                target_N = None
                if len(de_layers_per_head[i]) > 0 and isinstance(de_layers_per_head[i][0], torch.Tensor):
                    target_N = int(de_layers_per_head[i][0].shape[0])
                elif self.topic_pe and isinstance(topic_entity_one_hot, torch.Tensor):
                    target_N = int(topic_entity_one_hot.shape[0])

                if target_N is not None and int(h_e_i.shape[0]) != target_N:
                    N_src = int(h_e_i.shape[0])
                    if N_src > target_N:
                        # Truncate to target_N (preserve ID mapping).
                        h_e_i = h_e_i[:target_N]
                    else:
                        # Zero-pad to target_N rows.
                        pad_rows = target_N - N_src
                        pad = torch.zeros((pad_rows, h_e_i.shape[1]), dtype=h_e_i.dtype, device=h_e_i.device)
                        h_e_i = torch.cat([h_e_i, pad], dim=0)

                h_e_list = [h_e_i]
                if self.topic_pe:
                    h_e_list.append(topic_entity_one_hot)
                # Append per-layer DE outputs [N, C].
                for layer_nc in de_layers_per_head[i]:
                    h_e_list.append(layer_nc)
                h_e_cat_i = torch.cat(h_e_list, dim=1)  # [N_align, D + L*C + (topic_pe?2:0)]

                # Relation and query for this head.
                h_r_i = r_heads[i]                  # [E,D]
                # Ensure concatenation dtypes match (topic PE and DE layers to match h_e_i dtype)
                if self.topic_pe and isinstance(topic_entity_one_hot, torch.Tensor) and topic_entity_one_hot.dtype != h_e_i.dtype:
                    topic_entity_one_hot = topic_entity_one_hot.to(h_e_i.dtype)
                # Cast DE per-layer outputs to match h_e_i dtype
                for idx_layer, layer_nc in enumerate(de_layers_per_head[i]):
                    if isinstance(layer_nc, torch.Tensor) and layer_nc.dtype != h_e_i.dtype:
                        de_layers_per_head[i][idx_layer] = layer_nc.to(h_e_i.dtype)

                q_i = q_heads_D[i]                 # [D]

                # Triple scoring (shared predictor).
                h_triple_i = torch.cat([
                    q_i.expand(len(h_r_i), -1),
                    h_e_cat_i[h_id_tensor],
                    h_r_i,
                    h_e_cat_i[t_id_tensor]
                ], dim=1)  # [E, pred_in_size]
                logit_i = self.pred(h_triple_i).reshape(-1)  # [E]
                head_logits_list.append(logit_i)

            head_logits = torch.stack(head_logits_list, dim=1)  # [E,H]
            # Base logits: mean over heads for compatibility.
            base_logits = head_logits.mean(dim=1, keepdim=True)  # [E,1]
            return base_logits, head_logits
