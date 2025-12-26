import os, sys
import os
import torch
from html import escape

from tqdm import tqdm

from supervise.utils import set_seed, prepare_sample
from supervise.retriever import Retriever
from supervise.retriever_dataset import RetrieverDataset, collate_retriever
import torch.nn.functional as F
import csv
from collections import defaultdict

class HeadHopLogger:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sample_id", "hop_id", "head_id", "cand_id", "score", "rank",
                            "is_gold", "was_selected", "came_from_head", "dataset_split"])

    def log(self, **row):
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([row.get(k) for k in ["sample_id", "hop_id", "head_id", "cand_id", "score",
                                             "rank", "is_gold", "was_selected", "came_from_head",
                                             "dataset_split"]])

def compute_edge_hops(h_ids, t_ids, topic_entity_ids, num_entities):
    """Compute hop distance per edge using BFS from topic entities."""
    import collections

    adj = collections.defaultdict(list)
    for i, (h, t) in enumerate(zip(h_ids.tolist(), t_ids.tolist())):
        adj[h].append((t, i))

    queue = collections.deque([(entity_id, 0) for entity_id in topic_entity_ids])
    entity_hops = [-1] * num_entities
    for entity_id in topic_entity_ids:
        if entity_id < num_entities:
            entity_hops[entity_id] = 0

    while queue:
        curr_entity, curr_hop = queue.popleft()

        for neighbor_entity, _ in adj[curr_entity]:
            if neighbor_entity < num_entities and entity_hops[neighbor_entity] == -1:
                entity_hops[neighbor_entity] = curr_hop + 1
                queue.append((neighbor_entity, curr_hop + 1))

    edge_hops = [-1] * len(h_ids)
    for i, h_id in enumerate(h_ids.tolist()):
        if h_id < num_entities and entity_hops[h_id] != -1:
            edge_hops[i] = entity_hops[h_id] + 1

    return edge_hops

def rerank_with_multihead_features(final_scores, head_logits, q_emb,
                                   h_id_tensor, r_id_tensor, t_id_tensor,
                                   relation_embs, max_K=500, head_topk=64,
                                   weights_str=None):
    E = final_scores.shape[0]
    K_base = min(int(max_K), E)
    m = min(int(head_topk), E)

    # 1) Union candidates U.
    base_idx = torch.topk(final_scores, k=K_base).indices  # [K_base]
    head_scores = torch.softmax(head_logits, dim=0)  # [E,H]
    H = head_scores.shape[1]
    per_head_idx = []
    for i in range(H):
        per_head_idx.append(torch.topk(head_scores[:, i], k=m).indices)
    per_head_idx = torch.unique(torch.cat(per_head_idx)) if len(per_head_idx) > 0 else base_idx
    U = torch.unique(torch.cat([base_idx, per_head_idx]))  # [U]

    # 2) Lightweight features.
    S_base = final_scores[U]
    S_head_max = head_logits[U].max(dim=1).values
    # Head coverage: top-m threshold via head_scores > 1/E.
    cov_thresh = max(1.0 / max(1, E), 1e-6)
    head_cov = (head_scores[U] > cov_thresh).sum(dim=1).float()  # [U]

    # Relation similarity (q_emb vs relation_emb[r]).
    rel_vec = relation_embs[r_id_tensor[U]]  # [U, D]
    rel_sim = F.cosine_similarity(q_emb.view(1, -1).expand_as(rel_vec), rel_vec, dim=-1)  # [U]

    # Path support: count composable edges within U (approx).
    H_u, T_u = h_id_tensor[U], t_id_tensor[U]
    # Edge composition for length-2 paths: t(e_i)==h(e_j) or h(e_i)==t(e_j).
    # Broadcasted comparison; U is typically a few thousand.
    with torch.no_grad():
        cond1 = (T_u.view(-1, 1) == H_u.view(1, -1))
        cond2 = (H_u.view(-1, 1) == T_u.view(1, -1))
        path_supp = (cond1 | cond2).float().sum(dim=1)  # [U]

    # 3) Weighted scoring.
    # Default weights can be overridden by --rerank_weights (e.g., "base=1.0,hmax=0.2,hcov=0.1,rel=0.1,path=0.1").
    lam = dict(base=1.0, hmax=0.2, hcov=0.1, rel=0.1, path=0.1)
    if isinstance(weights_str, str) and len(weights_str) > 0:
        try:
            for kv in weights_str.split(','):
                k, v = kv.split('=')
                lam[k.strip()] = float(v)
        except Exception:
            pass

    rerank = lam['base']*S_base + lam['hmax']*S_head_max + lam['hcov']*head_cov + lam['rel']*rel_sim + lam['path']*path_supp
    order = torch.argsort(rerank, descending=True)
    U_sorted = U[order][:K_base]

    # Return reranked top-K scores and indices; scores follow rerank order.
    rerank_top_scores = rerank[order][:K_base]
    return rerank_top_scores, U_sorted.tolist()
@torch.no_grad()
def main(args):
    device = torch.device(f'cuda:0')

    cpt = torch.load(args.path, map_location='cpu')
    config = cpt['config']
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])

    infer_set = RetrieverDataset(
        config=config, split='test', skip_no_path=False)

    emb_size = infer_set[0]['q_emb'].shape[-1]
    # Infer num_heads/head_dim to match training-time settings.
    retr_cfg = dict(config['retriever'])
    sample_heads = infer_set[0].get('q_head_embs', None)
    if sample_heads is not None and hasattr(sample_heads, 'shape') and sample_heads.dim() == 2:
        H, d_h = int(sample_heads.shape[0]), int(sample_heads.shape[1])
        retr_cfg['num_heads'] = retr_cfg.get('num_heads') or H
        retr_cfg['head_dim'] = retr_cfg.get('head_dim') or d_h
    model = Retriever(emb_size, **retr_cfg).to(device)
    # Use strict=False for checkpoint compatibility (inference does not need head_proj).
    model.load_state_dict(cpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    logger = HeadHopLogger(args.log_path) if args.log_path else None
    pred_dict = dict()
    for i in tqdm(range(len(infer_set))):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, a_entity_id_list, entity_head_embs, relation_head_embs, q_head_embs = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']
        top_K_triples = []
        target_relevant_triples = []
        head_attn_dump = None

        if len(h_id_tensor) != 0:
            # Eval-only forward for stable ranking.
            outputs = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot,
                entity_head_embs, relation_head_embs, q_head_embs)

            if isinstance(outputs, tuple):
                pred_triple_logits, head_logits = outputs
            else:
                pred_triple_logits, head_logits = outputs, None

            # Apply the same gated aggregation used in training.
            use_gate = bool(config.get('train', {}).get('eval_use_gate', True))
            if use_gate and (head_logits is not None) and (q_head_embs is not None) and (q_head_embs.numel() > 0):
                # Compute gate weights and aggregate head scores.
                gate_topk = int(config.get('train', {}).get('gate_topk', 0)) or None
                alpha = model.gate_heads(q_emb, topk=gate_topk)
                final_scores = (head_logits * alpha.view(1, -1)).sum(dim=1).reshape(-1)
            else:
                # Fallback to base logits if heads or gate are unavailable.
                final_scores = pred_triple_logits.reshape(-1)

            # Optional rerank: union over heads + lightweight features.
            if args.rerank and use_gate and (head_logits is not None):
                final_scores, top_K_triple_IDs = rerank_with_multihead_features(
                    final_scores, head_logits, q_emb, h_id_tensor, r_id_tensor, t_id_tensor,
                    relation_embs, args.max_K, args.rerank_head_topk, args.rerank_weights
                )
                top_K_scores = final_scores[:len(top_K_triple_IDs)].detach().cpu().tolist()
            else:
                # Base ranking.
                top_K_results = torch.topk(final_scores,
                                           k=min(args.max_K, len(final_scores)))
                top_K_scores = top_K_results.values.detach().cpu().tolist()
                top_K_triple_IDs = top_K_results.indices.detach().cpu().tolist()

            for j, triple_id in enumerate(top_K_triple_IDs):
                top_K_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                    top_K_scores[j]
                ))

            # Gold triples.
            target_relevant_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
            for triple_id in target_relevant_triple_ids:
                target_relevant_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                ))

            # Optional: dump per-head score distributions over full graph.
            if args.dump_attn and (head_logits is not None) and (q_head_embs is not None) and (q_head_embs.numel() > 0):
                # Softmax over all edges to get per-head distributions.
                head_scores = torch.softmax(head_logits, dim=0)  # [E,H]
                H = head_scores.shape[1]
                K = min(int(args.attn_top_k), head_scores.shape[0])
                head_attn_dump = []
                for hi in range(H):
                    vals, idx = torch.topk(head_scores[:, hi], k=K)
                    vals = vals.detach().cpu().tolist()
                    idx = idx.detach().cpu().tolist()
                    triples = [(
                        entity_list[h_id_tensor[e].item()],
                        relation_list[r_id_tensor[e].item()],
                        entity_list[t_id_tensor[e].item()],
                        vals[k]
                    ) for k, e in enumerate(idx)]
                    head_attn_dump.append({
                        'head': hi,
                        'topK': triples
                    })

        sample_dict = {
            'question': raw_sample['question'],
            'scored_triples': top_K_triples,
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples,
        }
        if args.dump_attn and head_attn_dump is not None:
            sample_dict['head_scores_topK'] = head_attn_dump

        pred_dict[raw_sample['id']] = sample_dict

    # Save structured results.
    root_path = os.path.dirname(args.path)
    result_path = os.path.join(root_path, 'retrieval_result.pth')
    torch.save(pred_dict, result_path)

    root_path = os.path.dirname(args.path)
    torch.save(pred_dict, os.path.join(root_path, 'retrieval_result.pth'))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to a saved model checkpoint, e.g., webqsp_Nov08-01:14:47/cpt.pth')
    parser.add_argument('--max_K', type=int, default=500,
                        help='K in top-K triple retrieval')
    parser.add_argument('--attn_top_k', type=int, default=100,
                        help='When dumping attention, keep per-head top-K edges for compactness')
    parser.add_argument('--rerank', action='store_true', help='Enable option A: union over heads + lightweight feature rerank')
    parser.add_argument('--rerank_head_topk', type=int, default=64, help='Top-m per head for union')
    parser.add_argument('--rerank_weights', type=str, default='base=1.0,hmax=0.2,hcov=0.1,rel=0.1,path=0.1',
                        help='Rerank weights, comma-separated (e.g., "base=1.0,hmax=0.2,hcov=0.1,rel=0.1,path=0.1")')
    parser.add_argument('--log_path', type=str, default=None, help='If set, write detailed head-hop logs to this path')
    args = parser.parse_args()

    main(args)
