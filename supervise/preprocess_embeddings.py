import os
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from supervise.utils import load_processed_list
from src.dataset.preprocess.emb import text2embedding_with_multihead


def compute_and_save_embeddings(processed_file: str, save_file: str,
                                model, tokenizer, device, chunk_size: int = None) -> None:
    """Chunked embedding, dedup, and mapping to reduce memory pressure."""
    # 1) Chunk size (overridable via env).
    if chunk_size is None:
        try:
            # Larger chunks for >40GB GPUs.
            total_mem = torch.cuda.get_device_properties(device).total_memory if torch.cuda.is_available() else 0
        except Exception:
            total_mem = 0
        default_chunk = 20000 if total_mem >= 40 * 1024**3 else 5000
        chunk_size = int(os.getenv('EMB_CHUNK_SIZE', str(default_chunk)))

    # 2) Load processed list (.pkl/.json).
    processed_list: List[Dict[str, Any]] = load_processed_list(processed_file)

    if len(processed_list) == 0:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        torch.save({}, save_file)
        return

    # 3) Collect question/entity/relation text (fallback to graph when missing).
    def _normalize_text(x: Any) -> str:
        try:
            if isinstance(x, str):
                return x.strip()
            if isinstance(x, dict):
                for k in ('label', 'text', 'name', 'id', 'qid', 'pid', 'value'):
                    v = x.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            if isinstance(x, (list, tuple)):
                for it in x:
                    s = _normalize_text(it)
                    if s:
                        return s
            s = str(x)
            return s.strip()
        except Exception:
            return ''

    questions: List[str] = []
    uniq_entity_set: set = set()
    uniq_relation_set: set = set()
    any_graph_present = False
    for s in processed_list:
        questions.append(s.get('question', ''))
        # Prefer explicit fields.
        src_entities = list((s.get('text_entity_list', []) or []))
        src_relations = list((s.get('relation_list', []) or []))
        # Always parse graph to enrich global dedup sets.
        graph_entities, graph_relations = [], []
        graph = s.get('graph', []) or []
        if graph:
            any_graph_present = True
            for trip in graph:
                if isinstance(trip, (list, tuple)) and len(trip) >= 3:
                    h, p, t = trip[0], trip[1], trip[2]
                    h_txt = _normalize_text(h)
                    t_txt = _normalize_text(t)
                    p_txt = _normalize_text(p)
                    if h_txt:
                        graph_entities.append(h_txt)
                    if t_txt:
                        graph_entities.append(t_txt)
                    if p_txt:
                        graph_relations.append(p_txt)
                elif isinstance(trip, dict):
                    h_txt = _normalize_text(trip.get('head') or trip.get('h'))
                    t_txt = _normalize_text(trip.get('tail') or trip.get('t'))
                    p_txt = _normalize_text(trip.get('relation') or trip.get('rel') or trip.get('p'))
                    if h_txt:
                        graph_entities.append(h_txt)
                    if t_txt:
                        graph_entities.append(t_txt)
                    if p_txt:
                        graph_relations.append(p_txt)
        # Merge explicit fields and graph.
        for t in src_entities:
            if isinstance(t, str) and t.strip():
                uniq_entity_set.add(t)
        for r in src_relations:
            if isinstance(r, str) and r.strip():
                uniq_relation_set.add(r)
        for t in graph_entities:
            if isinstance(t, str) and t.strip():
                uniq_entity_set.add(t)
        for r in graph_relations:
            if isinstance(r, str) and r.strip():
                uniq_relation_set.add(r)

    if len(uniq_entity_set) == 0 and len(uniq_relation_set) == 0 and any_graph_present:
        print(f"[WARN] Detected graph in samples but extracted 0 unique entities/relations. Please check graph format.")

    # Stable sort for deterministic indexing.
    uniq_entities: List[str] = sorted(uniq_entity_set)
    uniq_relations: List[str] = sorted(uniq_relation_set)

    # 4) Encode questions with sentence + multi-head embeddings.
    print(f"Encoding {len(questions)} questions with multi-head outputs...")
    # No instruction template needed for BGE-M3.
    q_embs, multihead_embs_dict = text2embedding_with_multihead(
        questions, model, tokenizer, device, instruction_template=None, target_layers=None
    )
    # Use the last layer multi-head embeddings: [N, H, head_dim].
    if len(multihead_embs_dict) == 0:
        # Fallback: empty tensor if no multi-head output.
        q_head_embs = torch.zeros((q_embs.shape[0], 0, 0))
    else:
        # Single key (last layer).
        _layer_idx = sorted(multihead_embs_dict.keys())[-1]
        q_head_embs = multihead_embs_dict[_layer_idx]

    print(f"Encoding {len(uniq_entities)} unique entities with multi-head in chunks of {chunk_size}...")
    ent_embs_list = []
    ent_head_list = []
    for i in tqdm(range(0, len(uniq_entities), chunk_size), desc="Encoding entities"):
        chunk_entities = uniq_entities[i:i+chunk_size]
        chunk_embs, chunk_heads_dict = text2embedding_with_multihead(
            chunk_entities, model, tokenizer, device, instruction_template=None, target_layers=None
        )
        if not chunk_heads_dict and len(chunk_entities) > 0:
            raise RuntimeError("BGE-M3 multi-head outputs missing for entity chunks; please check text2embedding_with_multihead.")
        _layer_idx = sorted(chunk_heads_dict.keys())[-1]
        chunk_heads = chunk_heads_dict[_layer_idx]
        ent_embs_list.append(chunk_embs)
        ent_head_list.append(chunk_heads)
        torch.cuda.empty_cache()  # Clear CUDA cache.
    ent_all = torch.cat(ent_embs_list, dim=0) if ent_embs_list else torch.zeros((0, 1024))
    if len(uniq_entities) > 0 and not ent_head_list:
        raise RuntimeError("Entity multi-head list is empty while uniq_entities > 0.")
    ent_head_all = torch.cat(ent_head_list, dim=0) if ent_head_list else torch.zeros((0, 0, 0))

    print(f"Encoding {len(uniq_relations)} unique relations with multi-head in chunks of {chunk_size}...")
    rel_embs_list = []
    rel_head_list = []
    for i in tqdm(range(0, len(uniq_relations), chunk_size), desc="Encoding relations"):
        chunk_relations = uniq_relations[i:i+chunk_size]
        chunk_embs, chunk_heads_dict = text2embedding_with_multihead(
            chunk_relations, model, tokenizer, device, instruction_template=None, target_layers=None
        )
        if not chunk_heads_dict and len(chunk_relations) > 0:
            raise RuntimeError("BGE-M3 multi-head outputs missing for relation chunks; please check text2embedding_with_multihead.")
        _layer_idx = sorted(chunk_heads_dict.keys())[-1]
        chunk_heads = chunk_heads_dict[_layer_idx]
        rel_embs_list.append(chunk_embs)
        rel_head_list.append(chunk_heads)
        torch.cuda.empty_cache()  # Clear CUDA cache.
    rel_all = torch.cat(rel_embs_list, dim=0) if rel_embs_list else torch.zeros((0, 1024))
    if len(uniq_relations) > 0 and not rel_head_list:
        raise RuntimeError("Relation multi-head list is empty while uniq_relations > 0.")
    rel_head_all = torch.cat(rel_head_list, dim=0) if rel_head_list else torch.zeros((0, 0, 0))

    # 4) Build lookup tables: text -> row index.
    ent2row: Dict[str, int] = {t: i for i, t in enumerate(uniq_entities)}
    rel2row: Dict[str, int] = {t: i for i, t in enumerate(uniq_relations)}

    # 5) Map embeddings back to samples.
    emb_dict: Dict[Any, Dict[str, torch.Tensor]] = {}
    for idx, s in enumerate(tqdm(processed_list, desc=f"assembling {os.path.basename(processed_file)}")):
        # Support multiple ID fields with a fallback.
        sid = s.get('id')
        if sid is None or str(sid).strip() == '':
            sid = s.get('uid') or s.get('qid') or s.get('question_id') or s.get('questionId')
        if sid is None or str(sid).strip() == '':
            # Fallback to question MD5 or index.
            try:
                import hashlib
                q_text = str(s.get('question', ''))
                sid = hashlib.md5(q_text.encode('utf-8')).hexdigest() if q_text else f"sample_{idx}"
            except Exception:
                sid = f"sample_{idx}"
        # Build entity/relation lists; fallback to graph with stable order.
        text_entity_list = list((s.get('text_entity_list', []) or []))
        relation_list = list((s.get('relation_list', []) or []))
        if (not text_entity_list) or (not relation_list):
            graph = s.get('graph', []) or []
            if graph:
                ent_buf = []
                rel_buf = []
                for trip in graph:
                    if isinstance(trip, (list, tuple)) and len(trip) >= 3:
                        h, p, t = trip[0], trip[1], trip[2]
                        ent_buf.extend([_normalize_text(h), _normalize_text(t)])
                        rel_buf.append(_normalize_text(p))
                    elif isinstance(trip, dict):
                        ent_buf.extend([
                            _normalize_text(trip.get('head') or trip.get('h')),
                            _normalize_text(trip.get('tail') or trip.get('t')),
                        ])
                        rel_buf.append(_normalize_text(trip.get('relation') or trip.get('rel') or trip.get('p')))
                # De-dup while preserving order.
                if not text_entity_list:
                    text_entity_list = [x for i, x in enumerate(ent_buf) if isinstance(x, str) and x.strip() and x not in ent_buf[:i]]
                if not relation_list:
                    relation_list = [x for i, x in enumerate(rel_buf) if isinstance(x, str) and x.strip() and x not in rel_buf[:i]]

        # Question embedding for this sample.
        q_emb = q_embs[idx].contiguous() if q_embs.size(0) > 0 else torch.zeros(1024)
        # Multi-head embeddings for this sample.
        if q_head_embs.dim() == 3 and q_head_embs.size(0) > idx:
            q_heads = q_head_embs[idx].contiguous()
        else:
            q_heads = torch.zeros((0, 0))

        # Entity/relation indices in original order.
        if len(text_entity_list) > 0 and ent_all.size(0) > 0:
            ent_idx = []
            miss_ent = 0
            for t in text_entity_list:
                key = str(t)
                idx = ent2row.get(key)
                if idx is None:
                    k2 = key.strip()
                    idx = ent2row.get(k2)
                if idx is None:
                    miss_ent += 1
                    continue
                ent_idx.append(idx)
            if miss_ent > 0:
                print(f"[WARN] Sample {sid}: {miss_ent} entities not found in uniq set; they will be skipped.")
            entity_embs = ent_all[ent_idx].contiguous()
            entity_head_embs = ent_head_all[ent_idx].contiguous() if ent_head_all.size(0) > 0 else torch.zeros((0, 0, 0))
        else:
            entity_embs = torch.zeros((0, ent_all.size(1) if ent_all.dim() == 2 else 1024))
            entity_head_embs = torch.zeros((0, ent_head_all.size(1) if ent_head_all.dim() == 3 else 0,
                                           ent_head_all.size(2) if ent_head_all.dim() == 3 else 0))

        if len(relation_list) > 0 and rel_all.size(0) > 0:
            rel_idx = []
            miss_rel = 0
            for r in relation_list:
                key = str(r)
                idx = rel2row.get(key)
                if idx is None:
                    k2 = key.strip()
                    idx = rel2row.get(k2)
                if idx is None:
                    miss_rel += 1
                    continue
                rel_idx.append(idx)
            if miss_rel > 0:
                print(f"[WARN] Sample {sid}: {miss_rel} relations not found in uniq set; they will be skipped.")
            relation_embs = rel_all[rel_idx].contiguous()
            relation_head_embs = rel_head_all[rel_idx].contiguous() if rel_head_all.size(0) > 0 else torch.zeros((0, 0, 0))
        else:
            relation_embs = torch.zeros((0, rel_all.size(1) if rel_all.dim() == 2 else 1024))
            relation_head_embs = torch.zeros((0, rel_head_all.size(1) if rel_head_all.dim() == 3 else 0,
                                             rel_head_all.size(2) if rel_head_all.dim() == 3 else 0))

        emb_dict[sid] = {
            'q_emb': q_emb,
            'q_head_embs': q_heads,
            'entity_embs': entity_embs,
            'relation_embs': relation_embs,
            'entity_head_embs': entity_head_embs,
            'relation_head_embs': relation_head_embs,
        }

    # 6) Save output.
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    torch.save(emb_dict, save_file)
