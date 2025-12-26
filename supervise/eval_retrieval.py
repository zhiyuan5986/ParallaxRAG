import argparse
import os
import torch
from collections import defaultdict


def compute_metrics(pred_dict, k_list=(10, 50, 100, 200, 500), by_hop=False, hop_values=None):
    """
    Match eval_from_cpt.py -> eval_epoch behavior:
    - Recall-only metrics (triple_recall@K, ans_recall@K)
    - Skip samples with no gold triples
    - Answer recall uses head+tail union of top-K triples
    - Optional bucketing by max_path_length (e.g., hop=1,2)
    """
    # Overall metrics.
    collectors = {f'triple_recall@{int(K)}': [] for K in k_list}
    collectors.update({f'ans_recall@{int(K)}': [] for K in k_list})
    collectors.update({f'ans_recall_tail@{int(K)}': [] for K in k_list})
    collectors.update({f'ans_hit_tail@{int(K)}': [] for K in k_list})

    # Per-hop metrics.
    hop_values = set(int(h) for h in (hop_values or [])) if by_hop else set()
    hop_collectors = {}
    hop_counts = {h: 0 for h in hop_values}  # Sample count per hop.
    for h in hop_values:
        hc = {f'triple_recall@{int(K)}': [] for K in k_list}
        hc.update({f'ans_recall@{int(K)}': [] for K in k_list})
        hc.update({f'ans_recall_tail@{int(K)}': [] for K in k_list})
        hc.update({f'ans_hit_tail@{int(K)}': [] for K in k_list})
        hop_collectors[int(h)] = hc

    for _, sample in pred_dict.items():
        scored = sample.get('scored_triples', []) or []  # list of (h, r, t, score)
        gold_triples = sample.get('target_relevant_triples', []) or []  # list of (h, r, t)
        # Skip samples with no gold triples.
        if len(gold_triples) == 0:
            continue

        hop = int(sample.get('max_path_length') or 0)  # 0/None indicates no path.
        gold_answers = set(sample.get('a_entity_in_graph', []) or [])
        triples_only = [(h, r, t) for (h, r, t, *_ ) in scored]
        gold_triples_set = set(gold_triples)

        for K in k_list:
            K = int(K)
            K_eff = min(K, len(triples_only))
            topK = triples_only[:K_eff]
            topK_set = set(topK)

            # Triple recall.
            inter = topK_set & gold_triples_set
            recall_t = len(inter) / max(1, len(gold_triples_set))
            collectors[f'triple_recall@{K}'].append(recall_t)

            # Answer recall (head+tail union).
            entK = []
            for (h, _, t) in topK:
                entK.append(h)
                entK.append(t)
            entK_set = set(entK)
            interA = entK_set & gold_answers
            recall_a = len(interA) / max(1, len(gold_answers))
            collectors[f'ans_recall@{K}'].append(recall_a)

            # Tail-only metrics.
            tailsK = [t for (_, _, t) in topK]
            tail_set = set(tailsK)
            inter_tail = tail_set & gold_answers
            recall_tail = len(inter_tail) / max(1, len(gold_answers))
            hit_tail = 1.0 if len(inter_tail) > 0 else 0.0
            collectors[f'ans_recall_tail@{K}'].append(recall_tail)
            collectors[f'ans_hit_tail@{K}'].append(hit_tail)

            # Per-hop stats.
            if by_hop and (hop in hop_values):
                hop_collectors[hop][f'triple_recall@{K}'].append(recall_t)
                hop_collectors[hop][f'ans_recall@{K}'].append(recall_a)
                hop_collectors[hop][f'ans_recall_tail@{K}'].append(recall_tail)
                hop_collectors[hop][f'ans_hit_tail@{K}'].append(hit_tail)
                hop_counts[hop] += 1

    # Aggregate mean values to a flat dict.
    result = {}
    for name, vals in collectors.items():
        result[name] = None if len(vals) == 0 else float(torch.tensor(vals, dtype=torch.float32).mean().item())

    if by_hop:
        for h, d in hop_collectors.items():
            for name, vals in d.items():
                key = f"{name}|hop={h}"
                result[key] = None if len(vals) == 0 else float(torch.tensor(vals, dtype=torch.float32).mean().item())
        # Attach sample counts per hop.
        for h, c in hop_counts.items():
            result[f"count|hop={h}"] = int(c)

    return result


def format_result(res):
    # Match eval_from_cpt.py flat key=value formatting.
    lines = []
    for name in sorted(res.keys()):
        v = res[name]
        if isinstance(v, float):
            lines.append(f"{name}={v:.4f}")
        else:
            lines.append(f"{name}={v}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, help='Path to retrieval_result.pth')
    parser.add_argument('--k_list', type=str, default='10,50,100,200,500', help='Comma-separated K values')
    parser.add_argument('--by_hop', action='store_true', help='Bucket metrics by max_path_length')
    parser.add_argument('--hop_values', type=str, default='1,2', help='Hop values to bucket, e.g., 1,2')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError(args.path)

    k_list = tuple(int(s) for s in args.k_list.split(',') if s.strip())
    hop_values = tuple(int(s) for s in args.hop_values.split(',') if s.strip()) if args.by_hop else None
    pred_dict = torch.load(args.path, map_location='cpu')

    res = compute_metrics(pred_dict, k_list=k_list, by_hop=args.by_hop, hop_values=hop_values)
    print(format_result(res))


if __name__ == '__main__':
    main()
