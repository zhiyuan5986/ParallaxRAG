# aggregate_headhop.py
import argparse, csv, numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def load_rows(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for x in r:
            x["hop_id"] = int(x["hop_id"])
            x["head_id"] = int(x["head_id"])
            x["cand_id"] = int(x["cand_id"])
            x["score"] = float(x["score"])
            x["rank"] = int(x["rank"])
            x["is_gold"] = int(x["is_gold"])
            x["was_selected"] = int(x["was_selected"])
            x["came_from_head"] = int(x["came_from_head"])
            rows.append(x)
    return rows

def headhop_matrix(rows, H=None, T=None, metric="contribution"):
    # Build H x T matrix; metric: contribution / hit_rate / use_rate.
    max_h = max(r["head_id"] for r in rows)
    max_t = max(r["hop_id"] for r in rows)
    H = H or (max_h + 1)
    T = T or (max_t + 1)
    M = np.full((H, T), np.nan)

    by_ht = defaultdict(list)
    for r in rows:
        if r['hop_id'] > 0:
            by_ht[(r["head_id"], r["hop_id"]-1)].append(r) # hop_id is 1-indexed, matrix is 0-indexed

    for h in range(H):
        for t in range(T):
            logs = by_ht.get((h, t), [])
            if not logs:
                continue
            if metric == "contribution":
                # Contribution: share of selected gold items from this head at this hop.
                gold_sel_total_rows = [x for x in rows if x["hop_id"] == t+1 and x["was_selected"] == 1 and x["is_gold"] == 1]
                gold_sel_total = len(set( (r['sample_id'],r['cand_id']) for r in gold_sel_total_rows))

                gold_sel_from_h_rows = [x for x in logs if x["was_selected"] == 1 and x["is_gold"] == 1 and x['came_from_head'] == h]
                gold_sel_from_h = len(set( (r['sample_id'],r['cand_id']) for r in gold_sel_from_h_rows))

                M[h, t] = (gold_sel_from_h / gold_sel_total) if gold_sel_total > 0 else 0.0
            elif metric == "hit_rate":
                # Hit rate: gold share among selected candidates from this head at this hop.
                used_rows = [x for x in logs if x["was_selected"] == 1 and x['came_from_head'] == h]
                used = len(set( (r['sample_id'],r['cand_id']) for r in used_rows))
                if used == 0:
                    M[h, t] = np.nan
                else:
                    gold_in_used = len(set( (r['sample_id'],r['cand_id']) for r in used_rows if r['is_gold']==1))
                    M[h, t] = gold_in_used / used
            elif metric == "use_rate":
                # Use rate: fraction of samples where this head selected any candidate at this hop.
                all_samples = {x["sample_id"] for x in rows if x["hop_id"] == t+1}
                used_samples = {x["sample_id"] for x in logs if x["was_selected"] == 1 and x['came_from_head'] == h}
                M[h, t] = len(used_samples) / max(1, len(all_samples))
            else:
                raise ValueError(metric)
    return M

def draw_heatmap(M, title, out_path):
    H, T = M.shape
    heads = [f"H{h}" for h in range(H)]
    hops = [f"Hop-{t + 1}" for t in range(T)]
    fig, ax = plt.subplots(figsize=(1.2 * T + 3, 0.5 * H + 2))
    im = ax.imshow(M, aspect='auto', cmap='viridis')
    ax.set_xticks(np.arange(T));
    ax.set_yticks(np.arange(H))
    ax.set_xticklabels(hops, rotation=45, ha='right');
    ax.set_yticklabels(heads)
    for i in range(H):
        for j in range(T):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha='center', va='center', color='w', fontsize=8)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8);
    cbar.set_label(title)
    ax.set_title(f"Head–Hop Heatmap: {title}")
    ax.set_xlabel("Hop");
    ax.set_ylabel("Head");
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_csv", default="mh_headhop_log.csv")
    ap.add_argument("--metric", default="contribution", choices=["contribution", "hit_rate", "use_rate"])
    ap.add_argument("--out", default="headhop_heatmap.png")
    args = ap.parse_args()

    rows = load_rows(args.log_csv)
    M = headhop_matrix(rows, metric=args.metric)
    draw_heatmap(M, title=args.metric, out_path=args.out)
