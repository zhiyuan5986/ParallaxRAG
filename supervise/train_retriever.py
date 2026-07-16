import os, sys
import argparse
import time
import copy
import torch
import swanlab
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from collections import defaultdict
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from supervise.utils import set_seed, prepare_sample
from supervise.retriever import Retriever
from supervise.retriever_dataset import RetrieverDataset, collate_retriever
import numpy as np
from src.utils.lr_schedule import adjust_learning_rate
from supervise.config import HARDCODED_CONFIG

def eval_epoch(config, device, data_loader, model):
    model.eval()
    metric = defaultdict(list)
    with torch.no_grad():
        for sample in data_loader:
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list, entity_head_embs, relation_head_embs, q_head_embs = prepare_sample(device, sample)

            outputs = model(h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                           num_non_text_entities, relation_embs, topic_entity_one_hot,
                           entity_head_embs, relation_head_embs, q_head_embs)

            eval_use_gate = config['train'].get('eval_use_gate', True)
            if not eval_use_gate:
                raise RuntimeError("eval_use_gate must be True.")

            if not isinstance(outputs, tuple):
                raise RuntimeError("eval expects tuple outputs with head_logits; ensure model returns (logits, head_logits).")

            base, head_logits = outputs

            topk = int(config['train'].get('gate_topk', 0)) or None
            alpha = model.gate_heads(q_emb, topk=topk)
            if alpha is None:
                raise RuntimeError("gate not initialized or num_heads missing")
            logits = (head_logits * alpha.view(1, -1)).sum(dim=1)

            sorted_ids = torch.argsort(logits, descending=True).cpu()
            ranks = torch.empty_like(sorted_ids)
            ranks[sorted_ids] = torch.arange(len(ranks))

            target_ids = target_triple_probs.nonzero().squeeze(-1)
            if len(target_ids) == 0:
                continue

            
            num_total_entities = len(entity_embs) + num_non_text_entities
            for k in config['eval']['k_list']:
                k = int(k)
                recall_k = (ranks[target_ids] < k).sum().item() / len(target_ids)
                metric[f'triple_recall@{k}'].append(recall_k)

                mask_k = ranks < k
                ent_mask_k = torch.zeros(num_total_entities)
                ent_mask_k[h_id_tensor[mask_k]] = 1.
                ent_mask_k[t_id_tensor[mask_k]] = 1.
                ans_recall_k = ent_mask_k[a_entity_id_list].sum().item() / max(1, len(a_entity_id_list))
                metric[f'ans_recall@{k}'].append(ans_recall_k)

    return {k: float(torch.tensor(v).mean()) for k, v in metric.items()}


def train_epoch(config, device, data_loader, model, optimizer, epoch):
    model.train()
    total_loss = 0.0
    sum_main, num_batches = 0.0, 0
    log_every = int(config['train'].get('log_every', 0))

    gradient_accumulation_steps = config['train'].get('gradient_accumulation_steps', 1)

    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    P_list, E_list = [], []
    for b_idx, sample in enumerate(data_loader, 1):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
        num_non_text_entities, relation_embs, topic_entity_one_hot, \
        target_triple_probs, a_entity_id_list, entity_head_embs, relation_head_embs, q_head_embs = prepare_sample(device, sample)

        if len(h_id_tensor) == 0:
            continue

        if use_amp:
            try:
                autocast_ctx = autocast('cuda', enabled=True)
            except TypeError:
                autocast_ctx = autocast(enabled=True)
        else:
            autocast_ctx = autocast(enabled=False)

        with autocast_ctx:
            outputs = model(h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                            num_non_text_entities, relation_embs, topic_entity_one_hot,
                            entity_head_embs, relation_head_embs, q_head_embs)

            if isinstance(outputs, tuple):
                logits, head_logits = outputs
            else:
                logits, head_logits = outputs, None

            logits = logits.reshape(-1)
            pos = target_triple_probs.to(device).float()
            P = float(pos.sum().item())
            E = float(len(h_id_tensor))
            if E > 0:
                P_list.append(P)
                E_list.append(E)

        gold = None
        with torch.no_grad():
            if pos.sum() <= 0:
                raise RuntimeError("no positives in full set")
            gold = pos / pos.sum()

        if head_logits is None:
            raise RuntimeError("head_logits is None")

        if gold is None:
            raise RuntimeError("gold distribution is None")

        listwise_warm = int(config['train'].get('listwise_warmup_epochs', 0))
        use_gate_topk = (epoch >= listwise_warm)
        topk_cfg = int(config['train'].get('gate_topk', 0)) if use_gate_topk else 0
        topk = topk_cfg if topk_cfg > 0 else None
        alpha = model.gate_heads(q_emb, topk=topk)  # [H]
        if alpha is None:
            raise RuntimeError("gate not initialized or num_heads missing; please provide q_head_embs and ensure Retriever.gate is set.")

        head_mat = head_logits  # [E, H]
        S = (head_mat * alpha.view(1, -1)).sum(dim=1)  # [E' or E]
        P_pred = torch.softmax(S, dim=0)

        # ===== weighted listwise loss =====
        pos_mask = (gold > 0)
        if pos_mask.any():
            weights = torch.ones_like(gold)
            weights[pos_mask] = 10.0  # Tunable (5-20) to boost positives.
            gold = (gold * weights) / (gold * weights).sum()
        L_main = -(gold * (P_pred + 1e-12).log()).sum()
        loss = L_main

        loss_val = float(loss.item())
        main_val = float(L_main.item())

        loss = loss / gradient_accumulation_steps

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if b_idx % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if b_idx % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        num_batches += 1
        total_loss += loss_val
        sum_main += main_val

        if log_every > 0 and (b_idx % log_every) == 0:
            try:
                print(f"[Epoch {epoch} Batch {b_idx}] E={int(E)}, P={int(P)}, "
                      f"L_main={sum_main/num_batches:.4f}, Loss={total_loss/num_batches:.4f}")
            except Exception:
                pass

    try:
        avg_main, avg_total = sum_main/max(1,num_batches), total_loss/max(1,num_batches)
        avg_P = float(np.mean(P_list)) if len(P_list) else 0.0
        avg_E = float(np.mean(E_list)) if len(E_list) else 0.0
        avg_PE = (np.array(P_list)/np.array(E_list)).mean().item() if len(P_list)==len(E_list) and len(P_list)>0 else 0.0
        avg_N = avg_E - avg_P
        print(f"[Epoch {epoch}] Train Avg: L_main={avg_main:.4f}, Loss={avg_total:.4f}")
        print(f"[Epoch {epoch}] Stats: avg_P={avg_P:.2f}, avg_E={avg_E:.2f}, avg_N={avg_N:.2f}, avg_P/E={avg_PE:.6f}")

    except Exception:
        pass

    try:
        swanlab.log({
            'train/L_main': avg_main,
            'train/Loss': avg_total,
            'train/avg_P': avg_P,
            'train/avg_E': avg_E,
            'train/avg_N': avg_N,
            'train/avg_P_over_E': avg_PE,
        })
    except Exception:
        pass

    return total_loss / max(1, len(data_loader))


def main():
    config = copy.deepcopy(HARDCODED_CONFIG)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['webqsp', 'cwq', 'bioasq'], default=config['dataset']['name'])
    parser.add_argument('--output-dir', help='Directory in which to write cpt.pth')
    parser.add_argument('--epochs', type=int, help='Override the configured epoch count')
    parser.add_argument('--device', help='Torch device, e.g. cuda:0 or cpu')
    args = parser.parse_args()
    config['dataset']['name'] = args.dataset
    config['train']['save_prefix'] = args.dataset
    if args.epochs is not None:
        config['train']['num_epochs'] = args.epochs
    device = torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    train_set = RetrieverDataset(config=config, split='train')
    try:
        val_set = RetrieverDataset(config=config, split='val')
    except FileNotFoundError:
        print("validation split not found, fallback to 'test' split for validation.")
        val_set = RetrieverDataset(config=config, split='test')

    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever,
                              num_workers=num_workers, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_retriever,
                            num_workers=num_workers, pin_memory=True, persistent_workers=False)

    # Model
    emb_size = train_set[0]['q_emb'].shape[-1]
    sample_heads = train_set[0].get('q_head_embs', None)
    if sample_heads is None or not hasattr(sample_heads, 'shape') or sample_heads.dim() != 2:
        raise RuntimeError("q_head_embs not found or invalid in training data")

    H, d_h = sample_heads.shape[0], sample_heads.shape[1]
    retr_cfg = dict(config['retriever'])
    retr_cfg['num_heads'] = int(H)
    retr_cfg['head_dim'] = int(d_h)
    model = Retriever(emb_size, **retr_cfg).to(device)

    optimizer_config = dict(config['optimizer'])
    if torch.cuda.is_available():
        try:
            optimizer = AdamW(model.parameters(), fused=True, foreach=True, **optimizer_config)
            print("Using AdamW with fused=True, foreach=True")
        except (TypeError, RuntimeError):
            optimizer = AdamW(model.parameters(), **optimizer_config)
            print("Using standard AdamW (fused/foreach not available)")
    else:
        optimizer = AdamW(model.parameters(), **optimizer_config)
        print("Using standard AdamW (CPU mode)")

    lr_args = argparse.Namespace(
        warmup_epochs=int(config['train'].get('warmup_epochs', 0)),
        num_epochs=int(config['train'].get('num_epochs', 1)),
    )
    base_lr = float(config['optimizer'].get('lr', 1e-3))

    # Logging & Saving
    ts = time.strftime('%b%d-%H:%M:%S')
    save_dir = args.output_dir or f"{config['train']['save_prefix']}_{ts}"
    os.makedirs(save_dir, exist_ok=True)
    swanlab.init(project=f"{config['dataset']['name']}_retriever", experiment_name=save_dir, config=config)

    # Always persist the first evaluated model, even when recall is exactly zero.
    best = float('-inf')
    patient = 0
    for epoch in range(config['train']['num_epochs']):
        try:
            current_lrs = []
            for pg in optimizer.param_groups:
                lr_now = adjust_learning_rate(pg, base_lr, epoch, lr_args)
                current_lrs.append(lr_now)
            if len(current_lrs) > 0:
                swanlab.log({'train/lr': float(np.mean(current_lrs)), 'epoch': epoch})
            else:
                swanlab.log({'train/lr': base_lr, 'epoch': epoch})
        except Exception:
            pass
        val_metrics = eval_epoch(config, device, val_loader, model)
        triple_recall_100 = float(val_metrics.get('triple_recall@100', 0.0))
        if triple_recall_100 > best:
            best = triple_recall_100
            patient = 0
            torch.save({'config': config, 'model_state_dict': model.state_dict()}, os.path.join(save_dir, 'cpt.pth'))
        else:
            patient += 1
        swanlab.log({**{f"val/{k}": v for k, v in val_metrics.items()}, 'epoch': epoch})
        try:
            val_str = ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
            print(f"[Epoch {epoch}] Val: {val_str}")
        except Exception:
            pass

        # Train
        train_loss = train_epoch(config, device, train_loader, model, optimizer, epoch)
        swanlab.log({'train/loss': train_loss})
        try:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")
        except Exception:
            pass

        if patient >= config['train']['patience']:
            break

    print('Training done. Best triple_recall@100 =', best)
    try:
        swanlab.finish()
    except Exception:
        pass


if __name__ == '__main__':

    main()
