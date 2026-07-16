import os
import random
import pickle
import json
from typing import List, Dict, Any

import numpy as np
import torch
from datasets import load_dataset

from src.dataset.utils.emb import EmbInferDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_sample(device, sample):
    h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list, entity_head_embs, relation_head_embs, q_head_embs = sample

    h_id_tensor = h_id_tensor.to(device)
    r_id_tensor = r_id_tensor.to(device)
    t_id_tensor = t_id_tensor.to(device)
    q_emb = q_emb.to(device)
    entity_embs = entity_embs.to(device)
    relation_embs = relation_embs.to(device)
    topic_entity_one_hot = topic_entity_one_hot.to(device)
    entity_head_embs = entity_head_embs.to(device)
    relation_head_embs = relation_head_embs.to(device)
    q_head_embs = q_head_embs.to(device)

    return h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list, entity_head_embs, relation_head_embs, q_head_embs


def ensure_entity_identifiers(path_file: str) -> set:
    if os.path.exists(path_file):
        with open(path_file, 'r') as f:
            return set(l.strip() for l in f if l.strip())
    os.makedirs(os.path.dirname(path_file), exist_ok=True)
    # default: empty set
    with open(path_file, 'w') as f:
        f.write('')
    return set()


def build_processed_pickles(save_dir: str, dataset_name: str = 'webqsp') -> None:
    os.makedirs(save_dir, exist_ok=True)
    # Choose dataset
    if dataset_name.lower() == 'cwq':
        hf_name = 'rmanluo/RoG-cwq'
        base_dir = os.path.join(os.environ.get('PARALLAX_DATA_ROOT', '/data'), 'cwq')

    else:
        hf_name = 'ml1996/webqsp'
        base_dir = os.path.join(os.environ.get('PARALLAX_DATA_ROOT', '/data'), 'webqsp')

    # Load HF dataset (question/answer/q_entity/a_entity/graph)
    train_set = load_dataset(hf_name, split='train')
    val_set = load_dataset(hf_name, split='validation')
    test_set = load_dataset(hf_name, split='test')

    entity_identifiers = ensure_entity_identifiers(os.path.join(base_dir, 'entity_identifiers.txt'))

    EmbInferDataset(train_set, entity_identifiers, os.path.join(save_dir, 'train.pkl'))
    EmbInferDataset(val_set, entity_identifiers, os.path.join(save_dir, 'val.pkl'))
    EmbInferDataset(test_set, entity_identifiers, os.path.join(save_dir, 'test.pkl'),
                    skip_no_topic=False, skip_no_ans=False)


def load_processed_list(processed_file: str) -> List[Dict[str, Any]]:
    """Load processed list from .pkl or .json produced by step0 or EmbInferDataset."""
    if processed_file.lower().endswith('.pkl'):
        with open(processed_file, 'rb') as f:
            return pickle.load(f)
    if processed_file.lower().endswith('.json'):
        with open(processed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            raise ValueError(f"JSON at {processed_file} is not a list")
    raise ValueError(f"Unsupported file extension for {processed_file}")
