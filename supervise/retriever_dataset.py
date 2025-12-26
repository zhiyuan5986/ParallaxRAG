import os
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import pickle
from tqdm import tqdm


class RetrieverDataset:
    def __init__(
        self,
        config,
        split,
        skip_no_path=True
    ):
        # Load pre-processed data.
        dataset_name = config['dataset']['name']
        processed_dict_list = self._load_processed(dataset_name, split)

        # Extract directed shortest paths from topic entities to answer
        # entities or vice versa as weak supervision signals for triple scoring.
        triple_score_dict = self._get_triple_scores(
            dataset_name, split, processed_dict_list)

        # Load pre-computed embeddings.
        emb_dict = self._load_emb(
            dataset_name, config['dataset']['text_encoder_name'], split)

        # Put everything together.
        self._assembly(
            processed_dict_list, triple_score_dict, emb_dict, skip_no_path)

    def _load_processed(
        self,
        dataset_name,
        split
    ):
        # BioASQ uses JSON files under proc_resolved.
        if dataset_name.lower() == 'bioasq':
            processed_file = os.path.join(
                f'/data/BioASQ/proc_resolved/{split}.json')
            if not os.path.exists(processed_file):
                raise FileNotFoundError(
                    f"BioASQ processed file not found: {processed_file}\n"
                    f"Please run: python supervise/step0_bioasq_resolve_ids.py"
                )
            with open(processed_file, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            # Convert BioASQ format to RetrieverDataset schema.
            return self._convert_bioasq_format(data_list)
        else:
            # Default: load .pkl from processed directory.
            processed_file = os.path.join(
                f'/data/{dataset_name}/processed/{split}.pkl')
            with open(processed_file, 'rb') as f:
                return pickle.load(f)

    def _convert_bioasq_format(self, data_list):
        """
        Convert BioASQ format to RetrieverDataset schema.

        BioASQ format:
        - graph: [[h, r, t], ...]
        - q_entity: [...]
        - a_entity: [...]
        - text_entity_list: [...]

        Target fields:
        - h_id_list, r_id_list, t_id_list: triple index lists
        - q_entity_id_list, a_entity_id_list: entity index lists
        """
        converted = []
        for item in data_list:
            # 1) Build entity-to-ID map.
            text_entities = item.get('text_entity_list', [])
            if not text_entities:
                # Fallback: extract entities from graph.
                graph = item.get('graph', [])
                entities_set = set()
                for triple in graph:
                    if len(triple) >= 3:
                        entities_set.add(triple[0])
                        entities_set.add(triple[2])
                text_entities = sorted(list(entities_set))

            entity2id = {ent: idx for idx, ent in enumerate(text_entities)}

            # 2) Build relation-to-ID map.
            relations = item.get('relation_list', [])
            if not relations:
                # Fallback: extract relations from graph.
                graph = item.get('graph', [])
                relations_set = set()
                for triple in graph:
                    if len(triple) >= 3:
                        relations_set.add(triple[1])
                relations = sorted(list(relations_set))

            relation2id = {rel: idx for idx, rel in enumerate(relations)}

            # 3) Convert graph triples to ID lists.
            graph = item.get('graph', [])
            h_id_list = []
            r_id_list = []
            t_id_list = []

            for triple in graph:
                if len(triple) >= 3:
                    h, r, t = triple[0], triple[1], triple[2]
                    if h in entity2id and t in entity2id and r in relation2id:
                        h_id_list.append(entity2id[h])
                        r_id_list.append(relation2id[r])
                        t_id_list.append(entity2id[t])

            # 4) Convert q_entity/a_entity to ID lists.
            q_entities = item.get('q_entity', [])
            a_entities = item.get('a_entity', [])

            q_entity_id_list = [entity2id[ent] for ent in q_entities if ent in entity2id]
            a_entity_id_list = [entity2id[ent] for ent in a_entities if ent in entity2id]

            # 5) Build converted sample.
            converted_item = {
                'id': item.get('id', ''),
                'question': item.get('question', ''),
                'h_id_list': h_id_list,
                'r_id_list': r_id_list,
                't_id_list': t_id_list,
                'q_entity': q_entities,  # Keep text list.
                'a_entity': a_entities,  # Keep text list.
                'q_entity_id_list': q_entity_id_list,
                'a_entity_id_list': a_entity_id_list,
                'text_entity_list': text_entities,
                'relation_list': relations,
                'non_text_entity_list': [],  # BioASQ has no non-text entities.
            }

            # Preserve extra fields when present.
            for key in ['ideal_answer', 'snippets', 'question_type']:
                if key in item:
                    converted_item[key] = item[key]

            converted.append(converted_item)

        return converted

    def _get_triple_scores(
        self,
        dataset_name,
        split,
        processed_dict_list
    ):
        # BioASQ uses capitalized path.
        if dataset_name.lower() == 'bioasq':
            save_dir = os.path.join('/data/BioASQ/triple_scores')
        else:
            save_dir = os.path.join('/data', dataset_name, 'triple_scores')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pth')

        if os.path.exists(save_file):
            try:
                cached = torch.load(save_file)
                # Validate cached IDs cover current processed list.
                if isinstance(cached, dict):
                    cached_ids = set(cached.keys())
                    current_ids = set()
                    for s in processed_dict_list:
                        try:
                            current_ids.add(s['id'])
                        except Exception:
                            continue
                    if current_ids and current_ids.issubset(cached_ids):
                        return cached
                    else:
                        print(f"[WARN] triple_scores cache mismatch for {dataset_name}/{split} — will recompute.")
                else:
                    print(f"[WARN] triple_scores cache type invalid for {dataset_name}/{split} — will recompute.")
            except Exception:
                print(f"[WARN] failed to load triple_scores cache for {dataset_name}/{split} — will recompute.")

        triple_score_dict = dict()
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            triple_scores_i, max_path_length_i, hop_to_triples_i = self._extract_paths_and_score(
                sample_i)

            triple_score_dict[sample_i_id] = {
                'triple_scores': triple_scores_i,
                'max_path_length': max_path_length_i,
                'hop_to_triples': hop_to_triples_i
            }

        torch.save(triple_score_dict, save_file)

        return triple_score_dict

    def _extract_paths_and_score(
        self,
        sample
    ):
        nx_g = self._get_nx_g(
            sample['h_id_list'],
            sample['r_id_list'],
            sample['t_id_list']
        )

        # Each raw path is a list of entity IDs.
        path_list_ = []
        for q_entity_id in sample['q_entity_id_list']:
            for a_entity_id in sample['a_entity_id_list']:
                paths_q_a = self._shortest_path(nx_g, q_entity_id, a_entity_id)
                if len(paths_q_a) > 0:
                    path_list_.extend(paths_q_a)

        if len(path_list_) == 0:
            max_path_length = None
        else:
            max_path_length = 0

        # Each processed path is a list of triple IDs, grouped by hop.
        hop_to_triples = defaultdict(set)

        for path in path_list_:
            num_triples_path = len(path) - 1
            max_path_length = max(max_path_length, num_triples_path)

            for i in range(num_triples_path):
                h_id_i = path[i]
                t_id_i = path[i+1]
                # NOTE: This assumes a single edge between two nodes in the graph.
                # If multiple edges can exist, this logic might need adjustment.
                triple_id = nx_g[h_id_i][t_id_i]['triple_id']
                hop_id = i + 1  # Hops are 1-indexed
                hop_to_triples[hop_id].add(triple_id)

        # For backward compatibility, also create the flat triple_scores tensor
        num_triples = len(sample['h_id_list'])
        triple_scores = torch.zeros(num_triples)
        for _, triple_ids in hop_to_triples.items():
            for triple_id in triple_ids:
                triple_scores[triple_id] = 1.

        # Fallback supervision: if no paths, use topic/answer adjacent edges.
        if int(triple_scores.sum().item()) == 0:
            try:
                q_ids = set(sample.get('q_entity_id_list', []) or [])
                a_ids = set(sample.get('a_entity_id_list', []) or [])
                h_ids = sample.get('h_id_list', []) or []
                t_ids = sample.get('t_id_list', []) or []
                if len(h_ids) == num_triples and len(t_ids) == num_triples:
                    fallback_pos = set()
                    for i in range(num_triples):
                        h_i = h_ids[i]
                        t_i = t_ids[i]
                        if (h_i in q_ids) or (t_i in q_ids) or (h_i in a_ids) or (t_i in a_ids):
                            fallback_pos.add(i)
                    if len(fallback_pos) > 0:
                        for i in fallback_pos:
                            triple_scores[i] = 1.
                        hop_to_triples = {1: list(sorted(fallback_pos))}
                        max_path_length = 1
            except Exception:
                pass

        # Convert sets to lists for serialization
        hop_to_triples_serializable = {k: list(v) for k, v in hop_to_triples.items()}

        return triple_scores, max_path_length, hop_to_triples_serializable

    def _get_nx_g(
        self,
        h_id_list,
        r_id_list,
        t_id_list
    ):
        nx_g = nx.DiGraph()
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)

        return nx_g

    def _shortest_path(
        self,
        nx_g,
        q_entity_id,
        a_entity_id
    ):
        try:
            forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
        except Exception:
            forward_paths = []

        try:
            backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
        except Exception:
            backward_paths = []

        full_paths = forward_paths + backward_paths
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths

        min_path_len = min([len(path) for path in full_paths])
        refined_paths = []
        for path in full_paths:
            if len(path) == min_path_len:
                refined_paths.append(path)

        return refined_paths

    def _score_triples(
        self,
        path_list,
        num_triples
    ):
        triple_scores = torch.zeros(num_triples)

        for path in path_list:
            for triple_id_list in path:
                triple_scores[triple_id_list] = 1.

        return triple_scores

    def _load_emb(
        self,
        dataset_name,
        text_encoder_name,
        split
    ):
        # BioASQ uses capitalized path.
        if dataset_name.lower() == 'bioasq':
            file_path = f'/data/BioASQ/emb/{text_encoder_name}/{split}.pth'
        else:
            file_path = f'/data/{dataset_name}/emb/{text_encoder_name}/{split}.pth'

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Embedding file not found: {file_path}\n"
                f"Please run: python supervise/preprocess_retriever.py -d {dataset_name}"
            )
        # Use context manager and map_location to avoid leaks and device issues.
        with open(file_path, 'rb') as f:
            return torch.load(f, map_location='cpu')

    def _assembly(
        self,
        processed_dict_list,
        triple_score_dict,
        emb_dict,
        skip_no_path,
    ):
        self.processed_dict_list = []

        num_relevant_triples = []
        num_skipped = 0
        count_with_path = 0
        count_no_path = 0
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            assert sample_i_id in triple_score_dict

            triple_score_i = triple_score_dict[sample_i_id]['triple_scores']
            max_path_length_i = triple_score_dict[sample_i_id]['max_path_length']

            num_relevant_triples_i = len(triple_score_i.nonzero())
            num_relevant_triples.append(num_relevant_triples_i)

            sample_i['target_triple_probs'] = triple_score_i
            sample_i['max_path_length'] = max_path_length_i
            sample_i['hop_to_triples'] = triple_score_dict[sample_i_id].get('hop_to_triples', {})  # Use .get for backward compatibility

            # Path existence stats (independent of skipping).
            if max_path_length_i in [None, 0]:
                count_no_path += 1
            else:
                count_with_path += 1

            if skip_no_path and (max_path_length_i in [None, 0]):
                num_skipped += 1
                continue

            sample_i.update(emb_dict[sample_i_id])
            # Downcast large float tensors to float16 to reduce CPU memory footprint
            _float_keys = ['q_emb', 'entity_embs', 'relation_embs', 'entity_head_embs', 'relation_head_embs', 'q_head_embs']
            for _k in _float_keys:
                _v = sample_i.get(_k, None)
                if isinstance(_v, torch.Tensor) and _v.is_floating_point():
                    sample_i[_k] = _v.to(dtype=torch.float16)

            sample_i['a_entity'] = list(set(sample_i['a_entity']))
            sample_i['a_entity_id_list'] = list(set(sample_i['a_entity_id_list']))

            # PE for topic entities.
            num_entities_i = len(sample_i['text_entity_list']) + len(sample_i['non_text_entity_list'])
            topic_entity_mask = torch.zeros(num_entities_i)
            topic_entity_mask[sample_i['q_entity_id_list']] = 1.
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
            sample_i['topic_entity_one_hot'] = topic_entity_one_hot.float()

            self.processed_dict_list.append(sample_i)

        median_num_relevant = int(np.median(num_relevant_triples))
        mean_num_relevant = int(np.mean(num_relevant_triples))
        max_num_relevant = int(np.max(num_relevant_triples))

        print(f'# skipped samples: {num_skipped}')
        print(f'# relevant triples | median: {median_num_relevant} | mean: {mean_num_relevant} | max: {max_num_relevant}')
        # Shortest-path availability stats.
        total_seen = count_with_path + count_no_path
        try:
            ratio_no_path = (count_no_path / max(1, total_seen)) if total_seen else 0.0
        except Exception:
            ratio_no_path = 0.0
        print(f'# shortest-path stats | with_path: {count_with_path} | no_path: {count_no_path} | total_seen: {total_seen} | no_path_ratio: {ratio_no_path:.4f}')

    def __len__(self):
        return len(self.processed_dict_list)

    def __getitem__(self, i):
        return self.processed_dict_list[i]


def collate_retriever(data):
    sample = data[0]

    h_id_list = sample['h_id_list']
    h_id_tensor = torch.tensor(h_id_list)
    q_head_embs = sample.get('q_head_embs', torch.zeros(0, 0))

    r_id_list = sample['r_id_list']
    r_id_tensor = torch.tensor(r_id_list)

    t_id_list = sample['t_id_list']
    t_id_tensor = torch.tensor(t_id_list)

    num_non_text_entities = len(sample['non_text_entity_list'])

    entity_head_embs = sample.get('entity_head_embs', torch.zeros(0, 0, 0))
    relation_head_embs = sample.get('relation_head_embs', torch.zeros(0, 0, 0))

    return h_id_tensor, r_id_tensor, t_id_tensor, sample['q_emb'],\
        sample['entity_embs'], num_non_text_entities, sample['relation_embs'],\
        sample['topic_entity_one_hot'], sample['target_triple_probs'], sample['a_entity_id_list'],\
        entity_head_embs, relation_head_embs, q_head_embs
