Supervise
=========

This folder contains the retriever training, evaluation, inference, and
preprocessing utilities used in this project.

What This Folder Provides
-------------------------
- End-to-end retriever pipeline: preprocessing, training, evaluation, inference
- Multi-head retriever model with gated head aggregation
- Dataset loader that expects preprocessed files and embeddings under /data

Key Scripts
-----------
- preprocess_retriever.py: build processed files and embeddings
- train_retriever.py: train the retriever
- eval_from_cpt.py: evaluate a saved checkpoint
- inference.py: run inference over samples
- eval_retrieval.py: evaluation utilities
- aggregate_headhop.py: aggregate per-head hop statistics

Core Modules
------------
- retriever.py: model definitions
- retriever_dataset.py: RetrieverDataset + collate_retriever
- utils.py
- preprocess_embeddings.py: embedding computation and mapping
- config.py: default config values

Data Preprocess
---------------
The preprocessing output is produced by:

```
supervise/preprocess_embeddings.py
```

The pipeline uses `/data` by default. Set `PARALLAX_DATA_ROOT` (or use the
top-level reproduction script's `--data-root`) to relocate it. Make sure the dataset layout
matches the following structure:

```
/data/<dataset_name>/
  processed/
    train.pkl
    val.pkl
    test.pkl
  emb/
    bge/
      train.pth
      val.pth
      test.pth
```

Quick Start
-----------
1) Prepare processed data and embeddings
```
python supervise/preprocess_retriever.py -d webqsp --build_pkl
```

2) Train the retriever
```
python supervise/train_retriever.py -d webqsp
```

3) Run inference
```
python -m supervise.inference --path <path_to_cpt> --split test --output retrieval_result.pth
```

4) Run retrieval evaluation
```
python supervise/eval_retrieval.py --cpt_path <path_to_cpt> --split test
```

Configuration Notes
-------------------
- Default config values are in `supervise/config.py`.
- Training uses the dataset name to set save_prefix and to locate data under /data.
- The retriever requires multi-head embeddings (q_head_embs, entity_head_embs,
  relation_head_embs). Make sure preprocessing generates them.
