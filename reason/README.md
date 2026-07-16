Reason
======

This folder contains the KGQA reasoning pipeline that builds prompts from
retrieved triples, runs LLM inference, and evaluates results. It is designed
to run from the repository root.

What This Folder Provides
-------------------------
- Data assembly with graph substructures and scored triples
- Prompt construction for multiple modes (ICL, no-evidence, GPT variants)
- LLM inference (local vLLM or API-backed models via llm_utils.py)
- Evaluation scripts for CWQ/WebQSP and BioASQ

Key Scripts
-----------
- main.py: run the full inference and evaluation loop
- preprocess/prepare_data.py: attach subgraphs and scored triples
- preprocess/prepare_prompts.py: build prompts for CWQ/WebQSP
- preprocess/prepare_prompts_bioasq.py: build prompts for BioASQ
- metrics/evaluate_results.py: original evaluation
- metrics/evaluate_results_corrected.py: corrected evaluation (used in main.py)
- metrics/evaluate_bioasq.py: BioASQ evaluation

Inputs and Data Dependencies
----------------------------
The reasoning pipeline expects:
1) Retrieval results (predictions) JSONL file
2) Scored triples file (.pth)
3) Subgraphs dataset from Hugging Face: `rmanluo/RoG-<dataset>`

By default, `main.py` uses:
```
./results/KGQA/<dataset>/RoG/<split>/results_gen_rule_path_RoG-<dataset>_RoG_<split>_predictions_3_False_jsonl/predictions.jsonl
```

Quick Start
-----------
Run an end-to-end CWQ test inference:
```
python -m reason.main -d cwq --split test --prompt_mode scored_100 --llm_mode sys_icl_dc
```

Run WebQSP with a local vLLM model:
```
python -m reason.main -d webqsp --split test -m meta-llama/Meta-Llama-3.1-8B-Instruct
```

Use a custom scored triples file:
```
python -m reason.main -d cwq --split test -p ./scored_triples/my_scores.pth
```

Prompt Modes and LLM Modes
--------------------------
Prompt behavior is controlled by `prompt_mode` and `llm_mode`:
- Prompt templates are defined in `reason/prompts.py`
- Data and prompt assembly is done in:
  - `reason/preprocess/prepare_data.py`
  - `reason/preprocess/prepare_prompts.py`

To add a new prompt mode:
1) Add templates in `reason/prompts.py`
2) Add routing in `get_defined_prompts` in `reason/main.py`
3) Update prompt assembly in `reason/preprocess/prepare_prompts.py`

LLM Backends
------------
`llm_utils.py` selects the backend based on `model_name`:
- Local vLLM for local model names
- OpenAI-compatible API for models containing "gpt"

If you use GPT models, token usage and estimated cost are reported at the end
of a run.

Evaluation
----------
During `main.py` runs, evaluation happens:
- periodically every 500 samples
- once at the end

You can also run evaluation manually:
```
python reason/metrics/evaluate_results_corrected.py <predictions_jsonl>
```

BioASQ evaluation:
```
python reason/metrics/evaluate_bioasq.py <predictions_jsonl> <ground_truth_json>
```

Outputs
-------
Predictions are written under:
```
./results/KGQA/<dataset>/SubgraphRAG/<model_name>/
```

The run writes a `*-resume.jsonl` file during inference. Evaluation uses a
temporary copy to avoid overwriting the resume file.

Notes
-----
- `main.py` uses `swanlab` to log metrics.
- Subgraph data is loaded from Hugging Face datasets; ensure access is available.
