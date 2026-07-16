ParallaxRAG
========

This repository contains two main pipelines:

1) `supervise/`: retriever preprocessing, training, evaluation, inference  
2) `reason/`: KGQA reasoning pipeline that builds prompts, runs LLM inference, and evaluates results

Quick Navigation
----------------
- Retriever: `supervise/README.md`
- Reasoning: `reason/README.md`

Full Reproduction
-----------------
Run the complete preprocessing, retriever training, test retrieval, reasoning,
and evaluation flow from the repository root:

```bash
pip install -r requirements.txt
scripts/run_full_experiment.sh \
  --dataset webqsp \
  --data-root /data \
  --rog-predictions /path/to/RoG-webqsp-test/predictions.jsonl \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

The RoG predictions file is a required upstream input: it contains the
questions and reasoning paths to which this project's retrieved triples are
attached. Data is stored beneath `<data-root>/<dataset>/`, while all generated
checkpoints and predictions default to `runs/<dataset>/`. See
`scripts/run_full_experiment.sh --help` for resume options, custom devices,
epoch overrides, and output paths.

To reuse preprocessing and a trained checkpoint:

```bash
scripts/run_full_experiment.sh \
  --dataset cwq \
  --rog-predictions /path/to/cwq/predictions.jsonl \
  --skip-preprocess --skip-train \
  --checkpoint runs/cwq/retriever/cpt.pth
```
