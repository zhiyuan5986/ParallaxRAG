#!/usr/bin/env bash
# Reproduce the retriever training -> retrieval -> KGQA reasoning pipeline.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="webqsp"
DATA_ROOT="${PARALLAX_DATA_ROOT:-/data}"
OUTPUT_ROOT="$ROOT_DIR/runs"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
ROG_PREDICTIONS=""
EPOCHS=""
DEVICE=""
SKIP_PREPROCESS=0
SKIP_TRAIN=0
CHECKPOINT=""

usage() {
  cat <<'EOF'
Usage: scripts/run_full_experiment.sh --dataset {webqsp|cwq} --rog-predictions FILE [options]

Required:
  --rog-predictions FILE  RoG predictions.jsonl needed by reason/main.py

Options:
  --data-root DIR         Dataset/embedding root (default: /data)
  --output-root DIR       Checkpoints and predictions (default: ./runs)
  --model NAME            vLLM/Hugging Face or API model name
  --epochs N              Override retriever epochs
  --device DEVICE         Retriever torch device (default: cuda:0, or cpu)
  --skip-preprocess       Reuse processed data and embeddings
  --skip-train            Reuse --checkpoint (requires both options)
  --checkpoint FILE       Existing cpt.pth for --skip-train
  -h, --help              Show this help

The data tree is DATA_ROOT/<dataset>/{processed,emb/bge}. For API models,
export OPENAI_API_KEY (GPT) or DASHSCOPE_API_KEY (Qwen) before running.
EOF
}

while (($#)); do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --rog-predictions) ROG_PREDICTIONS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$DATASET" == webqsp || "$DATASET" == cwq ]] || { echo "Unsupported dataset: $DATASET" >&2; exit 2; }
[[ -n "$ROG_PREDICTIONS" ]] || { echo "--rog-predictions is required" >&2; exit 2; }
[[ -f "$ROG_PREDICTIONS" ]] || { echo "RoG predictions not found: $ROG_PREDICTIONS" >&2; exit 2; }
if ((SKIP_TRAIN)) && [[ -z "$CHECKPOINT" ]]; then
  echo "--skip-train requires --checkpoint" >&2; exit 2
fi

export PARALLAX_DATA_ROOT="$DATA_ROOT"
RUN_DIR="$OUTPUT_ROOT/$DATASET"
TRAIN_DIR="$RUN_DIR/retriever"
SCORE_FILE="$RUN_DIR/retrieval_result.pth"
REASON_DIR="$RUN_DIR/reason"
mkdir -p "$RUN_DIR"

if ((!SKIP_PREPROCESS)); then
  python -m supervise.preprocess_retriever --dataset_name "$DATASET" --build_pkl
fi

if ((!SKIP_TRAIN)); then
  train_args=(--dataset "$DATASET" --output-dir "$TRAIN_DIR")
  [[ -n "$EPOCHS" ]] && train_args+=(--epochs "$EPOCHS")
  [[ -n "$DEVICE" ]] && train_args+=(--device "$DEVICE")
  python -m supervise.train_retriever "${train_args[@]}"
  CHECKPOINT="$TRAIN_DIR/cpt.pth"
fi
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 1; }

infer_args=(--path "$CHECKPOINT" --split test --output "$SCORE_FILE")
[[ -n "$DEVICE" ]] && infer_args+=(--device "$DEVICE")
python -m supervise.inference "${infer_args[@]}"

python -m reason.main \
  --dataset_name "$DATASET" \
  --split test \
  --score_dict_path "$SCORE_FILE" \
  --rog-predictions "$ROG_PREDICTIONS" \
  --model_name "$MODEL" \
  --output-dir "$REASON_DIR"

printf '\nExperiment complete.\n  checkpoint: %s\n  retrieval:  %s\n  reasoning:  %s\n' \
  "$CHECKPOINT" "$SCORE_FILE" "$REASON_DIR"
