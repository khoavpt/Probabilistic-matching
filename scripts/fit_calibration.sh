#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <dataset-root> [extra calibration/fit.py args]" >&2
  echo "Env: MATCHERS, DATASET, SPLIT, OUTPUT, OUTPUT_DIR, PARAMS_JSON, PYTHON_BIN" >&2
  echo "     IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES, NUM_WORKERS, DEVICE, WEIGHTS" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE="$1"
shift

PYTHON_BIN="${PYTHON_BIN:-python}"
MATCHERS="${MATCHERS:-eloftr}"
DATASET="${DATASET:-megadepth}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/calibration}"
IMAGE_WIDTH="${IMAGE_WIDTH:-1280}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-960}"
SAMPLES="${SAMPLES:-200}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GATE_PENALTY="${GATE_PENALTY:-0.15}"
MAX_ITERATIONS="${MAX_ITERATIONS:-120}"

read -r -a matcher_list <<< "$MATCHERS"

for matcher in "${matcher_list[@]}"; do
  if [[ ${#matcher_list[@]} -eq 1 && -n "${OUTPUT:-}" ]]; then
    output="$OUTPUT"
  else
    output="$OUTPUT_DIR/${matcher}.json"
  fi

  args=(
    calibration/fit.py
    --matcher "$matcher"
    --output "$output"
    --width "$IMAGE_WIDTH"
    --height "$IMAGE_HEIGHT"
    --samples "$SAMPLES"
    --num-workers "$NUM_WORKERS"
    --gate-penalty "$GATE_PENALTY"
    --max-iterations "$MAX_ITERATIONS"
  )

  args+=(--data-root "$SOURCE" --dataset "$DATASET" --split "$SPLIT")

  if [[ -n "${PARAMS_JSON:-}" ]]; then
    args+=(--params-json "$PARAMS_JSON")
  fi
  if [[ -n "${DEVICE:-}" ]]; then
    args+=(--device "$DEVICE")
  fi
  if [[ -n "${WEIGHTS:-}" ]]; then
    args+=(--weights "$WEIGHTS")
  fi

  "$PYTHON_BIN" "${args[@]}" "$@"
done
