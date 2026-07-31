#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <images_root> <model_dir> <queries.txt> <retrieval.txt> [extra experiments/aachen.py args]" >&2
  echo "Env: PYTHON_BIN, MATCHER, WEIGHTS, PARAMS, RESIZE_W, RESIZE_H, TOP_K, ASSOCIATION_RADIUS, RANSAC_THRESHOLD, MAX_QUERIES, DEVICE, OUTPUT_PREFIX" >&2
}

if [[ $# -lt 4 ]]; then
  usage
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
IMAGES_ROOT="$1"
MODEL_DIR="$2"
QUERIES_FILE="$3"
RETRIEVAL_FILE="$4"
shift 4

MATCHER="${MATCHER:-eloftr}"
WEIGHTS="${WEIGHTS:-}"
PARAMS="${PARAMS:-calibration/params.json}"
RESIZE_W="${RESIZE_W:-1184}"
RESIZE_H="${RESIZE_H:-896}"
TOP_K="${TOP_K:-50}"
ASSOCIATION_RADIUS="${ASSOCIATION_RADIUS:-4}"
RANSAC_THRESHOLD="${RANSAC_THRESHOLD:-12}"
MAX_QUERIES="${MAX_QUERIES:-}"
DEVICE="${DEVICE:-}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/aachen}"

args=(
  experiments/aachen.py
  --images "$IMAGES_ROOT"
  --model "$MODEL_DIR"
  --queries "$QUERIES_FILE"
  --retrieval "$RETRIEVAL_FILE"
  --matcher "$MATCHER"
  --params "$PARAMS"
  --resize "$RESIZE_W" "$RESIZE_H"
  --top-k "$TOP_K"
  --association-radius "$ASSOCIATION_RADIUS"
  --ransac-threshold "$RANSAC_THRESHOLD"
  --output-prefix "$OUTPUT_PREFIX"
)

if [[ -n "$WEIGHTS" ]]; then
  args+=(--weights "$WEIGHTS")
fi

if [[ -n "$MAX_QUERIES" ]]; then
  args+=(--max-queries "$MAX_QUERIES")
fi

if [[ -n "$DEVICE" ]]; then
  args+=(--device "$DEVICE")
fi

exec "$PYTHON_BIN" "${args[@]}" "$@"
