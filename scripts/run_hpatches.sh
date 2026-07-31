#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <hpatches_root> [extra experiments/hpatches.py args]" >&2
  echo "Env: PYTHON_BIN, MATCHER, WEIGHTS, PARAMS, ESTIMATOR, METHODS, RESIZE_W, RESIZE_H, THRESHOLD, MAX_PAIRS, DEVICE, OUTPUT" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="$1"
shift

MATCHER="${MATCHER:-eloftr}"
WEIGHTS="${WEIGHTS:-}"
PARAMS="${PARAMS:-calibration/params.json}"
ESTIMATOR="${ESTIMATOR:-ransac}"
METHODS="${METHODS:-baseline,core}"
RESIZE_W="${RESIZE_W:-1280}"
RESIZE_H="${RESIZE_H:-960}"
THRESHOLD="${THRESHOLD:-2.0}"
MAX_PAIRS="${MAX_PAIRS:-}"
DEVICE="${DEVICE:-}"
OUTPUT="${OUTPUT:-outputs/hpatches.json}"

args=(
  experiments/hpatches.py
  --data-root "$DATA_ROOT"
  --matcher "$MATCHER"
  --params "$PARAMS"
  --estimator "$ESTIMATOR"
  --methods "$METHODS"
  --resize "$RESIZE_W" "$RESIZE_H"
  --threshold "$THRESHOLD"
  --output "$OUTPUT"
)

if [[ -n "$WEIGHTS" ]]; then
  args+=(--weights "$WEIGHTS")
fi

if [[ -n "$MAX_PAIRS" ]]; then
  args+=(--max-pairs "$MAX_PAIRS")
fi

if [[ -n "$DEVICE" ]]; then
  args+=(--device "$DEVICE")
fi

exec "$PYTHON_BIN" "${args[@]}" "$@"
