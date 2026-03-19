#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8000}"
MODEL_PATH="${MODEL_PATH:-}"
CANARY_FILE="${CANARY_FILE:-scripts/noise_canary.json}"
OUTPUT_JSON="${OUTPUT_JSON:-benchmark_results/noise_health/latest_noise_health.json}"
BASELINE_JSON="${BASELINE_JSON:-benchmark_results/noise_health/baseline_noise_health.json}"
DISTANCE_DRIFT_THRESHOLD="${DISTANCE_DRIFT_THRESHOLD:-0.20}"
MAX_DRIFTED_PROBE_RATIO="${MAX_DRIFTED_PROBE_RATIO:-0.25}"
MIN_MATCH_RATE="${MIN_MATCH_RATE:-0.90}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required"
  exit 2
fi

if [[ ! -f "$CANARY_FILE" ]]; then
  echo "CANARY_FILE not found: $CANARY_FILE"
  exit 2
fi

"$PYTHON_BIN" scripts/noise_health_check.py \
  --server-url "$SERVER_URL" \
  --model-path "$MODEL_PATH" \
  --canary-file "$CANARY_FILE" \
  --output-json "$OUTPUT_JSON" \
  --baseline-json "$BASELINE_JSON" \
  --distance-drift-threshold "$DISTANCE_DRIFT_THRESHOLD" \
  --max-drifted-probe-ratio "$MAX_DRIFTED_PROBE_RATIO" \
  --min-match-rate "$MIN_MATCH_RATE"
