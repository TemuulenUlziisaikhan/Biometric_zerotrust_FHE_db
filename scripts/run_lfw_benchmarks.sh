#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${PYTHONPATH:-.}"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
LFW_ROOT="${LFW_ROOT:-}"
PAIRS_FILE="${PAIRS_FILE:-}"
MODEL_PATH="${MODEL_PATH:-}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8000}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results/lfw}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
MAX_PAIRS="${MAX_PAIRS:-}"
MAX_IDENTITIES="${MAX_IDENTITIES:-}"
ITERATIONS="${ITERATIONS:-}"
WORKERS="${WORKERS:-}"
USE_KAGGLEHUB="${USE_KAGGLEHUB:-0}"
KAGGLE_DATASET="${KAGGLE_DATASET:-jessicali9530/lfw-dataset}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH environment variable is required"
  echo "Example: MODEL_PATH=models/arcface.onnx LFW_ROOT=/data/lfw-funneled scripts/run_lfw_benchmarks.sh"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: python interpreter not found at $PYTHON_BIN"
  exit 1
fi

if [[ -z "$LFW_ROOT" ]]; then
  if [[ "$USE_KAGGLEHUB" == "1" ]]; then
    echo "[prep] Resolving LFW dataset via kagglehub: $KAGGLE_DATASET"
    LFW_ROOT="$($PYTHON_BIN scripts/fetch_lfw_kaggle.py --dataset "$KAGGLE_DATASET" --print-root)"
    echo "[prep] Using LFW_ROOT=$LFW_ROOT"
  else
    echo "ERROR: set LFW_ROOT or set USE_KAGGLEHUB=1 for auto-download"
    exit 1
  fi
fi

if [[ -z "$PAIRS_FILE" ]]; then
  if [[ -f "$LFW_ROOT/pairs.txt" ]]; then
    PAIRS_FILE="$LFW_ROOT/pairs.txt"
  elif [[ -f "$ROOT_DIR/models/pairs.csv" ]]; then
    PAIRS_FILE="$ROOT_DIR/models/pairs.csv"
  elif [[ -f "$ROOT_DIR/models/pairs.txt" ]]; then
    PAIRS_FILE="$ROOT_DIR/models/pairs.txt"
  else
    PAIRS_FILE="$LFW_ROOT/pairs.txt"
  fi
fi

if [[ ! -d "$LFW_ROOT" ]]; then
  echo "ERROR: LFW_ROOT directory not found: $LFW_ROOT"
  exit 1
fi

if [[ ! -f "$PAIRS_FILE" ]]; then
  echo "ERROR: PAIRS_FILE not found: $PAIRS_FILE"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH file not found: $MODEL_PATH"
  exit 1
fi

if pgrep -f "uvicorn server.main_server:app" >/dev/null; then
  echo "ERROR: Existing server process detected. Stop it before running benchmarks."
  exit 1
fi

rm -rf server/biometric.rocks

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[1/4] Starting server with venv python"
"$PYTHON_BIN" -m uvicorn server.main_server:app --host 127.0.0.1 --port 8000 >"$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!

for _ in {1..40}; do
  if curl -sf "$SERVER_URL/health" >/dev/null; then
    break
  fi
  sleep 0.5
done

if ! curl -sf "$SERVER_URL/health" >/dev/null; then
  echo "ERROR: server did not become healthy"
  exit 1
fi

echo "[2/4] Running LFW official/custom pairs verification metrics"
PAIRS_ARGS=()
if [[ -n "$MAX_PAIRS" ]]; then
  PAIRS_ARGS+=(--max-pairs "$MAX_PAIRS")
fi

"$PYTHON_BIN" scripts/lfw_eval_pairs.py \
  --lfw-root "$LFW_ROOT" \
  --pairs-file "$PAIRS_FILE" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR/pairs" \
  --progress-every "$PROGRESS_EVERY" \
  "${PAIRS_ARGS[@]}"

echo "[3/4] Running LFW system benchmark (enroll/auth + latency)"
SYSTEM_ARGS=()
if [[ -n "$MAX_IDENTITIES" ]]; then
  SYSTEM_ARGS+=(--max-identities "$MAX_IDENTITIES")
fi
if [[ -n "$ITERATIONS" ]]; then
  SYSTEM_ARGS+=(--iterations "$ITERATIONS")
fi
if [[ -n "$WORKERS" ]]; then
  SYSTEM_ARGS+=(--workers "$WORKERS")
fi

"$PYTHON_BIN" scripts/lfw_benchmark_system.py \
  --lfw-root "$LFW_ROOT" \
  --model-path "$MODEL_PATH" \
  --server-url "$SERVER_URL" \
  --output-dir "$OUTPUT_DIR/system" \
  --progress-every "$PROGRESS_EVERY" \
  "${SYSTEM_ARGS[@]}"

echo "[4/4] Benchmark run complete"
echo "Artifacts written under: $OUTPUT_DIR"
