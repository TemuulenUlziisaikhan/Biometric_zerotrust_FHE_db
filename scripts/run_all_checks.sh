#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PY="./venv/bin/python"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ ! -x "$VENV_PY" ]]; then
  echo "[ERROR] venv python not found at $VENV_PY"
  echo "Create it with: python3 -m venv venv && ./venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

echo "[1/6] Installing/updating dependencies in venv"
"$VENV_PY" -m pip install -r requirements.txt >/dev/null

echo "[2/6] Resetting local RocksDB test data"
rm -rf server/biometric.rocks

echo "[3/6] Starting API server"
PYTHONPATH=. "$VENV_PY" -m uvicorn server.main_server:app --host 127.0.0.1 --port 8000 >/tmp/fhe_db_server.log 2>&1 &
SERVER_PID=$!

# Wait for readiness by polling /health
for _ in {1..60}; do
  if "$VENV_PY" - <<'PY' >/dev/null 2>&1
import requests
requests.get("http://127.0.0.1:8000/health", timeout=1).raise_for_status()
PY
  then
    break
  fi
  sleep 0.5
done

if ! "$VENV_PY" - <<'PY' >/dev/null 2>&1
import requests
requests.get("http://127.0.0.1:8000/health", timeout=1).raise_for_status()
PY
then
  echo "[ERROR] Server did not become healthy. See /tmp/fhe_db_server.log"
  exit 1
fi

echo "[4/6] Running smoke test"
"$VENV_PY" scripts/smoke_test.py

echo "[5/6] Running integration test"
"$VENV_PY" scripts/integration_test.py

echo "[6/6] Running security check"
"$VENV_PY" scripts/security_check.py

echo "[OK] All checks passed"
