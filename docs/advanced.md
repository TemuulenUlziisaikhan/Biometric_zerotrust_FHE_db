# Advanced Guide

This document collects deeper operational and evaluation workflows that are intentionally kept out of the main README.

## Architecture Notes

- Python 3.13 only.
- TenSEAL CKKS for encrypted vector operations.
- RocksDB (via `rocksdict`) for chunked encrypted template storage.
- FastAPI server + web client UI.
- LSH-based retrieval (`Bucket_<bits>`) with multiprobe + decoys.

## Data Model (Chunk-Oriented)

Each chunk stores up to `FACES_PER_CHUNK=16` templates.

Core chunk fields:

- `bucket_id`
- `packed_ciphertext`
- `current_face_count`
- `uuid_map`
- `face_ciphertexts` (newer records)
- `context_fingerprint` (set when context is bound)

Why both ciphertext forms exist:

- `packed_ciphertext`: fast auth-time distance eval.
- `face_ciphertexts`: per-label rebuild/delete safety.

## Context Compatibility

- Server/client `LSH_BITS` must match.
- Encrypted chunk operations depend on matching CKKS context.
- `context_fingerprint` prevents mixing incompatible contexts.

If context is rotated intentionally, reset DB and client shared context together.

## Local Validation Workflows

With server running:

```bash
./venv/bin/python scripts/smoke_test.py
./venv/bin/python scripts/integration_test.py
./venv/bin/python scripts/security_check.py
```

Full local validation script:

```bash
./scripts/run_all_checks.sh
```

## LFW Benchmarks

Run pair-eval + system benchmark pipeline:

```bash
LFW_ROOT=/path/to/lfw-funneled MODEL_PATH=/path/to/model.onnx ./scripts/run_lfw_benchmarks.sh
```

Kaggle auto-fetch mode:

```bash
USE_KAGGLEHUB=1 MODEL_PATH=/path/to/model.onnx ./scripts/run_lfw_benchmarks.sh
```

Outputs are written under `benchmark_results/lfw`.

## LFW Population Options

CLI population script:

```bash
./venv/bin/python scripts/lfw_populate_db.py --lfw-root /path/to/lfw-funneled --model-path /path/to/model.onnx --server-url http://127.0.0.1:8000
```

Optional flags include:

- `--max-images`
- `--state-file`
- `--no-resume`

Web Admin population supports:

- live progress
- pause/resume
- per-identity cap via `LFW_MAX_PER_IDENTITY`

## Persistence Details

Docker Compose volumes:

- `server_db` -> `/app/server/biometric.rocks`
- `client_context` -> `/app/client/.shared_context`

Data persists across stop/start and rebuild. It is removed when volumes are removed, e.g. `docker compose down -v`.

## Troubleshooting

- `INVALID_BUCKET_ID`: mismatch in `LSH_BITS` between client/server.
- Frequent `CONTEXT_MISMATCH`: contexts are inconsistent across enroll/auth/delete.
- Heavy DB viewer pages: reduce `page_size`, use label filtering.
- Docker exit code `137`: memory pressure/OOM; reduce payload sizes and page size.
