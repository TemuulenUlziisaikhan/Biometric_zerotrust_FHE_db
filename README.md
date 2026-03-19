# Biometric_zerotrust_FHE_db

Privacy-preserving biometric authentication prototype using TenSEAL CKKS + FastAPI + RocksDB.

## What this project does

- Extracts 512-d face embeddings from images using ArcFace ONNX.
- Encrypts embeddings client-side and stores encrypted templates server-side.
- Uses LSH bucket routing for sub-linear candidate retrieval.
- Provides a web UI for enroll/authenticate/admin/database viewing.

## Main components

- `server/`: FastAPI API + encrypted DB operations
- `client/`: embedding extraction, encryption, web UI
- `docker-compose.yml`: one-command deployment (server + web client)

## Quick start (Docker Compose)

### 1) Prerequisites

- Docker + Docker Compose installed.
- Place your model at `models/arcface.onnx`.
- (Optional) For one-click LFW population in Admin UI, place dataset at `models/lfw-deepfunneled/lfw-deepfunneled`.

### 2) Start services

```bash
docker compose up --build
```

### 3) Open endpoints

- Web UI: `http://127.0.0.1:8500`
- Admin page: `http://127.0.0.1:8500/admin`
- DB viewer: `http://127.0.0.1:8500/database`
- API health: `http://127.0.0.1:8000/health`

## Common usage flow

1. Open Web UI and enroll one user (single or multiple photos).
2. Authenticate with another photo.
3. Use Admin page for:
   - label deletion
   - full DB reset
   - LFW population (with progress + pause/resume)

## Persistence and reset

Compose uses named volumes:

- `server_db` -> `/app/server/biometric.rocks`
- `client_context` -> `/app/client/.shared_context`

This means data persists across container stop/start and image rebuild.

Data is removed only if you remove volumes, e.g.:

```bash
docker compose down -v
```

## Important config notes

- `LSH_BITS` must match on client and server.
- `MODEL_PATH` in compose points to `/models/arcface.onnx`.
- `LFW_ROOT` can be set to point to your mounted LFW directory.
- `LFW_MAX_PER_IDENTITY` controls per-identity image cap for LFW populate.

## Optional local run (without Docker)

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=. ./venv/bin/python -m uvicorn server.main_server:app --host 127.0.0.1 --port 8000
PYTHONPATH=. MODEL_PATH=/path/to/arcface.onnx SERVER_URL=http://127.0.0.1:8000 ./venv/bin/python -m uvicorn client.web_client:app --host 127.0.0.1 --port 8500
```

## Troubleshooting quick tips

- If auth requests fail with `INVALID_BUCKET_ID`, align `LSH_BITS` between server/client.
- If DB viewer is heavy, use pagination and smaller page size.
- If old data becomes unusable after context/key changes, reset DB + client shared context.

## More details

For benchmarks, deeper architecture notes, advanced scripts, and operational details, see [docs/advanced.md](docs/advanced.md).
