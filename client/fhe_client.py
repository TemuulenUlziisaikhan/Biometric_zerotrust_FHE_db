from __future__ import annotations

import base64
from pathlib import Path

import time
import numpy as np
import tenseal as ts

POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
GLOBAL_SCALE = 2**40
FACE_DIM = 512
FACES_PER_CHUNK = 16
CHUNK_SLOTS = FACE_DIM * FACES_PER_CHUNK
DISTANCE_TOLERANCE = 1e-3
DEFAULT_SHARED_CONTEXT_PATH = "client/.shared_context/ckks_context_with_secret.bin"


def create_ckks_context() -> ts.Context:
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    context.global_scale = GLOBAL_SCALE
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


def serialize_eval_context_without_secret(context: ts.Context) -> str:
    raw = context.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=True,
        save_relin_keys=True,
    )
    return base64.b64encode(raw).decode("utf-8")


def serialize_context_with_secret(context: ts.Context) -> bytes:
    return context.serialize(
        save_public_key=True,
        save_secret_key=True,
        save_galois_keys=True,
        save_relin_keys=True,
    )


def load_or_create_shared_context(context_path: str = DEFAULT_SHARED_CONTEXT_PATH) -> ts.Context:
    path = Path(context_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raw = path.read_bytes()
        return ts.context_from(raw)

    context = create_ckks_context()
    path.write_bytes(serialize_context_with_secret(context))
    return context


def load_or_create_shared_eval_context_b64(
    context: ts.Context,
    context_path: str = DEFAULT_SHARED_CONTEXT_PATH,
) -> str:
    eval_path = Path(context_path).with_suffix(".eval.b64")
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    if eval_path.exists():
        return eval_path.read_text(encoding="utf-8").strip()

    eval_b64 = serialize_eval_context_without_secret(context)
    eval_path.write_text(eval_b64, encoding="utf-8")
    return eval_b64


def build_sparse_enrollment_vector(embedding_512: np.ndarray, slot_offset: int) -> np.ndarray:
    emb = np.asarray(embedding_512, dtype=np.float64).reshape(-1)
    if emb.shape[0] != FACE_DIM:
        raise ValueError("Embedding must contain exactly 512 floats")
    if slot_offset < 0 or slot_offset >= FACES_PER_CHUNK:
        raise ValueError("slot_offset must be between 0 and 15")
    arr = np.zeros(CHUNK_SLOTS, dtype=np.float64)
    start = slot_offset * FACE_DIM
    end = start + FACE_DIM
    arr[start:end] = emb
    return arr


def build_replicated_probe_vector(embedding_512: np.ndarray) -> np.ndarray:
    emb = np.asarray(embedding_512, dtype=np.float64).reshape(-1)
    if emb.shape[0] != FACE_DIM:
        raise ValueError("Embedding must contain exactly 512 floats")
    return np.tile(emb, FACES_PER_CHUNK)


def encrypt_vector(context: ts.Context, vector_8192: np.ndarray) -> str:
    time_watch = time.time()
    arr = np.asarray(vector_8192, dtype=np.float64).reshape(-1)
    if arr.shape[0] != CHUNK_SLOTS:
        raise ValueError("Input vector must have exactly 8192 slots")
    encrypted = ts.ckks_vector(context, arr.tolist()).serialize()
    print(f"Encryption Time: {time.time() - time_watch:.2f} seconds")
    return base64.b64encode(encrypted).decode("utf-8")


def decrypt_distance_vector(context: ts.Context, encrypted_distance_b64: str) -> np.ndarray:
    raw = base64.b64decode(encrypted_distance_b64.encode("utf-8"))
    ct = ts.ckks_vector_from(context, raw)
    values = ct.decrypt()
    return np.asarray(values, dtype=np.float64)


def parse_distances_by_chunk_slots(distance_vector_8192: np.ndarray, face_count: int) -> list[float]:
    vec = np.asarray(distance_vector_8192, dtype=np.float64).reshape(-1)
    if vec.shape[0] != CHUNK_SLOTS:
        raise ValueError("Decrypted distance vector must have 8192 slots")
    if face_count < 0 or face_count > FACES_PER_CHUNK:
        raise ValueError("face_count must be between 0 and 16")
    distances: list[float] = []
    for idx in range(face_count):
        start = idx * FACE_DIM
        end = start + FACE_DIM
        distance = float(np.sum(vec[start:end]))
        if distance < -DISTANCE_TOLERANCE:
            raise ValueError(
                f"Encountered strongly negative squared distance {distance}; CKKS precision tolerance exceeded"
            )
        distances.append(max(distance, 0.0))
    return distances


def select_best_uuid(distance_rows: list[dict], threshold: float) -> str | None:
    best_uuid: str | None = None
    best_dist = float("inf")
    for row in distance_rows:
        uid = row["uuid"]
        dist = float(row["distance"])
        if dist < best_dist:
            best_dist = dist
            best_uuid = uid
    if best_uuid is None:
        return None
    return best_uuid if best_dist <= threshold else None