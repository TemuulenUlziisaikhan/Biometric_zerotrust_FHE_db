from __future__ import annotations

import csv
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import tenseal as ts

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from client.fhe_client import (  # noqa: E402
    CHUNK_SLOTS,
    COEFF_MOD_BIT_SIZES,
    FACE_DIM,
    GLOBAL_SCALE,
    POLY_MODULUS_DEGREE,
    build_replicated_probe_vector,
    build_sparse_enrollment_vector,
    create_ckks_context,
    decrypt_distance_vector,
    encrypt_vector,
    serialize_eval_context_without_secret,
)
from server.database import Database  # noqa: E402
from server.fhe_server import (  # noqa: E402
    deserialize_ciphertext,
    homomorphic_enroll_add,
    homomorphic_squared_distance,
    serialize_ciphertext,
)

BENCHMARK_SEED = 20260325
MICRO_ITERATIONS = 100
DB_SIZES = (1000, 5000, 10000)
LSH_CHUNK_COUNT = 4
TEMP_DB_PATH = REPO_ROOT / "server" / "test_benchmark.rocks"
CSV_OUTPUT_PATH = REPO_ROOT / "benchmark_results.csv"


def _set_deterministic_seed(seed: int = BENCHMARK_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _measure_average_ms(fn: Callable[[], object], iterations: int = MICRO_ITERATIONS) -> tuple[float, float]:
    fn()
    timings_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    arr = np.asarray(timings_ms, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def _measure_once_ms(fn: Callable[[], object]) -> float:
    started = time.perf_counter()
    fn()
    return (time.perf_counter() - started) * 1000.0


def _random_embedding(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(0.0, 1.0, size=(FACE_DIM,)).astype(np.float64)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return _random_embedding(rng)
    return vec / norm


def _bucket_id(index: int, bits: int = 9) -> str:
    suffix = format(index % (1 << bits), f"0{bits}b")
    return f"Bucket_{suffix}"


def _cleanup_temp_db(db_path: Path) -> None:
    if db_path.exists():
        shutil.rmtree(db_path, ignore_errors=True)


def _build_context_without_keys() -> ts.Context:
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    context.global_scale = GLOBAL_SCALE
    return context


def _manual_key_generation_once() -> None:
    context = _build_context_without_keys()
    context.generate_galois_keys()
    context.generate_relin_keys()


def _populate_mock_db(db: Database, n_chunks: int, packed_ciphertext_b64: str) -> tuple[str, list[str]]:
    target_chunk_key = db.make_chunk_key(_bucket_id(0), 1)
    decoy_chunk_keys: list[str] = []

    for i in range(n_chunks):
        bucket_id = _bucket_id(i)
        chunk_index = (i // (1 << 9)) + 1
        chunk_key = db.make_chunk_key(bucket_id, chunk_index)
        record = {
            "bucket_id": chunk_key,
            "packed_ciphertext": packed_ciphertext_b64,
            "current_face_count": 1,
            "uuid_map": [f"user_{i:06d}"],
            "face_ciphertexts": [packed_ciphertext_b64],
        }
        db.write_chunk_record(chunk_key, record)

        if i in {1, 2, 3}:
            decoy_chunk_keys.append(chunk_key)

    if len(decoy_chunk_keys) < 3:
        all_keys = db.all_chunk_keys()
        for key in all_keys:
            if key != target_chunk_key and key not in decoy_chunk_keys:
                decoy_chunk_keys.append(key)
            if len(decoy_chunk_keys) == 3:
                break

    return target_chunk_key, decoy_chunk_keys[:3]


def _linear_scan_time_ms(db: Database, context: ts.Context, probe_ct: ts.CKKSVector) -> tuple[float, int]:
    def _run() -> int:
        processed = 0
        for chunk_key in db.all_chunk_keys():
            record = db.read_chunk_record(chunk_key)
            if not record or not record.get("packed_ciphertext"):
                continue
            chunk_ct = deserialize_ciphertext(context, str(record["packed_ciphertext"]))
            _ = homomorphic_squared_distance(chunk_ct, probe_ct)
            processed += 1
        return processed

    started = time.perf_counter()
    processed_chunks = _run()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, processed_chunks


def _lsh_scan_time_ms(
    db: Database,
    context: ts.Context,
    probe_ct: ts.CKKSVector,
    target_chunk_key: str,
    decoy_chunk_keys: list[str],
) -> tuple[float, int]:
    selected_keys = [target_chunk_key, *decoy_chunk_keys[:3]]
    selected_keys = selected_keys[:LSH_CHUNK_COUNT]

    def _run() -> int:
        processed = 0
        for chunk_key in selected_keys:
            record = db.read_chunk_record(chunk_key)
            if not record or not record.get("packed_ciphertext"):
                continue
            chunk_ct = deserialize_ciphertext(context, str(record["packed_ciphertext"]))
            _ = homomorphic_squared_distance(chunk_ct, probe_ct)
            processed += 1
        return processed

    started = time.perf_counter()
    processed_chunks = _run()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, processed_chunks


def _size_overheads(context: ts.Context, packed_ciphertext_b64: str) -> dict:
    rng = np.random.default_rng(BENCHMARK_SEED + 111)
    plaintext_embedding = _random_embedding(rng).astype(np.float32)

    eval_context_b64 = serialize_eval_context_without_secret(context)

    return {
        "plaintext_512_nbytes": int(plaintext_embedding.nbytes),
        "plaintext_512_sys_getsizeof": int(sys.getsizeof(plaintext_embedding)),
        "ciphertext_chunk_b64_bytes": int(len(packed_ciphertext_b64.encode("utf-8"))),
        "eval_context_b64_bytes": int(len(eval_context_b64.encode("utf-8"))),
    }


def main() -> None:
    _set_deterministic_seed(BENCHMARK_SEED)

    print("=" * 88)
    print("Sub-Linear FHE Biometric Benchmark")
    print("=" * 88)
    print(f"Seed: {BENCHMARK_SEED}")
    print(f"Microbenchmark iterations per metric: {MICRO_ITERATIONS}")
    print(f"Database sizes tested: {DB_SIZES}")
    print(f"Temporary benchmark DB: {TEMP_DB_PATH}")
    print("=" * 88)

    rows: list[dict[str, object]] = []

    rng = np.random.default_rng(BENCHMARK_SEED)
    enrollment_embedding = _random_embedding(rng)
    auth_embedding = _random_embedding(rng)

    print("\n[1/4] Building CKKS context and benchmark fixtures...")
    context = create_ckks_context()
    sparse_vector = build_sparse_enrollment_vector(enrollment_embedding, slot_offset=0)
    probe_vector = build_replicated_probe_vector(auth_embedding)

    sparse_ciphertext_b64 = encrypt_vector(context, sparse_vector)
    chunk_ct = deserialize_ciphertext(context, sparse_ciphertext_b64)
    probe_ciphertext_b64 = encrypt_vector(context, probe_vector)
    probe_ct = deserialize_ciphertext(context, probe_ciphertext_b64)
    distance_ct = homomorphic_squared_distance(chunk_ct, probe_ct)
    distance_b64 = serialize_ciphertext(distance_ct)

    print("\n[2/4] Running microbenchmarks (average of 100 iterations each)...")

    context_create_mean, context_create_std = _measure_average_ms(create_ckks_context)
    rows.append(
        {
            "section": "microbenchmark",
            "metric": "create_ckks_context_ms",
            "group": "context",
            "value": context_create_mean,
            "stddev": context_create_std,
            "unit": "ms",
            "notes": "Includes CKKS context init + galois/relin key generation",
        }
    )

    keygen_mean, keygen_std = _measure_average_ms(_manual_key_generation_once)
    rows.append(
        {
            "section": "microbenchmark",
            "metric": "key_generation_only_ms",
            "group": "context",
            "value": keygen_mean,
            "stddev": keygen_std,
            "unit": "ms",
            "notes": "Manual keygen on raw CKKS context for isolation",
        }
    )

    encrypt_sparse_mean, encrypt_sparse_std = _measure_average_ms(lambda: encrypt_vector(context, sparse_vector))
    rows.append(
        {
            "section": "microbenchmark",
            "metric": "encrypt_vector_sparse_ms",
            "group": "client",
            "value": encrypt_sparse_mean,
            "stddev": encrypt_sparse_std,
            "unit": "ms",
            "notes": "8192-slot sparse enrollment vector",
        }
    )

    encrypt_repl_mean, encrypt_repl_std = _measure_average_ms(lambda: encrypt_vector(context, probe_vector))
    rows.append(
        {
            "section": "microbenchmark",
            "metric": "encrypt_vector_replicated_ms",
            "group": "client",
            "value": encrypt_repl_mean,
            "stddev": encrypt_repl_std,
            "unit": "ms",
            "notes": "8192-slot replicated authentication vector",
        }
    )

    enroll_add_mean, enroll_add_std = _measure_average_ms(lambda: homomorphic_enroll_add(chunk_ct, chunk_ct))
    rows.append(
        {
            "section": "microbenchmark",
            "metric": "homomorphic_enroll_add_ms",
            "group": "server",
            "value": enroll_add_mean,
            "stddev": enroll_add_std,
            "unit": "ms",
            "notes": "SIMD chunk append (ciphertext + ciphertext)",
        }
    )

    sqdist_mean, sqdist_std = _measure_average_ms(lambda: homomorphic_squared_distance(chunk_ct, probe_ct))
    rows.append(
        {
            "section": "microbenchmark",
            "metric": "homomorphic_squared_distance_ms",
            "group": "server",
            "value": sqdist_mean,
            "stddev": sqdist_std,
            "unit": "ms",
            "notes": "D^2=(x-y)^2 over 8192 encrypted slots",
        }
    )

    decrypt_mean, decrypt_std = _measure_average_ms(lambda: decrypt_distance_vector(context, distance_b64))
    rows.append(
        {
            "section": "microbenchmark",
            "metric": "decrypt_distance_vector_ms",
            "group": "client",
            "value": decrypt_mean,
            "stddev": decrypt_std,
            "unit": "ms",
            "notes": "Includes ciphertext deserialize + CKKS decrypt",
        }
    )

    print("\n[3/4] Running scalability proof benchmarks (O(N) vs O(K=4))...")
    _cleanup_temp_db(TEMP_DB_PATH)
    db = Database(db_path=str(TEMP_DB_PATH))

    for n_chunks in DB_SIZES:
        cleared = db.clear_all_chunk_records()
        print(f"  - Preparing mock DB for N={n_chunks} chunks (cleared {cleared} previous chunks)")
        target_chunk_key, decoy_chunk_keys = _populate_mock_db(db, n_chunks, sparse_ciphertext_b64)

        linear_ms, linear_processed = _linear_scan_time_ms(db, context, probe_ct)
        lsh_ms, lsh_processed = _lsh_scan_time_ms(db, context, probe_ct, target_chunk_key, decoy_chunk_keys)

        speedup = linear_ms / max(lsh_ms, 1e-9)

        rows.extend(
            [
                {
                    "section": "scalability",
                    "metric": "linear_scan_time_ms",
                    "group": f"N={n_chunks}",
                    "value": linear_ms,
                    "stddev": 0.0,
                    "unit": "ms",
                    "notes": f"Processed {linear_processed} chunks with homomorphic_squared_distance",
                },
                {
                    "section": "scalability",
                    "metric": "lsh_scan_time_ms",
                    "group": f"N={n_chunks}",
                    "value": lsh_ms,
                    "stddev": 0.0,
                    "unit": "ms",
                    "notes": f"Processed {lsh_processed} chunks (1 target + 3 decoys)",
                },
                {
                    "section": "scalability",
                    "metric": "speedup_linear_over_lsh",
                    "group": f"N={n_chunks}",
                    "value": speedup,
                    "stddev": 0.0,
                    "unit": "x",
                    "notes": "Higher is better; demonstrates sub-linear query cost",
                },
            ]
        )

    print("\n[4/4] Measuring storage and payload overhead...")
    overhead = _size_overheads(context, sparse_ciphertext_b64)
    for key, value in overhead.items():
        rows.append(
            {
                "section": "overhead",
                "metric": key,
                "group": "sizes",
                "value": float(value),
                "stddev": 0.0,
                "unit": "bytes",
                "notes": "payload size metric",
            }
        )

    with CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "metric", "group", "value", "stddev", "unit", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 88)
    print("Benchmark Complete")
    print("=" * 88)
    print("Microbenchmarks:")
    for row in rows:
        if row["section"] == "microbenchmark":
            print(f"  {row['metric']:<38} {float(row['value']):>10.3f} {row['unit']}")

    print("\nScalability (Linear vs LSH):")
    for n_chunks in DB_SIZES:
        grp = f"N={n_chunks}"
        linear_row = next(r for r in rows if r["section"] == "scalability" and r["group"] == grp and r["metric"] == "linear_scan_time_ms")
        lsh_row = next(r for r in rows if r["section"] == "scalability" and r["group"] == grp and r["metric"] == "lsh_scan_time_ms")
        speedup_row = next(r for r in rows if r["section"] == "scalability" and r["group"] == grp and r["metric"] == "speedup_linear_over_lsh")
        print(
            f"  {grp:<8} linear={float(linear_row['value']):>10.3f} ms  "
            f"lsh(K=4)={float(lsh_row['value']):>10.3f} ms  "
            f"speedup={float(speedup_row['value']):>8.2f}x"
        )

    print("\nStorage / Payload Overhead:")
    for row in rows:
        if row["section"] == "overhead":
            print(f"  {row['metric']:<38} {int(float(row['value'])):>10d} bytes")

    print("\nArtifacts:")
    print(f"  CSV: {CSV_OUTPUT_PATH}")
    print(f"  Temporary RocksDB used: {TEMP_DB_PATH}")
    print("=" * 88)


if __name__ == "__main__":
    main()
