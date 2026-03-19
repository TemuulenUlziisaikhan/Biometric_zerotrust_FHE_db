from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.main_client import BiometricClient
from client.lsh_engine import LSHEngine


def _make_client(seed: int = 7, threshold: float = 10.0) -> BiometricClient:
    return BiometricClient(
        server_url="http://127.0.0.1:8000",
        model_path=None,
        threshold=threshold,
        lsh_seed=seed,
    )


def run_winner_test(tmp_dir: Path) -> None:
    rng = np.random.default_rng(42)
    emb_a = rng.standard_normal(512).astype(np.float32)
    emb_b = emb_a + (rng.standard_normal(512).astype(np.float32) * 0.001)

    emb_a_path = tmp_dir / "winner_a.npy"
    emb_b_path = tmp_dir / "winner_b.npy"
    np.save(emb_a_path, emb_a)
    np.save(emb_b_path, emb_b)

    client = _make_client(seed=1701, threshold=10.0)
    client.enroll_user(str(emb_a_path), "usr_smoke_a")
    winner = client.authenticate_user(str(emb_b_path))

    assert winner == "usr_smoke_a", f"Expected usr_smoke_a, got {winner}"


def run_chunk_rollover_test(tmp_dir: Path) -> None:
    rng = np.random.default_rng(99)
    base = rng.standard_normal(512).astype(np.float32)

    enroll_paths: list[Path] = []
    for idx in range(17):
        emb = base.copy()
        emb_path = tmp_dir / f"roll_{idx:02d}.npy"
        np.save(emb_path, emb)
        enroll_paths.append(emb_path)

    probe_path = tmp_dir / "roll_probe.npy"
    np.save(probe_path, base)

    client = _make_client(seed=3101, threshold=10.0)

    responses = []
    for idx, emb_path in enumerate(enroll_paths):
        response = client.enroll_user(str(emb_path), f"usr_roll_{idx:02d}")
        responses.append(response)

    first_16 = responses[:16]
    seventeenth = responses[16]

    assert all(row["chunk_index"] == 1 for row in first_16), "First 16 enrollments should be in chunk 1"
    assert [row["current_face_count"] for row in first_16] == list(range(1, 17)), "Chunk 1 counts should be 1..16"
    assert seventeenth["chunk_index"] == 2, "17th enrollment should roll over to chunk 2"
    assert seventeenth["current_face_count"] == 1, "Chunk 2 should start with count 1"

    winner = client.authenticate_user(str(probe_path))
    assert winner is not None, "Expected non-empty authentication winner"


def run_lsh_bucket_stability_test() -> None:
    engine = LSHEngine(bits=12, seed=77, in_dim=512)
    embedding = np.linspace(-1.0, 1.0, 512, dtype=np.float32)

    bucket_a = engine.hash_embedding_to_bucket(embedding)
    bucket_b = engine.hash_embedding_to_bucket(embedding)
    assert bucket_a == bucket_b, "LSH bucket hashing must be deterministic for same input"

    expected_bucket = "Bucket_000001101001"
    assert bucket_a == expected_bucket, f"Expected fixed test-vector bucket {expected_bucket}, got {bucket_a}"


def run_multiprobe_decoy_contract_test() -> None:
    engine = LSHEngine(bits=12, seed=7, in_dim=512)
    embedding = np.linspace(-0.5, 0.5, 512, dtype=np.float32)
    primary = engine.hash_embedding_to_bucket(embedding)

    multiprobe = engine.generate_multiprobe_buckets(
        primary,
        probes=4,
        strategy="margin",
        embedding=embedding,
        hamming_radius=2,
    )
    decoys = engine.generate_decoy_buckets(exclude={primary, *multiprobe}, count=5)
    request_ids = engine.build_auth_bucket_request(primary, multiprobe, decoys)

    assert len(multiprobe) == 4, "Expected exactly 4 multiprobe buckets"
    assert len(set(multiprobe)) == len(multiprobe), "Multiprobe buckets must be unique"
    assert all(bucket != primary for bucket in multiprobe), "Multiprobe buckets must exclude primary"
    assert len(decoys) == 5, "Expected exactly 5 decoy buckets"
    assert all(bucket not in {primary, *multiprobe} for bucket in decoys), "Decoys must exclude target buckets"
    assert request_ids[0] == primary, "Primary bucket should be first in request list"
    assert len(request_ids) == 1 + len(multiprobe) + len(decoys), "Request list should contain all unique buckets"


def run_threshold_sweep_diagnostics(tmp_dir: Path) -> None:
    rng = np.random.default_rng(2026)

    enroll = rng.standard_normal(512).astype(np.float32)
    positive_probe = enroll + (rng.standard_normal(512).astype(np.float32) * 0.001)
    negative_probe = rng.standard_normal(512).astype(np.float32)

    enroll_path = tmp_dir / "thr_enroll.npy"
    pos_path = tmp_dir / "thr_positive.npy"
    neg_path = tmp_dir / "thr_negative.npy"
    np.save(enroll_path, enroll)
    np.save(pos_path, positive_probe)
    np.save(neg_path, negative_probe)

    client = _make_client(seed=5501, threshold=10.0)
    client.enroll_user(str(enroll_path), "usr_threshold")

    pos_diag = client.authenticate_embedding_diagnostics(np.load(pos_path), threshold=10.0)
    neg_diag = client.authenticate_embedding_diagnostics(np.load(neg_path), threshold=10.0)

    assert pos_diag["best_distance"] is not None, "Expected positive diagnostics to include best_distance"

    thresholds = [0.5, 1.0, 2.0, 5.0, 10.0]
    rows: list[dict] = []
    false_rejects = 0
    false_matches = 0
    for threshold in thresholds:
        pos_winner = client.authenticate_embedding_diagnostics(np.load(pos_path), threshold=threshold)["winner_uuid"]
        neg_winner = client.authenticate_embedding_diagnostics(np.load(neg_path), threshold=threshold)["winner_uuid"]
        is_positive_match = pos_winner == "usr_threshold"
        is_negative_match = neg_winner is not None
        false_rejects += int(not is_positive_match)
        false_matches += int(is_negative_match)
        rows.append(
            {
                "threshold": threshold,
                "positive_match": is_positive_match,
                "negative_match": is_negative_match,
            }
        )

    print(
        {
            "diagnostics": {
                "positive_best_distance": pos_diag["best_distance"],
                "negative_best_distance": neg_diag["best_distance"],
                "false_reject_count": false_rejects,
                "false_match_count": false_matches,
                "threshold_sweep": rows,
            }
        }
    )


def main() -> None:
    data_dir = Path("scripts/.tmp")
    data_dir.mkdir(parents=True, exist_ok=True)

    run_lsh_bucket_stability_test()
    run_multiprobe_decoy_contract_test()
    run_winner_test(data_dir)
    run_chunk_rollover_test(data_dir)
    run_threshold_sweep_diagnostics(data_dir)

    print({
        "status": "ok",
        "tests": [
            "lsh_bucket_stability",
            "multiprobe_decoy_contract",
            "winner",
            "chunk_rollover",
            "threshold_sweep",
        ],
    })


if __name__ == "__main__":
    main()
