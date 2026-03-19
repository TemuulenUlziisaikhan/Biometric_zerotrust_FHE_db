from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.main_client import BiometricClient


def main() -> None:
    data_dir = Path("scripts/.tmp")
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    emb_a = rng.standard_normal(512).astype(np.float32)
    emb_b = emb_a + (rng.standard_normal(512).astype(np.float32) * 0.001)

    emb_a_path = data_dir / "emb_a.npy"
    emb_b_path = data_dir / "emb_b.npy"
    np.save(emb_a_path, emb_a)
    np.save(emb_b_path, emb_b)

    client = BiometricClient(
        server_url="http://127.0.0.1:8000",
        model_path=None,
        threshold=10.0,
        lsh_seed=7,
    )

    enroll_result = client.enroll_user(str(emb_a_path), "usr_smoke_a")
    winner = client.authenticate_user(str(emb_b_path))
    if winner != "usr_smoke_a":
        raise AssertionError(f"Expected winner usr_smoke_a, got {winner}")

    print({"status": "ok", "enroll": enroll_result, "winner": winner})


if __name__ == "__main__":
    main()
