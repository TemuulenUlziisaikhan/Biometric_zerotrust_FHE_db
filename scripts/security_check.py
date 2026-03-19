from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def main() -> None:
    client_fhe = _read("client/fhe_client.py")
    server_main = _read("server/main_server.py")

    assert "save_secret_key=False" in client_fhe, "Client context serialization must exclude secret key"
    assert "logger.info" in server_main, "Server audit logs should be present"

    forbidden_log_fields = ["eval_context_b64", "sparse_ciphertext_b64", "probe_ciphertext_b64", "packed_ciphertext"]
    for field in forbidden_log_fields:
        assert f"{field}=%" not in server_main, f"Audit logs should not include sensitive field {field}"

    print({"status": "ok", "checks": ["no_secret_key_serialization", "safe_audit_logging"]})


if __name__ == "__main__":
    main()
