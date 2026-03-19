from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from client.main_client import BiometricClient, BiometricClientError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def _print_progress(done: int, total: int, started: float) -> None:
    elapsed = max(1e-9, time.perf_counter() - started)
    pct = (done / max(1, total)) * 100.0
    rate = done / elapsed
    eta = (total - done) / max(rate, 1e-9)
    print(
        f"[populate] {done}/{total} ({pct:.1f}%) elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
        flush=True,
    )


def _collect_lfw_images(lfw_root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for identity_dir in sorted([path for path in lfw_root.iterdir() if path.is_dir()]):
        label = identity_dir.name
        for file_path in sorted(identity_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((label, file_path))
    return rows


def _load_state(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
        completed = payload.get("completed", [])
        if isinstance(completed, list):
            return {str(item) for item in completed}
    except Exception:
        return set()
    return set()


def _write_state(state_file: Path, completed: set[str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"completed": sorted(completed)}
    state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate DB by enrolling all LFW faces")
    parser.add_argument("--lfw-root", required=True, help="Path to LFW root directory")
    parser.add_argument("--model-path", required=True, help="Path to ArcFace ONNX model")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--threshold", type=float, default=800.0, help="Client threshold (not used for enroll, kept for consistency)")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N enrollments")
    parser.add_argument(
        "--state-file",
        default="benchmark_results/lfw_populate_state.json",
        help="Resume state file (stores successfully enrolled image paths)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume; process every discovered image in this run",
    )
    args = parser.parse_args()

    lfw_root = Path(args.lfw_root)
    if not lfw_root.exists() or not lfw_root.is_dir():
        raise ValueError(f"Invalid --lfw-root: {lfw_root}")

    state_file = Path(args.state_file)
    completed = set() if args.no_resume else _load_state(state_file)

    discovered = _collect_lfw_images(lfw_root)
    if args.max_images is not None:
        discovered = discovered[: max(0, int(args.max_images))]

    if not discovered:
        raise ValueError("No LFW images found to enroll")

    rows = [(label, path) for label, path in discovered if str(path.resolve()) not in completed]

    print(
        {
            "status": "starting",
            "discovered_images": len(discovered),
            "already_completed": len(discovered) - len(rows),
            "to_enroll": len(rows),
            "server_url": args.server_url,
        },
        flush=True,
    )

    client = BiometricClient(server_url=args.server_url, model_path=args.model_path, threshold=float(args.threshold))

    started = time.perf_counter()
    enrolled = 0
    failed = 0
    total = len(rows)
    failures: list[dict] = []

    for index, (label, image_path) in enumerate(rows, start=1):
        image_key = str(image_path.resolve())
        try:
            client.enroll_user(str(image_path), label)
            completed.add(image_key)
            enrolled += 1
        except BiometricClientError as exc:
            failed += 1
            failures.append({"image_path": image_key, "label": label, "error": str(exc)})
        except Exception as exc:
            failed += 1
            failures.append({"image_path": image_key, "label": label, "error": str(exc)})

        if index == 1 or index == total or index % max(1, int(args.progress_every)) == 0:
            _print_progress(index, total, started)

        if not args.no_resume and (index % 25 == 0 or index == total):
            _write_state(state_file, completed)

    elapsed = time.perf_counter() - started

    summary = {
        "status": "ok" if failed == 0 else "completed_with_failures",
        "elapsed": _format_duration(elapsed),
        "attempted": total,
        "enrolled": enrolled,
        "failed": failed,
        "state_file": None if args.no_resume else str(state_file),
    }
    print(summary, flush=True)

    if failures:
        fail_path = Path("benchmark_results/lfw_populate_failures.json")
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print({"failures_file": str(fail_path), "failure_count": len(failures)}, flush=True)


if __name__ == "__main__":
    main()
