from __future__ import annotations

import argparse
from pathlib import Path
import sys

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _looks_like_lfw_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    pairs_file = path / "pairs.txt"
    if not pairs_file.exists():
        return False

    identity_dirs = [item for item in path.iterdir() if item.is_dir()]
    if not identity_dirs:
        return False

    image_count = 0
    for identity_dir in identity_dirs[:30]:
        image_count += sum(1 for child in identity_dir.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS)
    return image_count > 0


def _find_lfw_root(download_root: Path) -> Path | None:
    if _looks_like_lfw_root(download_root):
        return download_root

    candidates: list[Path] = []
    for pairs_path in download_root.rglob("pairs.txt"):
        candidates.append(pairs_path.parent)

    # Prefer shallow paths first for determinism.
    candidates = sorted(set(candidates), key=lambda p: (len(p.parts), str(p)))

    for candidate in candidates:
        if _looks_like_lfw_root(candidate):
            return candidate

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/resolve LFW dataset using kagglehub")
    parser.add_argument(
        "--dataset",
        default="jessicali9530/lfw-dataset",
        help="Kaggle dataset slug for LFW (owner/dataset)",
    )
    parser.add_argument(
        "--cache-dir",
        default="benchmark_results/kagglehub",
        help="Optional cache directory hint (used for diagnostics only)",
    )
    parser.add_argument("--print-root", action="store_true", help="Print only resolved LFW root path")
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Install in venv with: ./venv/bin/python -m pip install kagglehub"
        ) from exc

    download_path = Path(kagglehub.dataset_download(args.dataset)).resolve()
    lfw_root = _find_lfw_root(download_path)
    if lfw_root is None:
        raise RuntimeError(
            f"Could not locate a valid LFW root (with pairs.txt + identity images) under downloaded path: {download_path}"
        )

    if args.print_root:
        print(str(lfw_root))
        return

    print(
        {
            "status": "ok",
            "dataset": args.dataset,
            "download_path": str(download_path),
            "lfw_root": str(lfw_root),
            "cache_dir_hint": str(Path(args.cache_dir).resolve()),
        }
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print({"status": "error", "message": str(exc)}, file=sys.stderr)
        raise
