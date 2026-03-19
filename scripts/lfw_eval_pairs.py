from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from client.ml_extractor import extract_embedding

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def _print_progress(stage: str, done: int, total: int, started: float) -> None:
    elapsed = max(1e-9, time.perf_counter() - started)
    pct = (done / max(1, total)) * 100.0
    rate = done / elapsed
    eta = (total - done) / max(rate, 1e-9)
    print(
        f"[{stage}] {done}/{total} ({pct:.1f}%) elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
        flush=True,
    )


def _resolve_lfw_image(lfw_root: Path, identity: str, image_index: int) -> Path:
    base = f"{identity}_{image_index:04d}"
    for ext in IMAGE_EXTENSIONS:
        candidate = lfw_root / identity / f"{base}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate image for identity={identity}, index={image_index}")


def _parse_pairs_file(pairs_file: Path, max_pairs: int | None = None) -> tuple[list[dict], bool]:
    if pairs_file.suffix.lower() == ".csv":
        return _parse_pairs_csv(pairs_file, max_pairs=max_pairs), False

    lines = [line.strip() for line in pairs_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("pairs file is empty")

    start_index = 0
    is_official_10_fold = False
    try:
        header_tokens = [int(x) for x in lines[0].split()]
        if len(header_tokens) in {1, 2}:
            start_index = 1
            is_official_10_fold = True
    except ValueError:
        start_index = 0

    pairs: list[dict] = []
    for line_index, line in enumerate(lines[start_index:], start=start_index):
        tokens = line.split()
        fold = (line_index - start_index) // 600 if is_official_10_fold else 0
        if len(tokens) == 3:
            identity, idx1, idx2 = tokens
            pairs.append(
                {
                    "fold": fold,
                    "same": 1,
                    "identity_1": identity,
                    "index_1": int(idx1),
                    "identity_2": identity,
                    "index_2": int(idx2),
                }
            )
        elif len(tokens) == 4:
            identity1, idx1, identity2, idx2 = tokens
            pairs.append(
                {
                    "fold": fold,
                    "same": 0,
                    "identity_1": identity1,
                    "index_1": int(idx1),
                    "identity_2": identity2,
                    "index_2": int(idx2),
                }
            )
        else:
            raise ValueError(f"Malformed pairs line: {line}")

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    return pairs, is_official_10_fold


def _parse_pairs_csv(pairs_file: Path, max_pairs: int | None = None) -> list[dict]:
    pairs: list[dict] = []
    with pairs_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            cells = [cell.strip() for cell in row if cell is not None and cell.strip() != ""]
            if not cells:
                continue

            lowered = [cell.lower() for cell in cells]
            if "name" in lowered or "imagenum1" in lowered:
                continue

            if len(cells) == 3:
                identity, idx1, idx2 = cells
                pairs.append(
                    {
                        "fold": 0,
                        "same": 1,
                        "identity_1": identity,
                        "index_1": int(idx1),
                        "identity_2": identity,
                        "index_2": int(idx2),
                    }
                )
            elif len(cells) == 4:
                identity1, idx1, identity2, idx2 = cells
                pairs.append(
                    {
                        "fold": 0,
                        "same": 0,
                        "identity_1": identity1,
                        "index_1": int(idx1),
                        "identity_2": identity2,
                        "index_2": int(idx2),
                    }
                )
            else:
                raise ValueError(f"Malformed CSV pairs row: {row}")

    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def _cache_key(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()


def _load_embedding_with_cache(model_path: str, image_path: Path, cache_dir: Path | None) -> np.ndarray:
    if cache_dir is None:
        return extract_embedding(model_path, str(image_path))

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key(image_path)}.npy"
    if cache_file.exists():
        return np.load(cache_file)

    emb = extract_embedding(model_path, str(image_path))
    np.save(cache_file, emb)
    return emb


def _compute_distances(labels: np.ndarray, distances: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positives = labels == 1
    negatives = labels == 0

    tpr = np.zeros_like(thresholds, dtype=np.float64)
    fpr = np.zeros_like(thresholds, dtype=np.float64)
    accuracy = np.zeros_like(thresholds, dtype=np.float64)

    for index, threshold in enumerate(thresholds):
        preds = distances <= threshold
        tp = np.sum(preds & positives)
        tn = np.sum((~preds) & negatives)
        fp = np.sum(preds & negatives)
        fn = np.sum((~preds) & positives)

        tpr[index] = tp / max(1, np.sum(positives))
        fpr[index] = fp / max(1, np.sum(negatives))
        accuracy[index] = (tp + tn) / max(1, labels.shape[0])

    return tpr, fpr, accuracy


def _roc_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def _eer(thresholds: np.ndarray, fpr: np.ndarray, tpr: np.ndarray) -> tuple[float, float]:
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fnr - fpr)))
    eer_value = float((fnr[index] + fpr[index]) / 2.0)
    return eer_value, float(thresholds[index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LFW verification metrics using pairs protocol")
    parser.add_argument("--lfw-root", required=True, help="Path to LFW root directory")
    parser.add_argument("--pairs-file", required=True, help="Path to LFW pairs.txt")
    parser.add_argument("--model-path", required=True, help="Path to ArcFace ONNX model")
    parser.add_argument("--output-dir", default="benchmark_results/lfw_pairs", help="Output directory")
    parser.add_argument("--cache-dir", default="benchmark_results/embedding_cache", help="Embedding cache dir")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap for quick runs")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N pairs (also prints first/last)",
    )
    args = parser.parse_args()

    lfw_root = Path(args.lfw_root)
    pairs_file = Path(args.pairs_file)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    pairs, is_official_10_fold = _parse_pairs_file(pairs_file, max_pairs=args.max_pairs)

    records: list[dict] = []
    labels: list[int] = []
    distances: list[float] = []

    total_pairs = len(pairs)
    if total_pairs == 0:
        raise RuntimeError("No pairs to evaluate")
    print(f"[pairs] starting evaluation for {total_pairs} pairs", flush=True)

    for index, pair in enumerate(pairs, start=1):
        image_1 = _resolve_lfw_image(lfw_root, pair["identity_1"], pair["index_1"])
        image_2 = _resolve_lfw_image(lfw_root, pair["identity_2"], pair["index_2"])

        emb_1 = _load_embedding_with_cache(args.model_path, image_1, cache_dir)
        emb_2 = _load_embedding_with_cache(args.model_path, image_2, cache_dir)

        dist = float(np.sum((emb_1 - emb_2) ** 2))

        labels.append(int(pair["same"]))
        distances.append(dist)
        records.append(
            {
                "fold": int(pair["fold"]),
                "same": int(pair["same"]),
                "image_1": str(image_1),
                "image_2": str(image_2),
                "distance": dist,
            }
        )

        if index == 1 or index == total_pairs or index % max(1, args.progress_every) == 0:
            _print_progress("pairs", index, total_pairs, started)

    labels_np = np.asarray(labels, dtype=np.int8)
    distances_np = np.asarray(distances, dtype=np.float64)

    thresholds = np.unique(distances_np)
    if thresholds.shape[0] == 0:
        raise RuntimeError("No thresholds generated; check pairs input")

    tpr, fpr, accuracy = _compute_distances(labels_np, distances_np, thresholds)
    best_index = int(np.argmax(accuracy))
    best_threshold = float(thresholds[best_index])
    auc = _roc_auc(fpr, tpr)
    eer_value, eer_threshold = _eer(thresholds, fpr, tpr)

    fold_summary: list[dict] = []
    if is_official_10_fold:
        fold_ids = sorted({record["fold"] for record in records})
        for fold in fold_ids:
            fold_mask = np.asarray([record["fold"] == fold for record in records], dtype=bool)
            train_mask = ~fold_mask

            train_tpr, train_fpr, train_acc = _compute_distances(
                labels_np[train_mask], distances_np[train_mask], thresholds
            )
            train_best_index = int(np.argmax(train_acc))
            train_best_threshold = float(thresholds[train_best_index])

            test_preds = distances_np[fold_mask] <= train_best_threshold
            test_labels = labels_np[fold_mask]
            test_accuracy = float(np.mean((test_preds.astype(np.int8) == test_labels).astype(np.float64)))

            positives = test_labels == 1
            negatives = test_labels == 0
            tp = int(np.sum(test_preds & positives))
            fp = int(np.sum(test_preds & negatives))
            tn = int(np.sum((~test_preds) & negatives))
            fn = int(np.sum((~test_preds) & positives))

            fold_summary.append(
                {
                    "fold": int(fold),
                    "threshold": train_best_threshold,
                    "accuracy": test_accuracy,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }
            )

    completed_ms = (time.perf_counter() - started) * 1000.0

    summary = {
        "pair_count": int(labels_np.shape[0]),
        "positive_count": int(np.sum(labels_np == 1)),
        "negative_count": int(np.sum(labels_np == 0)),
        "best_threshold": best_threshold,
        "best_accuracy": float(accuracy[best_index]),
        "roc_auc": auc,
        "eer": eer_value,
        "eer_threshold": eer_threshold,
        "official_10_fold": bool(is_official_10_fold),
        "fold_summary": fold_summary,
        "runtime_ms": completed_ms,
    }

    (output_dir / "lfw_pairs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "lfw_pairs_distances.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "same", "image_1", "image_2", "distance"])
        writer.writeheader()
        writer.writerows(records)

    with (output_dir / "lfw_pairs_roc.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "tpr", "fpr", "accuracy"])
        writer.writeheader()
        for threshold, tpr_row, fpr_row, acc_row in zip(thresholds, tpr, fpr, accuracy):
            writer.writerow(
                {
                    "threshold": float(threshold),
                    "tpr": float(tpr_row),
                    "fpr": float(fpr_row),
                    "accuracy": float(acc_row),
                }
            )

    print({"status": "ok", "output_dir": str(output_dir), "summary": summary})


if __name__ == "__main__":
    main()
