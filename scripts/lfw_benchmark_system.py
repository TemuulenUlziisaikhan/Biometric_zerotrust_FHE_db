from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from client.fhe_client import build_replicated_probe_vector, encrypt_vector
from client.main_client import API_VERSION, BiometricClient
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


def _list_identity_images(identity_dir: Path) -> list[Path]:
    images: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        images.extend(identity_dir.glob(f"*{extension}"))
    images = sorted(images)
    return images


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _parse_thresholds(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("No thresholds provided")
    return sorted(values)


def _prepare_population(
    lfw_root: Path,
    max_identities: int,
    gallery_per_identity: int,
    probe_per_identity: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    identity_dirs = sorted([path for path in lfw_root.iterdir() if path.is_dir()])
    eligible = []
    for identity_dir in identity_dirs:
        images = _list_identity_images(identity_dir)
        if len(images) >= (gallery_per_identity + probe_per_identity):
            eligible.append((identity_dir.name, images))

    if len(eligible) < max_identities:
        raise ValueError(f"Not enough eligible identities. Requested={max_identities}, available={len(eligible)}")

    selected = eligible[:max_identities]

    gallery: list[dict] = []
    probes: list[dict] = []
    for identity, images in selected:
        gallery_images = images[:gallery_per_identity]
        probe_images = images[gallery_per_identity : gallery_per_identity + probe_per_identity]
        for image in gallery_images:
            gallery.append({"label": identity, "image_path": str(image)})
        for image in probe_images:
            probes.append({"label": identity, "image_path": str(image), "is_genuine": True})

    selected_labels = {identity for identity, _ in selected}
    impostors: list[dict] = []
    for identity, images in eligible[max_identities:]:
        if identity in selected_labels:
            continue
        impostors.append({"label": identity, "image_path": str(images[0]), "is_genuine": False})
        if len(impostors) >= max(1, max_identities // 2):
            break

    return gallery, probes, impostors


def _prediction_for_threshold(distance_rows: list[dict], threshold: float) -> str | None:
    if not distance_rows:
        return None
    best_row = min(distance_rows, key=lambda row: float(row["distance"]))
    return best_row["uuid"] if float(best_row["distance"]) <= threshold else None


def _build_auth_payload(client: BiometricClient, embedding: np.ndarray) -> dict:
    plan = client._build_auth_plan(embedding)
    probe = build_replicated_probe_vector(embedding)
    encrypted_probe = encrypt_vector(client.context, probe)
    return {
        "api_version": API_VERSION,
        "bucket_ids": plan["bucket_ids"],
        "eval_context_b64": client.eval_context_b64,
        "probe_ciphertext_b64": encrypted_probe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FHE biometric system using LFW identities")
    parser.add_argument("--lfw-root", required=True, help="Path to LFW root directory")
    parser.add_argument("--model-path", required=True, help="Path to ArcFace ONNX model")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--output-dir", default="benchmark_results/lfw_system", help="Output directory")
    parser.add_argument("--max-identities", type=int, default=100, help="Number of enrolled identities")
    parser.add_argument("--gallery-per-identity", type=int, default=1, help="Images per enrolled identity")
    parser.add_argument("--probe-per-identity", type=int, default=1, help="Probe images per enrolled identity")
    parser.add_argument(
        "--thresholds",
        default="50,100,200,300,400,500,600,700,800",
        help="Comma-separated threshold sweep",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent worker count for API latency test")
    parser.add_argument("--iterations", type=int, default=200, help="Concurrent latency request count")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print progress every N items per stage (also prints first/last)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_thresholds(args.thresholds)
    threshold_max = max(thresholds)

    gallery, genuine_probes, impostor_probes = _prepare_population(
        lfw_root=Path(args.lfw_root),
        max_identities=args.max_identities,
        gallery_per_identity=args.gallery_per_identity,
        probe_per_identity=args.probe_per_identity,
    )

    client = BiometricClient(server_url=args.server_url, model_path=args.model_path, threshold=threshold_max)

    enroll_latencies_ms: list[float] = []
    enroll_started = time.perf_counter()
    total_gallery = len(gallery)
    print(f"[enroll] starting enrollment for {total_gallery} gallery images", flush=True)
    for index, row in enumerate(gallery, start=1):
        started = time.perf_counter()
        client.enroll_user(row["image_path"], row["label"])
        enroll_latencies_ms.append((time.perf_counter() - started) * 1000.0)
        if index == 1 or index == total_gallery or index % max(1, args.progress_every) == 0:
            _print_progress("enroll", index, total_gallery, enroll_started)

    probe_rows = genuine_probes + impostor_probes
    diagnostics_rows: list[dict] = []
    extract_latencies_ms: list[float] = []
    auth_total_latencies_ms: list[float] = []
    auth_request_latencies_ms: list[float] = []

    probes_started = time.perf_counter()
    total_probes = len(probe_rows)
    print(f"[probes] starting diagnostics for {total_probes} probes", flush=True)
    for index, row in enumerate(probe_rows, start=1):
        extract_started = time.perf_counter()
        embedding = extract_embedding(args.model_path, row["image_path"])
        extract_ms = (time.perf_counter() - extract_started) * 1000.0

        auth_started = time.perf_counter()
        diagnostics = client.authenticate_embedding_diagnostics(embedding, threshold=threshold_max)
        auth_total_ms = (time.perf_counter() - auth_started) * 1000.0

        extract_latencies_ms.append(extract_ms)
        auth_total_latencies_ms.append(auth_total_ms)
        auth_request_latencies_ms.append(float(diagnostics["timings_ms"]["request_roundtrip"]))

        diagnostics_rows.append(
            {
                "label": row["label"],
                "image_path": row["image_path"],
                "is_genuine": bool(row["is_genuine"]),
                "best_distance": diagnostics["best_distance"],
                "distance_rows": diagnostics["distance_rows"],
                "winner_uuid": diagnostics["winner_uuid"],
                "timings_ms": diagnostics["timings_ms"],
            }
        )
        if index == 1 or index == total_probes or index % max(1, args.progress_every) == 0:
            _print_progress("probes", index, total_probes, probes_started)

    threshold_table: list[dict] = []
    for threshold in thresholds:
        genuine_total = 0
        genuine_errors = 0
        genuine_correct = 0
        impostor_total = 0
        impostor_false_accept = 0

        for row in diagnostics_rows:
            predicted = _prediction_for_threshold(row["distance_rows"], threshold)
            if row["is_genuine"]:
                genuine_total += 1
                if predicted == row["label"]:
                    genuine_correct += 1
                else:
                    genuine_errors += 1
            else:
                impostor_total += 1
                if predicted is not None:
                    impostor_false_accept += 1

        top1 = genuine_correct / max(1, genuine_total)
        frr = genuine_errors / max(1, genuine_total)
        far = impostor_false_accept / max(1, impostor_total)

        threshold_table.append(
            {
                "threshold": float(threshold),
                "top1": float(top1),
                "frr": float(frr),
                "far": float(far),
                "genuine_total": int(genuine_total),
                "impostor_total": int(impostor_total),
            }
        )

    # Concurrent API latency benchmark (request+server response only)
    all_probe_embeddings = [extract_embedding(args.model_path, row["image_path"]) for row in probe_rows]
    payloads = [_build_auth_payload(client, embedding) for embedding in all_probe_embeddings]

    sampled_payloads = [payloads[index % len(payloads)] for index in range(args.iterations)]
    random.shuffle(sampled_payloads)

    concurrent_latencies_ms: list[float] = []

    def _run_one(payload: dict) -> float:
        started = time.perf_counter()
        client._post_json("/authenticate", payload, timeout=120)
        return (time.perf_counter() - started) * 1000.0

    concurrent_started = time.perf_counter()
    total_iterations = len(sampled_payloads)
    print(
        f"[concurrent] starting request benchmark for {total_iterations} iterations (workers={args.workers})",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, latency in enumerate(executor.map(_run_one, sampled_payloads), start=1):
            concurrent_latencies_ms.append(float(latency))
            if index == 1 or index == total_iterations or index % max(1, args.progress_every) == 0:
                _print_progress("concurrent", index, total_iterations, concurrent_started)

    summary = {
        "gallery_count": len(gallery),
        "genuine_probe_count": len(genuine_probes),
        "impostor_probe_count": len(impostor_probes),
        "threshold_table": threshold_table,
        "latency_ms": {
            "enroll": _percentiles(enroll_latencies_ms),
            "extract": _percentiles(extract_latencies_ms),
            "auth_total": _percentiles(auth_total_latencies_ms),
            "auth_request_roundtrip": _percentiles(auth_request_latencies_ms),
            "concurrent_auth_request_roundtrip": _percentiles(concurrent_latencies_ms),
            "concurrent_workers": int(args.workers),
            "concurrent_iterations": int(args.iterations),
        },
    }

    (output_dir / "lfw_system_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "lfw_system_threshold_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["threshold", "top1", "frr", "far", "genuine_total", "impostor_total"],
        )
        writer.writeheader()
        writer.writerows(threshold_table)

    print({"status": "ok", "output_dir": str(output_dir), "summary": summary})


if __name__ == "__main__":
    main()
