from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

from client.main_client import BiometricClient
from client.ml_extractor import extract_embedding


def _load_canary_file(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Canary file must be a JSON array")

    rows: list[dict] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Canary item {index} must be an object")
        label = str(item.get("label", "")).strip()
        image_path = str(item.get("image_path", "")).strip()
        if not label or not image_path:
            raise ValueError(f"Canary item {index} must include label and image_path")
        rows.append({"label": label, "image_path": image_path})
    return rows


def _safe_rel_delta(current: float, baseline: float) -> float:
    denominator = max(abs(baseline), 1e-9)
    return abs(current - baseline) / denominator


def _distance_stats(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
        }
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p95": float(ordered[p95_index]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily CKKS noise/quality health check with canary probes")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--model-path", required=True, help="Path to ArcFace ONNX model")
    parser.add_argument("--canary-file", required=True, help="JSON list of canary probes: [{label,image_path}]")
    parser.add_argument(
        "--output-json",
        default="benchmark_results/noise_health/latest_noise_health.json",
        help="Output report JSON",
    )
    parser.add_argument(
        "--baseline-json",
        default="benchmark_results/noise_health/baseline_noise_health.json",
        help="Baseline report JSON used for drift comparison if it exists",
    )
    parser.add_argument(
        "--distance-drift-threshold",
        type=float,
        default=0.20,
        help="Relative distance drift threshold per probe (e.g. 0.20 = 20%%)",
    )
    parser.add_argument(
        "--max-drifted-probe-ratio",
        type=float,
        default=0.25,
        help="Fail if drifted probes ratio exceeds this",
    )
    parser.add_argument(
        "--min-match-rate",
        type=float,
        default=0.90,
        help="Fail if winner==label ratio drops below this",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1_000_000.0,
        help="Auth threshold used during health checks (high default to force best-candidate selection)",
    )
    args = parser.parse_args()

    canary_file = Path(args.canary_file)
    canary_rows = _load_canary_file(canary_file)

    client = BiometricClient(server_url=args.server_url, model_path=args.model_path, threshold=float(args.threshold))

    started = time.time()
    per_probe: list[dict] = []
    distances: list[float] = []
    success = 0
    matches = 0
    precision_failures = 0
    exceptions: list[dict] = []

    for row in canary_rows:
        label = row["label"]
        image_path = row["image_path"]
        probe_entry = {
            "label": label,
            "image_path": image_path,
            "winner_uuid": None,
            "best_distance": None,
            "ok": False,
            "error": None,
        }
        try:
            embedding = extract_embedding(args.model_path, image_path)
            diagnostics = client.authenticate_embedding_diagnostics(embedding, threshold=float(args.threshold))
            winner = diagnostics.get("winner_uuid")
            best_distance = diagnostics.get("best_distance")

            probe_entry["winner_uuid"] = winner
            probe_entry["best_distance"] = float(best_distance) if best_distance is not None else None
            probe_entry["ok"] = True

            success += 1
            if winner == label:
                matches += 1
            if best_distance is not None:
                distances.append(float(best_distance))
        except Exception as exc:
            message = str(exc)
            probe_entry["error"] = message
            exceptions.append({"label": label, "image_path": image_path, "error": message})
            if "negative squared distance" in message.lower() or "precision tolerance" in message.lower():
                precision_failures += 1

        per_probe.append(probe_entry)

    baseline_path = Path(args.baseline_json)
    baseline_by_path: dict[str, float] = {}
    if baseline_path.exists():
        try:
            baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            for item in baseline_payload.get("per_probe", []):
                if item.get("best_distance") is None:
                    continue
                baseline_by_path[str(item.get("image_path"))] = float(item["best_distance"])
        except Exception:
            baseline_by_path = {}

    drifted = 0
    comparable = 0
    for item in per_probe:
        distance = item.get("best_distance")
        baseline_distance = baseline_by_path.get(str(item.get("image_path")))
        if distance is None or baseline_distance is None:
            item["relative_drift"] = None
            continue
        comparable += 1
        rel = _safe_rel_delta(float(distance), float(baseline_distance))
        item["relative_drift"] = float(rel)
        if rel > float(args.distance_drift_threshold):
            drifted += 1

    total = len(canary_rows)
    match_rate = matches / max(1, success)
    drift_ratio = drifted / max(1, comparable)

    checks = {
        "precision_failures_zero": precision_failures == 0,
        "match_rate_ok": match_rate >= float(args.min_match_rate),
        "drift_ratio_ok": drift_ratio <= float(args.max_drifted_probe_ratio),
    }
    status = "ok" if all(checks.values()) else "alert"

    report = {
        "status": status,
        "started_at_epoch": int(started),
        "server_url": args.server_url,
        "model_path": args.model_path,
        "canary_file": str(canary_file),
        "summary": {
            "total_probes": total,
            "successful_probes": success,
            "failed_probes": total - success,
            "match_count": matches,
            "match_rate": float(match_rate),
            "precision_failure_count": precision_failures,
            "comparable_for_drift": comparable,
            "drifted_probe_count": drifted,
            "drifted_probe_ratio": float(drift_ratio),
            "distance_stats": _distance_stats(distances),
        },
        "thresholds": {
            "distance_drift_threshold": float(args.distance_drift_threshold),
            "max_drifted_probe_ratio": float(args.max_drifted_probe_ratio),
            "min_match_rate": float(args.min_match_rate),
        },
        "checks": checks,
        "per_probe": per_probe,
        "exceptions": exceptions,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"status": status, "report": str(output_path), "summary": report["summary"]}, indent=2))

    if status != "ok":
        sys.exit(2)


if __name__ == "__main__":
    main()
