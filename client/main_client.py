from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import requests

from client.client_models import API_VERSION, BiometricClientError, ServerError, TransportError
from client.fhe_client import (
    DEFAULT_SHARED_CONTEXT_PATH,
    build_replicated_probe_vector,
    build_sparse_enrollment_vector,
    create_ckks_context,
    decrypt_distance_vector,
    encrypt_vector,
    load_or_create_shared_context,
    load_or_create_shared_eval_context_b64,
    parse_distances_by_chunk_slots,
    select_best_uuid,
    serialize_eval_context_without_secret,
)
from client.lsh_engine import LSHEngine
from client.ml_extractor import extract_embedding


@dataclass
class BiometricClient:
    server_url: str
    model_path: str | None = None
    threshold: float = 1.0
    lsh_bits: int = 9
    lsh_seed: int = 7
    multiprobe_count: int = 2
    multiprobe_strategy: str = "margin"
    multiprobe_hamming_radius: int = 1
    decoy_count: int = 3
    max_standalone_enrollments: int = 5
    miss_retry_enabled: bool = True
    miss_retry_probe_count: int = 8
    miss_retry_hamming_radius: int = 2
    use_shared_context: bool = True
    context_path: str = DEFAULT_SHARED_CONTEXT_PATH

    def __post_init__(self) -> None:
        if self.use_shared_context:
            self.context = load_or_create_shared_context(self.context_path)
            self.eval_context_b64 = load_or_create_shared_eval_context_b64(self.context, self.context_path)
        else:
            self.context = create_ckks_context()
            self.eval_context_b64 = serialize_eval_context_without_secret(self.context)
        self.lsh = LSHEngine(bits=self.lsh_bits, seed=self.lsh_seed)

    def _post_json(self, path: str, payload: dict, timeout: int) -> dict:
        url = f"{self.server_url}{path}"
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            raise TransportError(f"Request failed for {path}: {exc}") from exc

        if response.status_code >= 400:
            try:
                payload_in = response.json()
            except Exception:
                payload_in = {}

            request_id_header = response.headers.get("X-Request-ID", "")
            detail = payload_in.get("detail", {}) if isinstance(payload_in, dict) else {}
            if isinstance(detail, dict):
                code = detail.get("code", "SERVER_ERROR")
                message = detail.get("message", response.text)
                request_id = str(detail.get("request_id", "")).strip() or request_id_header
            else:
                code = "SERVER_ERROR"
                message = str(detail)
                request_id = request_id_header

            request_id_suffix = f" (request_id={request_id})" if request_id else ""
            raise ServerError(f"{code}: {message}{request_id_suffix}")

        payload_out = response.json()
        if isinstance(payload_out, dict):
            nested_data = payload_out.get("data")
            if isinstance(nested_data, dict):
                merged = dict(payload_out)
                merged.update(nested_data)
                payload_out = merged
        api_version = payload_out.get("api_version")
        if api_version is not None and api_version != API_VERSION:
            raise ServerError(f"API_VERSION_MISMATCH: expected {API_VERSION}, got {api_version}")
        return payload_out

    def _collect_distance_rows(self, payload: dict, in_scope_buckets: set[str] | None = None) -> tuple[list[dict], int]:
        distance_rows: list[dict] = []
        ignored_chunk_count = 0
        for chunk in payload["results"]:
            if in_scope_buckets is not None and chunk["bucket_id"] not in in_scope_buckets:
                ignored_chunk_count += 1
                continue
            decrypted = decrypt_distance_vector(self.context, chunk["encrypted_distance_b64"])
            uuid_map = list(chunk["uuid_map"])
            distances = parse_distances_by_chunk_slots(decrypted, face_count=len(uuid_map))
            for uid, dist in zip(uuid_map, distances):
                distance_rows.append(
                    {
                        "uuid": uid,
                        "distance": float(dist),
                        "bucket_id": chunk["bucket_id"],
                        "chunk_index": int(chunk["chunk_index"]),
                    }
                )
        return distance_rows, ignored_chunk_count

    def enroll_user(self, image_path: str, user_uuid: str) -> dict:
        embedding = extract_embedding(self.model_path, image_path)
        return self.enroll_embedding(embedding, user_uuid)

    def enroll_user_images(self, image_paths: list[str], user_uuid: str) -> dict:
        if not image_paths:
            raise BiometricClientError("At least one image is required for enrollment")
        embeddings = [extract_embedding(self.model_path, path) for path in image_paths]
        return self.enroll_embeddings(embeddings, user_uuid)

    def enroll_embeddings(self, embeddings, user_uuid: str) -> dict:
        vectors = [np.asarray(embedding, dtype=np.float32).reshape(-1) for embedding in embeddings]
        if not vectors:
            raise BiometricClientError("At least one embedding is required for enrollment")

        normalized_vectors = []
        for vector in vectors:
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise BiometricClientError("Embedding norm is zero")
            normalized_vectors.append(vector / norm)

        if len(normalized_vectors) == 1:
            single_result = self.enroll_embedding(normalized_vectors[0], user_uuid)
            return {
                "status": "enrolled",
                "user_uuid": user_uuid,
                "source_image_count": 1,
                "fused_embedding_sent": False,
                "standalone_embeddings_sent": 1,
                "enrollment_calls": 1,
                "responses": [single_result],
            }

        stacked = np.stack(normalized_vectors, axis=0)
        fused = np.mean(stacked, axis=0)
        fused_norm = float(np.linalg.norm(fused))
        if fused_norm == 0.0:
            raise BiometricClientError("Fused embedding norm is zero")
        fused = fused / fused_norm

        max_extras = max(0, int(self.max_standalone_enrollments))
        cosine_scores = [float(np.dot(vector, fused)) for vector in normalized_vectors]
        ranked_indexes = sorted(range(len(normalized_vectors)), key=lambda idx: cosine_scores[idx])
        selected_indexes = ranked_indexes[:max_extras]
        extras = [normalized_vectors[idx] for idx in selected_indexes]

        responses: list[dict] = []
        responses.append(self.enroll_embedding(fused, user_uuid))
        for extra in extras:
            responses.append(self.enroll_embedding(extra, user_uuid))

        return {
            "status": "enrolled",
            "user_uuid": user_uuid,
            "source_image_count": len(normalized_vectors),
            "fused_embedding_sent": True,
            "standalone_embeddings_sent": len(extras),
            "enrollment_calls": len(responses),
            "responses": responses,
        }

    def enroll_embedding(self, embedding, user_uuid: str) -> dict:
        bucket_id = self.lsh.hash_embedding_to_bucket(embedding)
        offset_payload = {"api_version": API_VERSION, "bucket_id": bucket_id}
        offset = self._post_json("/enroll/offset", offset_payload, timeout=30)

        sparse = build_sparse_enrollment_vector(embedding, slot_offset=int(offset["slot_offset"]))
        encrypted_sparse = encrypt_vector(self.context, sparse)
        enroll_payload = {
            "api_version": API_VERSION,
            "bucket_id": bucket_id,
            "user_uuid": user_uuid,
            "eval_context_b64": self.eval_context_b64,
            "sparse_ciphertext_b64": encrypted_sparse,
            "chunk_index": int(offset["chunk_index"]),
        }
        return self._post_json("/enroll", enroll_payload, timeout=60)

    def authenticate_user(self, image_path: str) -> str | None:
        embedding = extract_embedding(self.model_path, image_path)
        return self.authenticate_embedding(embedding)

    def _build_auth_plan(self, embedding) -> dict:
        primary = self.lsh.hash_embedding_to_bucket(embedding)
        multiprobe = self.lsh.generate_multiprobe_buckets(
            primary,
            probes=self.multiprobe_count,
            strategy=self.multiprobe_strategy,
            embedding=embedding,
            hamming_radius=self.multiprobe_hamming_radius,
        )
        decoys = self.lsh.generate_decoy_buckets(exclude={primary, *multiprobe}, count=self.decoy_count)
        bucket_ids = self.lsh.build_auth_bucket_request(primary, multiprobe, decoys)
        return {
            "primary": primary,
            "multiprobe": multiprobe,
            "decoys": decoys,
            "bucket_ids": bucket_ids,
        }

    def _build_miss_retry_buckets(self, embedding, primary: str, already_used: set[str]) -> list[str]:
        if not self.miss_retry_enabled or self.miss_retry_probe_count <= 0:
            return []
        retry_radius = max(self.miss_retry_hamming_radius, self.multiprobe_hamming_radius)
        retry_candidates = self.lsh.generate_multiprobe_buckets(
            primary,
            probes=self.miss_retry_probe_count + len(already_used),
            strategy=self.multiprobe_strategy,
            embedding=embedding,
            hamming_radius=retry_radius,
        )
        filtered = [bucket for bucket in retry_candidates if bucket not in already_used]
        return filtered[: self.miss_retry_probe_count]

    def _make_auth_payload(self, bucket_ids: list[str], encrypted_probe: str) -> dict:
        return {
            "api_version": API_VERSION,
            "bucket_ids": bucket_ids,
            "eval_context_b64": self.eval_context_b64,
            "probe_ciphertext_b64": encrypted_probe,
        }

    def authenticate_embedding_diagnostics(self, embedding, threshold: float | None = None) -> dict:
        total_start = time.perf_counter()

        plan_start = time.perf_counter()
        plan = self._build_auth_plan(embedding)
        plan_ms = (time.perf_counter() - plan_start) * 1000.0

        primary = plan["primary"]
        multiprobe = plan["multiprobe"]
        decoys = plan["decoys"]
        bucket_ids = plan["bucket_ids"]
        in_scope_buckets = {primary, *multiprobe}
        requested_bucket_ids_effective: list[str] = []

        encrypt_start = time.perf_counter()
        probe = build_replicated_probe_vector(embedding)
        encrypted_probe = encrypt_vector(self.context, probe)
        encrypt_ms = (time.perf_counter() - encrypt_start) * 1000.0

        primary_request_bucket_ids = list(dict.fromkeys([primary, *decoys]))
        auth_payload = self._make_auth_payload(primary_request_bucket_ids, encrypted_probe)

        request_start = time.perf_counter()
        payload = self._post_json("/authenticate", auth_payload, timeout=120)
        request_ms = (time.perf_counter() - request_start) * 1000.0
        requested_bucket_ids_effective.extend(primary_request_bucket_ids)

        effective_threshold = self.threshold if threshold is None else float(threshold)
        decrypt_parse_start = time.perf_counter()
        distance_rows, ignored_chunk_count = self._collect_distance_rows(payload, in_scope_buckets={primary})
        winner = select_best_uuid(distance_rows, effective_threshold)

        fallback_used = False
        fallback_bucket_count = 0
        fallback_requested_buckets: list[str] = []
        fallback_existing_buckets: list[str] = []
        miss_retry_used = False
        miss_retry_bucket_count = 0
        miss_retry_requested_buckets: list[str] = []
        miss_retry_existing_buckets: list[str] = []
        if not distance_rows or winner is None:
            fallback_bucket_ids = multiprobe[: self.multiprobe_count]
            fallback_requested_buckets = list(fallback_bucket_ids)
            if fallback_bucket_ids:
                fallback_request_bucket_ids = list(dict.fromkeys([*fallback_bucket_ids, *decoys]))
                fallback_payload = self._make_auth_payload(fallback_request_bucket_ids, encrypted_probe)
                fallback_response = self._post_json("/authenticate", fallback_payload, timeout=120)
                fallback_existing_buckets = sorted(
                    {
                        str(chunk.get("bucket_id"))
                        for chunk in fallback_response.get("results", [])
                        if str(chunk.get("bucket_id")) in set(fallback_bucket_ids)
                    }
                )
                fallback_rows, _ = self._collect_distance_rows(fallback_response, in_scope_buckets=set(fallback_bucket_ids))
                distance_rows.extend(fallback_rows)
                winner = select_best_uuid(distance_rows, effective_threshold)
                fallback_used = True
                fallback_bucket_count = len(fallback_bucket_ids)
                requested_bucket_ids_effective.extend(fallback_request_bucket_ids)

        if not distance_rows or winner is None:
            already_used = {primary, *multiprobe}
            miss_retry_buckets = self._build_miss_retry_buckets(embedding, primary=primary, already_used=already_used)
            miss_retry_requested_buckets = list(miss_retry_buckets)
            if miss_retry_buckets:
                miss_retry_request_bucket_ids = list(dict.fromkeys([*miss_retry_buckets, *decoys]))
                miss_retry_payload = self._make_auth_payload(miss_retry_request_bucket_ids, encrypted_probe)
                miss_retry_response = self._post_json("/authenticate", miss_retry_payload, timeout=120)
                miss_retry_existing_buckets = sorted(
                    {
                        str(chunk.get("bucket_id"))
                        for chunk in miss_retry_response.get("results", [])
                        if str(chunk.get("bucket_id")) in set(miss_retry_buckets)
                    }
                )
                miss_retry_rows, _ = self._collect_distance_rows(
                    miss_retry_response,
                    in_scope_buckets=set(miss_retry_buckets),
                )
                distance_rows.extend(miss_retry_rows)
                winner = select_best_uuid(distance_rows, effective_threshold)
                miss_retry_used = True
                miss_retry_bucket_count = len(miss_retry_buckets)
                requested_bucket_ids_effective.extend(miss_retry_request_bucket_ids)

        decrypt_parse_ms = (time.perf_counter() - decrypt_parse_start) * 1000.0
        best_distance = min((float(row["distance"]) for row in distance_rows), default=None)
        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "winner_uuid": winner,
            "threshold": effective_threshold,
            "best_distance": best_distance,
            "candidate_count": len(distance_rows),
            "ignored_chunk_count": ignored_chunk_count,
            "fallback_used": fallback_used,
            "fallback_bucket_count": fallback_bucket_count,
            "nearby_requested_buckets": fallback_requested_buckets,
            "nearby_requested_bucket_count": len(fallback_requested_buckets),
            "nearby_existing_buckets": fallback_existing_buckets,
            "nearby_existing_bucket_count": len(fallback_existing_buckets),
            "miss_retry_used": miss_retry_used,
            "miss_retry_bucket_count": miss_retry_bucket_count,
            "miss_retry_requested_buckets": miss_retry_requested_buckets,
            "miss_retry_requested_bucket_count": len(miss_retry_requested_buckets),
            "miss_retry_existing_buckets": miss_retry_existing_buckets,
            "miss_retry_existing_bucket_count": len(miss_retry_existing_buckets),
            "primary_bucket": primary,
            "multiprobe_buckets": multiprobe,
            "decoy_buckets": decoys,
            "requested_bucket_ids": list(dict.fromkeys(requested_bucket_ids_effective)),
            "result_chunk_count": len(payload["results"]),
            "distance_rows": distance_rows,
            "timings_ms": {
                "build_plan": plan_ms,
                "encrypt_probe": encrypt_ms,
                "request_roundtrip": request_ms,
                "decrypt_and_parse": decrypt_parse_ms,
                "total": total_ms,
            },
        }

    def authenticate_embedding(self, embedding) -> str | None:
        diagnostics = self.authenticate_embedding_diagnostics(embedding)
        return diagnostics["winner_uuid"]