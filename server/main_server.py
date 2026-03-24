from __future__ import annotations

import logging
import math
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from server.api_models import (
    API_VERSION,
    BUCKET_BITS,
    BUCKET_LEN,
    AuthenticateChunkResult,
    AuthenticateRequest,
    AuthenticateResponse,
    DBLabelsChunkRow,
    DBLabelsResponse,
    DBRecordRow,
    DBRecordsResponse,
    DeleteLabelRequest,
    DeleteLabelResponse,
    EnrollOffsetRequest,
    EnrollOffsetResponse,
    EnrollRequest,
    EnrollResponse,
    HealthResponse,
    ResetDBRequest,
    ResetDBResponse,
)
from server.database import DataConsistencyError, Database
from server.fhe_server import (
    deserialize_ciphertext,
    deserialize_eval_context,
    eval_context_fingerprint,
    homomorphic_enroll_add,
    homomorphic_sum_ciphertexts,
    homomorphic_squared_distance,
    serialize_ciphertext,
)

logger = logging.getLogger("server.audit")
logger_debug = logging.getLogger("server.debug")


def _error_detail(request: Request, code: str, message: str, errors: list[dict] | None = None) -> dict:
    detail = {
        "code": code,
        "message": message,
        "request_id": getattr(request.state, "request_id", "unknown"),
    }
    if errors is not None:
        detail["errors"] = errors
    return detail


def _http_error(request: Request, status_code: int, code: str, message: str) -> HTTPException:
    # Never expose raw exception details to clients; use generic messages
    return HTTPException(status_code=status_code, detail=_error_detail(request, code, message))


def _sanitize_exception(exc: Exception) -> tuple[str, str]:
    """
    Convert raw exception to generic error code + safe message for client.
    Full exception is logged server-side at debug level for troubleshooting.
    
    Returns: (error_code, safe_message)
    """
    exc_type = type(exc).__name__
    logger_debug.debug(f"Exception type={exc_type} message={str(exc)}", exc_info=True)
    
    # Map exception types to generic codes; never expose details to client
    if "Crypto" in exc_type or "tenseal" in str(exc).lower():
        return "INVALID_CRYPTO_PAYLOAD", "Invalid cryptographic payload"
    elif "Validation" in exc_type:
        return "INVALID_PAYLOAD", "Invalid request payload"
    else:
        return "INTERNAL_ERROR", "An error occurred processing your request"


def _storage_error(request: Request, action: str, exc: Exception) -> HTTPException:
    logger_debug.error("Storage inconsistency in %s: %s", action, str(exc), exc_info=True)
    return _http_error(request, 500, "STORAGE_ERROR", "Failed to access storage")


def _crypto_payload_error(request: Request, exc: Exception) -> HTTPException:
    code, msg = _sanitize_exception(exc)
    return _http_error(request, 400, code, msg)


app = FastAPI(title="Sub-Linear FHE Biometric Server")
DB_PATH = os.getenv("DB_PATH", "server/biometric.rocks")
db = Database(db_path=DB_PATH)


@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "request_done request_id=%s method=%s path=%s status=%d duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response

@app.exception_handler(RequestValidationError)
def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": _error_detail(request, "VALIDATION_ERROR", "Invalid request payload", errors=exc.errors())
        },
    )


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "HTTP_ERROR", "message": str(exc.detail)}
    code = str(detail.get("code", "HTTP_ERROR"))
    message = str(detail.get("message", "Request failed"))
    errors = detail.get("errors") if isinstance(detail.get("errors"), list) else None
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": _error_detail(request, code, message, errors=errors)},
    )


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "unhandled_exception request_id=%s method=%s path=%s",
        getattr(request.state, "request_id", "unknown"),
        request.method,
        request.url.path,
    )
    code, message = _sanitize_exception(exc)
    return JSONResponse(
        status_code=500,
        content={"detail": _error_detail(request, code, message)},
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    logger.info("health_ok request_id=%s", request.state.request_id)
    return HealthResponse(api_version=API_VERSION, status="ok")


@app.get("/db/labels", response_model=DBLabelsResponse)
def db_labels(request: Request) -> DBLabelsResponse:
    try:
        records = db.list_all_chunk_records()
    except DataConsistencyError as exc:
        raise _storage_error(request, "db_labels", exc)

    rows: list[DBLabelsChunkRow] = []
    total_labels = 0
    for chunk_key, record in records:
        labels = [str(item) for item in record.get("uuid_map", [])]
        total_labels += len(labels)
        rows.append(
            DBLabelsChunkRow(
                bucket_id=record["bucket_id"].split("_Chunk_")[0],
                chunk_index=db.parse_chunk_index(chunk_key),
                current_face_count=int(record["current_face_count"]),
                labels=labels,
            )
        )

    logger.info(
        "db_labels request_id=%s chunk_count=%d label_count=%d",
        request.state.request_id,
        len(rows),
        total_labels,
    )
    return DBLabelsResponse(
        api_version=API_VERSION,
        total_chunks=len(rows),
        total_labels=total_labels,
        rows=rows,
    )


@app.get("/db/records", response_model=DBRecordsResponse)
def db_records(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    label: str = Query(default="", max_length=128),
) -> DBRecordsResponse:
    try:
        records, total_pages, total_chunks, effective_page, total_labels = db.list_numbered_records(page=page, page_size=page_size)
    except DataConsistencyError as exc:
        raise _storage_error(request, "db_records", exc)

    label_query = label.strip()

    rows: list[DBRecordRow] = []
    for chunk_key, record in records:
        labels = [str(item) for item in record.get("uuid_map", [])]
        rows.append(
            DBRecordRow(
                chunk_key=chunk_key,
                bucket_id=record["bucket_id"].split("_Chunk_")[0],
                labels=labels,
            )
        )

    logger.info(
        "db_records request_id=%s chunk_count=%d label_count=%d page=%d page_size=%d total_pages=%d label_query=%s",
        request.state.request_id,
        total_chunks,
        total_labels,
        effective_page,
        page_size,
        total_pages,
        label_query,
    )
    
    return DBRecordsResponse(
        api_version=API_VERSION,
        total_chunks=total_chunks,
        total_labels=total_labels,
        page=effective_page,
        page_size=page_size,
        total_pages=total_pages,
        label_query=label_query,
        rows=rows,
    )


@app.post("/admin/delete-label", response_model=DeleteLabelResponse)
def admin_delete_label(request: Request, req: DeleteLabelRequest) -> DeleteLabelResponse:
    try:
        ctx_fingerprint = eval_context_fingerprint(req.eval_context_b64)
        context = deserialize_eval_context(req.eval_context_b64)
    except Exception as exc:
        raise _crypto_payload_error(request, exc)

    try:
        records = db.list_all_chunk_records()
    except DataConsistencyError as exc:
        raise _storage_error(request, "delete-label", exc)

    deleted_count = 0
    affected_chunks = 0
    blocked_chunks: list[str] = []

    for chunk_key, record in records:
        labels = [str(item) for item in record.get("uuid_map", [])]
        if req.user_uuid not in labels:
            continue

        existing_fp = record.get("context_fingerprint")
        if existing_fp is not None and existing_fp != ctx_fingerprint:
            blocked_chunks.append(chunk_key)
            continue

        keep_indexes = [idx for idx, label in enumerate(labels) if label != req.user_uuid]
        delete_indexes = [idx for idx, label in enumerate(labels) if label == req.user_uuid]
        if not delete_indexes:
            continue

        face_ciphertexts = record.get("face_ciphertexts")
        if face_ciphertexts is None:
            if keep_indexes:
                blocked_chunks.append(chunk_key)
                continue
            db.delete_chunk_record(chunk_key)
            deleted_count += len(delete_indexes)
            affected_chunks += 1
            continue

        remaining_face_ciphertexts = [face_ciphertexts[idx] for idx in keep_indexes]
        if any(item is None for item in remaining_face_ciphertexts):
            blocked_chunks.append(chunk_key)
            continue

        if keep_indexes:
            rebuilt_ciphertexts = [
                deserialize_ciphertext(context, str(face_ciphertexts[idx]))
                for idx in keep_indexes
            ]
            rebuilt = homomorphic_sum_ciphertexts(rebuilt_ciphertexts)
            record["packed_ciphertext"] = serialize_ciphertext(rebuilt)
            record["uuid_map"] = [labels[idx] for idx in keep_indexes]
            record["face_ciphertexts"] = [face_ciphertexts[idx] for idx in keep_indexes]
            record["current_face_count"] = len(keep_indexes)
            db.write_chunk_record(chunk_key, record)
        else:
            db.delete_chunk_record(chunk_key)

        deleted_count += len(delete_indexes)
        affected_chunks += 1

    logger.info(
        "admin_delete_label request_id=%s user_uuid=%s deleted_count=%d affected_chunks=%d blocked_chunks=%d",
        request.state.request_id,
        req.user_uuid,
        deleted_count,
        affected_chunks,
        len(blocked_chunks),
    )
    return DeleteLabelResponse(
        api_version=API_VERSION,
        deleted_count=deleted_count,
        affected_chunks=affected_chunks,
        blocked_chunks=blocked_chunks,
    )


@app.post("/admin/reset", response_model=ResetDBResponse)
def admin_reset(request: Request, req: ResetDBRequest) -> ResetDBResponse:
    if req.confirm_text != "RESET_DB":
        raise _http_error(request, 400, "INVALID_CONFIRM_TEXT", "confirm_text must equal RESET_DB")

    deleted_chunks = db.clear_all_chunk_records()
    logger.info(
        "admin_reset request_id=%s deleted_chunks=%d",
        request.state.request_id,
        deleted_chunks,
    )
    return ResetDBResponse(api_version=API_VERSION, deleted_chunks=deleted_chunks, status="reset")


@app.post("/enroll/offset", response_model=EnrollOffsetResponse)
def enroll_offset(request: Request, req: EnrollOffsetRequest) -> EnrollOffsetResponse:
    chunk_index, slot_offset, current_face_count = db.resolve_enroll_position(req.bucket_id)
    logger.info(
        "enroll_offset request_id=%s bucket_id=%s chunk_index=%d slot_offset=%d face_count=%d",
        request.state.request_id,
        req.bucket_id,
        chunk_index,
        slot_offset,
        current_face_count,
    )
    return EnrollOffsetResponse(
        api_version=API_VERSION,
        bucket_id=req.bucket_id,
        chunk_index=chunk_index,
        slot_offset=slot_offset,
        current_face_count=current_face_count,
    )


@app.post("/enroll", response_model=EnrollResponse)
def enroll(request: Request, req: EnrollRequest) -> EnrollResponse:
    try:
        ctx_fingerprint = eval_context_fingerprint(req.eval_context_b64)
        context = deserialize_eval_context(req.eval_context_b64)
        incoming_ct = deserialize_ciphertext(context, req.sparse_ciphertext_b64)
    except Exception as exc:
        raise _crypto_payload_error(request, exc)

    chunk_key = db.make_chunk_key(req.bucket_id, req.chunk_index)
    try:
        record = db.read_chunk_record(chunk_key)
        if record is None:
            record = db.create_empty_chunk_record(req.bucket_id, req.chunk_index)
    except DataConsistencyError as exc:
        raise _storage_error(request, "enroll", exc)

    try:
        existing_fp = record.get("context_fingerprint")
        if existing_fp is not None and existing_fp != ctx_fingerprint:
            raise _http_error(request, 409, "CONTEXT_MISMATCH", "Evaluation context does not match existing chunk")

        if "face_ciphertexts" not in record:
            record["face_ciphertexts"] = [None] * int(record["current_face_count"])

        if record["packed_ciphertext"]:
            existing_ct = deserialize_ciphertext(context, record["packed_ciphertext"])
            merged_ct = homomorphic_enroll_add(existing_ct, incoming_ct)
        else:
            merged_ct = incoming_ct
        record["packed_ciphertext"] = serialize_ciphertext(merged_ct)
        record["context_fingerprint"] = ctx_fingerprint
        record = db.append_uuid_and_increment(record, req.user_uuid)
        record["face_ciphertexts"].append(req.sparse_ciphertext_b64)
        db.write_chunk_record(chunk_key, record)
    except ValueError as exc:
        raise _http_error(request, 409, "CHUNK_FULL", str(exc))

    logger.info(
        "enroll request_id=%s bucket_id=%s chunk_index=%d face_count=%d",
        request.state.request_id,
        req.bucket_id,
        req.chunk_index,
        int(record["current_face_count"]),
    )

    return EnrollResponse(
        api_version=API_VERSION,
        status="enrolled",
        bucket_id=req.bucket_id,
        chunk_index=req.chunk_index,
        current_face_count=int(record["current_face_count"]),
    )


@app.post("/authenticate", response_model=AuthenticateResponse)
def authenticate(request: Request, req: AuthenticateRequest) -> AuthenticateResponse:
    for bucket_id in req.bucket_ids:
        if len(bucket_id) != BUCKET_LEN or not bucket_id.startswith("Bucket_"):
            raise _http_error(
                request,
                400,
                "INVALID_BUCKET_ID",
                f"bucket_ids entries must match Bucket_[01]{{{BUCKET_BITS}}}",
            )
        suffix = bucket_id.split("Bucket_", maxsplit=1)[-1]
        if len(suffix) != BUCKET_BITS or any(ch not in {"0", "1"} for ch in suffix):
            raise _http_error(
                request,
                400,
                "INVALID_BUCKET_ID",
                f"bucket_ids entries must match Bucket_[01]{{{BUCKET_BITS}}}",
            )

    try:
        ctx_fingerprint = eval_context_fingerprint(req.eval_context_b64)
        context = deserialize_eval_context(req.eval_context_b64)
        probe_ct = deserialize_ciphertext(context, req.probe_ciphertext_b64)
    except Exception as exc:
        raise _crypto_payload_error(request, exc)

    try:
        records = db.load_bucket_chunks(req.bucket_ids)
    except DataConsistencyError as exc:
        raise _storage_error(request, "authenticate", exc)

    results: list[AuthenticateChunkResult] = []
    for chunk_key, record in records:
        existing_fp = record.get("context_fingerprint")
        if existing_fp is not None and existing_fp != ctx_fingerprint:
            raise _http_error(request, 409, "CONTEXT_MISMATCH", "Evaluation context mismatch for one or more chunks")

        chunk_ct = deserialize_ciphertext(context, record["packed_ciphertext"])
        d2_ct = homomorphic_squared_distance(chunk_ct, probe_ct)
        results.append(
            AuthenticateChunkResult(
                bucket_id=record["bucket_id"].split("_Chunk_")[0],
                chunk_index=db.parse_chunk_index(chunk_key),
                uuid_map=record["uuid_map"],
                encrypted_distance_b64=serialize_ciphertext(d2_ct),
            )
        )

    logger.info(
        "authenticate request_id=%s bucket_count=%d chunk_count=%d",
        request.state.request_id,
        len(req.bucket_ids),
        len(results),
    )
    return AuthenticateResponse(api_version=API_VERSION, results=results)