from __future__ import annotations

import os

from pydantic import BaseModel, Field

API_VERSION = "v1"
MAX_B64_LEN = 100_000_000
BUCKET_BITS = int(os.getenv("LSH_BITS", "9"))
BUCKET_LEN = len("Bucket_") + BUCKET_BITS
BUCKET_PATTERN = rf"^Bucket_[01]{{{BUCKET_BITS}}}$"


class VersionedRequest(BaseModel):
    api_version: str = Field(pattern=r"^v1$")


class VersionedResponse(BaseModel):
    api_version: str = API_VERSION


class EnrollOffsetRequest(VersionedRequest):
    bucket_id: str = Field(min_length=BUCKET_LEN, max_length=BUCKET_LEN, pattern=BUCKET_PATTERN)


class EnrollOffsetResponse(VersionedResponse):
    bucket_id: str
    chunk_index: int
    slot_offset: int
    current_face_count: int


class EnrollRequest(VersionedRequest):
    bucket_id: str = Field(min_length=BUCKET_LEN, max_length=BUCKET_LEN, pattern=BUCKET_PATTERN)
    user_uuid: str = Field(min_length=1, max_length=128)
    sparse_ciphertext_b64: str = Field(min_length=64, max_length=MAX_B64_LEN)
    chunk_index: int = Field(ge=1, le=1_000_000)


class EnrollResponse(VersionedResponse):
    status: str
    bucket_id: str
    chunk_index: int
    current_face_count: int


class AuthenticateRequest(VersionedRequest):
    bucket_ids: list[str] = Field(min_length=1, max_length=256)
    probe_ciphertext_b64: str = Field(min_length=64, max_length=MAX_B64_LEN)


class AuthenticateChunkResult(BaseModel):
    bucket_id: str
    chunk_index: int
    uuid_map: list[str]
    encrypted_distance_b64: str


class AuthenticateResponse(VersionedResponse):
    results: list[AuthenticateChunkResult]


class HealthResponse(VersionedResponse):
    status: str


class DBLabelsChunkRow(BaseModel):
    bucket_id: str
    chunk_index: int
    current_face_count: int
    labels: list[str]


class DBLabelsResponse(VersionedResponse):
    total_chunks: int
    total_labels: int
    rows: list[DBLabelsChunkRow]


class DBRecordRow(BaseModel):
    """Database record row for admin viewing. Does NOT expose encrypted data or internal structures."""
    chunk_key: str
    bucket_id: str
    labels: list[str]


class DBRecordsResponse(VersionedResponse):
    total_chunks: int
    total_labels: int
    page: int
    page_size: int
    total_pages: int
    label_query: str
    rows: list[DBRecordRow]


class DeleteLabelRequest(VersionedRequest):
    user_uuid: str = Field(min_length=1, max_length=128)


class DeleteLabelResponse(VersionedResponse):
    deleted_count: int
    affected_chunks: int
    blocked_chunks: list[str]


class ResetDBRequest(VersionedRequest):
    confirm_text: str = Field(min_length=1, max_length=64)


class ResetDBResponse(VersionedResponse):
    deleted_chunks: int
    status: str
