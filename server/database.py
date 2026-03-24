from __future__ import annotations

import json
from pathlib import Path

from rocksdict import Rdict

FACES_PER_CHUNK = 16


class DataConsistencyError(ValueError):
    pass


class Database:
    def __init__(self, db_path: str = "server_data.rocks") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = Rdict(db_path)

    @staticmethod
    def make_chunk_key(bucket_id: str, chunk_index: int) -> str:
        return f"{bucket_id}_Chunk_{chunk_index}"

    @staticmethod
    def parse_chunk_index(chunk_key: str) -> int:
        return int(chunk_key.rsplit("_Chunk_", maxsplit=1)[-1])

    def read_chunk_record(self, chunk_key: str) -> dict | None:
        raw = self.db.get(chunk_key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        record = json.loads(raw)
        self.validate_chunk_record(chunk_key, record)
        return record

    def write_chunk_record(self, chunk_key: str, record: dict) -> None:
        self.db[chunk_key] = json.dumps(record)

    def delete_chunk_record(self, chunk_key: str) -> None:
        try:
            del self.db[chunk_key]
        except KeyError:
            return

    def create_empty_chunk_record(self, bucket_id: str, chunk_index: int) -> dict:
        return {
            "bucket_id": self.make_chunk_key(bucket_id, chunk_index),
            "packed_ciphertext": None,
            "current_face_count": 0,
            "uuid_map": [],
            "face_ciphertexts": [],
        }

    def validate_chunk_record(self, chunk_key: str, record: dict) -> None:
        required = {"bucket_id", "packed_ciphertext", "current_face_count", "uuid_map"}
        missing = required - set(record.keys())
        if missing:
            raise DataConsistencyError(f"Record {chunk_key} missing required keys: {sorted(missing)}")

        if record["bucket_id"] != chunk_key:
            raise DataConsistencyError(
                f"Record {chunk_key} has mismatched bucket_id {record['bucket_id']}"
            )

        face_count = int(record["current_face_count"])
        if face_count < 0 or face_count > FACES_PER_CHUNK:
            raise DataConsistencyError(f"Record {chunk_key} has invalid current_face_count={face_count}")

        uuid_map = record["uuid_map"]
        if not isinstance(uuid_map, list):
            raise DataConsistencyError(f"Record {chunk_key} has non-list uuid_map")

        if len(uuid_map) != face_count:
            raise DataConsistencyError(
                f"Record {chunk_key} uuid_map length {len(uuid_map)} does not match current_face_count {face_count}"
            )

        face_ciphertexts = record.get("face_ciphertexts")
        if face_ciphertexts is not None:
            if not isinstance(face_ciphertexts, list):
                raise DataConsistencyError(f"Record {chunk_key} has non-list face_ciphertexts")
            if len(face_ciphertexts) != face_count:
                raise DataConsistencyError(
                    f"Record {chunk_key} face_ciphertexts length {len(face_ciphertexts)} does not match current_face_count {face_count}"
                )

        if face_count > 0 and not record["packed_ciphertext"]:
            raise DataConsistencyError(f"Record {chunk_key} has faces but missing packed_ciphertext")

    def _bucket_chunk_keys(self, bucket_id: str) -> list[str]:
        keys: list[str] = []
        prefix = f"{bucket_id}_Chunk_"
        for key in self.db.keys():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            if str(key).startswith(prefix):
                keys.append(str(key))
        keys.sort(key=self.parse_chunk_index)
        return keys

    def all_chunk_keys(self) -> list[str]:
        keys: list[str] = []
        for key in self.db.keys():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            key_str = str(key)
            if "_Chunk_" in key_str:
                keys.append(key_str)
        keys.sort(key=lambda value: (value.split("_Chunk_", maxsplit=1)[0], self.parse_chunk_index(value)))
        return keys

    def list_all_chunk_records(self) -> list[tuple[str, dict]]:
        rows: list[tuple[str, dict]] = []
        for key in self.all_chunk_keys():
            record = self.read_chunk_record(key)
            if record is not None:
                rows.append((key, record))
        return rows
    
    def list_numbered_records(self, page: int = 1, page_size: int = 100) -> tuple[list[tuple[str, dict]], int, int, int, int]:
        keys = self.all_chunk_keys()
        print(len(keys), "total chunk records found in database")
        total_chunks = len(keys)
        safe_page_size = max(1, int(page_size))
        total_pages = max(1, (total_chunks + safe_page_size - 1) // safe_page_size)
        safe_page = max(1, min(int(page), total_pages))
        start = (safe_page - 1) * safe_page_size
        end = start + safe_page_size
        keys_slice = keys[start:end]
        
        rows: list[tuple[str, dict]] = []
        total_labels = 0
        for key in keys_slice:
            record = self.read_chunk_record(key)
            if record is not None:
                record = dict(record)
                labels = record.get("uuid_map", [])
                total_labels += len(labels)
                record.pop("face_ciphertexts", None)
                record.pop("packed_ciphertext", None)
                rows.append((key, record))
        return rows, total_pages, total_chunks, safe_page, total_labels

    def clear_all_chunk_records(self) -> int:
        keys = self.all_chunk_keys()
        for key in keys:
            self.delete_chunk_record(key)
        return len(keys)

    def resolve_enroll_position(self, bucket_id: str) -> tuple[int, int, int]:
        keys = self._bucket_chunk_keys(bucket_id)
        if not keys:
            return 1, 0, 0
        last_key = keys[-1]
        record = self.read_chunk_record(last_key)
        if record is None:
            return 1, 0, 0
        current_face_count = int(record["current_face_count"])
        last_chunk_index = self.parse_chunk_index(last_key)
        if current_face_count >= FACES_PER_CHUNK:
            return last_chunk_index + 1, 0, 0
        return last_chunk_index, current_face_count, current_face_count

    def append_uuid_and_increment(self, record: dict, user_uuid: str) -> dict:
        if int(record["current_face_count"]) >= FACES_PER_CHUNK:
            raise ValueError("Chunk is full")
        record["uuid_map"].append(user_uuid)
        record["current_face_count"] = int(record["current_face_count"]) + 1
        return record

    def load_bucket_chunks(self, bucket_ids: list[str]) -> list[tuple[str, dict]]:
        rows: list[tuple[str, dict]] = []
        for bucket in bucket_ids:
            for key in self._bucket_chunk_keys(bucket):
                record = self.read_chunk_record(key)
                if record is not None and record.get("packed_ciphertext"):
                    rows.append((key, record))
        return rows