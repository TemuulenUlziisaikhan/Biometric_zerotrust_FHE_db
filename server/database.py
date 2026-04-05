from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock

from rocksdict import Rdict

FACES_PER_CHUNK = 16


class DataConsistencyError(ValueError):
    pass


class Database:
    def __init__(self, db_path: str = "server_data.rocks") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = Rdict(db_path)
        self._bucket_key_cache: dict[str, list[str]] | None = None
        self._cache_lock = Lock()

    @staticmethod
    def make_chunk_key(bucket_id: str, chunk_index: int) -> str:
        return f"{bucket_id}_Chunk_{chunk_index}"

    @staticmethod
    def parse_chunk_index(chunk_key: str) -> int:
        return int(chunk_key.rsplit("_Chunk_", maxsplit=1)[-1])

    @staticmethod
    def _bucket_from_chunk_key(chunk_key: str) -> str:
        return chunk_key.split("_Chunk_", maxsplit=1)[0]

    def _build_bucket_key_cache(self) -> dict[str, list[str]]:
        bucket_map: dict[str, list[str]] = {}
        for key in self.db.keys():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            key_str = str(key)
            if "_Chunk_" not in key_str:
                continue
            bucket_id = self._bucket_from_chunk_key(key_str)
            bucket_map.setdefault(bucket_id, []).append(key_str)
        for bucket_id, keys in bucket_map.items():
            keys.sort(key=self.parse_chunk_index)
            bucket_map[bucket_id] = keys
        return bucket_map

    def _ensure_bucket_key_cache(self) -> dict[str, list[str]]:
        with self._cache_lock:
            if self._bucket_key_cache is None:
                cache_started = time.time()
                self._bucket_key_cache = self._build_bucket_key_cache()
                total_keys = sum(len(keys) for keys in self._bucket_key_cache.values())
                print(
                    f"Built in-memory bucket key cache for {len(self._bucket_key_cache)} buckets "
                    f"({total_keys} chunk keys) in {time.time() - cache_started:.2f} seconds"
                )
            return self._bucket_key_cache

    def _cache_add_chunk_key(self, chunk_key: str) -> None:
        with self._cache_lock:
            if self._bucket_key_cache is None:
                return
            bucket_id = self._bucket_from_chunk_key(chunk_key)
            keys = self._bucket_key_cache.setdefault(bucket_id, [])
            if chunk_key not in keys:
                keys.append(chunk_key)
                keys.sort(key=self.parse_chunk_index)

    def _cache_remove_chunk_key(self, chunk_key: str) -> None:
        with self._cache_lock:
            if self._bucket_key_cache is None:
                return
            bucket_id = self._bucket_from_chunk_key(chunk_key)
            keys = self._bucket_key_cache.get(bucket_id)
            if not keys:
                return
            try:
                keys.remove(chunk_key)
            except ValueError:
                return
            if not keys:
                self._bucket_key_cache.pop(bucket_id, None)

    def read_chunk_record(self, chunk_key: str) -> dict | None:
        chunk_read_time = time.time()
        raw = self.db.get(chunk_key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        record = json.loads(raw)
        print(f"Read chunk record {chunk_key} from database in {time.time() - chunk_read_time:.2f} seconds")
        self.validate_chunk_record(chunk_key, record)
        print(f"Read and validated chunk record {chunk_key} in {time.time() - chunk_read_time:.2f} seconds")
        return record

    def write_chunk_record(self, chunk_key: str, record: dict) -> None:
        self.db[chunk_key] = json.dumps(record)
        self._cache_add_chunk_key(chunk_key)

    def delete_chunk_record(self, chunk_key: str) -> None:
        try:
            del self.db[chunk_key]
        except KeyError:
            return
        self._cache_remove_chunk_key(chunk_key)

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
        bucket_map = self._ensure_bucket_key_cache()
        return list(bucket_map.get(bucket_id, []))

    def all_chunk_keys(self) -> list[str]:
        bucket_map = self._ensure_bucket_key_cache()
        keys: list[str] = []
        for bucket_id in sorted(bucket_map.keys()):
            keys.extend(bucket_map[bucket_id])
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
        with self._cache_lock:
            self._bucket_key_cache = {}
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
        bucket_read_time = time.time()
        print("Loading chunk records for buckets", bucket_ids)
        for bucket in bucket_ids:
            for key in self._bucket_chunk_keys(bucket):
                record = self.read_chunk_record(key)
                if record is not None and record.get("packed_ciphertext"):
                    rows.append((key, record))
        print(f"Loaded {len(rows)} chunk records for buckets {bucket_ids} in {time.time() - bucket_read_time:.2f} seconds")
        return rows