from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np


@dataclass
class LSHEngine:
    bits: int = 12
    seed: int = 7
    in_dim: int = 512

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._projection = rng.standard_normal((self.bits, self.in_dim), dtype=np.float32)

    def hash_embedding_to_bucket(self, embedding: np.ndarray) -> str:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.in_dim:
            raise ValueError(f"Expected embedding dimension {self.in_dim}, got {vec.shape[0]}")
        dots = self._projection @ vec
        bits = (dots >= 0).astype(np.int8)
        bit_str = "".join(str(int(x)) for x in bits.tolist())
        return f"Bucket_{bit_str}"

    def _bit_flip_priority(
        self,
        primary_bucket: str,
        strategy: str,
        embedding: np.ndarray | None,
    ) -> list[int]:
        bit_str = primary_bucket.split("Bucket_")[-1]
        if len(bit_str) != self.bits:
            raise ValueError("Invalid primary bucket bit-length")

        if strategy == "sequential":
            return list(range(self.bits))

        if strategy == "margin":
            if embedding is None:
                raise ValueError("embedding is required for margin probe strategy")
            vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if vec.shape[0] != self.in_dim:
                raise ValueError(f"Expected embedding dimension {self.in_dim}, got {vec.shape[0]}")
            dots = self._projection @ vec
            return np.argsort(np.abs(dots)).astype(np.int32).tolist()

        raise ValueError("Unsupported probe strategy. Use 'sequential' or 'margin'.")

    def generate_multiprobe_buckets(
        self,
        primary_bucket: str,
        probes: int = 2,
        strategy: str = "sequential",
        embedding: np.ndarray | None = None,
        hamming_radius: int = 1,
    ) -> list[str]:
        if probes < 0:
            raise ValueError("probes must be non-negative")
        if hamming_radius < 1:
            raise ValueError("hamming_radius must be >= 1")

        bit_str = primary_bucket.split("Bucket_")[-1]
        if len(bit_str) != self.bits:
            raise ValueError("Invalid primary bucket bit-length")

        priority = self._bit_flip_priority(primary_bucket, strategy=strategy, embedding=embedding)
        buckets: list[str] = []

        for radius in range(1, hamming_radius + 1):
            for flip_indexes in itertools.combinations(priority, radius):
                mutated = list(bit_str)
                for index in flip_indexes:
                    mutated[index] = "1" if mutated[index] == "0" else "0"
                candidate = f"Bucket_{''.join(mutated)}"
                if candidate == primary_bucket or candidate in buckets:
                    continue
                buckets.append(candidate)
                if len(buckets) >= probes:
                    return buckets
        return buckets

    def generate_decoy_buckets(self, exclude: set[str], count: int = 3) -> list[str]:
        rng = np.random.default_rng(self.seed + 991)
        decoys: list[str] = []
        while len(decoys) < count:
            bits = "".join(str(int(x)) for x in rng.integers(0, 2, self.bits).tolist())
            candidate = f"Bucket_{bits}"
            if candidate in exclude or candidate in decoys:
                continue
            decoys.append(candidate)
        return decoys

    def build_auth_bucket_request(self, primary: str, multiprobe: list[str], decoys: list[str]) -> list[str]:
        all_ids = [primary, *multiprobe, *decoys]
        return list(dict.fromkeys(all_ids))