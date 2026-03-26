from __future__ import annotations

import base64
import hashlib
import time

import tenseal as ts


def eval_context_fingerprint(context_b64: str) -> str:
    raw = base64.b64decode(context_b64.encode("utf-8"))
    return hashlib.sha256(raw).hexdigest()


def deserialize_eval_context(context_b64: str) -> ts.Context:
    raw = base64.b64decode(context_b64.encode("utf-8"))
    return ts.context_from(raw)


def deserialize_ciphertext(context: ts.Context, ciphertext_b64: str) -> ts.CKKSVector:
    time_watch = time.time()
    raw = base64.b64decode(ciphertext_b64.encode("utf-8"))
    print(f"Deserialization took {time.time() - time_watch:.2f} seconds")
    return ts.ckks_vector_from(context, raw)


def serialize_ciphertext(ciphertext: ts.CKKSVector) -> str:
    raw = ciphertext.serialize()
    return base64.b64encode(raw).decode("utf-8")


def homomorphic_enroll_add(existing_chunk_ct: ts.CKKSVector, incoming_sparse_ct: ts.CKKSVector) -> ts.CKKSVector:
    return existing_chunk_ct + incoming_sparse_ct


def homomorphic_sum_ciphertexts(ciphertexts: list[ts.CKKSVector]) -> ts.CKKSVector:
    if not ciphertexts:
        raise ValueError("ciphertexts must not be empty")
    acc = ciphertexts[0]
    for ciphertext in ciphertexts[1:]:
        acc = acc + ciphertext
    return acc


def homomorphic_squared_distance(chunk_ct: ts.CKKSVector, probe_ct: ts.CKKSVector) -> ts.CKKSVector:
    diff = probe_ct - chunk_ct
    return diff * diff