from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import time
from PIL import Image


_SESSION_CACHE: dict[str, ort.InferenceSession] = {}


def load_arcface_model(model_path: str) -> ort.InferenceSession:
    cached = _SESSION_CACHE.get(model_path)
    if cached is not None:
        return cached
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _SESSION_CACHE[model_path] = session
    return session


def preprocess_image(image_path: str) -> np.ndarray:
    file_path = Path(image_path)
    if file_path.suffix.lower() == ".npy":
        arr = np.load(file_path)
        return np.asarray(arr, dtype=np.float32)

    if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = Image.open(file_path).convert("RGB").resize((112, 112), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32)
        arr = (arr - 127.5) / 128.0
        chw = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    raise ValueError("Unsupported image format. Use .npy or common image files (jpg/png/bmp/webp)")


def load_embedding_file(embedding_path: str) -> np.ndarray:
    file_path = Path(embedding_path)
    if file_path.suffix.lower() != ".npy":
        raise ValueError("Embedding file must be .npy")
    arr = np.load(file_path)
    return validate_embedding_shape(arr)


def extract_embedding_512(session: ort.InferenceSession, preprocessed: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    input_tensor = np.asarray(preprocessed, dtype=np.float32)
    input_shape = session.get_inputs()[0].shape

    if len(input_shape) == 4 and input_tensor.shape == (1, 3, 112, 112):
        channels_index_1 = input_shape[1] == 3 or str(input_shape[1]) == "3"
        channels_index_3 = input_shape[3] == 3 or str(input_shape[3]) == "3"
        if channels_index_3 and not channels_index_1:
            input_tensor = np.transpose(input_tensor, (0, 2, 3, 1))

    raw = session.run(None, {input_name: input_tensor})[0]
    emb = np.asarray(raw, dtype=np.float32).reshape(-1)
    return validate_embedding_shape(emb)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("Embedding norm is zero")
    return vec / norm


def validate_embedding_shape(embedding: np.ndarray) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vec.shape[0] != 512:
        raise ValueError(f"Expected 512-float embedding, got {vec.shape[0]}")
    return vec


def extract_from_image(model_path: str, image_path: str) -> np.ndarray:
    time_watch = time.time()
    session = load_arcface_model(model_path)
    preprocessed = preprocess_image(image_path)
    emb = extract_embedding_512(session, preprocessed)
    print(f"AI Extraction Time: {time.time() - time_watch:.2f} seconds")
    return normalize_embedding(emb)


def extract_embedding(model_path: str | None, image_or_embedding_path: str) -> np.ndarray:
    # time_watch = time.time()
    if model_path:
        return extract_from_image(model_path, image_or_embedding_path)
    embedding = load_embedding_file(image_or_embedding_path)
    return normalize_embedding(embedding)