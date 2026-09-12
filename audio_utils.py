"""Audio normalization helpers used before Qwen voice prompt creation."""

import numpy as np


def normalize_reference_audio(audio: np.ndarray) -> np.ndarray:
    normalized = np.asarray(audio)
    if normalized.ndim > 1:
        normalized = np.mean(normalized, axis=-1)
    return normalized.astype(np.float32, copy=False)
