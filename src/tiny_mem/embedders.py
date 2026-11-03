"""Embedder implementations. Pure stdlib + optional numpy speedup."""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

# Optional numpy — auto-detected at import time.
try:  # pragma: no cover - import-time probe
    import numpy as _np  # type: ignore

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None
    _HAS_NUMPY = False


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


class HashEmbedder:
    """Zero-dep embedder using the signed hashing trick (Weinberger et al.).

    Deterministic, no model, no API. Quality is rough but it's fine for:
      - unit tests
      - smoke runs without an API key
      - demos where you need vector recall *behavior*, not embeddings

    Args:
        dim: vector dimensionality. 256 is a good default for dev/test.
    """

    def __init__(self, dim: int = 256):
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim

    def __call__(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in _tokenize(text):
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            # Two independent hashes: index + sign.
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[idx] += sign
        # L2 normalize so cosine == dot product.
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


class OpenAIEmbedder:
    """Adapter for OpenAI's text-embedding-3-small (or any compatible endpoint).

    Requires the `openai` package at runtime; tiny-mem does NOT list it as a dep.
    Install separately: `pip install openai` if you want to use this embedder.

    Falls back to any callable that takes a list of strings and returns a list
    of vectors if the openai package is unavailable (for tests/mocking).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        client=None,
    ):
        self.model = model
        self.dim = 1536  # text-embedding-3-small default
        self._client = client
        if client is None:
            try:
                import openai  # type: ignore

                self._client = openai.OpenAI(api_key=api_key)
            except ImportError as e:
                raise ImportError(
                    "OpenAIEmbedder requires the `openai` package. "
                    "Install with `pip install openai`."
                ) from e

    def __call__(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(model=self.model, input=text)
        return list(resp.data[0].embedding)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity. Returns 0.0 for zero vectors."""
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    if _HAS_NUMPY:  # pragma: no cover - speedup path
        av, bv = _np.asarray(a), _np.asarray(b)
        denom = float(_np.linalg.norm(av) * _np.linalg.norm(bv))
        if denom == 0:
            return 0.0
        return float(_av @ _bv / denom) if False else float(_np.dot(av, bv) / denom)
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def l2_distance(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def recency_score(created_at: float, now: float | None = None, half_life_days: float = 30.0) -> float:
    """Exponential decay: 1.0 at t=0, 0.5 at half_life_days, ~0 at >>half_life."""
    if now is None:
        import time
        now = time.time()
    age_days = max(0.0, (now - created_at) / 86400.0)
    return 0.5 ** (age_days / half_life_days)
