from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from mythings.embed import Vector

# Real semantic Embedder implementations backed by open-source model libraries.
# my-embedder is to embed.Embedder what my-guard is to policy.Policy: a package
# that fills a core Protocol with real machinery, so the heavy dependency
# (fastembed's onnxruntime, sentence-transformers' torch) lives in *one* repo and
# never leaks into the dependency-free fleet. Anything that only needs vectors
# points core's ApiEmbedder at this package's `serve` endpoint (server.py)
# instead of importing it.

FASTEMBED_DEFAULT = "BAAI/bge-small-en-v1.5"
SENTENCE_TRANSFORMERS_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"

# The headline default: English, small, ONNX (no torch). MULTILINGUAL_MODEL is
# the one to reach for on the Italian/English study corpus — it closes the
# cross-language gap ADR 0003 measured. Both are fastembed-supported.
DEFAULT_MODEL = FASTEMBED_DEFAULT
MULTILINGUAL_MODEL = "BAAI/bge-m3"

# An Encoder maps a batch of texts to one row of floats each, in order. It is the
# injectable seam that keeps the heavy library out of the test path: a fake
# encoder stands in, same discipline as engine.Runner / github.Runner. The
# default encoders below soft-import their library on first use.
Encoder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def _normalize(values: list[float]) -> Vector:
    norm = math.sqrt(sum(v * v for v in values))
    if not norm:
        return tuple(values)
    return tuple(v / norm for v in values)


class _LibraryEmbedder:
    # Shared plumbing for the library-backed backends: normalise every row to
    # unit length, and never silently drop or reorder rows — one vector per input
    # text, in order, or raise. Subclasses supply only the default encoder, which
    # is soft-imported and built once on the first embed().
    def __init__(self, model: str, *, encoder: Encoder | None = None) -> None:
        self.model = model
        self._encoder = encoder

    def _default_encoder(self) -> Encoder:  # pragma: no cover - overridden
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        items = list(texts)
        if not items:
            return []
        if self._encoder is None:
            self._encoder = self._default_encoder()
        rows = list(self._encoder(items))
        # A backend that drops or reorders a row silently misaligns every caller
        # that zips vectors back to its documents (MyCartographer does exactly
        # this) — so one vector per input text, in order, or fail loud.
        if len(rows) != len(items):
            raise ValueError(f"encoder returned {len(rows)} vectors for {len(items)} texts")
        return [_normalize([float(x) for x in row]) for row in rows]


class FastEmbedEmbedder(_LibraryEmbedder):
    # Wraps fastembed (Qdrant) — ONNX runtime, no PyTorch, ARM-friendly.
    def __init__(self, model: str = FASTEMBED_DEFAULT, *, encoder: Encoder | None = None) -> None:
        super().__init__(model, encoder=encoder)

    def _default_encoder(self) -> Encoder:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is not installed — run `pip install 'my-embedder[fastembed]'`"
            ) from exc
        model = TextEmbedding(model_name=self.model)
        return lambda texts: [list(row) for row in model.embed(list(texts))]


class SentenceTransformerEmbedder(_LibraryEmbedder):
    # Wraps sentence-transformers — the de-facto standard, widest model choice,
    # but pulls in PyTorch.
    def __init__(
        self, model: str = SENTENCE_TRANSFORMERS_DEFAULT, *, encoder: Encoder | None = None
    ) -> None:
        super().__init__(model, encoder=encoder)

    def _default_encoder(self) -> Encoder:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed — run "
                "`pip install 'my-embedder[sentence-transformers]'`"
            ) from exc
        model = SentenceTransformer(self.model)
        return lambda texts: [list(row) for row in model.encode(list(texts))]


_BACKENDS: dict[str, type[_LibraryEmbedder]] = {
    "fastembed": FastEmbedEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
}

BACKEND_NAMES = tuple(_BACKENDS)


def build_backend(
    name: str, *, model: str | None = None, encoder: Encoder | None = None
) -> _LibraryEmbedder:
    try:
        cls = _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown backend {name!r}; choose from {', '.join(BACKEND_NAMES)}"
        ) from None
    return cls(model, encoder=encoder) if model else cls(encoder=encoder)
