from __future__ import annotations

import math
from importlib.util import find_spec

import pytest

from myembedder.backends import (
    BACKEND_NAMES,
    DEFAULT_MODEL,
    FastEmbedEmbedder,
    SentenceTransformerEmbedder,
    build_backend,
)


def _fake_encoder(texts):
    # A callable encoder: one deterministic raw (unnormalised) row per text.
    return [[float(len(t)), 3.0] for t in texts]


def test_embed_normalises_to_unit_length() -> None:
    vectors = FastEmbedEmbedder("m", encoder=_fake_encoder).embed(["abcd"])  # [4,3] -> (0.8,0.6)
    assert vectors[0] == pytest.approx((0.8, 0.6))
    assert math.isclose(sum(v * v for v in vectors[0]), 1.0)


def test_embed_is_deterministic_and_ordered() -> None:
    e = SentenceTransformerEmbedder("m", encoder=_fake_encoder)
    a = e.embed(["one", "three"])
    assert a == e.embed(["one", "three"])
    assert a[0][0] < a[1][0]  # order preserved (3 vs 5 chars)


def test_embed_empty_input_returns_empty() -> None:
    assert FastEmbedEmbedder("m", encoder=_fake_encoder).embed([]) == []


def test_embed_handles_a_zero_vector() -> None:
    e = FastEmbedEmbedder("m", encoder=lambda texts: [[0.0, 0.0] for _ in texts])
    assert e.embed(["x"]) == [(0.0, 0.0)]


def test_embed_raises_on_row_count_mismatch() -> None:
    bad = FastEmbedEmbedder("m", encoder=lambda texts: [[1.0, 0.0]])  # one row, many inputs
    with pytest.raises(ValueError, match="2 texts"):
        bad.embed(["a", "b"])


def test_build_backend_selects_by_name() -> None:
    assert isinstance(build_backend("fastembed"), FastEmbedEmbedder)
    assert isinstance(build_backend("sentence-transformers"), SentenceTransformerEmbedder)


def test_build_backend_applies_the_model_override() -> None:
    assert build_backend("fastembed", model="custom-model").model == "custom-model"


def test_build_backend_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        build_backend("word2vec")


def test_backend_names_and_default_model_are_exposed() -> None:
    assert BACKEND_NAMES == ("fastembed", "sentence-transformers")
    assert DEFAULT_MODEL


@pytest.mark.skipif(find_spec("fastembed") is not None, reason="fastembed is installed")
def test_missing_library_raises_an_install_hint() -> None:
    # No injected encoder and the library absent: using the backend points the
    # user at the right pip extra. Tolerant of ImportError/RuntimeError so the
    # exact raise type is the backend's call.
    with pytest.raises((ImportError, RuntimeError), match=r"my-embedder\[fastembed\]"):
        FastEmbedEmbedder("m").embed(["x"])
