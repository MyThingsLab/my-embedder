from __future__ import annotations

import json

import pytest

from myembedder.cli import main


class _FakeEmbedder:
    def embed(self, texts):
        return [(float(len(t)), 2.0) for t in texts]


def _factory(backend: str, model: str | None) -> _FakeEmbedder:
    return _FakeEmbedder()


def test_embed_command_prints_vectors(capsys: pytest.CaptureFixture) -> None:
    code = main(["embed", "hi", "world"], factory=_factory)
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["vectors"] == [[2.0, 2.0], [5.0, 2.0]]


def test_embed_command_passes_backend_and_model_through() -> None:
    seen: dict[str, str | None] = {}

    def spy(backend: str, model: str | None) -> _FakeEmbedder:
        seen["backend"] = backend
        seen["model"] = model
        return _FakeEmbedder()

    main(["--backend", "sentence-transformers", "--model", "bge-m3", "embed", "x"], factory=spy)
    assert seen == {"backend": "sentence-transformers", "model": "bge-m3"}


def test_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([], factory=_factory)


def test_rejects_an_unknown_backend() -> None:
    with pytest.raises(SystemExit):
        main(["--backend", "word2vec", "embed", "x"], factory=_factory)
