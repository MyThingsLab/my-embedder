from __future__ import annotations

from collections.abc import Sequence

# Shared fakes come from the core plugin — see my-things-core/docs/CONVENTIONS.md
# "Shared test fixtures". This suite is pure in-process (no git worktrees), so it
# only needs a couple of domain-specific fakes for the model libraries.
pytest_plugins = ("mythings.testing",)


class FakeFastEmbed:
    # Stands in for fastembed.TextEmbedding: .embed(list) yields one vector per
    # input, in order. Records calls so a test can assert lazy single-load.
    def __init__(self, table: dict[str, list[float]] | None = None) -> None:
        self.table = table or {}
        self.calls = 0

    def embed(self, texts: Sequence[str]):
        self.calls += 1
        for t in texts:
            yield self.table.get(t, [float(len(t)), 1.0])


class FakeSentenceTransformer:
    # Stands in for sentence_transformers.SentenceTransformer: .encode(list)
    # returns a 2-D array-like (list of rows).
    def __init__(self, table: dict[str, list[float]] | None = None) -> None:
        self.table = table or {}
        self.calls = 0

    def encode(self, texts: Sequence[str]):
        self.calls += 1
        return [self.table.get(t, [float(len(t)), 1.0]) for t in texts]
