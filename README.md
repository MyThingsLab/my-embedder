# my-embedder

[![CI](https://github.com/MyThingsLab/my-embedder/actions/workflows/ci.yml/badge.svg)](https://github.com/MyThingsLab/my-embedder/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/MyThingsLab/my-embedder/branch/main/graph/badge.svg)](https://codecov.io/gh/MyThingsLab/my-embedder) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Real semantic-embedding backends for the
[`mythings.embed.Embedder`](../my-things-core/docs/adr/0003-embedding-seam.md)
protocol, backed by open-source model libraries — plus a local
**OpenAI-compatible `/v1/embeddings` server**.

**my-embedder is to `embed.Embedder` what [my-guard](../my-guard) is to
`policy.Policy`**: a package that fills a core protocol with real machinery, so
the heavy dependency (fastembed's ONNX runtime, sentence-transformers' PyTorch)
lives in *one* repo and never leaks into the dependency-free fleet.

## Two ways the fleet uses it

**1. Directly** — a tool that accepts the dependency imports a backend:

```python
from myembedder import FastEmbedEmbedder
embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
# hand it to any core consumer:
from mythings import corpus
corpus.shortlist(chunks, query, embedder=embedder)     # hybrid retrieval
```

**2. Over HTTP, staying dependency-free** — run the server on one machine and
point core's `ApiEmbedder` (stdlib `urllib`, no dependency) at it:

```bash
myembedder serve --backend fastembed --model BAAI/bge-small-en-v1.5 --port 8080
# then, anywhere in the fleet:
export MYTHINGS_EMBED_URL=http://that-host:8080/v1/embeddings
mycartographer map --corpus ~/papers --embed api        # real semantic clustering
```

The server speaks the OpenAI embeddings shape, so it also works with any other
OpenAI-compatible client.

## Backends (pip extras)

The base install is light; the model library is an opt-in extra (only one is
used at a time, and both are heavy):

| Backend | Install | Notes |
|---|---|---|
| **fastembed** (default) | `pip install 'my-embedder[fastembed]'` | ONNX, **no PyTorch**, small + fast, ARM-friendly. |
| sentence-transformers | `pip install 'my-embedder[sentence-transformers]'` | The standard; widest model choice; pulls in PyTorch. |

Default model `BAAI/bge-small-en-v1.5` (English). For the Italian/English study
corpus use `--model BAAI/bge-m3` (`MULTILINGUAL_MODEL`) — it closes the
cross-language gap [core ADR 0003](../my-things-core/docs/adr/0003-embedding-seam.md)
measured.

```bash
myembedder embed --model BAAI/bge-m3 "apprendimento non supervisionato"
```

## Design notes

- **No Engine call** — this tool *is* an embedding backend; it never touches the
  LLM Engine seam. Deterministic.
- **Testable with zero downloads.** Every backend takes an injectable `loader`,
  so the whole suite runs with neither model library installed — the same
  mock-the-boundary discipline as `engine.Runner` / `github.Runner`. The heavy
  libraries are exercised only on the machines that actually serve.
- **One vector per input, in order, or it raises** — a dropped/reordered row
  would silently misalign a caller that zips vectors back to its documents.

## Requires

`my-things-core` with the `mythings.embed` seam
([core#113](https://github.com/MyThingsLab/my-things-core/pull/113)), plus one
backend extra (or an injected `loader`) for real embeddings.
