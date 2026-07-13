# my-embedder — agent instructions

You are developing **my-embedder**, a MyThingsLab My[X] tool.

**Inherited rules:** obey [`./HARNESS.md`](./HARNESS.md) in full — the vendored
MyThingsLab build-harness rules. Do not restate or override them. Anything not
covered here defers to `HARNESS.md`, then `my-things-core/docs/CONVENTIONS.md`.

## This tool

- **Purpose:** concrete `Embedder` backends for the `mythings.embed.Embedder`
  protocol, backed by open-source model libraries (fastembed, sentence-
  transformers), plus a local OpenAI-compatible `/v1/embeddings` server so the
  dependency-free fleet can reach real semantic vectors via core's `ApiEmbedder`.
  It is to `embed.Embedder` what `my-guard` is to `policy.Policy`.
- **The single Engine call:** none — deterministic. This tool *is* an embedding
  backend; it never calls the LLM Engine seam.
- **Invariants / rules:** the heavy model libraries (ONNX/PyTorch) stay optional
  pip extras and are soft-imported — the base package and the whole test suite
  must import and run with neither installed (inject an `encoder=`). A backend
  never silently drops or reorders rows: one normalised vector per input text, in
  order, or it raises. `embed()` output is L2-normalised.
- **Backlog label:** my-embedder

## Testing

Fakes come from `mythings.testing` (opt-in via `pytest_plugins` in
`tests/conftest.py`; see `my-things-core/docs/CONVENTIONS.md`, "Shared test
fixtures"). Never copy fixture code into a conftest — only domain-specific
helpers live there.
