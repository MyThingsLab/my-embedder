from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from mythings.embed import Embedder

from myembedder.backends import BACKEND_NAMES, build_backend
from myembedder.server import serve

EmbedderFactory = Callable[[str, str | None], Embedder]


def _default_factory(backend: str, model: str | None) -> Embedder:
    return build_backend(backend, model=model)


def main(argv: list[str] | None = None, *, factory: EmbedderFactory = _default_factory) -> int:
    parser = argparse.ArgumentParser(
        prog="myembedder",
        description="Open-source-library embedding backends for the mythings.embed protocol.",
    )
    parser.add_argument("--backend", choices=BACKEND_NAMES, default="fastembed")
    parser.add_argument("--model", default=None, help="model name (default: the backend's own)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    emb = sub.add_parser("embed", help="embed one or more texts and print the vectors as JSON")
    emb.add_argument("text", nargs="+", help="text(s) to embed")

    srv = sub.add_parser("serve", help="serve an OpenAI-compatible /v1/embeddings endpoint")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8080)

    args = parser.parse_args(argv)
    embedder = factory(args.backend, args.model)
    model = getattr(embedder, "model", args.model)

    if args.cmd == "embed":
        vectors = embedder.embed(args.text)
        print(json.dumps({"model": model, "vectors": [list(v) for v in vectors]}))
        return 0

    server = serve(embedder, host=args.host, port=args.port)
    print(f"serving {args.backend}:{model} on http://{args.host}:{args.port}/v1/embeddings")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
