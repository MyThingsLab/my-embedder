from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mythings.embed import Embedder

# A minimal OpenAI-compatible /v1/embeddings endpoint. This is the bridge that
# keeps the rest of the fleet dependency-free: a tool sets MYTHINGS_EMBED_URL to
# this server and reaches real semantic vectors through core's ApiEmbedder
# (stdlib urllib), so the ONNX/PyTorch dependency lives only in this one process.
# The request/response shape matches OpenAI so the same server also works with
# any other OpenAI-compatible client.

_PATH = "/v1/embeddings"


def handle_embeddings(embedder: Embedder, body: bytes) -> tuple[int, dict[str, Any]]:
    # Pure request handling, separate from the socket layer so it is testable
    # without a server. Returns (status, json-body).
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return 400, _error("request body is not valid JSON")
    if not isinstance(payload, dict):
        return 400, _error("request body must be a JSON object")

    raw = payload.get("input")
    # OpenAI accepts a single string or a list of strings; normalise to a list.
    if isinstance(raw, str):
        texts = [raw]
    elif isinstance(raw, list) and all(isinstance(t, str) for t in raw):
        texts = raw
    else:
        return 400, _error("'input' must be a string or a list of strings")

    vectors = embedder.embed(texts)
    data = [
        {"object": "embedding", "index": i, "embedding": list(vec)}
        for i, vec in enumerate(vectors)
    ]
    total = sum(len(t.split()) for t in texts)
    return 200, {
        "object": "list",
        "data": data,
        "model": payload.get("model", ""),
        "usage": {"prompt_tokens": total, "total_tokens": total},
    }


def _error(message: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": "invalid_request_error"}}


def make_handler(embedder: Embedder) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path.rstrip("/") != _PATH:
                self._respond(404, _error(f"unknown path {self.path!r}"))
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            status, payload = handle_embeddings(embedder, body)
            self._respond(status, payload)

        def do_GET(self) -> None:  # noqa: N802 - health check for readiness probes
            if self.path.rstrip("/") == "/health":
                self._respond(200, {"status": "ok"})
            else:
                self._respond(404, _error(f"unknown path {self.path!r}"))

        def _respond(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:  # keep the server quiet
            pass

    return Handler


def serve(embedder: Embedder, *, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    # Returns the bound server; the caller runs serve_forever() (the CLI) or
    # handles one request in a test. Threading so a slow model call doesn't block
    # concurrent readers.
    return ThreadingHTTPServer((host, port), make_handler(embedder))
