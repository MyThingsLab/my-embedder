from __future__ import annotations

import json
import threading
import urllib.request

from myembedder.server import handle_embeddings, serve


class _FakeEmbedder:
    def embed(self, texts):
        return [(float(len(t)), 1.0) for t in texts]


def test_handle_embeddings_accepts_a_single_string() -> None:
    status, body = handle_embeddings(_FakeEmbedder(), json.dumps({"input": "hello"}).encode())
    assert status == 200
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["index"] == 0
    assert body["data"][0]["embedding"] == [5.0, 1.0]


def test_handle_embeddings_accepts_a_list_and_preserves_order() -> None:
    status, body = handle_embeddings(_FakeEmbedder(), json.dumps({"input": ["a", "bbb"]}).encode())
    assert status == 200
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert body["data"][0]["embedding"][0] == 1.0
    assert body["data"][1]["embedding"][0] == 3.0


def test_handle_embeddings_echoes_model_and_reports_usage() -> None:
    _, body = handle_embeddings(
        _FakeEmbedder(), json.dumps({"model": "bge", "input": "two words"}).encode()
    )
    assert body["model"] == "bge"
    assert body["usage"]["total_tokens"] == 2


def test_handle_embeddings_rejects_bad_json() -> None:
    status, body = handle_embeddings(_FakeEmbedder(), b"not json")
    assert status == 400
    assert "error" in body


def test_handle_embeddings_rejects_non_object_body() -> None:
    status, _ = handle_embeddings(_FakeEmbedder(), b"[1, 2, 3]")
    assert status == 400


def test_handle_embeddings_rejects_a_bad_input_type() -> None:
    status, _ = handle_embeddings(_FakeEmbedder(), json.dumps({"input": 42}).encode())
    assert status == 400
    status, _ = handle_embeddings(_FakeEmbedder(), json.dumps({"input": [1, 2]}).encode())
    assert status == 400


def _post(port: int, path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def test_server_answers_over_http() -> None:
    server = serve(_FakeEmbedder(), host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(port, "/v1/embeddings", {"input": "hi"})
        assert status == 200
        assert body["data"][0]["embedding"] == [2.0, 1.0]
        # health check
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as resp:
            assert json.loads(resp.read())["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_server_404s_an_unknown_path() -> None:
    server = serve(_FakeEmbedder(), host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        try:
            _post(port, "/nope", {"input": "x"})
            raise AssertionError("expected an HTTP error")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
