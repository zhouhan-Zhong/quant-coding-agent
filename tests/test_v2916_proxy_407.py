"""2.9.16: an HTTP 407 (or a ProxyError on CONNECT) comes from the egress proxy, not from the House route.
The request was never admitted, so the agent gives the slot back, backs off and retries; it gives up after 4 tries."""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import agent  # noqa: E402

REPLY = {"choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _server(fail_first: int):
    state = {"seen": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            state["seen"] += 1
            if state["seen"] <= fail_first:
                body = b'{"error": "Proxy Authentication Required"}'
                self.send_response(407)
            else:
                body = json.dumps(REPLY).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def _house(port: int) -> agent.House:
    cfg = agent.Config(endpoint=f"http://127.0.0.1:{port}", token="t", max_requests=25, request_timeout=30.0)
    return agent.House(cfg, agent.Budget(600.0, 900.0))


def test_407_is_refunded_and_retried(monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    srv, state = _server(fail_first=2)
    try:
        h = _house(srv.server_address[1])
        content, finish = h.chat([{"role": "user", "content": "hi"}], label="t")
    finally:
        srv.shutdown()
    assert (content, finish) == ("hello", "stop")
    assert state["seen"] == 3
    assert h.requests_made == 1          # the two refused requests were not charged
    assert h.failures == 2


def test_persistent_407_gives_up_after_four_tries(monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    srv, state = _server(fail_first=99)
    try:
        h = _house(srv.server_address[1])
        with pytest.raises(agent.ModelError, match="407"):
            h.chat([{"role": "user", "content": "hi"}], label="t")
    finally:
        srv.shutdown()
    assert state["seen"] == 4
    assert h.requests_made == 0          # nothing was admitted, nothing stays charged


def test_proxy_error_on_connect_is_refunded_and_retried(monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class FakeResp:
        status_code = 200
        text = json.dumps(REPLY)

        def json(self):
            return REPLY

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise requests.exceptions.ProxyError("407 on CONNECT")
        return FakeResp()

    monkeypatch.setattr(requests, "post", fake_post)
    h = _house(1)
    content, finish = h.chat([{"role": "user", "content": "hi"}], label="t")
    assert (content, finish) == ("hello", "stop")
    assert calls["n"] == 3
    assert h.requests_made == 1


def test_other_request_errors_keep_the_old_policy(monkeypatch):
    """A ConnectionError may happen after admission: no refund, 3 tries, then ModelError."""
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("reset")

    monkeypatch.setattr(requests, "post", fake_post)
    h = _house(1)
    with pytest.raises(agent.ModelError):
        h.chat([{"role": "user", "content": "hi"}], label="t")
    assert calls["n"] == 3
    assert h.requests_made == 3
