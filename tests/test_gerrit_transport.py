# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Transport tests for the Gerrit REST client.

Covers the pieces ``_client_transport`` adds around python-gerrit-api: the
session that never reads netrc implicitly, the per-request session pool,
and the environment-driven proxy and CA bundle settings the session must
keep.  The loopback tests run real HTTP(S) servers and proxies on
127.0.0.1; certificates are generated per run with the ``openssl`` CLI.
Every credential here is synthetic.
"""

from __future__ import annotations

import base64
import contextlib
import json
import select
import shutil
import socket
import ssl
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest
import responses

from dependamerge.gerrit._client_transport import (
    GerritSession,
    SessionPool,
    decode_response,
)
from dependamerge.gerrit.client import GerritRestClient, GerritRestError

USER = "synthetic-user"
SECRET = "synthetic-Secret-123"
PROXY_USER = "proxy-user"
PROXY_SECRET = "proxy-Secret-456"

_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NETRC",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _write_netrc(path: Path, machine: str) -> Path:
    path.write_text(f"machine {machine} login {USER} password {SECRET}\n")
    path.chmod(0o600)
    return path


class TestGerritSession:
    """Netrc is never read implicitly; everything else from env is kept."""

    def test_redirect_does_not_add_netrc_credentials(self, clean_env, monkeypatch):
        other = "elsewhere.example.net"
        monkeypatch.setenv("NETRC", str(_write_netrc(clean_env / "n", other)))
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "https://gerrit.example.org/changes/1",
                status=302,
                headers={"Location": f"https://{other}/changes/1"},
            )
            rsps.add(responses.GET, f"https://{other}/changes/1", json={})
            GerritRestClient(base_url="https://gerrit.example.org/").get("/changes/1")
            assert "Authorization" not in rsps.calls[1].request.headers

    def test_same_host_redirect_keeps_credentials(self, clean_env):
        base = "https://gerrit.example.org"
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                f"{base}/a/changes/1",
                status=302,
                headers={"Location": f"{base}/a/changes/1/"},
            )
            rsps.add(responses.GET, f"{base}/a/changes/1/", json={})
            client = GerritRestClient(base_url=f"{base}/", auth=(USER, SECRET))
            client.get("/changes/1")
            assert "Authorization" in rsps.calls[1].request.headers

    def test_environment_settings_are_kept(self, clean_env, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.net:3128")
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/requests/ca.pem")
        monkeypatch.setenv("CURL_CA_BUNDLE", "/curl/ca.pem")
        session = GerritSession()
        settings = session.merge_environment_settings(
            "https://gerrit.example.org/changes/", {}, None, None, None
        )
        assert session.trust_env is True
        assert settings["proxies"]["https"] == "http://proxy.example.net:3128"
        assert settings["verify"] == "/requests/ca.pem"

    def test_curl_ca_bundle_is_the_fallback(self, clean_env, monkeypatch):
        monkeypatch.setenv("CURL_CA_BUNDLE", "/curl/ca.pem")
        settings = GerritSession().merge_environment_settings(
            "https://gerrit.example.org/", {}, None, None, None
        )
        assert settings["verify"] == "/curl/ca.pem"

    def test_verification_is_on_by_default(self, clean_env):
        settings = GerritSession().merge_environment_settings(
            "https://gerrit.example.org/", {}, None, None, None
        )
        assert settings["verify"] is True

    def test_all_proxy_fallback(self, clean_env, monkeypatch):
        monkeypatch.setenv("ALL_PROXY", "http://all.example.net:3128")
        settings = GerritSession().merge_environment_settings(
            "https://gerrit.example.org/", {}, None, None, None
        )
        assert settings["proxies"]["all"] == "http://all.example.net:3128"

    def test_no_proxy_ipv4_cidr_bypass(self, clean_env, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.net:3128")
        monkeypatch.setenv("NO_PROXY", "10.0.0.0/8")
        session = GerritSession()
        bypassed = session.merge_environment_settings(
            "https://10.1.2.3/changes/", {}, None, None, None
        )
        proxied = session.merge_environment_settings(
            "https://192.0.2.10/changes/", {}, None, None, None
        )
        assert "https" not in bypassed["proxies"]
        assert proxied["proxies"]["https"] == "http://proxy.example.net:3128"


class TestSessionPool:
    """Concurrent requests never share a backend."""

    def test_concurrent_holders_get_distinct_backends(self):
        made: list[object] = []

        def factory() -> object:
            made.append(object())
            return made[-1]

        pool: SessionPool[object] = SessionPool(factory, lambda _b: None)
        barrier = threading.Barrier(3)
        seen: list[object] = []
        lock = threading.Lock()

        def hold() -> None:
            with pool.acquire() as backend:
                with lock:
                    seen.append(backend)
                barrier.wait(timeout=5)

        threads = [threading.Thread(target=hold) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len({id(b) for b in seen}) == 3
        with pool.acquire() as backend:
            assert backend in made
        assert len(pool) == 3

    def test_close_closes_idle_and_returned_backends(self):
        closed: list[str] = []
        names = iter(["first", "second"])
        pool: SessionPool[str] = SessionPool(lambda: next(names), closed.append)
        with pool.acquire():
            pass
        with pool.acquire() as busy:
            assert busy == "first"
            with pool.acquire():
                pass
            pool.close()
            assert closed == ["second"]
        assert closed == ["second", "first"]
        with pytest.raises(RuntimeError), pool.acquire():
            pass

    def test_client_runs_parallel_requests_on_separate_sessions(self, clean_env):
        barrier = threading.Barrier(3)

        def slow(_request: Any) -> tuple[int, dict[str, str], str]:
            barrier.wait(timeout=5)
            return 200, {"Content-Type": "application/json"}, "{}"

        client = GerritRestClient(base_url="https://gerrit.example.org/")
        with responses.RequestsMock() as rsps:
            rsps.add_callback(
                responses.GET, "https://gerrit.example.org/changes/1", callback=slow
            )
            threads = [
                threading.Thread(target=client.get, args=("/changes/1",))
                for _ in range(3)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert len(rsps.calls) == 3
        assert len(client._backends) == 3
        client.close()


class TestDecodeResponse:
    """Decoding is independent of the library's LF-only guard handling."""

    @pytest.mark.parametrize("guard", [")]}'\n", ")]}'\r\n"])
    def test_guard_variants(self, guard):
        response = _response(guard + '{"a": 1}', "application/json")
        assert decode_response(response) == {"a": 1}

    def test_empty_body(self):
        assert decode_response(_response("", "application/json")) == ""


def _response(body: str, content_type: str) -> Any:
    import requests

    response = requests.Response()
    response._content = body.encode()
    response.encoding = "utf-8"
    response.headers["Content-Type"] = content_type
    response.status_code = 200
    return response


# --- Loopback servers -------------------------------------------------------


class _GerritHandler(BaseHTTPRequestHandler):
    """Minimal Gerrit stand-in returning a guarded JSON body."""

    seen: ClassVar[list[dict[str, str]]]

    def do_GET(self) -> None:
        type(self).seen.append({"path": self.path, **dict(self.headers)})
        body = b")]}'\n" + json.dumps({"_number": 1}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class _ProxyHandler(BaseHTTPRequestHandler):
    """Forward proxy: absolute-URI HTTP requests and CONNECT tunnels."""

    seen: ClassVar[list[dict[str, str]]]

    def do_GET(self) -> None:
        type(self).seen.append({"line": self.requestline, **dict(self.headers)})
        body = json.dumps({"via": "proxy"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_CONNECT(self) -> None:
        type(self).seen.append({"line": self.requestline, **dict(self.headers)})
        host, port = self.path.rsplit(":", 1)
        upstream = socket.create_connection((host, int(port)), timeout=5)
        self.send_response(200, "Connection established")
        self.end_headers()
        _pipe(self.connection, upstream)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _pipe(a: socket.socket, b: socket.socket) -> None:
    sockets = [a, b]
    try:
        while True:
            readable, _, _ = select.select(sockets, [], [], 5)
            if not readable:
                return
            for sock in readable:
                data = sock.recv(65536)
                if not data:
                    return
                (b if sock is a else a).sendall(data)
    finally:
        b.close()


@contextlib.contextmanager
def _serve(
    handler: type[BaseHTTPRequestHandler], context: ssl.SSLContext | None = None
) -> Iterator[tuple[int, list[dict[str, str]]]]:
    seen: list[dict[str, str]] = []
    bound = type(handler.__name__, (handler,), {"seen": seen})
    server = ThreadingHTTPServer(("127.0.0.1", 0), bound)
    server.daemon_threads = True
    if context is not None:
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], seen
    finally:
        server.shutdown()
        server.server_close()


class _Pki:
    def __init__(self, directory: Path, name: str) -> None:
        self.ca = directory / f"{name}-ca.pem"
        self.cert = directory / f"{name}-cert.pem"
        self.key = directory / f"{name}-key.pem"
        ca_key = directory / f"{name}-ca-key.pem"
        csr = directory / f"{name}.csr"
        ext = directory / f"{name}.ext"
        # Extensions Python 3.13+ requires under VERIFY_X509_STRICT.
        ext.write_text(
            "subjectAltName=IP:127.0.0.1\n"
            "basicConstraints=critical,CA:FALSE\n"
            "keyUsage=critical,digitalSignature,keyEncipherment\n"
            "extendedKeyUsage=serverAuth\n"
            "subjectKeyIdentifier=hash\n"
            "authorityKeyIdentifier=keyid\n"
        )
        commands = [
            [
                "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                "-subj", f"/CN={name} test CA", "-keyout", str(ca_key),
                "-out", str(self.ca),
                "-addext", "basicConstraints=critical,CA:TRUE",
                "-addext", "keyUsage=critical,keyCertSign,cRLSign",
            ],
            [
                "req", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=127.0.0.1",
                "-keyout", str(self.key), "-out", str(csr),
            ],
            [
                "x509", "-req", "-in", str(csr), "-CA", str(self.ca),
                "-CAkey", str(ca_key), "-CAcreateserial", "-days", "1",
                "-extfile", str(ext), "-out", str(self.cert),
            ],
        ]  # fmt: skip
        for command in commands:
            subprocess.run(
                ["openssl", *command], check=True, capture_output=True, timeout=60
            )

    def server_context(self) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.cert, self.key)
        return context


@pytest.fixture(scope="module")
def pki(tmp_path_factory) -> tuple[_Pki, _Pki]:
    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI not available")
    directory = tmp_path_factory.mktemp("pki")
    return _Pki(directory, "trusted"), _Pki(directory, "untrusted")


class TestLoopbackProxies:
    """Real sockets: proxy selection, bypass and proxy credentials."""

    def test_http_proxy_with_credentials(self, clean_env, monkeypatch):
        with _serve(_ProxyHandler) as (port, seen):
            monkeypatch.setenv(
                "HTTP_PROXY", f"http://{PROXY_USER}:{PROXY_SECRET}@127.0.0.1:{port}"
            )
            client = GerritRestClient(base_url="http://gerrit.invalid/", max_attempts=1)
            assert client.get("/changes/1") == {"via": "proxy"}
        assert seen[0]["line"].startswith("GET http://gerrit.invalid/changes/1 ")
        token = base64.b64encode(f"{PROXY_USER}:{PROXY_SECRET}".encode()).decode()
        assert seen[0]["Proxy-Authorization"] == f"Basic {token}"

    def test_all_proxy_is_used_without_scheme_proxy(self, clean_env, monkeypatch):
        with _serve(_ProxyHandler) as (port, seen):
            monkeypatch.setenv("ALL_PROXY", f"http://127.0.0.1:{port}")
            client = GerritRestClient(base_url="http://gerrit.invalid/", max_attempts=1)
            assert client.get("/changes/1") == {"via": "proxy"}
        assert len(seen) == 1

    def test_no_proxy_target_never_sees_proxy_credentials(self, clean_env, monkeypatch):
        with (
            _serve(_ProxyHandler) as (proxy_port, proxy_seen),
            _serve(_GerritHandler) as (port, seen),
        ):
            monkeypatch.setenv(
                "HTTP_PROXY",
                f"http://{PROXY_USER}:{PROXY_SECRET}@127.0.0.1:{proxy_port}",
            )
            monkeypatch.setenv("NO_PROXY", "127.0.0.0/8")
            client = GerritRestClient(
                base_url=f"http://127.0.0.1:{port}/", max_attempts=1
            )
            assert client.get("/changes/1") == {"_number": 1}
        assert proxy_seen == []
        assert "Proxy-Authorization" not in seen[0]


class TestLoopbackTls:
    """Real TLS: custom CA trust and a TLS-first HTTPS proxy."""

    def test_custom_ca_bundle_is_trusted(self, clean_env, monkeypatch, pki):
        trusted, _ = pki
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(trusted.ca))
        with _serve(_GerritHandler, trusted.server_context()) as (port, _seen):
            client = GerritRestClient(
                base_url=f"https://127.0.0.1:{port}/", max_attempts=1
            )
            assert client.get("/changes/1") == {"_number": 1}

    def test_untrusted_certificate_is_rejected(self, clean_env, monkeypatch, pki):
        trusted, untrusted = pki
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(trusted.ca))
        with _serve(_GerritHandler, untrusted.server_context()) as (port, seen):
            client = GerritRestClient(
                base_url=f"https://127.0.0.1:{port}/", max_attempts=1
            )
            with pytest.raises(GerritRestError):
                client.get("/changes/1")
        assert seen == []

    def test_https_proxy_speaks_tls_before_connect(self, clean_env, monkeypatch, pki):
        trusted, _ = pki
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(trusted.ca))
        with (
            _serve(_GerritHandler, trusted.server_context()) as (port, seen),
            _serve(_ProxyHandler, trusted.server_context()) as (proxy_port, via),
        ):
            monkeypatch.setenv(
                "HTTPS_PROXY",
                f"https://{PROXY_USER}:{PROXY_SECRET}@127.0.0.1:{proxy_port}",
            )
            client = GerritRestClient(
                base_url=f"https://127.0.0.1:{port}/",
                auth=(USER, SECRET),
                max_attempts=1,
            )
            assert client.get("/changes/1") == {"_number": 1}
        # The proxy only parsed CONNECT because the client opened TLS first:
        # its listening socket accepts nothing else.
        assert via[0]["line"].startswith(f"CONNECT 127.0.0.1:{port} HTTP/")
        assert "Proxy-Authorization" in via[0]
        assert "Authorization" not in via[0]
        assert seen[0]["path"] == "/a/changes/1"
        assert "Proxy-Authorization" not in seen[0]
