# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Wire-level tests for the Gerrit REST client.

These drive ``GerritRestClient`` and ``build_client`` through the real
backend down to Requests, with ``responses`` standing in for the network,
so they pin what actually goes over the wire: URLs, credentials, payloads,
decoding, error data and retry accounting.  Every credential here is
synthetic.
"""

from __future__ import annotations

import base64
import json
import logging
from unittest.mock import patch

import pytest
import requests
import responses

from dependamerge.gerrit.client import (
    GerritAuthError,
    GerritNotFoundError,
    GerritRestClient,
    GerritRestError,
    build_client,
)

HOST = "gerrit.example.org"
BASE = f"https://{HOST}"
USER = "synthetic-user"
SECRET = "synthetic-Secret-123"
JSON = "application/json; charset=utf-8"


def _basic(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


@pytest.fixture
def mock_http():
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        yield rsps


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Remove every ambient credential source the client could consult."""
    for name in (
        "GERRIT_USERNAME",
        "GERRIT_PASSWORD",
        "GERRIT_HTTP_USER",
        "GERRIT_HTTP_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    empty = tmp_path / "empty-netrc"
    empty.write_text("")
    monkeypatch.setenv("NETRC", str(empty))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def no_sleep():
    with patch("time.sleep") as sleep:
        yield sleep


def _client(base_url: str = f"{BASE}/", **kwargs) -> GerritRestClient:
    return GerritRestClient(base_url=base_url, **kwargs)


class TestUrlConstruction:
    """Endpoint URLs, deployment prefixes and the /a/ auth prefix."""

    @pytest.mark.parametrize(
        ("base_url", "auth", "path", "expected"),
        [
            (f"{BASE}/", None, "/changes/1", f"{BASE}/changes/1"),
            (f"{BASE}/", (USER, SECRET), "/changes/1", f"{BASE}/a/changes/1"),
            (f"{BASE}/infra/", None, "/changes/1", f"{BASE}/infra/changes/1"),
            (
                f"{BASE}/infra/",
                (USER, SECRET),
                "changes/1",
                f"{BASE}/infra/a/changes/1",
            ),
            (f"{BASE}/r", (USER, SECRET), "/changes/1", f"{BASE}/r/a/changes/1"),
            (
                f"{BASE}/infra/",
                (USER, SECRET),
                "//changes/1",
                f"{BASE}/infra/a/changes/1",
            ),
        ],
    )
    def test_endpoint_url(self, mock_http, clean_env, base_url, auth, path, expected):
        mock_http.add(responses.GET, expected, json={})
        _client(base_url, auth=auth).get(path)
        assert mock_http.calls[0].request.url == expected

    def test_existing_escapes_are_not_double_encoded(self, mock_http, clean_env):
        url = f"{BASE}/a/changes/releng%2Fbuilder~main~I0123abcd/detail"
        mock_http.add(responses.GET, url, json={})
        _client(auth=(USER, SECRET)).get(
            "/changes/releng%2Fbuilder~main~I0123abcd/detail"
        )
        assert mock_http.calls[0].request.url == url

    def test_query_options_are_preserved(self, mock_http, clean_env):
        query = "q=status%3Aopen+topic%3Afoo&o=CURRENT_REVISION&o=LABELS&n=25&S=50"
        mock_http.add(responses.GET, f"{BASE}/changes/", json=[])
        _client().get(f"/changes/?{query}")
        assert mock_http.calls[0].request.url == f"{BASE}/changes/?{query}"


class TestCredentialsOnTheWire:
    """What the client sends must match what ``is_authenticated`` says."""

    def test_authenticated_sends_basic_auth(self, mock_http, clean_env):
        mock_http.add(responses.GET, f"{BASE}/a/changes/1", json={})
        client = _client(auth=(USER, SECRET))
        client.get("/changes/1")
        headers = mock_http.calls[0].request.headers
        assert headers["Authorization"] == _basic(USER, SECRET)

    def test_anonymous_sends_no_auth(self, mock_http, clean_env):
        mock_http.add(responses.GET, f"{BASE}/changes/1", json={})
        _client().get("/changes/1")
        assert "Authorization" not in mock_http.calls[0].request.headers

    def test_anonymous_client_ignores_ambient_netrc(self, mock_http, clean_env):
        netrc = clean_env / "ambient-netrc"
        netrc.write_text(f"machine {HOST} login {USER} password {SECRET}\n")
        netrc.chmod(0o600)
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("NETRC", str(netrc))
            mock_http.add(responses.GET, f"{BASE}/changes/1", json={})
            client = _client()
            client.get("/changes/1")
        assert client.is_authenticated is False
        request = mock_http.calls[0].request
        assert request.url == f"{BASE}/changes/1"
        assert "Authorization" not in request.headers

    def test_unsafe_redirect_drops_credentials(self, mock_http, clean_env):
        other = "https://elsewhere.example.net/landing"
        mock_http.add(
            responses.GET,
            f"{BASE}/a/changes/1",
            status=302,
            headers={"Location": other},
        )
        mock_http.add(responses.GET, other, json={})
        _client(auth=(USER, SECRET)).get("/changes/1")
        assert "Authorization" in mock_http.calls[0].request.headers
        assert "Authorization" not in mock_http.calls[1].request.headers

    def test_logs_and_repr_hold_no_secret(self, mock_http, clean_env, caplog):
        mock_http.add(responses.GET, f"{BASE}/a/changes/1", json={})
        caplog.set_level(logging.DEBUG)
        client = _client(auth=(USER, SECRET))
        client.get("/changes/1")
        assert SECRET not in caplog.text
        assert SECRET not in repr(client)


class TestBuildClientCredentials:
    """Credential precedence, observed on the request that is sent."""

    def _auth_header(self, mock_http, client: GerritRestClient) -> str | None:
        prefix = "a/" if client.is_authenticated else ""
        mock_http.add(responses.GET, f"{BASE}/{prefix}config/server/version", json="3")
        client.get("/config/server/version")
        header: str | None = mock_http.calls[-1].request.headers.get("Authorization")
        return header

    def _netrc(self, directory, body: str):
        path = directory / "creds-netrc"
        path.write_text(body)
        path.chmod(0o600)
        return path

    def test_explicit_beats_netrc_and_env(self, mock_http, clean_env, monkeypatch):
        monkeypatch.setenv("GERRIT_USERNAME", "env-user")
        monkeypatch.setenv("GERRIT_PASSWORD", "env-pass")
        netrc = self._netrc(clean_env, f"machine {HOST} login n-user password n-pass\n")
        client = build_client(HOST, username=USER, password=SECRET, netrc_file=netrc)
        assert self._auth_header(mock_http, client) == _basic(USER, SECRET)

    def test_netrc_beats_env(self, mock_http, clean_env, monkeypatch):
        monkeypatch.setenv("GERRIT_USERNAME", "env-user")
        monkeypatch.setenv("GERRIT_PASSWORD", "env-pass")
        netrc = self._netrc(
            clean_env, f"machine {HOST} login {USER} password {SECRET}\n"
        )
        client = build_client(HOST, netrc_file=netrc)
        assert self._auth_header(mock_http, client) == _basic(USER, SECRET)

    def test_netrc_password_with_hash(self, mock_http, clean_env):
        password = "synthetic#hash"
        netrc = self._netrc(
            clean_env, f'machine {HOST} login {USER} password "{password}"\n'
        )
        client = build_client(HOST, netrc_file=netrc)
        assert self._auth_header(mock_http, client) == _basic(USER, password)

    def test_netrc_from_environment_variable(self, mock_http, clean_env, monkeypatch):
        netrc = self._netrc(
            clean_env, f"machine {HOST} login {USER} password {SECRET}\n"
        )
        monkeypatch.setenv("NETRC", str(netrc))
        client = build_client(HOST)
        assert client.is_authenticated is True
        assert self._auth_header(mock_http, client) == _basic(USER, SECRET)

    def test_netrc_account_only_entry(self, mock_http, clean_env, monkeypatch):
        netrc = self._netrc(
            clean_env, f"machine {HOST} account {USER} password {SECRET}\n"
        )
        monkeypatch.setenv("NETRC", str(netrc))
        client = build_client(HOST)
        assert client.is_authenticated is True
        assert self._auth_header(mock_http, client) == _basic(USER, SECRET)

    def test_explicit_netrc_file_is_not_bypassed(
        self, mock_http, clean_env, monkeypatch
    ):
        ambient = self._netrc(
            clean_env, f"machine {HOST} login {USER} password {SECRET}\n"
        )
        monkeypatch.setenv("NETRC", str(ambient))
        chosen = clean_env / "chosen-netrc"
        chosen.write_text("machine other.example.net login x password y\n")
        chosen.chmod(0o600)
        client = build_client(HOST, netrc_file=chosen)
        assert client.is_authenticated is False
        assert self._auth_header(mock_http, client) is None

    @pytest.mark.parametrize(
        ("user_var", "pass_var"),
        [
            ("GERRIT_USERNAME", "GERRIT_PASSWORD"),
            ("GERRIT_HTTP_USER", "GERRIT_HTTP_PASSWORD"),
        ],
    )
    def test_env_fallback(self, mock_http, clean_env, monkeypatch, user_var, pass_var):
        monkeypatch.setenv(user_var, USER)
        monkeypatch.setenv(pass_var, SECRET)
        client = build_client(HOST)
        assert self._auth_header(mock_http, client) == _basic(USER, SECRET)

    def test_use_netrc_false_sends_nothing_from_netrc(
        self, mock_http, clean_env, monkeypatch
    ):
        netrc = self._netrc(
            clean_env, f"machine {HOST} login {USER} password {SECRET}\n"
        )
        monkeypatch.setenv("NETRC", str(netrc))
        client = build_client(HOST, use_netrc=False)
        assert client.is_authenticated is False
        assert self._auth_header(mock_http, client) is None


class TestPayloads:
    """Request bodies are JSON, including an explicit empty object."""

    def test_post_sends_json(self, mock_http, clean_env):
        url = f"{BASE}/a/changes/1/revisions/current/review"
        mock_http.add(responses.POST, url, json={"labels": {"Code-Review": 2}})
        payload = {"labels": {"Code-Review": 2}, "message": "ok"}
        result = _client(auth=(USER, SECRET)).post(
            "/changes/1/revisions/current/review", payload
        )
        request = mock_http.calls[0].request
        assert json.loads(request.body) == payload
        assert request.headers["Content-Type"].startswith("application/json")
        assert result == {"labels": {"Code-Review": 2}}

    def test_post_sends_empty_object(self, mock_http, clean_env):
        url = f"{BASE}/a/changes/1/rebase"
        mock_http.add(responses.POST, url, json={})
        _client(auth=(USER, SECRET)).post("/changes/1/rebase", {})
        request = mock_http.calls[0].request
        assert json.loads(request.body) == {}
        assert request.headers["Content-Type"].startswith("application/json")

    def test_post_without_data_sends_no_body(self, mock_http, clean_env):
        url = f"{BASE}/a/changes/1/submit"
        mock_http.add(responses.POST, url, json={"status": "MERGED"})
        result = _client(auth=(USER, SECRET)).post("/changes/1/submit")
        assert not mock_http.calls[0].request.body
        assert result == {"status": "MERGED"}

    def test_put_sends_json(self, mock_http, clean_env):
        url = f"{BASE}/a/changes/1/topic"
        mock_http.add(responses.PUT, url, json="feature-x")
        result = _client(auth=(USER, SECRET)).put(
            "/changes/1/topic", {"topic": "feature-x"}
        )
        assert json.loads(mock_http.calls[0].request.body) == {"topic": "feature-x"}
        assert result == "feature-x"

    def test_delete_returns_decoded_body(self, mock_http, clean_env):
        url = f"{BASE}/a/changes/1/topic"
        mock_http.add(responses.DELETE, url, status=204, body="")
        assert _client(auth=(USER, SECRET)).delete("/changes/1/topic") == ""

    def test_timeout_reaches_transport(self, mock_http, clean_env):
        mock_http.add(responses.GET, f"{BASE}/changes/1", json={})
        _client(timeout=7.5).get("/changes/1")
        assert mock_http.calls[0].request.req_kwargs["timeout"] == 7.5


class TestResponseDecoding:
    """Gerrit's XSSI guard and non-JSON bodies."""

    def test_xssi_guard_lf(self, mock_http, clean_env):
        body = ')]}\'\n[{"_number": 1}]'
        mock_http.add(responses.GET, f"{BASE}/changes/", body=body, content_type=JSON)
        assert _client().get("/changes/") == [{"_number": 1}]

    def test_xssi_guard_crlf(self, mock_http, clean_env):
        body = ')]}\'\r\n{"_number": 1}'
        mock_http.add(responses.GET, f"{BASE}/changes/1", body=body, content_type=JSON)
        assert _client().get("/changes/1") == {"_number": 1}

    def test_html_body_is_returned_as_text(self, mock_http, clean_env):
        mock_http.add(
            responses.GET,
            f"{BASE}/changes/",
            body="<html>Sign in</html>",
            content_type="text/html",
        )
        result = _client().get("/changes/")
        assert result == "<html>Sign in</html>"
        assert not isinstance(result, list)

    def test_invalid_json_raises(self, mock_http, clean_env):
        mock_http.add(
            responses.GET, f"{BASE}/changes/1", body=")]}'\nnot json", content_type=JSON
        )
        with pytest.raises(GerritRestError):
            _client(max_attempts=1).get("/changes/1")


class TestErrors:
    """HTTP failures keep their status (and body) in the project's errors."""

    @pytest.mark.parametrize(
        ("status", "error"),
        [(401, GerritAuthError), (403, GerritAuthError), (404, GerritNotFoundError)],
    )
    def test_auth_and_not_found(self, mock_http, clean_env, status, error):
        mock_http.add(responses.GET, f"{BASE}/changes/1", status=status, body="no")
        with pytest.raises(error) as info:
            _client().get("/changes/1")
        assert info.value.status_code == status

    @pytest.mark.parametrize("status", [400, 429, 500, 502, 503, 504])
    def test_status_is_exact(self, mock_http, clean_env, status):
        mock_http.add(responses.GET, f"{BASE}/changes/1", status=status, body="err")
        with pytest.raises(GerritRestError) as info:
            _client(max_attempts=1).get("/changes/1")
        assert info.value.status_code == status
        assert not isinstance(info.value, (GerritAuthError, GerritNotFoundError))

    def test_conflict_keeps_body(self, mock_http, clean_env):
        body = "merge conflict in src/a.py\nmerge conflict in src/b.py"
        url = f"{BASE}/a/changes/1/rebase"
        mock_http.add(responses.POST, url, status=409, body=body)
        with pytest.raises(GerritRestError) as info:
            _client(auth=(USER, SECRET), max_attempts=1).post("/changes/1/rebase", {})
        assert info.value.status_code == 409
        assert info.value.response_body == body

    def test_digits_in_a_network_error_are_not_a_status(self, mock_http, clean_env):
        mock_http.add(
            responses.GET,
            f"{BASE}/changes/404",
            body=requests.ConnectionError("cannot reach port 403"),
        )
        with pytest.raises(GerritRestError) as info:
            _client(max_attempts=1).get("/changes/404")
        assert type(info.value) is GerritRestError

    def test_empty_path_is_rejected(self, mock_http, clean_env):
        with pytest.raises(GerritRestError, match="path is required"):
            _client().get("")
        assert len(mock_http.calls) == 0

    def test_timeout_raises_rest_error(self, mock_http, clean_env):
        mock_http.add(
            responses.GET,
            f"{BASE}/changes/1",
            body=requests.ReadTimeout("Read timed out"),
        )
        with pytest.raises(GerritRestError) as info:
            _client(max_attempts=1).get("/changes/1")
        assert info.value.status_code is None


class TestRetryAccounting:
    """The client owns retries; each attempt is one request on the wire."""

    def test_retries_then_succeeds(self, mock_http, clean_env, no_sleep):
        url = f"{BASE}/changes/1"
        mock_http.add(responses.GET, url, status=503, body="busy")
        mock_http.add(responses.GET, url, json={"ok": True})
        assert _client(max_attempts=3).get("/changes/1") == {"ok": True}
        assert len(mock_http.calls) == 2
        assert no_sleep.call_count == 1

    def test_429_is_retried(self, mock_http, clean_env, no_sleep):
        url = f"{BASE}/changes/1"
        mock_http.add(responses.GET, url, status=429, body="slow down")
        mock_http.add(responses.GET, url, json={})
        _client(max_attempts=3).get("/changes/1")
        assert len(mock_http.calls) == 2

    def test_exhausted_attempts(self, mock_http, clean_env, no_sleep):
        mock_http.add(responses.GET, f"{BASE}/changes/1", status=500, body="boom")
        with pytest.raises(GerritRestError) as info:
            _client(max_attempts=3).get("/changes/1")
        assert info.value.status_code == 500
        assert len(mock_http.calls) == 3
        assert no_sleep.call_count == 2

    @pytest.mark.parametrize("status", [401, 403, 404, 409])
    def test_not_retried(self, mock_http, clean_env, no_sleep, status):
        mock_http.add(responses.GET, f"{BASE}/changes/1", status=status, body="x")
        with pytest.raises(GerritRestError):
            _client(max_attempts=3).get("/changes/1")
        assert len(mock_http.calls) == 1
        assert no_sleep.call_count == 0

    def test_http_error_body_does_not_make_it_transient(
        self, mock_http, clean_env, no_sleep
    ):
        mock_http.add(
            responses.GET,
            f"{BASE}/changes/1",
            status=400,
            body="upstream connection reset by peer",
        )
        with pytest.raises(GerritRestError):
            _client(max_attempts=3).get("/changes/1")
        assert len(mock_http.calls) == 1

    def test_transient_network_error_is_retried(self, mock_http, clean_env, no_sleep):
        url = f"{BASE}/changes/1"
        mock_http.add(
            responses.GET,
            url,
            body=requests.ConnectionError("Connection reset by peer"),
        )
        mock_http.add(responses.GET, url, json={})
        _client(max_attempts=3).get("/changes/1")
        assert len(mock_http.calls) == 2

    def test_submit_is_replayed_after_server_error(
        self, mock_http, clean_env, no_sleep
    ):
        url = f"{BASE}/a/changes/1/submit"
        mock_http.add(responses.POST, url, status=503, body="busy")
        mock_http.add(responses.POST, url, json={"status": "MERGED"})
        result = _client(auth=(USER, SECRET), max_attempts=3).post("/changes/1/submit")
        assert result == {"status": "MERGED"}
        assert len(mock_http.calls) == 2
