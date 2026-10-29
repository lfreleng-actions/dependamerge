# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Tests for Gerrit REST client module.

This module tests the GerritRestClient's initialization, request handling,
retry helpers, and credential resolution.  Wire-level behaviour through
the python-gerrit-api backend is covered in ``test_gerrit_backend``.
"""

import importlib.util
from unittest.mock import MagicMock, patch

from requests.exceptions import HTTPError

from dependamerge.gerrit.client import (
    GerritRestClient,
    _Auth,
    _calculate_backoff,
    _extract_status_code,
    _is_transient_error,
    _mask_secret,
    build_client,
)


class TestMaskSecret:
    """Tests for the _mask_secret helper function."""

    def test_empty_string(self):
        """Test masking empty string."""
        assert _mask_secret("") == ""

    def test_short_string(self):
        """Test masking short strings (4 or fewer chars)."""
        assert _mask_secret("abc") == "****"
        assert _mask_secret("abcd") == "****"

    def test_normal_string(self):
        """Test masking normal length strings."""
        assert _mask_secret("password123") == "pa*******23"
        assert _mask_secret("secret") == "se**et"

    def test_long_string(self):
        """Test masking long strings."""
        result = _mask_secret("verylongsecretpassword")
        assert result.startswith("ve")
        assert result.endswith("rd")
        assert "****" in result


class TestIsTransientError:
    """Tests for transient error detection."""

    def test_timeout_error(self):
        """Test that timeout errors are detected as transient."""
        exc = Exception("Connection timed out")
        assert _is_transient_error(exc) is True

    def test_connection_reset(self):
        """Test that connection reset is detected as transient."""
        exc = Exception("Connection reset by peer")
        assert _is_transient_error(exc) is True

    def test_service_unavailable(self):
        """Test that service unavailable is detected as transient."""
        exc = Exception("Service unavailable")
        assert _is_transient_error(exc) is True

    def test_non_transient_error(self):
        """Test that non-transient errors are not flagged."""
        exc = Exception("Invalid request")
        assert _is_transient_error(exc) is False


class TestCalculateBackoff:
    """Tests for backoff calculation."""

    def test_first_attempt(self):
        """Test backoff for first attempt."""
        delay = _calculate_backoff(0, base_delay=1.0, max_delay=30.0, jitter=0.0)
        assert delay == 1.0

    def test_exponential_growth(self):
        """Test that delay grows exponentially."""
        delay0 = _calculate_backoff(0, base_delay=1.0, max_delay=60.0, jitter=0.0)
        delay1 = _calculate_backoff(1, base_delay=1.0, max_delay=60.0, jitter=0.0)
        delay2 = _calculate_backoff(2, base_delay=1.0, max_delay=60.0, jitter=0.0)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        delay = _calculate_backoff(10, base_delay=1.0, max_delay=30.0, jitter=0.0)
        assert delay == 30.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds variability to delay."""
        # With 50% jitter, delays should vary
        delays = [
            _calculate_backoff(1, base_delay=1.0, max_delay=30.0, jitter=0.5)
            for _ in range(10)
        ]
        # Base delay at attempt 1 is 2.0, with 50% jitter it can be 2.0-3.0
        assert all(2.0 <= d <= 3.0 for d in delays)
        # With random jitter, not all delays should be identical
        # (statistically very unlikely)
        assert len(set(delays)) > 1


class TestExtractStatusCode:
    """Tests for status code extraction from exceptions."""

    def test_extract_from_response_attribute(self):
        """Test extracting status code from response attribute."""
        exc = HTTPError()
        exc.response = MagicMock()
        exc.response.status_code = 404
        assert _extract_status_code(exc) == 404

    def test_digits_in_message_are_not_a_status(self):
        """Digits in an exception message are never taken for a status."""
        exc = Exception("Server returned 503 Service Unavailable")
        assert _extract_status_code(exc) is None

    def test_no_status_code(self):
        """Test when no status code can be extracted."""
        exc = Exception("Some random error")
        assert _extract_status_code(exc) is None


class TestGerritRestClientInit:
    """Tests for GerritRestClient initialization."""

    def test_basic_init(self):
        """Test basic client initialization."""
        client = GerritRestClient(base_url="https://gerrit.example.org/")

        assert client.base_url == "https://gerrit.example.org/"
        assert client.is_authenticated is False

    def test_init_normalizes_base_url(self):
        """Test that base URL is normalized to end with slash."""
        client = GerritRestClient(base_url="https://gerrit.example.org")
        assert client.base_url == "https://gerrit.example.org/"

    def test_init_with_auth(self):
        """Test client initialization with authentication."""
        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            auth=("user", "password"),
        )

        assert client.is_authenticated is True
        assert client._auth == _Auth("user", "password")

    def test_init_with_empty_auth(self):
        """Test that empty auth credentials are ignored."""
        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            auth=("", ""),
        )

        assert client.is_authenticated is False

    def test_init_with_partial_auth(self):
        """Test that partial auth credentials are ignored."""
        client = GerritRestClient(
            base_url="https://gerrit.example.org/",
            auth=("user", ""),
        )

        assert client.is_authenticated is False

    def test_repr(self):
        """Test string representation."""
        client = GerritRestClient(base_url="https://gerrit.example.org/")
        repr_str = repr(client)
        assert repr_str == "GerritRestClient(base_url='https://gerrit.example.org/')"


class TestBuildClient:
    """Tests for the build_client factory function."""

    def test_build_client_basic(self):
        """Test building client with just hostname."""
        client = build_client("gerrit.example.org")

        assert client.base_url == "https://gerrit.example.org/"
        assert client.is_authenticated is False

    def test_build_client_with_base_path(self):
        """Test building client with base path."""
        client = build_client("gerrit.example.org", base_path="infra")

        assert client.base_url == "https://gerrit.example.org/infra/"

    def test_build_client_with_credentials(self):
        """Test building client with explicit credentials."""
        client = build_client(
            "gerrit.example.org",
            username="testuser",
            password="testpass",
        )

        assert client.is_authenticated is True
        assert client._auth == _Auth("testuser", "testpass")

    def test_build_client_from_env(self, monkeypatch):
        """Test building client with credentials from environment."""
        monkeypatch.setenv("GERRIT_USERNAME", "envuser")
        monkeypatch.setenv("GERRIT_PASSWORD", "envpass")

        client = build_client("gerrit.example.org")

        assert client.is_authenticated is True
        assert client._auth == _Auth("envuser", "envpass")

    def test_build_client_env_fallback(self, monkeypatch):
        """Test credential fallback from GERRIT_HTTP_USER/PASSWORD."""
        monkeypatch.delenv("GERRIT_USERNAME", raising=False)
        monkeypatch.delenv("GERRIT_PASSWORD", raising=False)
        monkeypatch.setenv("GERRIT_HTTP_USER", "httpuser")
        monkeypatch.setenv("GERRIT_HTTP_PASSWORD", "httppass")

        client = build_client("gerrit.example.org")

        assert client.is_authenticated is True
        assert client._auth == _Auth("httpuser", "httppass")

    def test_build_client_explicit_overrides_env(self, monkeypatch):
        """Test that explicit credentials override environment."""
        monkeypatch.setenv("GERRIT_USERNAME", "envuser")
        monkeypatch.setenv("GERRIT_PASSWORD", "envpass")

        client = build_client(
            "gerrit.example.org",
            username="explicit",
            password="credentials",
        )

        assert client.is_authenticated is True
        assert client._auth == _Auth("explicit", "credentials")

    @patch("dependamerge.gerrit.client.get_credentials_for_host")
    def test_build_client_from_netrc(self, mock_get_creds, monkeypatch):
        """Test building client with credentials from .netrc file."""
        # Clear environment variables
        monkeypatch.delenv("GERRIT_USERNAME", raising=False)
        monkeypatch.delenv("GERRIT_PASSWORD", raising=False)
        monkeypatch.delenv("GERRIT_HTTP_USER", raising=False)
        monkeypatch.delenv("GERRIT_HTTP_PASSWORD", raising=False)

        # Mock netrc credentials
        from dependamerge.netrc import NetrcCredentials

        mock_get_creds.return_value = NetrcCredentials(
            machine="gerrit.example.org",
            login="netrcuser",
            password="netrcpass",
        )

        client = build_client("gerrit.example.org")

        assert client.is_authenticated is True
        assert client._auth == _Auth("netrcuser", "netrcpass")
        mock_get_creds.assert_called_once()

    @patch("dependamerge.gerrit.client.get_credentials_for_host")
    def test_build_client_explicit_overrides_netrc(self, mock_get_creds, monkeypatch):
        """Test that explicit credentials override netrc."""
        # Clear environment variables
        monkeypatch.delenv("GERRIT_USERNAME", raising=False)
        monkeypatch.delenv("GERRIT_PASSWORD", raising=False)

        # Mock netrc credentials (should not be used)
        from dependamerge.netrc import NetrcCredentials

        mock_get_creds.return_value = NetrcCredentials(
            machine="gerrit.example.org",
            login="netrcuser",
            password="netrcpass",
        )

        client = build_client(
            "gerrit.example.org",
            username="explicit",
            password="credentials",
        )

        assert client.is_authenticated is True
        assert client._auth == _Auth("explicit", "credentials")
        # netrc should not be queried when explicit creds provided
        mock_get_creds.assert_not_called()

    @patch("dependamerge.gerrit.client.get_credentials_for_host")
    def test_build_client_netrc_overrides_env(self, mock_get_creds, monkeypatch):
        """Test that netrc credentials take priority over environment."""
        monkeypatch.setenv("GERRIT_USERNAME", "envuser")
        monkeypatch.setenv("GERRIT_PASSWORD", "envpass")

        # Mock netrc credentials
        from dependamerge.netrc import NetrcCredentials

        mock_get_creds.return_value = NetrcCredentials(
            machine="gerrit.example.org",
            login="netrcuser",
            password="netrcpass",
        )

        client = build_client("gerrit.example.org")

        assert client.is_authenticated is True
        # netrc should take priority
        assert client._auth == _Auth("netrcuser", "netrcpass")

    @patch("dependamerge.gerrit.client.get_credentials_for_host")
    def test_build_client_use_netrc_false(self, mock_get_creds, monkeypatch):
        """Test that use_netrc=False skips netrc lookup."""
        # Clear environment variables
        monkeypatch.delenv("GERRIT_USERNAME", raising=False)
        monkeypatch.delenv("GERRIT_PASSWORD", raising=False)

        client = build_client("gerrit.example.org", use_netrc=False)

        assert client.is_authenticated is False
        mock_get_creds.assert_not_called()

    @patch("dependamerge.gerrit.client.get_credentials_for_host")
    def test_build_client_netrc_with_explicit_file(
        self, mock_get_creds, monkeypatch, tmp_path
    ):
        """Test that netrc_file parameter is passed correctly."""
        # Clear environment variables
        monkeypatch.delenv("GERRIT_USERNAME", raising=False)
        monkeypatch.delenv("GERRIT_PASSWORD", raising=False)

        # Create a dummy netrc file path
        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text("machine gerrit.example.org login user password pass")

        # Mock netrc credentials
        from dependamerge.netrc import NetrcCredentials

        mock_get_creds.return_value = NetrcCredentials(
            machine="gerrit.example.org",
            login="fileuser",
            password="filepass",
        )

        client = build_client("gerrit.example.org", netrc_file=netrc_path)

        assert client.is_authenticated is True
        assert client._auth == _Auth("fileuser", "filepass")
        # Verify netrc_file was passed
        call_kwargs = mock_get_creds.call_args[1]
        assert call_kwargs.get("netrc_file") == netrc_path


class TestBackend:
    """The transport is python-gerrit-api, configured by the client."""

    def test_pygerrit2_is_gone(self):
        """pygerrit2 is no longer installed or imported."""
        assert importlib.util.find_spec("pygerrit2") is None

    @patch("dependamerge.gerrit.client.GerritClient")
    def test_backend_configuration(self, mock_backend):
        """Netrc, retries and timeout are owned by GerritRestClient."""
        client = GerritRestClient(
            base_url="https://gerrit.example.org/infra",
            auth=("user", "password"),
            timeout=12.0,
        )
        with client._backends.acquire():
            pass
        kwargs = mock_backend.call_args.kwargs
        assert kwargs["base_url"] == "https://gerrit.example.org/infra/"
        assert kwargs["username"] == "user"
        assert kwargs["use_netrc"] is False
        assert kwargs["max_retries"] is None
        assert kwargs["timeout"] == 12.0
        assert kwargs["ssl_verify"] is True
