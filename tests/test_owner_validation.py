# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests that an owner is validated however it was written.

``lfreleng-actions`` and ``https://github.com/lfreleng-actions`` are the
same request, but only the bare form was checked.  ``not a url`` was
therefore refused as a bare owner and accepted through a URL, reaching
the API as ``orgs/not%20a%20url`` and returning 404 --- so the operator
saw a not-found error where the identical input given bare produced a
fail-fast message naming the problem.

The regression this guards against is the opposite mistake.  Tightening
the URL path is only safe because the login grammar no longer polices
hyphen *placement*: ``johan--`` is a real user with about two thousand
repositories and ``a--b--t`` a real organisation, and an earlier version
of this gate refused both.  Every entry point is asserted against them.
"""

from __future__ import annotations

import pytest

from dependamerge.github_client import GitHubClient
from dependamerge.url_parser import UrlParseError
from dependamerge.url_parser.change import parse_change_url
from dependamerge.url_parser.repos import (
    parse_org_url,
    parse_owner_arg,
    parse_owner_target,
    parse_repo_url,
)

#: The seven ways an owner reaches the parsers, as callables taking the
#: owner alone.  Parametrising over them is what stops one entry point
#: drifting from the others again.
ENTRY_POINTS = {
    "bare": lambda owner: parse_owner_arg(owner),
    "bare-with-host": lambda owner: parse_owner_target(owner)[0],
    "owner-url": lambda owner: parse_org_url(f"https://github.com/{owner}").owner,
    "orgs-url": lambda owner: parse_org_url(f"https://github.com/orgs/{owner}").owner,
    "repo-url": lambda owner: (
        parse_repo_url(f"https://github.com/{owner}/widget").owner
    ),
    "pr-url": lambda owner: parse_change_url(
        f"https://github.com/{owner}/widget/pull/7"
    ).project.split("/", 1)[0],
    "pr-url-client": lambda owner: GitHubClient(token="t").parse_pr_url(
        f"https://github.com/{owner}/widget/pull/7"
    )[0],
}

REFUSED = [
    "not a url",
    "has$dollar",
    "has_underscore",
    "-leading",
    "x" * 60,
]

#: Real accounts. ``johan--`` is a user with ~1,960 public repositories
#: created in 2008; ``a--b--t`` an organisation (id 7857740) created in
#: 2014. GitHub's signup form would refuse both today, which is why a
#: gate modelled on that form is the wrong gate.
RESOLVABLE = ["johan--", "a--b--t", "trailing-", "lfreleng-actions", "a"]


@pytest.mark.parametrize("entry", ENTRY_POINTS.keys())
class TestEveryEntryPointAgrees:
    """Bare and URL forms of one argument must answer the same."""

    @pytest.mark.parametrize("owner", REFUSED)
    def test_rubbish_is_refused(self, entry: str, owner: str) -> None:
        with pytest.raises(UrlParseError):
            ENTRY_POINTS[entry](owner)

    @pytest.mark.parametrize("owner", RESOLVABLE)
    def test_a_real_account_resolves(self, entry: str, owner: str) -> None:
        assert ENTRY_POINTS[entry](owner) == owner


class TestTheRefusalIsWordedOnce:
    """A different message per entry point is a divergence waiting to happen."""

    @pytest.mark.parametrize("entry", ENTRY_POINTS.keys())
    def test_the_same_guidance_wherever_it_came_from(self, entry: str) -> None:
        with pytest.raises(UrlParseError) as caught:
            ENTRY_POINTS[entry]("not a url")

        assert "Not a valid GitHub owner name: 'not a url'" in str(caught.value)
        assert "alphanumerics and hyphens, at most 39 characters" in str(caught.value)


class TestNothingReachesTheApi:
    """The point of failing fast is that no request is made.

    The transport percent-encodes the value, so an unchecked owner
    becomes a real call to ``orgs/not%20a%20url`` and a 404 the operator
    then has to interpret.
    """

    def test_refusal_happens_during_parsing(self) -> None:
        # Raised by the parser, with no client constructed and no
        # network available to it.
        with pytest.raises(UrlParseError):
            parse_org_url("https://github.com/not a url")


class TestConsecutiveHyphensStayValid:
    """Settled during #475 review and re-pinned here.

    Owner-wide merging is mostly an organisation operation, and
    ``a--b--t`` is a real organisation, so the grammar deliberately
    permits consecutive hyphens.
    """

    @pytest.mark.parametrize("owner", ["a--b--t", "acme--tools", "johan--"])
    def test_they_resolve_through_a_url(self, owner: str) -> None:
        assert parse_org_url(f"https://github.com/{owner}").owner == owner


class TestSurroundingWhitespaceIsRefused:
    """``looks_like_owner`` strips before matching; this must not.

    Accepting the stripped form and returning the original would send
    ``" acme"`` to the API --- the same bypass by a quieter route, and
    invisible to the eye in a URL.

    Only whitespace that survives URL parsing is worth asserting, and
    most does not: ``urlsplit`` removes tabs from the whole URL, and
    whitespace around the entire target is trimmed with it.  What does
    survive is a *leading* space inside the path, and a trailing one
    with another segment behind it.
    """

    @pytest.mark.parametrize(
        "entry", ["owner-url", "orgs-url", "repo-url", "pr-url", "pr-url-client"]
    )
    def test_a_leading_space_is_refused(self, entry: str) -> None:
        with pytest.raises(UrlParseError):
            ENTRY_POINTS[entry](" acme")

    def test_a_trailing_space_before_another_segment_is_refused(self) -> None:
        with pytest.raises(UrlParseError):
            parse_repo_url("https://github.com/acme /widget")

    def test_trimming_the_whole_target_is_not_a_bypass(self) -> None:
        """The control: that whitespace never reaches the gate."""
        assert parse_org_url("https://github.com/acme ").owner == "acme"

    def test_the_bare_form_still_trims(self) -> None:
        """Trimming a shell-quoting artefact there is deliberate."""
        assert parse_owner_arg("  lfreleng-actions  ") == "lfreleng-actions"


class TestTheHostIsStillCheckedFirst:
    """Owner validation must not displace the declaration guard.

    An undeclared host is refused before the owner is looked at, so a
    mistyped host is never reported as a bad owner name. The owner here
    is invalid too, so only the ordering can decide which error wins.
    """

    def test_the_host_error_wins_over_the_owner_error(self) -> None:
        with pytest.raises(UrlParseError) as caught:
            parse_org_url("https://ghe.example.com/not a url")

        message = str(caught.value)
        assert "ghe.example.com" in message
        assert "declare it first" in message
        assert "owner name" not in message.lower()


class TestEnterpriseOwnersKeepTheirGrammar:
    """The dotcom grammar is not the grammar of an Enterprise directory.

    Accounts there are provisioned over LDAP or SAML, which the comment
    beside ``_OWNER_RE`` records explicitly. Applying the dotcom rules
    to a declared Enterprise host would report ``team_name`` as invalid
    on an install where it is somebody's actual login -- and before this
    change, that URL parsed.

    So the gate is host-aware: github.com gets the full grammar, an
    Enterprise host gets only what no directory can produce.
    """

    @pytest.fixture(autouse=True)
    def _declared(self, monkeypatch):
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "ghe.example.com")

    @staticmethod
    def _ghe(owner: str) -> str:
        return parse_org_url(f"https://ghe.example.com/{owner}").owner

    @pytest.mark.parametrize("owner", ["team_name", "has$dollar", "UPPER_case"])
    def test_a_directory_login_is_accepted(self, owner: str) -> None:
        assert self._ghe(owner) == owner

    @pytest.mark.parametrize("owner", ["not a url", " leading", "x" * 60])
    def test_the_unambiguous_rubbish_is_still_refused(self, owner: str) -> None:
        with pytest.raises(UrlParseError):
            self._ghe(owner)

    def test_dotcom_is_unaffected(self) -> None:
        """The control: the same owner, judged by the dotcom grammar."""
        with pytest.raises(UrlParseError):
            parse_org_url("https://github.com/team_name")


class TestGuidanceMatchesTheRulesApplied:
    """Telling an operator to fix what was never wrong wastes their time.

    A single message would have to describe the stricter rule, so an
    Enterprise name refused for whitespace or length was told to remove
    underscores its own directory issues.
    """

    def test_dotcom_names_the_dotcom_grammar(self) -> None:
        with pytest.raises(UrlParseError) as caught:
            parse_org_url("https://github.com/not a url")

        assert "alphanumerics and hyphens" in str(caught.value)

    def test_enterprise_does_not_blame_the_character_set(self, monkeypatch) -> None:
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "ghe.example.com")

        with pytest.raises(UrlParseError) as caught:
            parse_org_url("https://ghe.example.com/not a url")

        message = str(caught.value)
        # The real fault, not a character set the directory may allow.
        assert "whitespace" in message
        assert "ghe.example.com" in message
        assert "alphanumerics and hyphens" not in message


class TestPercentEncodingIsDecodedFirst:
    """A path segment is encoded, so the gate must read what it means.

    ``not%20a%20url`` carries no literal whitespace, so a check written
    for the decoded form let it straight through -- the reported bypass,
    merely spelled differently.

    Decoding also keeps encoded-but-valid URLs working. Nothing in this
    codebase decodes an owner, so the escape reaches GitHub verbatim and
    GitHub resolves it; gating the raw segment would refuse a URL that
    addresses a real account.
    """

    ENCODED_URL_ENTRIES = [
        "owner-url",
        "orgs-url",
        "repo-url",
        "pr-url",
        "pr-url-client",
    ]

    @pytest.mark.parametrize("entry", ENCODED_URL_ENTRIES)
    def test_encoded_whitespace_is_refused(self, entry: str) -> None:
        with pytest.raises(UrlParseError):
            ENTRY_POINTS[entry]("not%20a%20url")

    @pytest.mark.parametrize("entry", ENCODED_URL_ENTRIES)
    def test_an_encoded_separator_is_refused(self, entry: str) -> None:
        with pytest.raises(UrlParseError):
            ENTRY_POINTS[entry]("acme%2Fwidget")

    @pytest.mark.parametrize("entry", ENCODED_URL_ENTRIES)
    def test_an_encoded_valid_name_resolves_decoded(self, entry: str) -> None:
        assert ENTRY_POINTS[entry]("lfreleng%2Dactions") == "lfreleng-actions"

    def test_an_enterprise_directory_name_decodes_too(self, monkeypatch) -> None:
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "ghe.example.com")

        assert parse_org_url("https://ghe.example.com/team%5Fname").owner == (
            "team_name"
        )

    def test_a_bare_argument_keeps_its_percent(self) -> None:
        """A shell can pass a literal ``%``; decoding would retarget the run."""
        with pytest.raises(UrlParseError):
            parse_owner_arg("lfreleng%2Dactions")


class TestAnOwnerMustSurviveBeingAPathSegment:
    """The owner is interpolated into REST paths throughout the client.

    So a name carrying a URL-structural character does not fail -- it
    succeeds against a *different* request. ``acme?admin=true`` turns
    ``/orgs/{owner}/repos`` into ``/orgs/acme`` with the query
    ``admin=true/repos``.

    github.com cannot reach this, its grammar being alphanumerics and
    hyphens. A directory issuing arbitrary names can, which is why the
    Enterprise branch needs the check the dotcom grammar gives for free.
    """

    @pytest.fixture(autouse=True)
    def _declared(self, monkeypatch):
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "ghe.example.com")

    @staticmethod
    def _ghe(encoded: str) -> str:
        return parse_org_url(f"https://ghe.example.com/{encoded}").owner

    @pytest.mark.parametrize(
        ("encoded", "decodes_to"),
        [
            ("acme%3Fadmin=true", "a query"),
            ("acme%23frag", "a fragment"),
            ("acme%252Fwidget", "a residual escape"),
            ("acme%5Cwin", "a backslash separator"),
        ],
    )
    def test_a_structural_character_is_refused(
        self, encoded: str, decodes_to: str
    ) -> None:
        with pytest.raises(UrlParseError):
            self._ghe(encoded)

    @pytest.mark.parametrize("encoded", ["team%5Fname", "acme.tools", "acme@corp"])
    def test_an_ordinary_directory_name_still_resolves(self, encoded: str) -> None:
        """The control: only structure is policed, not the character set."""
        assert self._ghe(encoded) == encoded.replace("%5F", "_")

    def test_the_guidance_names_the_real_fault(self) -> None:
        with pytest.raises(UrlParseError) as caught:
            self._ghe("acme%3Fadmin=true")

        assert "change what a URL means" in str(caught.value)
