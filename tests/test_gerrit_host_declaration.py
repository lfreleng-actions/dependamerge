# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for declaring a Gerrit host explicitly.

GitHub Enterprise hosts are declared before they are addressed; Gerrit
hosts were only inferred, from the shape of a URL path.  So routing
consulted the path alone, and a host declared as GitHub Enterprise was
still handed to the Gerrit client whenever a URL happened to carry
``/c/…/+/`` or ``/q/topic:``::

    $ export DEPENDAMERGE_GITHUB_HOSTS=ghe.corp.example.com
    $ dependamerge merge https://ghe.corp.example.com/c/project/+/123
    WARNING - No netrc credentials found for ghe.corp.example.com
    ❌ Gerrit credentials not found.

A declaration is a statement about the *host*; a path shape is a guess
about the *URL*.  The declaration wins, which is the rule
``local_repo._looks_like_gerrit_remote`` already applies to remotes.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from dependamerge.cli import app
from dependamerge.cli._merge_target import _parse_merge_target, _resolve_target_url
from dependamerge.local_remote import _looks_like_gerrit_remote
from dependamerge.local_repo import LocalTarget
from dependamerge.url_parser import (
    detect_source,
    is_declared_gerrit_host,
    looks_like_topic_search,
    parse_change_url,
    parse_gerrit_topic_url,
    parse_org_url,
    parse_owner_arg,
    parse_owner_target,
    parse_repo_url,
    set_gerrit_host,
)
from dependamerge.url_parser.models import (
    ChangeSource,
    HostDeclarationError,
    UrlParseError,
)

GHE = "ghe.corp.example.com"
REVIEW = "review.corp.example.com"


@pytest.fixture
def declared_github(monkeypatch):
    monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", GHE)
    return GHE


@pytest.fixture
def declared_gerrit(monkeypatch):
    monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOSTS", REVIEW)
    return REVIEW


class TestAGerritHostCanBeDeclared:
    """From the command line and from the environment, as GitHub is."""

    def test_an_undeclared_host_is_not_gerrit(self) -> None:
        assert is_declared_gerrit_host(REVIEW) is False

    def test_the_environment_declares_one(self, declared_gerrit) -> None:
        assert is_declared_gerrit_host(REVIEW) is True

    def test_the_list_is_split_on_commas(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "DEPENDAMERGE_GERRIT_HOSTS", " first.example.com , second.example.com "
        )
        assert is_declared_gerrit_host("first.example.com") is True
        assert is_declared_gerrit_host("second.example.com") is True
        assert is_declared_gerrit_host("third.example.com") is False

    def test_the_single_host_variable_declares_too(self, monkeypatch) -> None:
        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOST", REVIEW)
        assert is_declared_gerrit_host(REVIEW) is True

    def test_the_flag_declares_one(self) -> None:
        set_gerrit_host(REVIEW)
        assert is_declared_gerrit_host(REVIEW) is True

    def test_a_subdomain_does_not_inherit_the_declaration(
        self, declared_gerrit
    ) -> None:
        """An arbitrary hostname says nothing about its subdomains.

        The treatment a declared Enterprise host already gets, for the
        same reason.
        """
        assert is_declared_gerrit_host(f"sub.{REVIEW}") is False

    def test_a_host_naming_a_port_is_refused(self) -> None:
        """The same rule the GitHub declaration applies, worded for Gerrit."""
        with pytest.raises(UrlParseError, match="Gerrit host"):
            set_gerrit_host("review.example.com:8443")

    def test_the_bare_gerrit_host_variable_is_not_read(self, monkeypatch) -> None:
        """``GERRIT_HOST`` already means something else here.

        ``merge_manager/_gerrit_submit.py`` reads it to choose which
        server a GitHub2Gerrit change is submitted on.  Consuming it
        here would silently reinterpret an existing setting as a
        routing declaration.
        """
        monkeypatch.setenv("GERRIT_HOST", REVIEW)
        assert is_declared_gerrit_host(REVIEW) is False


class TestADeclaredGitHubHostIsNotRoutedToGerrit:
    """The gap #478 reports, in both Gerrit URL shapes."""

    def test_a_change_url_is_refused(self, declared_github) -> None:
        with pytest.raises(UrlParseError, match="Refusing to treat"):
            parse_change_url(f"https://{GHE}/c/project/+/123")

    def test_a_topic_url_is_refused(self, declared_github) -> None:
        """The second shape, which lives in a different module.

        ``/q/topic:`` is not in ``_is_gerrit_url`` at all --- it is
        reached only because ``parse_change_url`` raised first --- so
        the rule has to be applied in both places or declaring a host
        stops one shape and not the other.
        """
        with pytest.raises(UrlParseError, match="Refusing to treat"):
            parse_gerrit_topic_url(f"https://{GHE}/q/topic:release")

    def test_a_pull_request_url_still_parses(self, declared_github) -> None:
        """The control: the declaration did not break GitHub routing."""
        parsed = parse_change_url(f"https://{GHE}/owner/repo/pull/7")
        assert parsed.source is ChangeSource.GITHUB
        assert parsed.host == GHE

    def test_dotcom_is_covered_without_a_declaration(self) -> None:
        """github.com is a GitHub host whether or not anyone says so."""
        with pytest.raises(UrlParseError) as caught:
            parse_gerrit_topic_url("https://github.com/q/topic:release")

        message = str(caught.value)
        assert "cannot be routed to Gerrit" in message
        # Its GitHub-ness is intrinsic, so there is no declaration to
        # remove --- and following that advice would only produce the
        # "declared as both" refusal.
        assert "remove it from the GitHub declarations" not in message

    def test_a_declared_host_keeps_the_removable_remedy(self, declared_github) -> None:
        """The control: an Enterprise declaration *can* be removed."""
        with pytest.raises(UrlParseError, match="remove it from the GitHub"):
            parse_gerrit_topic_url(f"https://{GHE}/q/topic:release")


class TestAnUndeclaredHostKeepsStructuralInference:
    """No regression for anyone who declares nothing.

    Gerrit has no equivalent of github.com, so requiring a declaration
    before any Gerrit URL parsed would break every existing user.
    """

    def test_a_project_containing_pull_is_not_a_github_url(self) -> None:
        """A Gerrit project may carry ``pull/<digits>`` of its own.

        ``/c/team/pull/123/+/456`` is a change on the project
        ``team/pull/123``, but the ``/pull/`` heuristic claimed it first
        and parsed the GitHub pull request ``c/team#123`` --- a wrong
        target rather than a failed parse, and one that slipped past the
        declaration refusal too.  ``/c/…/+/`` cannot occur in a GitHub
        pull request URL, so it is tested first.
        """
        parsed = parse_change_url("https://gerrit.example.org/c/team/pull/123/+/456")

        assert parsed.source is ChangeSource.GERRIT
        assert parsed.project == "team/pull/123"
        assert parsed.change_number == 456

    def test_detect_source_agrees_about_that_project(self) -> None:
        """The exported detector repeated the same ordering."""
        assert (
            detect_source("https://gerrit.example.org/c/team/pull/123/+/456")
            is ChangeSource.GERRIT
        )

    def test_an_ordinary_pull_request_is_unaffected(self) -> None:
        """The control: the heuristic still decides when nothing outranks it."""
        assert (
            parse_change_url("https://github.com/acme/widget/pull/7").project
            == "acme/widget"
        )

    def test_gerrit_markers_in_a_pull_request_suffix_are_not_a_change(self) -> None:
        """The shape test is anchored, not a substring search.

        A GitHub pull request URL tolerates trailing segments, and those
        can spell ``/c/`` and ``/+/`` without the path ever being a
        change.  Testing for the markers anywhere stopped
        ``/owner/repo/pull/7/c/foo/+/1`` resolving pull request 7 ---
        trading one wrong answer for another.
        """
        parsed = parse_change_url("https://github.com/owner/repo/pull/7/c/foo/+/1")

        assert parsed.source is ChangeSource.GITHUB
        assert parsed.project == "owner/repo"
        assert parsed.change_number == 7

    def test_a_base_path_before_the_change_still_parses(self) -> None:
        """The anchored shape is the one the Gerrit parser accepts."""
        parsed = parse_change_url(
            "https://gerrit.example.org/infra/c/project/name/+/12345"
        )

        assert parsed.project == "project/name"
        assert parsed.change_number == 12345

    def test_a_change_url_still_routes_to_gerrit(self) -> None:
        parsed = parse_change_url("https://gerrit.example.org/c/project/+/12345")
        assert parsed.source is ChangeSource.GERRIT

    def test_a_topic_url_still_routes_to_gerrit(self) -> None:
        parsed = parse_gerrit_topic_url("https://gerrit.example.org/q/topic:release")
        assert parsed.source is ChangeSource.GERRIT
        assert parsed.topic == "release"


class TestADeclaredGerritHostRoutesRegardlessOfPath:
    """The half that needs the declaration to do something positive."""

    def test_a_change_url_routes_to_gerrit(self, declared_gerrit) -> None:
        parsed = parse_change_url(f"https://{REVIEW}/c/project/+/123")
        assert parsed.source is ChangeSource.GERRIT
        assert parsed.host == REVIEW

    def test_a_topic_url_routes_to_gerrit(self, declared_gerrit) -> None:
        parsed = parse_gerrit_topic_url(f"https://{REVIEW}/q/topic:release")
        assert parsed.source is ChangeSource.GERRIT

    def test_a_port_is_refused_rather_than_dropped(self, declared_gerrit) -> None:
        """``urlparse`` reports the hostname without it.

        So a target naming ``:8443`` would be parsed for the bare host
        and then addressed on the default port --- a different server
        than the operator named.  Every GitHub boundary already refuses
        one; these two now do as well.
        """
        with pytest.raises(UrlParseError, match="does not support a port"):
            parse_change_url(f"https://{REVIEW}:8443/c/project/+/123")
        with pytest.raises(UrlParseError, match="does not support a port"):
            parse_gerrit_topic_url(f"https://{REVIEW}:8443/q/topic:release")

    def test_detect_source_refuses_a_port_too(self, declared_gerrit) -> None:
        """The detector must not answer where the parser refuses."""
        with pytest.raises(UrlParseError, match="does not support a port"):
            detect_source(f"https://{REVIEW}:8443/c/project/+/123")

    def test_a_malformed_change_url_keeps_the_gerrit_diagnostic(
        self, declared_gerrit
    ) -> None:
        """The cascade must not answer for a declared Gerrit host.

        Falling through ended with the repository parser telling the
        operator to declare a GitHub host --- for a host they had just
        declared as Gerrit.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with pytest.raises(typer.Exit):
                _parse_merge_target(f"https://{REVIEW}/acme/widget")

        printed = " ".join(buffer.getvalue().split())
        assert "Invalid Gerrit change URL format" in printed
        assert "DEPENDAMERGE_GITHUB_HOSTS" not in printed

    def test_a_query_string_is_not_a_topic_search(self) -> None:
        """Scanning the raw target matched a query and refused a repo URL.

        ``?next=/q/topic:release`` is not a topic search, and exiting on
        it stopped ``parse_repo_url`` accepting a perfectly good
        repository URL.
        """
        target = _parse_merge_target(
            "https://github.com/acme/widget?next=/q/topic:release"
        )

        assert target.repo is not None
        assert target.repo.project == "acme/widget"

    def test_a_fragment_mentioning_a_topic_is_not_one(self) -> None:
        """Only the legacy ``#/q/`` form is a topic search.

        Matching the marker anywhere in a fragment refused
        ``/acme/widget#notes/q/topic:release`` --- a repository URL
        whose fragment merely mentions one.
        """
        target = _parse_merge_target(
            "https://github.com/acme/widget#notes/q/topic:release"
        )

        assert target.repo is not None
        assert target.repo.project == "acme/widget"

    def test_hostless_shorthand_is_not_a_topic_search(self) -> None:
        """A topic search must name its own server, as the parser says.

        Normalising first expanded ``q/topic:release`` as *owner
        shorthand* against the GitHub default host, which then read as a
        topic search on github.com --- so the selector claimed a target
        the parser refuses, and the cascade stopped on a Gerrit error
        for input that was never Gerrit.
        """
        assert not looks_like_topic_search("q/topic:release")
        assert not looks_like_topic_search("topic:release")
        # The control: naming a host makes it one.
        assert looks_like_topic_search("gerrit.example.org/q/topic:release")

    def test_a_topic_after_another_term_is_recognised(self) -> None:
        """The parser supports it, so the selector must agree.

        Approximating the query grammar missed
        ``status:open+topic:release`` and claimed ``notopic:release``;
        both now go through the parser's own term regex.
        """
        assert looks_like_topic_search(
            "https://gerrit.example.org/q/status:open+topic:release"
        )
        assert not looks_like_topic_search(
            "https://github.com/acme/widget#/q/notopic:release"
        )

    def test_the_legacy_fragment_form_still_counts(self) -> None:
        """The control: ``#/q/topic:`` is what the parser accepts."""
        assert looks_like_topic_search("https://gerrit.example.org/#/q/topic:release")

    def test_an_encoded_topic_url_is_recognised(self) -> None:
        """The parser supports ``%3A``, so the selector must see it too."""
        assert looks_like_topic_search("https://gerrit.example.org/q/topic%3Arelease")

    def test_an_undeclared_topic_url_keeps_its_own_error(self) -> None:
        """A topic-shaped target reports the topic parser's complaint.

        The cascade does not recognise ``/q/topic:`` as Gerrit, so it
        printed GitHub repository guidance for a Gerrit dashboard URL
        --- burying real faults the parser had found, an unsupported
        port among them.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with pytest.raises(typer.Exit):
                _parse_merge_target("https://gerrit.example.org:8443/q/topic:release")

        printed = " ".join(buffer.getvalue().split())
        assert "does not support a port" in printed
        assert "DEPENDAMERGE_GITHUB_HOSTS" not in printed

    def test_a_topic_url_survives_that_diagnostic(self, declared_gerrit) -> None:
        """Both Gerrit shapes have to be tried before the change error wins.

        On a declared host every path routes to the change parser, so a
        perfectly good ``/q/topic:`` URL arrives carrying a
        change-format error.  Reporting it before the topic parser ran
        refused a target the tool supports.
        """
        target = _parse_merge_target(f"https://{REVIEW}/q/topic:release")

        assert target.topic is not None
        assert target.topic.topic == "release"


class TestTheOverlappingGrammarPrefersTheHost:
    """``/c/team/pull/123/+/456`` satisfies both grammars.

    The PR regex tolerates trailing segments, so that path is a GitHub
    pull request *and* a Gerrit change by shape alone.  On a host that
    is GitHub the Gerrit reading is meaningless, and preferring it
    refused a URL ``GitHubClient.parse_pr_url`` still resolves --- two
    public entry points disagreeing about one address.
    """

    OVERLAPPING = "/c/team/pull/123/+/456"

    def test_github_wins_on_a_github_host(self) -> None:
        parsed = parse_change_url(f"https://github.com{self.OVERLAPPING}")

        assert parsed.source is ChangeSource.GITHUB
        assert parsed.project == "c/team"
        assert parsed.change_number == 123

    def test_every_entry_point_agrees(self) -> None:
        """The disagreement was the fault, so it is asserted directly."""
        from dependamerge.github_client import GitHubClient

        url = f"https://github.com{self.OVERLAPPING}"
        client = GitHubClient(token="test_token")

        assert detect_source(url) is ChangeSource.GITHUB
        assert client.parse_pr_url(url) == ("c", "team", 123)

    def test_a_gerrit_only_path_on_dotcom_names_the_real_problem(self) -> None:
        """``/c/project/+/123`` is Gerrit-shaped and nothing else.

        Deferring to GitHub on the host alone sent it to the pull
        request parser, which answered with a generic format error
        instead of the refusal that says what is actually wrong.
        """
        for call in (parse_change_url, detect_source):
            with pytest.raises(UrlParseError, match="cannot be routed to Gerrit"):
                call("https://github.com/c/project/+/123")

    def test_gerrit_wins_on_a_host_that_is_not_github(self) -> None:
        """The control: off github.com the Gerrit reading is the right one."""
        parsed = parse_change_url(f"https://gerrit.example.org{self.OVERLAPPING}")

        assert parsed.source is ChangeSource.GERRIT
        assert parsed.project == "team/pull/123"
        assert parsed.change_number == 456


class TestDeclaringAHostAsBothIsReported:
    """Reported rather than resolved by precedence.

    Whichever way a precedence rule fell it would silently ignore one of
    two explicit statements about where the operator's credentials may
    go, and the ignored one would be invisible at the point it mattered.
    """

    @pytest.fixture
    def declared_twice(self, monkeypatch):
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", REVIEW)
        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOSTS", REVIEW)
        return REVIEW

    def test_a_change_url_reports_the_conflict(self, declared_twice) -> None:
        with pytest.raises(UrlParseError, match="declared as both"):
            parse_change_url(f"https://{REVIEW}/c/project/+/123")

    def test_a_topic_url_reports_the_conflict(self, declared_twice) -> None:
        with pytest.raises(UrlParseError, match="declared as both"):
            parse_gerrit_topic_url(f"https://{REVIEW}/q/topic:release")

    def test_a_github_shaped_url_reports_it_too(self, declared_twice) -> None:
        """The contradiction is about the host, not about one URL."""
        with pytest.raises(UrlParseError, match="declared as both"):
            parse_change_url(f"https://{REVIEW}/owner/repo/pull/7")

    def test_declaring_dotcom_as_gerrit_conflicts(self, monkeypatch) -> None:
        """github.com needs no declaration to be a GitHub host."""
        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOSTS", "github.com")
        with pytest.raises(UrlParseError, match="declared as both"):
            parse_change_url("https://github.com/owner/repo/pull/7")


class TestUnusableGitHubConfigurationDoesNotAbortGerrit:
    """A Gerrit target must not fail over settings it never reads.

    ``is_supported_github_host`` raises when a declaration names a port.
    Asked during routing that would abort a Gerrit URL on unrelated
    configuration --- the reasoning ``local_repo`` already records for
    the local-checkout path, which swallows the identical error.
    """

    def test_a_gerrit_url_parses_despite_a_bad_github_host(self, monkeypatch) -> None:
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "ghe.example.com:8443")

        parsed = parse_change_url("https://gerrit.example.org/c/project/+/1")

        assert parsed.source is ChangeSource.GERRIT

    def test_a_github_url_parses_despite_a_bad_gerrit_host(self, monkeypatch) -> None:
        """The mirror, and the one the first push got wrong.

        Routing consults the Gerrit declarations for *every* target, so
        propagating an unusable value there failed an ordinary pull
        request URL over configuration it never reads.
        """
        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOSTS", "review.example.com:8443")

        parsed = parse_change_url("https://github.com/acme/widget/pull/7")

        assert parsed.source is ChangeSource.GITHUB

    def test_the_flag_still_reports_an_unusable_value(self) -> None:
        """Swallowing during routing must not silence the flag."""
        with pytest.raises(UrlParseError, match="Gerrit host"):
            set_gerrit_host("review.example.com:8443")


class TestTheRefusalSurvivesTheParserCascade:
    """A refusal about the host stops the cascade rather than joining it.

    ``_parse_merge_target`` tries change, then topic, then owner, then
    repository.  Every other failure means "not this shape, try the
    next"; a declaration refusal means "not this host, whatever the
    shape".  Letting it join the cascade showed the operator whichever
    parser complained last -- for a topic URL, an invalid *repository
    name* of ``topic:``, which describes neither the fault nor its
    remedy.
    """

    @staticmethod
    def _reported(url: str) -> str:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with pytest.raises(typer.Exit):
                _parse_merge_target(url)
        return " ".join(buffer.getvalue().split())

    def test_a_topic_url_reports_the_routing_refusal(self, declared_github) -> None:
        message = self._reported(f"https://{GHE}/q/topic:release")

        assert "Refusing to treat" in message
        assert "repository name" not in message

    def test_a_change_url_reports_the_routing_refusal(self, declared_github) -> None:
        message = self._reported(f"https://{GHE}/c/project/+/123")

        assert "Refusing to treat" in message

    @pytest.mark.parametrize(
        "path", ["acme/widget", "acme", "acme/widget/pull/7", "orgs/acme"]
    )
    def test_an_ordinary_github_target_is_unaffected(
        self, declared_github, path: str
    ) -> None:
        """The regression the first push introduced.

        The topic parser is tried *before* the owner and repository
        ones, so an ordinary ``owner/repo`` URL passes through it on the
        way past.  Refusing at the top of that parser therefore refused
        every non-PR target on a declared GitHub host.  The rules belong
        to a URL that really is asking for Gerrit.
        """
        assert _parse_merge_target(f"https://{GHE}/{path}") is not None


class TestEveryGitHubEntryPointChecksTheConflict:
    """A contradiction is about the host, so every route must see it.

    Checking it in the change and topic parsers alone left the cascade
    able to fall past a refusal into ``parse_repo_url`` and succeed.
    """

    @pytest.fixture
    def declared_twice(self, monkeypatch):
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", REVIEW)
        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOSTS", REVIEW)
        return REVIEW

    def test_the_repository_parser_reports_it(self, declared_twice) -> None:
        with pytest.raises(UrlParseError, match="declared as both"):
            parse_repo_url(f"https://{REVIEW}/acme/widget")

    def test_the_owner_parser_reports_it(self, declared_twice) -> None:
        with pytest.raises(UrlParseError, match="declared as both"):
            parse_org_url(f"https://{REVIEW}/acme")

    def test_the_cascade_cannot_fall_past_it(self, declared_twice) -> None:
        """End to end: no parser below accepts what the first refused."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with pytest.raises(typer.Exit):
                _parse_merge_target(f"https://{REVIEW}/acme/widget")

        assert "declared as both" in " ".join(buffer.getvalue().split())

    def test_the_client_parser_reports_it(self, declared_twice) -> None:
        """Its own host-policy boundary, reached without ``url_parser``."""
        from dependamerge.github_client import GitHubClient

        client = GitHubClient(token="test_token")

        with pytest.raises(UrlParseError, match="declared as both"):
            client.parse_pr_url(f"https://{REVIEW}/acme/widget/pull/7")

    @pytest.mark.parametrize("resolve", [parse_owner_target, parse_owner_arg])
    def test_a_bare_owner_reports_it(self, monkeypatch, resolve) -> None:
        """``status acme`` and ``blocked acme`` resolve a default host.

        The bare branch never reaches ``parse_org_url``, so a check
        placed only there let those commands run under contradictory
        declarations.
        """
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOST", REVIEW)
        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOSTS", REVIEW)

        with pytest.raises(UrlParseError, match="declared as both"):
            resolve("acme")


class TestDetectSourceAgreesWithTheParser:
    """``detect_source`` is exported, so it must not disagree.

    It answered from the path heuristics alone, which put it at odds
    with ``parse_change_url`` in both directions once declarations
    entered the picture.
    """

    def test_a_declared_gerrit_host_reports_gerrit(self, declared_gerrit) -> None:
        """Even for a pull-request-shaped path, which parsing routes to Gerrit."""
        assert (
            detect_source(f"https://{REVIEW}/owner/repo/pull/7") is ChangeSource.GERRIT
        )

    def test_a_declared_github_host_refuses_gerrit_shapes(
        self, declared_github
    ) -> None:
        with pytest.raises(UrlParseError, match="Refusing to treat"):
            detect_source(f"https://{GHE}/c/project/+/123")

    def test_an_undeclared_host_is_unchanged(self) -> None:
        """The control: inference still decides when nothing is declared."""
        assert (
            detect_source("https://gerrit.example.org/c/p/+/1") is ChangeSource.GERRIT
        )
        assert (
            detect_source("https://github.com/acme/widget/pull/7")
            is ChangeSource.GITHUB
        )


class TestALocalCheckoutHonoursTheDeclaration:
    """The omitted-target flow reads a remote, not a URL.

    ``dependamerge merge`` with no argument classifies the checkout from
    its remote.  A plain HTTPS remote on a declared Gerrit host carries
    none of the other evidence --- no port 29418, and an arbitrary
    hostname --- so without the declaration it was classified as GitHub
    and then failed asking the operator to declare a host they had
    already declared, for the other platform.
    """

    def test_a_declared_host_makes_an_https_remote_gerrit(
        self, declared_gerrit
    ) -> None:
        assert _looks_like_gerrit_remote(f"https://{REVIEW}/acme/widget") is True

    @pytest.mark.parametrize(
        "authority",
        [f"{REVIEW}:8443", f"{REVIEW}:notaport", f"user@{REVIEW}", REVIEW],
    )
    def test_the_authority_is_read_not_pattern_matched(
        self, declared_gerrit, authority: str
    ) -> None:
        """Stripping a trailing ``:digits`` missed two authority forms.

        A malformed port left ``host:notaport`` matching no declaration
        at all, and a bracketed IPv6 authority kept its brackets.  Both
        fall out of reading the URL rather than pattern-matching it.
        """
        assert _looks_like_gerrit_remote(f"https://{authority}/acme/widget") is True

    def test_an_scp_style_remote_keeps_its_host(self) -> None:
        """The control the hand-rolled version existed to protect.

        ``git@host:owner/repo.git`` puts the *path* after the colon, so
        a host reader must not mistake it for a port.
        """
        from dependamerge.local_remote import _gerrit_identity_from_remote

        host, project = _gerrit_identity_from_remote("git@github.com:acme/widget.git")

        assert host == "github.com"
        assert project == "acme/widget"

    def test_the_reported_host_is_canonical_too(self, declared_gerrit) -> None:
        """Classification and ``LocalTarget.host`` must agree.

        Reducing the authority for the lookup alone left the reported
        host carrying a port, so the conflict check the CLI runs against
        it missed declarations for the bare host.
        """
        from dependamerge.local_remote import _gerrit_identity_from_remote

        host, project = _gerrit_identity_from_remote(
            f"https://{REVIEW}:8443/acme/widget"
        )

        assert host == REVIEW
        assert project == "acme/widget"

    def test_a_port_does_not_hide_the_declaration(self, declared_gerrit) -> None:
        """Userinfo and a port belong to the authority, not the host.

        Comparing them along with it defeated every declaration, since
        ``review.corp.example.com:8443`` matches no declared host --- so
        a remote naming a port fell through to the name guess.
        """
        assert _looks_like_gerrit_remote(f"https://{REVIEW}:8443/acme/widget") is True
        assert _looks_like_gerrit_remote(f"https://user@{REVIEW}/acme/widget") is True

    def test_an_undeclared_host_is_unchanged(self) -> None:
        """The control: a plain remote on an unknown host stays GitHub-ish."""
        assert _looks_like_gerrit_remote(f"https://{REVIEW}/acme/widget") is False

    def test_the_ssh_port_still_decides_on_its_own(self) -> None:
        """The stronger evidence keeps working without any declaration."""
        assert (
            _looks_like_gerrit_remote("ssh://user@anything.example.com:29418/project")
            is True
        )

    def test_a_github_declaration_still_wins_over_a_name_guess(
        self, monkeypatch
    ) -> None:
        """The precedent this builds on, asserted so it is not lost."""
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", "gerrit.corp.example.com")

        assert (
            _looks_like_gerrit_remote("https://gerrit.corp.example.com/acme/widget")
            is False
        )

    def test_a_conflict_is_reported_before_the_checkout_guidance(
        self, monkeypatch
    ) -> None:
        """The Gerrit branch exits without reaching any parser.

        So this is the only place the omitted target is ever checked.
        Reporting the contradiction after the checkout guidance --- or
        not at all --- would leave the operator reading advice about
        Gerrit targets while the real fault was their configuration.
        """
        monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", REVIEW)
        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOSTS", REVIEW)
        monkeypatch.setattr(
            "dependamerge.cli._merge_target.detect_local_target",
            lambda: LocalTarget(
                source=ChangeSource.GERRIT,
                url="",
                remote="origin",
                root=Path("/tmp"),
                host=REVIEW,
            ),
        )

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with pytest.raises(typer.Exit):
                _resolve_target_url("")

        printed = " ".join(buffer.getvalue().split())
        assert "declared as both" in printed
        assert "not addressable as a repository" not in printed


class TestTheFlagIsWiredIntoEveryCommand:
    """The declaration is useless if the flag does not reach it.

    Everything above exercises the declaration through
    ``set_gerrit_host``, which a miswired option would still satisfy:
    the flag could be missing from a command, or declared and never
    applied, and the parser-level suite would stay green.

    Mirrors the coverage ``--github-host`` already has.
    """

    runner = CliRunner()

    @staticmethod
    def _option_names(command: str) -> set[str]:
        """Return the option strings a command accepts.

        Asks the command object rather than parsing ``--help``, which
        is a rendering: Rich colours the flag when it detects a
        terminal, so the literal text is absent from output that
        displays it perfectly.
        """
        cli = typer.main.get_command(app)
        subcommand = cli.commands[command]  # type: ignore[attr-defined]
        names: set[str] = set()
        for param in subcommand.params:
            names.update(param.opts)
        return names

    @pytest.mark.parametrize("command", ["merge", "close", "status", "blocked"])
    def test_every_target_taking_command_offers_it(self, command: str) -> None:
        assert "--gerrit-host" in self._option_names(command)

    def test_the_flag_reaches_the_declaration(self) -> None:
        """End to end: the declaration changes how a target routes.

        Without the flag this host is undeclared, so a Gerrit-shaped
        path is the only thing that would route it to Gerrit.  With the
        flag, an ordinary repository path does too --- and a Gerrit
        target is refused as unaddressable from a repository URL, which
        is the message that proves the routing changed.
        """
        result = self.runner.invoke(
            app,
            [
                "merge",
                f"https://{REVIEW}/acme/widget",
                "--gerrit-host",
                REVIEW,
                "--token",
                "t",
                "--dry-run",
            ],
        )

        assert result.exit_code == 1
        assert "Invalid Gerrit change URL format" in result.stdout

    def test_without_the_flag_the_same_target_is_not_gerrit(self) -> None:
        """The control, so the test above cannot pass on any failure."""
        result = self.runner.invoke(
            app,
            [
                "merge",
                f"https://{REVIEW}/acme/widget",
                "--token",
                "t",
                "--dry-run",
            ],
        )

        assert "Invalid Gerrit change URL format" not in result.stdout

    def test_an_unusable_value_is_reported_not_raised(self) -> None:
        """A port cannot reach the API base URL, so it is refused.

        Reported as a message through ``apply_gerrit_host``; without
        that the configuration reader would surface a traceback on an
        ordinary mistake.
        """
        result = self.runner.invoke(
            app,
            ["merge", "acme/widget", "--gerrit-host", f"{REVIEW}:8443", "--token", "t"],
        )

        assert result.exit_code == 1
        assert "names a port" in result.stdout
        assert "Traceback" not in result.stdout


class TestAMalformedDeclarationOfTheTargetIsReported:
    """Ignoring it routed a host the operator meant as Gerrit to GitHub."""

    def test_a_port_on_the_target_host_is_reported(self, monkeypatch) -> None:
        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOST", f"{REVIEW}:8443")

        with pytest.raises(HostDeclarationError, match="names a port"):
            parse_repo_url(f"https://{REVIEW}/acme/widget")

    @pytest.mark.parametrize(
        "value",
        [f"https://{REVIEW}:8443@evil.example", f"{REVIEW}:8443/path"],
    )
    def test_extra_syntax_is_not_blamed_on_the_target(
        self, monkeypatch, value: str
    ) -> None:
        """``review:8443@evil.example`` names ``evil.example``, not review."""
        from dependamerge.url_parser.gerrit_hosts import malformed_declaration_for

        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOST", value)

        assert malformed_declaration_for(REVIEW) is None

    def test_a_hostless_target_matches_nothing(self, monkeypatch) -> None:
        from dependamerge.url_parser.gerrit_hosts import malformed_declaration_for

        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOST", ":8443")

        assert malformed_declaration_for("") is None

    def test_one_for_another_host_is_still_tolerated(self, monkeypatch) -> None:
        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOST", "other.example.com:8443")

        assert parse_change_url("https://github.com/a/b/pull/1").change_number == 1


class TestADeclaredSingleLabelScpRemoteIsAServer:
    """``review:acme/widget.git`` is ambiguous until a declaration names it."""

    def test_the_declaration_settles_it(self) -> None:
        set_gerrit_host("review")

        assert _looks_like_gerrit_remote("review:acme/widget.git") is True

    def test_undeclared_it_is_still_a_local_path(self) -> None:
        assert _looks_like_gerrit_remote("review:acme/widget.git") is False

    def test_a_drive_letter_stays_local_whatever_is_declared(self) -> None:
        from dependamerge.local_remote import _names_a_server

        set_gerrit_host("c")

        assert _names_a_server("C:/repos/widget.git") is False


class TestAMalformedAuthorityIsNotGerritEvidence:
    """A textual port check read ``[bad]:29418`` as Gerrit, then crashed."""

    def test_it_is_not_classified(self) -> None:
        assert _looks_like_gerrit_remote("ssh://[bad]:29418/project") is False

    def test_the_identity_does_not_raise(self) -> None:
        from dependamerge.local_remote import _gerrit_identity_from_remote

        assert _gerrit_identity_from_remote("ssh://[bad]:29418/project") == ("", "")


class TestAPathParameterFollowsTheDeclarationChecks:
    """A ``;suffix`` must not mask what the declarations say, nor be
    accepted by the detector the parser refuses it in.
    """

    def test_declared_as_both_is_still_reported(
        self, declared_github, monkeypatch
    ) -> None:
        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOSTS", GHE)

        with pytest.raises(HostDeclarationError, match="declared as both"):
            parse_change_url(f"https://{GHE}/c/project/+/123;other")

    def test_a_conflict_outranks_a_stray_git_suffix(
        self, declared_github, monkeypatch
    ) -> None:
        """As :func:`detect_source` already ordered it."""
        monkeypatch.setenv("DEPENDAMERGE_GERRIT_HOSTS", GHE)
        url = f"https://{GHE}/c/project/+/1.git"

        for check in (parse_change_url, detect_source):
            with pytest.raises(HostDeclarationError, match="declared as both"):
                check(url)

    def test_a_github_host_is_still_refused_as_gerrit(self, declared_github) -> None:
        with pytest.raises(HostDeclarationError, match="it is a GitHub host"):
            parse_change_url(f"https://{GHE}/c/project/+/123;other")

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/acme/widget/pull/7;other",
            "https://gerrit.example.org/c/project/+/1;other",
        ],
    )
    def test_the_detector_refuses_what_the_parser_refuses(self, url: str) -> None:
        with pytest.raises(UrlParseError, match="path parameter"):
            parse_change_url(url)
        with pytest.raises(UrlParseError, match="path parameter"):
            detect_source(url)


class TestAnEmptyTopicIsAGerritError:
    """``/q/topic:`` used to fall through to unrelated GitHub guidance."""

    @pytest.mark.parametrize("query", ["topic:", 'topic:""', "topic:+status:open"])
    def test_it_is_refused_as_empty(self, query: str) -> None:
        url = f"https://{REVIEW}/q/{query}"

        assert looks_like_topic_search(url) is True
        with pytest.raises(UrlParseError, match="topic cannot be empty"):
            parse_gerrit_topic_url(url)

    @pytest.mark.parametrize("query", ['topic:"abc', 'topic:a"b'])
    def test_a_malformed_term_matches_no_prefix(self, query: str) -> None:
        with pytest.raises(UrlParseError, match="Only topic searches"):
            parse_gerrit_topic_url(f"https://{REVIEW}/q/{query}")


class TestATopicRefusalDoesNotEchoTheQuery:
    """The decoded query bypassed both the redaction and the ``;`` guard."""

    def test_an_encoded_session_is_withheld(self) -> None:
        secret = "SESSION-TOKEN-9d3b"

        with pytest.raises(UrlParseError) as caught:
            parse_gerrit_topic_url(
                f"https://{REVIEW}/q/status:open%3Bjsessionid={secret}"
            )

        assert secret not in str(caught.value)


def test_the_detector_redacts_what_it_cannot_classify() -> None:
    """Its refusal echoed the target whole, query and all."""
    with pytest.raises(UrlParseError) as caught:
        detect_source("https://example.com/foo?token=SECRET-5e7a")

    assert "SECRET-5e7a" not in str(caught.value)
