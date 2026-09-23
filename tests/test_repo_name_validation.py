# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for gating the repository segment of a URL.

#477 found that an owner taken from a URL bypassed the login gate, and
#498 closed it.  The repository segment beside it was never gated, so
every value below was accepted on github.com itself and sent to the API,
where it comes back 404 -- the same shape of failure, one path segment
along.

The gap mattered more than a wasted request, because ``parse_repo_url``
is the **last** parser the CLI tries.  Anything the change, topic and
owner parsers decline lands there, so a malformed Gerrit search URL on a
declared Enterprise host was accepted as the repository ``q/topic:``.
"""

from __future__ import annotations

import pytest

from dependamerge.url_parser import parse_change_url, parse_org_url, parse_repo_url
from dependamerge.url_parser.models import UrlParseError
from dependamerge.url_parser.names import require_repo

GHE = "ghe.corp.example.com"


@pytest.fixture
def declared_ghe(monkeypatch):
    """Declare a GHE host for the duration of a test."""
    monkeypatch.delenv("DEPENDAMERGE_GITHUB_HOST", raising=False)
    monkeypatch.delenv("GH_HOST", raising=False)
    monkeypatch.setenv("DEPENDAMERGE_GITHUB_HOSTS", GHE)
    return GHE


class TestARepositoryNameIsGated:
    """The values #514 reported, each refused before any request."""

    @pytest.mark.parametrize(
        "name",
        [
            "has:colon",
            "has space",
            "has$dollar",
            "has/slash",
            "has?query",
            "has#fragment",
            "x" * 101,
            "",
        ],
    )
    def test_a_malformed_name_is_refused(self, name: str) -> None:
        with pytest.raises(UrlParseError, match="valid GitHub repository name"):
            require_repo(name)

    @pytest.mark.parametrize("name", [".", ".."])
    def test_a_dot_only_name_is_refused(self, name: str) -> None:
        """They satisfy the grammar but address a different resource.

        ``.`` and ``..`` carry nothing the character rule objects to, so
        they are named separately: the objection is to what the segment
        *is*, not to what it contains.
        """
        with pytest.raises(UrlParseError, match="valid GitHub repository name"):
            require_repo(name)

    @pytest.mark.parametrize(
        "name",
        [
            "widget",
            ".github",
            "my_repo",
            "docs.example.com",
            "a-b-c",
            # Three or more dots carry no navigation meaning, and GitHub
            # serves a repository by exactly this name (``ingydotnet/...``).
            # Only ``.`` and ``..`` are reserved.
            "...",
            "....",
            # GitHub permits a leading hyphen in a *repository* name,
            # unlike a login.  Pinned so the two grammars cannot be
            # conflated back together by a later tightening.
            "-leading",
            "a.b_c-d",
            "x" * 100,
        ],
    )
    def test_an_ordinary_name_is_accepted(self, name: str) -> None:
        """The control, and the reason the login grammar is not reused.

        ``looks_like_owner`` refuses ``.`` and ``_`` and caps at 39, so
        borrowing it here would reject repositories this tool is pointed
        at daily.
        """
        assert require_repo(name) == name

    def test_surrounding_whitespace_is_refused_not_trimmed(self) -> None:
        """Trimming would send the untrimmed value to the API."""
        with pytest.raises(UrlParseError):
            require_repo(" widget")


class TestAWikiIsNotARepository:
    """``owner/foo.wiki`` names the wiki *of* ``foo``, not a repository.

    ``git ls-remote https://github.com/neovim/neovim.wiki.git`` answers
    with the neovim wiki, while the REST API has no repository by that
    name, so a wiki clone URL used to parse as a repository that every
    later request 404s on.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/neovim/neovim.wiki",
            "https://github.com/neovim/neovim.wiki.git",
            "git@github.com:neovim/neovim.wiki.git",
            "neovim/neovim.wiki",
        ],
    )
    def test_every_form_of_a_wiki_url_is_refused(self, url: str) -> None:
        with pytest.raises(UrlParseError, match="names a wiki"):
            parse_repo_url(url)

    def test_a_wiki_pull_request_url_is_refused(self) -> None:
        with pytest.raises(UrlParseError, match="names a wiki"):
            parse_change_url("https://github.com/neovim/neovim.wiki/pull/1")

    def test_the_client_parser_refuses_one_too(self) -> None:
        from dependamerge.github_client import GitHubClient

        client = GitHubClient(token="test_token")

        with pytest.raises(UrlParseError, match="names a wiki"):
            client.parse_pr_url("https://github.com/neovim/neovim.wiki/pull/1")

    def test_the_suffix_is_matched_without_regard_to_case(self) -> None:
        """GitHub matches names case-insensitively, so none can end so."""
        with pytest.raises(UrlParseError, match="names a wiki"):
            require_repo("neovim.WIKI")

    def test_the_message_names_the_repository_it_belongs_to(self) -> None:
        with pytest.raises(UrlParseError, match="use 'neovim' to address"):
            require_repo("neovim.wiki")

    @pytest.mark.parametrize("name", [".wiki", "has:colon.wiki"])
    def test_no_invalid_base_is_suggested(self, name: str) -> None:
        """Pointing the operator at ``''`` or ``has:colon`` would mislead."""
        with pytest.raises(UrlParseError, match="valid GitHub repository name"):
            require_repo(name)

    @pytest.mark.parametrize(
        "name", ["wiki", "my-wiki", "wikipedia", "foo.wikis", "foo.wiki-docs"]
    )
    def test_a_name_merely_containing_wiki_is_accepted(self, name: str) -> None:
        assert require_repo(name) == name


class TestEveryUrlFormReachesTheGate:
    """Both parsers that take a repository segment, and the decode."""

    def test_a_repository_url_is_gated(self) -> None:
        with pytest.raises(UrlParseError, match="valid GitHub repository name"):
            parse_repo_url("https://github.com/acme/has:colon")

    def test_a_pull_request_url_is_gated(self) -> None:
        """The most common form an operator types."""
        with pytest.raises(UrlParseError, match="valid GitHub repository name"):
            parse_change_url("https://github.com/acme/has:colon/pull/7")

    def test_an_encoded_separator_does_not_survive_decoding(self) -> None:
        """``a%2Fb`` names no repository, encoded or not."""
        with pytest.raises(UrlParseError, match="valid GitHub repository name"):
            parse_repo_url("https://github.com/acme/a%2Fb")

    def test_a_percent_is_never_decoded(self) -> None:
        """Unlike an owner, a repository segment is taken literally.

        Decoding an owner earns its keep --- ``lfreleng%2Dactions``
        addresses a real dotcom account.  A repository name's whole
        character set survives a URL untouched, so no legitimate URL
        encodes one, and decoding would turn a literal ``%`` the shell
        passed into a *different*, valid repository.
        """
        with pytest.raises(UrlParseError, match="valid GitHub repository name"):
            parse_repo_url("acme/my%5Frepo")

    @pytest.mark.parametrize("suffix", [";other", ";"])
    @pytest.mark.parametrize(
        ("parser", "target"),
        [
            (parse_repo_url, "https://github.com/acme/widget{}"),
            (parse_org_url, "https://github.com/acme{}"),
            (parse_change_url, "https://github.com/acme/widget/pull/7{}"),
        ],
    )
    def test_a_path_parameter_is_refused(
        self, parser, target: str, suffix: str
    ) -> None:
        """``urlparse`` splits ``;suffix`` off into ``params``.

        So no parser ever saw it: ``acme/widget;other`` arrived as the
        repository ``widget`` and ``/pull/7;other`` as pull request 7
        --- a different target, addressed confidently, rather than a
        refusal.

        Asserted **per parser** rather than by trying each in turn: a
        loop that stops at the first refusal would let any one of these
        guards be removed without failing.

        A bare ``;`` is covered too.  ``params`` is empty for one, yet
        ``urlparse`` still strips it from the path, so a check that
        only asked whether a parameter had content let ``widget;``
        through.
        """
        with pytest.raises(UrlParseError, match="path parameter"):
            parser(target.format(suffix))

    def test_the_bare_form_is_refused_too(self) -> None:
        """Shorthand reaches the same parser and must answer alike."""
        with pytest.raises(UrlParseError, match="path parameter"):
            parse_repo_url("acme/widget;other")

    def test_the_client_parser_refuses_one_too(self) -> None:
        """Reached by ``close`` without passing through ``url_parser``."""
        from dependamerge.github_client import GitHubClient

        client = GitHubClient(token="test_token")

        with pytest.raises(UrlParseError, match="path parameter"):
            client.parse_pr_url("https://github.com/acme/widget/pull/7;other")

    def test_the_client_puts_the_host_first_too(self) -> None:
        """The client parser must not answer differently.

        It reached the path-parameter check before its own
        undeclared-host gate, so the same target got a path error from
        the client and declaration guidance from ``parse_repo_url`` ---
        two entry points disagreeing about one URL.
        """
        from dependamerge.github_client import GitHubClient

        client = GitHubClient(token="test_token")

        with pytest.raises(UrlParseError) as caught:
            client.parse_pr_url("https://undeclared.example/acme/widget/pull/7;other")

        message = str(caught.value)
        assert "undeclared.example" in message
        assert "path parameter" not in message

    def test_the_host_error_still_wins(self) -> None:
        """Host policy is checked before the path, as it is for owners.

        ``TestTheHostIsStillCheckedFirst`` pins that ordering for the
        owner gate: a mistyped host must never be reported as a bad
        name.  Checking path parameters first would have obscured it.
        """
        with pytest.raises(UrlParseError) as caught:
            parse_repo_url("https://undeclared.example/acme/widget;other")

        message = str(caught.value)
        assert "undeclared.example" in message
        assert "path parameter" not in message

    def test_the_ordinary_forms_are_unaffected(self) -> None:
        assert parse_repo_url("https://github.com/acme/.github").repo == ".github"
        assert (
            parse_change_url("https://github.com/acme/.github/pull/7").project
            == "acme/.github"
        )

    @pytest.mark.parametrize("trailing", ["", "/"])
    def test_a_trailing_slash_does_not_hide_one(self, trailing: str) -> None:
        """The final segment is judged as the parsers see it, slash-free."""
        with pytest.raises(UrlParseError, match="path parameter"):
            parse_change_url(f"https://github.com/acme/widget/pull/7;other{trailing}")

    def test_a_semicolon_inside_a_gerrit_project_is_kept(self) -> None:
        """Gerrit permits ``;`` in a project name; only the final segment
        is ever split by ``urlparse``, so only it is checked.
        """
        parsed = parse_change_url("https://gerrit.example.org/c/team;tools/+/123")

        assert (parsed.project, parsed.change_number) == ("team;tools", 123)

    def test_a_gerrit_change_number_still_refuses_one(self) -> None:
        with pytest.raises(UrlParseError, match="path parameter"):
            parse_change_url("https://gerrit.example.org/c/team/+/123;other")


class TestAnEncodedSemicolonCannotReachAnOwner:
    """``;`` begins a path parameter, so it changes what a URL means.

    A literal one is refused by the parsers before any name is
    extracted, but percent-encoded it survives to
    ``require_owner_from_path``, which decodes it.  The Enterprise owner
    rules police structure rather than a character set, so ``team;admin``
    was accepted --- and interpolated unescaped, ``/orgs/team;admin/repos``
    addresses ``/orgs/team`` with a parameter.  The same class of fault
    as ``?`` and ``#``, arriving by a quieter route.
    """

    def test_an_encoded_semicolon_is_refused(self, declared_ghe) -> None:
        with pytest.raises(UrlParseError, match="valid GitHub owner name"):
            parse_repo_url(f"https://{GHE}/team%3Badmin/widget")

    @pytest.mark.parametrize("owner", ["team_name", "acme%40corp", "has%24dollar"])
    def test_enterprise_leniency_is_untouched(self, declared_ghe, owner: str) -> None:
        """The control: only *structure* is policed, not the character set.

        An LDAP-backed directory may issue any of these, and none of
        them changes what a URL means.
        """
        assert parse_repo_url(f"https://{GHE}/{owner}/widget").owner


class TestTheGrammarIsNotSplitByHost:
    """Unlike the owner gate, and deliberately so.

    An Enterprise *login* is lenient because accounts are provisioned
    over LDAP or SAML and never pass through GitHub's validation.  A
    repository is not: an Enterprise Server install creates one through
    the same API and the same rules as github.com, so there is no
    directory whose naming has to be deferred to.

    Applying the grammar everywhere is what closes the gap completely --
    a host split left ``topic:`` a legal repository name on a declared
    install, which is exactly where the fall-through was observed.
    """

    def test_an_enterprise_host_gets_the_same_rule(self, declared_ghe) -> None:
        with pytest.raises(UrlParseError, match="valid GitHub repository name"):
            parse_repo_url(f"https://{GHE}/acme/has:colon")

    def test_an_enterprise_repository_still_parses(self, declared_ghe) -> None:
        """The control: ordinary names keep working on a declared host."""
        assert parse_repo_url(f"https://{GHE}/acme/widget").repo == "widget"

    def test_the_owner_keeps_its_enterprise_leniency(self, declared_ghe) -> None:
        """The asymmetry, asserted rather than left to the docstring.

        ``team_name`` is a login an LDAP directory may well issue, and
        #498 accepts it on a declared host.  Gating repositories must
        not have tightened that.
        """
        assert parse_repo_url(f"https://{GHE}/team_name/widget").owner == "team_name"


class TestTheParserCascadeHasNoLandingPlace:
    """The consequence that made this worth fixing rather than noting.

    ``_parse_merge_target`` tries change, then topic, then owner, then
    repository.  A malformed Gerrit search URL was declined by the first
    three and accepted by the fourth, so the run would have addressed
    ``q/topic:`` on the operator's Enterprise host.
    """

    @pytest.mark.parametrize("path", ["q/topic:", "q/owner:self"])
    def test_a_malformed_search_url_is_no_longer_a_repository(
        self, declared_ghe, path: str
    ) -> None:
        with pytest.raises(UrlParseError):
            parse_repo_url(f"https://{GHE}/{path}")


class TestTheClientParserIsGatedToo:
    """``GitHubClient.parse_pr_url`` is a second, independent parser.

    ``close`` and the merge client setup call it directly, without
    passing through ``url_parser`` at all, so a gate added only there
    leaves this route open.  Its owner has been gated since #498; the
    repository beside it had not.
    """

    def test_a_malformed_repository_is_refused(self) -> None:
        from dependamerge.github_client import GitHubClient

        client = GitHubClient(token="test_token")

        with pytest.raises(UrlParseError, match="valid GitHub repository name"):
            client.parse_pr_url("https://github.com/acme/has:colon/pull/7")

    def test_an_ordinary_repository_still_parses(self) -> None:
        from dependamerge.github_client import GitHubClient

        client = GitHubClient(token="test_token")

        owner, repo, number = client.parse_pr_url(
            "https://github.com/acme/.github/pull/7"
        )

        assert (owner, repo, number) == ("acme", ".github", 7)


class TestTheFormerImportPathStillWorks:
    """``owner.py`` was public surface before the rules moved.

    ``require_owner`` carries no underscore, so
    ``dependamerge.url_parser.owner`` was an importable path for it.
    Removing the module would be a breaking change wearing a refactor's
    clothes, which is why it survives as a shim.
    """

    def test_the_owner_gate_is_still_importable_from_owner(self) -> None:
        from dependamerge.url_parser import names
        from dependamerge.url_parser.owner import require_owner, require_owner_from_path

        assert require_owner is names.require_owner
        assert require_owner_from_path is names.require_owner_from_path


class TestThePullRequestErrorReachesTheOperator:
    """The CLI cascade must not answer for a PR-shaped github.com URL.

    ``parse_change_url`` refuses ``/acme/has:colon/pull/7`` with the
    specific fault, then ``_parse_merge_target`` falls through to
    ``parse_repo_url``, whose fallback for this shape advises passing
    the full PR URL --- which the operator has just done.  Guidance that
    contradicts the input reads as the tool not having looked at it.
    """

    @staticmethod
    def _reported(url: str) -> str:
        import contextlib
        import io

        import typer

        from dependamerge.cli._merge_target import _parse_merge_target

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with pytest.raises(typer.Exit):
                _parse_merge_target(url)
        return " ".join(buffer.getvalue().split())

    def test_a_bad_repository_name_is_named(self) -> None:
        message = self._reported("https://github.com/acme/has:colon/pull/7")

        assert "valid GitHub repository name" in message
        assert "Pass the full PR URL" not in message

    def test_a_non_numeric_number_is_named(self) -> None:
        """The other way a PR-shaped path fails on github.com."""
        message = self._reported("https://github.com/acme/widget/pull/abc")

        assert "Invalid GitHub PR URL format" in message
        assert "Pass the full PR URL" not in message

    def test_a_repository_url_still_gets_repository_guidance(self) -> None:
        """The control: only PR-shaped paths take the change error."""
        message = self._reported("https://github.com/acme/widget/extra/bits")

        assert "repository url" in message.lower()


class TestARefusedParameterIsNotEchoed:
    """The error that refuses a ``;suffix`` must not print it back.

    ``;jsessionid=…`` is a long-standing way to carry a session in a
    URL, and ``redact_target`` removed userinfo, queries and fragments
    but not path parameters --- so the one message written to refuse a
    parameter was the one that exposed it.
    """

    SECRET = "SESSION-TOKEN-4f2a"

    @pytest.mark.parametrize(
        ("parser", "target"),
        [
            (parse_repo_url, "https://github.com/acme/widget;jsessionid={}"),
            (parse_repo_url, "acme/widget;jsessionid={}"),
            (parse_change_url, "https://github.com/acme/widget/pull/7;jsessionid={}"),
        ],
    )
    def test_the_parameter_value_never_reaches_the_message(
        self, parser, target: str
    ) -> None:
        """Asserted per parser, and the refusal itself is required.

        Catching the error and checking it inside the handler would pass
        silently if a parser stopped refusing the target altogether ---
        the shape this review has already caught once.
        """
        with pytest.raises(UrlParseError) as caught:
            parser(target.format(self.SECRET))

        assert self.SECRET not in str(caught.value)

    def test_the_client_parser_does_not_echo_it_either(self) -> None:
        from dependamerge.github_client import GitHubClient

        client = GitHubClient(token="test_token")

        with pytest.raises(UrlParseError) as caught:
            client.parse_pr_url(
                f"https://github.com/acme/widget/pull/7;jsessionid={self.SECRET}"
            )

        assert self.SECRET not in str(caught.value)

    def test_the_message_still_names_the_fault(self) -> None:
        """The control: redacting must not blank out the diagnosis."""
        with pytest.raises(UrlParseError) as caught:
            parse_repo_url(f"https://github.com/acme/widget;jsessionid={self.SECRET}")

        message = str(caught.value)
        assert "path parameter" in message
        assert "github.com/acme/widget" in message

    def test_redaction_is_central_rather_than_per_message(self) -> None:
        """Every caller gets it, not only the message that prompted it."""
        from dependamerge.url_parser import redact_target

        assert redact_target(f"https://h/a/b;jsessionid={self.SECRET}") == (
            "https://h/a/b;\u2026"
        )
        # Userinfo is replaced first, so a ';' inside it cannot truncate
        # the host out of the message.
        assert redact_target("https://u;x:pw@h/a/b") == "https://***@h/a/b"

    def test_an_encoded_delimiter_does_not_echo_it(self) -> None:
        """``%3B`` misses the literal guard, fails the PR shape, and the
        client parser then reports the whole target it refused.
        """
        from dependamerge.github_client import GitHubClient

        client = GitHubClient(token="test_token")

        with pytest.raises(UrlParseError) as caught:
            client.parse_pr_url(
                f"https://github.com/acme/widget/pull/7%3Bjsessionid={self.SECRET}"
            )

        assert self.SECRET not in str(caught.value)

    def test_the_cut_at_a_percent_is_marked(self) -> None:
        """An unmarked prefix would read as a different, complete target."""
        from dependamerge.url_parser import redact_target

        assert redact_target(f"https://h/a/7%3Bjsessionid={self.SECRET}") == (
            "https://h/a/7%…"
        )
        # Userinfo is replaced first, so an encoded character inside it
        # cannot truncate the host out of the message.
        assert redact_target("https://u%40x:pw@h/a/b") == "https://***@h/a/b"

    def test_a_mid_path_session_is_still_withheld(self) -> None:
        """A servlet container honours ``;jsessionid=`` on any segment."""
        from dependamerge.url_parser import redact_target

        assert self.SECRET not in redact_target(
            f"https://h/a;jsessionid={self.SECRET}/b"
        )

    def test_a_gerrit_project_cut_is_marked(self) -> None:
        """``/c/team`` alone would read as a different, complete target."""
        from dependamerge.url_parser import redact_target

        assert (
            redact_target("https://h/c/team;tools/+/abc") == "https://h/c/team;\u2026"
        )


class TestARefusedSegmentIsNotEchoedWhole:
    """The name gates echo the segment they refused.

    That used to be safe by the reasoning that a path segment cannot
    carry userinfo or a query.  Decoding broke it: ``team%3Bjsessionid=``
    arrives as ``team;jsessionid=``, and the refusal printed the session
    back --- a second leak path that ``redact_target`` never sees,
    because these messages interpolate the name directly.
    """

    SECRET = "SESSION-TOKEN-7c1e"

    def test_a_decoded_owner_does_not_leak(self, declared_ghe) -> None:
        with pytest.raises(UrlParseError) as caught:
            parse_repo_url(f"https://{GHE}/team%3Bjsessionid={self.SECRET}/widget")

        message = str(caught.value)
        assert self.SECRET not in message
        # The fault is still visible: the delimiter is kept.
        assert "team;" in message

    def test_an_encoded_repository_does_not_leak(self) -> None:
        """A repository segment is not decoded, so the encoded form matters."""
        with pytest.raises(UrlParseError) as caught:
            parse_repo_url(f"https://github.com/acme/wid%3Bjsessionid={self.SECRET}")

        assert self.SECRET not in str(caught.value)

    @pytest.mark.parametrize("encoded", ["%2F", "%5C", "%20", "%09", "%0A", "%7F"])
    def test_every_character_the_gate_refuses_stops_the_echo(
        self, declared_ghe, encoded: str
    ) -> None:
        """The gate and the echo share one definition of unsafe.

        They once kept separate sets, and the echo's covered only
        ``; ? # %``: ``team%2FSECRET`` decoded to ``team/SECRET``, the
        gate refused it, and the error printed it whole.
        """
        with pytest.raises(UrlParseError) as caught:
            parse_repo_url(f"https://{declared_ghe}/team{encoded}{self.SECRET}/widget")

        assert self.SECRET not in str(caught.value)

    def test_the_same_holds_on_github_com(self) -> None:
        with pytest.raises(UrlParseError) as caught:
            parse_repo_url(f"https://github.com/team%2F{self.SECRET}/widget")

        assert self.SECRET not in str(caught.value)

    def test_an_ordinary_bad_name_is_still_echoed_whole(self) -> None:
        """The control: withholding applies only past a delimiter.

        ``has:colon`` carries nothing sensitive, and echoing it is what
        tells the operator which name was refused.
        """
        with pytest.raises(UrlParseError, match="'has:colon'"):
            parse_repo_url("https://github.com/acme/has:colon")


class TestTheTopicParserGuardsItself:
    """``parse_gerrit_topic_url`` is public, so it cannot lean on the cascade.

    ``urlparse`` splits ``;other`` off into ``params``, so
    ``/q/topic:release;other`` read as the topic ``release`` --- a
    different search, returned without complaint.  The merge cascade
    trying ``parse_change_url`` first does nothing for a direct caller.
    """

    @pytest.mark.parametrize("suffix", [";other", ";"])
    def test_a_path_parameter_is_refused(self, suffix: str) -> None:
        from dependamerge.url_parser import parse_gerrit_topic_url

        with pytest.raises(UrlParseError, match="path parameter"):
            parse_gerrit_topic_url(
                f"https://gerrit.example.org/q/topic:release{suffix}"
            )

    def test_an_ordinary_topic_still_parses(self) -> None:
        from dependamerge.url_parser import parse_gerrit_topic_url

        parsed = parse_gerrit_topic_url("https://gerrit.example.org/q/topic:release")

        assert parsed.topic == "release"
