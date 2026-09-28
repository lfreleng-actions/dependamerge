# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
What a path segment may name, and how to say so when it may not.

One decision and one wording per question, reached by every route a
name arrives on: a bare argument, an owner URL, a repository URL, and
both pull request parsers.  They did not agree before this module
existed --- ``not a url`` was refused as a bare owner and accepted
through a URL, then sent to the API as ``orgs/not%20a%20url`` and
answered with a 404, where the identical input given bare named the
problem at once.

Owners and repositories are answered together because the *structural*
rule is the same for both, and is the half that matters: a segment
carrying ``?`` or ``%`` changes what a URL means rather than failing,
whichever position it occupies.  Only the grammars differ --- and they
differ on both axes, by host and by what GitHub permits a login versus
a repository.  Splitting them would duplicate the structural rule and
the host-aware shape built around it.

Separated from :mod:`repos`, which answers "what does this URL
address".  This answers the narrower question the parsers ask of one
segment, and is pure, so both the rules and their guidance are
directly testable.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

from .host_config import DEFAULT_GITHUB_HOST
from .hosts import _host_matches
from .models import UrlParseError
from .shorthand import looks_like_owner

__all__ = [
    "require_owner",
    "require_owner_from_path",
    "require_repo",
    "require_repo_from_path",
]

#: Longest a GitHub login can be.  Enterprise directories are not bound
#: by the dotcom grammar, but they are bound by the column this maps to.
_MAX_OWNER_LENGTH = 39

#: Longest a GitHub repository name can be --- a different limit from a
#: login's, and a different column, so the two are not interchangeable.
_MAX_REPO_LENGTH = 100

#: What GitHub permits in a repository name: alphanumerics and the three
#: punctuation characters it allows freely.  Deliberately *not*
#: :func:`looks_like_owner`, which would be wrong in both directions ---
#: it refuses ``.`` and ``_``, ordinary here (``.github``, ``my_repo``,
#: ``docs.example.com``), and it caps at 39.
_REPO_RE = re.compile(r"\A[A-Za-z0-9._-]+\Z")

#: The two names with path-navigation meaning.  ``.`` and ``..`` satisfy
#: the grammar above but address the current and parent directory, so a
#: path built from one addresses somewhere else entirely.  GitHub refuses
#: them for the same reason.
#:
#: Exactly these two, not "any run of dots".  ``...`` has no navigation
#: meaning, and GitHub serves repositories by that name
#: (``ingydotnet/...``), so refusing it would make a real repository
#: unreachable on the strength of a resemblance.
_RESERVED_REPO_NAMES = frozenset({".", ".."})

#: The suffix GitHub uses to address a repository's wiki.
#: ``owner/foo.wiki`` does not name a repository called ``foo.wiki``: it
#: names the wiki *of* ``foo`` --- ``git ls-remote`` against
#: ``neovim/neovim.wiki.git`` answers with the neovim wiki, while the
#: REST API has no repository by that name.  So it is the same class of
#: fault as ``.`` and ``..``, a name that addresses a different resource,
#: rather than a naming policy to mirror.
#:
#: Matched case-insensitively.  Only lowercase addresses the wiki, but
#: GitHub matches repository names without regard to case, so no
#: repository can be called ``foo.WIKI`` either.
_WIKI_SUFFIX = ".wiki"

#: Characters that change what a URL *means* when a segment carrying one
#: is interpolated into a path unescaped.  ``/`` separates path
#: components, ``?`` begins a query, ``#`` a fragment, ``;`` begins a
#: path parameter, ``\`` is folded to a separator by some parsers, and
#: ``%`` reintroduces an escape --- the route a doubly-encoded
#: separator would take to arrive intact after one round of decoding.
#:
#: An owner or a repository is interpolated into REST paths throughout
#: the client, so a name carrying one of these does not merely fail:
#: ``acme?admin=true`` turns ``/orgs/{owner}/repos`` into ``/orgs/acme``
#: with a query, which is a different request rather than a failing one.
#: ``;`` does the same more quietly --- ``/orgs/team;admin/repos``
#: addresses ``/orgs/team`` with a path parameter --- and reaches here
#: only percent-encoded, since a literal one is refused by the parsers
#: before any name is extracted.  github.com cannot reach this, its
#: grammars being narrower; a directory that issues arbitrary names can.
_URL_STRUCTURAL = frozenset("/?#%;\\")


def _is_unsafe_character(character: str) -> bool:
    """Whether *character* changes what a path segment means.

    The single definition both the gate and the echo use.  They once
    kept separate sets, and the echo's fell behind: a decoded ``%2F``
    was refused by the gate yet printed in full by the error.
    """
    return (
        character in _URL_STRUCTURAL
        or character.isspace()
        or ord(character) < 0x20
        or ord(character) == 0x7F
    )


def _shown(name: str) -> str:
    """Return *name* as safe to put in an error message.

    A rejected segment used to be echoed whole, on the reasoning that a
    path segment cannot carry the userinfo or query a credential would
    hide in.  Decoding broke that: ``team%3Bjsessionid=SECRET`` arrives
    as ``team;jsessionid=SECRET``, and ``team%2FSECRET`` as
    ``team/SECRET``, and the refusal printed the value back.  An
    undecoded repository segment carries the same value in its encoded
    form, which is why ``%`` stops the echo too.

    Everything from the first unsafe character is withheld, but that
    character itself is kept, so the operator still sees the fault.
    Callers format the result with ``!r``, so a kept newline or tab is
    shown escaped rather than breaking the message.
    """
    for index, character in enumerate(name):
        if _is_unsafe_character(character):
            return name[: index + 1] + "\u2026"
    return name


def _is_safe_path_segment(name: str) -> bool:
    """Whether *name* can be placed in a URL path as it stands."""
    return not any(_is_unsafe_character(character) for character in name)


def _acceptable_owner(name: str, host: str) -> bool:
    """Whether *name* could be a login **on that host**.

    github.com gets the full grammar, which is what makes the bare and
    URL forms of a dotcom target answer identically --- the point of the
    gate.

    An Enterprise host gets only the checks no directory can fail.
    Accounts there are provisioned over LDAP or SAML and need not follow
    the dotcom grammar at all, as the comment beside
    ``shorthand._OWNER_RE`` records, so applying it would report ``team_name`` as invalid on an
    install where it is somebody's actual login.  What is refused is
    over-length input and anything that would not survive being placed
    in a URL path --- neither of which a directory produces, and both of
    which change the request rather than failing it.
    """
    if _host_matches(host, DEFAULT_GITHUB_HOST):
        return looks_like_owner(name)
    return bool(name) and len(name) <= _MAX_OWNER_LENGTH and _is_safe_path_segment(name)


def _owner_guidance(host: str) -> str:
    """What to tell an operator whose owner was refused, on *that* host.

    One message for both hosts would have to describe the stricter rule,
    and would then contradict the validator: an Enterprise name refused
    for whitespace or length would be told to remove underscores that
    its own directory issues.  Guidance that names the wrong fault sends
    the reader to edit something that was never the problem.
    """
    if _host_matches(host, DEFAULT_GITHUB_HOST):
        return "Logins are alphanumerics and hyphens, at most 39 characters."
    return (
        f"Logins on {host} follow that install's own directory, but cannot "
        "contain whitespace, or characters that would change what a URL "
        "means, and are at most 39 characters."
    )


def require_owner(name: str, host: str) -> str:
    """Return *name*, or refuse it as a login on *host*.

    The single decision and the single wording, so the bare and URL
    forms of the same argument cannot answer differently.  They did:
    ``lfreleng-actions`` and ``https://github.com/lfreleng-actions`` are
    the same request, but only the first was checked, so ``not a url``
    was refused as a bare owner and accepted through a URL --- reaching
    the API as ``orgs/not%20a%20url`` and coming back 404, where the
    operator saw a not-found error instead of the fail-fast message.

    Surrounding whitespace is refused rather than trimmed.
    :func:`looks_like_owner` strips before matching, which suits the
    shorthand expansion it was written for, but accepting the stripped
    form here and returning the original would send ``" acme"`` to the
    API --- the same bypass by a quieter route.  A bare argument is
    already stripped by its caller, so this only bites a URL path
    segment, where whitespace is never meant.

    The echoed segment goes through :func:`_shown`.  It is a *path*
    segment, so it cannot carry userinfo, but once decoded it can carry
    a path parameter --- and a session identifier travels in exactly
    that position.
    """
    if name != name.strip() or not _acceptable_owner(name, host):
        raise UrlParseError(
            f"Not a valid GitHub owner name: {_shown(name)!r}. {_owner_guidance(host)}"
        )
    return name


def require_owner_from_path(segment: str, host: str) -> str:
    """Decode a URL path segment, then gate it as a login.

    A path segment is percent-encoded, so ``not%20a%20url`` carries no
    literal whitespace and slipped through a check written for the
    decoded form --- the reported bypass, merely spelled differently.

    Decoding also keeps ``lfreleng%2Dactions`` working.  Nothing in this
    codebase decodes an owner, so the escape reaches GitHub verbatim and
    GitHub resolves it; gating the raw segment would refuse a URL that
    does address a real account.

    The *decoded* value is returned, which is the login itself.  The
    transport re-encodes what needs it, so the request is unchanged for
    an ordinary name and corrected for an encoded one.

    Bare CLI arguments deliberately do not come through here.  A shell
    can pass a literal ``%``, and decoding it would silently retarget
    the run.
    """
    return require_owner(unquote(segment), host)


def _acceptable_repo(name: str) -> bool:
    """Whether *name* could be a repository, on any host.

    No host split, unlike :func:`_acceptable_owner`, and the asymmetry
    is deliberate rather than an oversight.  An Enterprise *login* is
    lenient because accounts there are provisioned over LDAP or SAML and
    never pass through GitHub's own validation, so the dotcom grammar
    would refuse names a directory legitimately issues.  A repository is
    not provisioned that way: an Enterprise Server install creates one
    through the same API and the same rules as github.com, so there is
    no directory whose naming has to be deferred to.

    Applying the grammar everywhere is what closes the reported gap
    completely.  With a host split, ``topic:`` remained a legal
    repository name on a declared install --- and since
    :func:`~dependamerge.url_parser.repos.parse_repo_url` is the last
    parser the CLI tries, ``/q/topic:`` still landed there as the
    repository ``q/topic:``.

    ``.`` and ``..`` are named separately, as is a ``.wiki`` suffix.
    They satisfy the grammar and carry nothing structurally unsafe, yet
    a path built from any of them addresses a different resource, which
    is a fact about what the segment *is* rather than what it contains.
    """
    if not name or len(name) > _MAX_REPO_LENGTH:
        return False
    if name in _RESERVED_REPO_NAMES:
        return False
    if name.lower().endswith(_WIKI_SUFFIX):
        return False
    return bool(_REPO_RE.match(name))


def require_repo_from_path(segment: str, *, from_url: bool) -> str:
    """Gate a repository segment taken from a target's path.

    Decoded first when the target is a URL, as an owner is.  GitHub
    resolves an encoded repository segment ---
    ``github.com/pypa/get%2Dpip`` answers 200, as does
    ``repos/pypa/get%2Dpip`` --- so gating the raw segment refused URLs
    that address real repositories (#522).  The *decoded* value is
    gated, so ``a%2Fb`` and ``%2E%2E`` are still refused, and it is the
    decoded value that is returned.

    Taken literally when the target is bare shorthand.  Shorthand is
    typed rather than copied from a browser, so it is not percent-
    encoded: ``acme/my%5Frepo`` names a repository with a literal ``%``,
    which the gate then refuses, rather than being quietly read as
    ``my_repo``.

    Args:
        segment: The repository segment, as it appears in the path.
        from_url: Whether the target was a URL or remote rather than
            shorthand, per
            :func:`~dependamerge.url_parser.shorthand.is_shorthand`.
    """
    return require_repo(unquote(segment) if from_url else segment)


def require_repo(name: str) -> str:
    """Return *name*, or refuse it as a repository name.

    The owner beside it has been gated since #498; this segment was not,
    so ``https://github.com/acme/has:colon`` was accepted and sent to
    the API, where a percent-encoded colon comes back 404 --- the same
    shape of failure #477 reported for owners, one path segment along.

    It matters more than the wasted request suggests, because
    :func:`~dependamerge.url_parser.repos.parse_repo_url` is the last
    parser the CLI tries.  Anything the change, topic and owner parsers
    decline lands here, so a segment this gate would refuse was
    otherwise accepted as a repository: on a declared Enterprise host
    ``/q/topic:`` parsed as the repository ``q/topic:``.

    *name* is gated as given.  A segment taken from a target's path goes
    through :func:`require_repo_from_path`, which decides whether it is
    percent-encoded first.

    Surrounding whitespace is refused rather than trimmed, as for an
    owner --- accepting the stripped form and returning the original
    would send the untrimmed value to the API.
    """
    if name != name.strip() or not _acceptable_repo(name):
        base = name[: -len(_WIKI_SUFFIX)]
        if name.lower().endswith(_WIKI_SUFFIX) and _acceptable_repo(base):
            # Named separately because the fix is different: the operator
            # usually has the wiki's clone URL, and the repository they
            # want is the one it belongs to.  Only offered when that
            # base is itself valid, so it never suggests another refusal.
            raise UrlParseError(
                f"{name!r} names a wiki, not a repository. GitHub serves a "
                f"repository's wiki at <repo>.wiki; use {base!r} to address "
                "the repository itself."
            )
        raise UrlParseError(
            f"Not a valid GitHub repository name: {_shown(name)!r}. Repository "
            "names are alphanumerics, hyphens, underscores and dots, at "
            "most 100 characters, and cannot be '.' or '..' or end in "
            "'.wiki'."
        )
    return name
