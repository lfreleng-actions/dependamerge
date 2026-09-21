# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Making a target safe to show before it reaches a message.

Every parser reports the target it refused, which is what makes the
message actionable --- and a target is operator input that may carry a
credential.  A credential has now escaped four separate times on this branch: a
git remote logged with its password, a configuration value echoed verbatim,
a network-path reference that skipped stripping, and an API authority
read from ``netloc``.  Each was fixed where it was found, and a fifth
set appeared in the errors added to fix the fourth.

Fixing them one at a time is not working, so the sanitising lives here
and every site that interpolates a target calls it.  A new message is
then safe by default rather than safe if its author remembered.
"""

from __future__ import annotations

import re

__all__ = ["redact_target"]

#: The userinfo a URL may carry, in both the ``scheme://`` and the
#: scheme-less ``//host/path`` forms.  Both name an authority, so both
#: can hide a credential in front of it; requiring an authority marker
#: is what leaves a bare ``a@b`` *path* segment alone.
_USERINFO_RE = re.compile(r"\A((?:[A-Za-z][A-Za-z0-9+.-]*:)?//)[^/@\s]+@")

#: Where a redacted target stops mid-segment: a path parameter, or an
#: encoded character that may be hiding one.
_MARKED_CUT_RE = re.compile(r"[;%]")


def redact_target(value: str) -> str:
    """Strip anything credential-bearing from a target for display.

    Removes URL userinfo, path parameters, the query string and the
    fragment --- the places a token is conventionally written --- while
    keeping the scheme, host and path, which are what make a message
    useful.

    The query and fragment go even though normalisation deliberately
    *preserves* them: they are kept so a Gerrit search URL parses, and
    that is a parsing concern.  Nothing downstream of an error needs
    them, and a token in ``?token=`` is the likeliest way one reaches
    a terminal.

    Path parameters go for the same reason.  ``;jsessionid=…`` is a
    long-standing way to carry a session in a URL, and the error that
    refuses a ``;suffix`` is precisely the one that would otherwise
    print it back.  Nothing is lost by it: every parser now refuses a
    target carrying one, so the suffix is never part of a name the
    operator needs to see.

    The target also stops at the first ``%``, the boundary the name
    gates already use when echoing a segment.  ``7%3Bjsessionid=…`` is
    the same session with its delimiter encoded, and a parser that
    fails to match the shape reports the target it refused, so an
    encoded delimiter would otherwise print the value back.

    A ``;`` or ``%`` is cut wherever it falls, not only in the final
    segment: a servlet container honours ``;jsessionid=`` on any
    segment, so narrowing the cut would reopen the leak.  Either can
    fall mid-segment --- Gerrit permits ``;`` in a project name --- so
    the character is kept and marked with an ellipsis.  An unmarked
    prefix would read as a different, complete target: ``/c/team;tools``
    shown as ``/c/team``.  ``?`` and ``#`` always begin a whole
    component, so their cut needs no mark.

    Args:
        value: The target as the operator supplied it, or as
            normalisation rewrote it.

    Returns:
        The target with credentials removed.
    """
    if not value:
        return value
    redacted = _USERINFO_RE.sub(r"\1***@", value)
    redacted = redacted.split("?", 1)[0].split("#", 1)[0]
    cut = _MARKED_CUT_RE.search(redacted)
    return redacted if cut is None else f"{redacted[: cut.end()]}…"
