# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Reducing an error to the one line a failure report can show.

``httpx`` builds a two-line message for every ``HTTPStatusError``::

    Server error '500 Internal Server Error' for url '.../pulls/323/reviews'
    For more information check: https://developer.mozilla.org/.../500

The second line explains what HTTP 500 means, which tells an operator
nothing about the run that just failed.  The newline in front of it does
real damage: the failure report indents per *element* it is handed, so a
reason carrying an embedded newline has only its first line indented and
the remainder lands at column 0, outdented past the bullet it belongs
to.

The status line is kept whole.  It names the operation, the status and
the URL, and several classifiers elsewhere match on that wording, so
only the trailer --- which no reader and no classifier wants --- goes.

Applied at the points that *display* a reason rather than the points
that compose one.  A reason reaches the report from more routes than can
be enumerated --- ``_report_processing_error`` assigns ``str(e)`` for any
exception that escapes --- and sanitising each composer would leave
whichever route nobody thought of.  Logs keep the raw text deliberately:
a log is read to diagnose, and there the trailer is merely inert.

Pure and free of I/O, so the wording can be tested directly.
"""

from __future__ import annotations

import re

__all__ = ["summarise_error"]

#: ``httpx`` appends this to the message of every ``HTTPStatusError``.
#: Matched from the preceding newline so removing it cannot strand a
#: blank line, and the host is left open because httpx has changed where
#: it points --- ``httpstatuses.com`` before 0.21, MDN since --- and a
#: message from either is the same noise.  A URL is *required* though:
#: without that, a line reading "For more information check: the run
#: log" would lose its first word and keep the rest.
_HTTPX_TRAILER_RE = re.compile(r"\s*\n\s*For more information check:\s*https?://\S*")

#: A newline and whatever whitespace surrounds it.  Replaced by a single
#: space rather than removed, so the words either side do not run
#: together.  Only newlines: collapsing every whitespace run would
#: rewrite spacing the message chose for itself, to no benefit here.
_NEWLINE_RUN_RE = re.compile(r"\s*\n\s*")


def summarise_error(error: BaseException | str) -> str:
    """Render *error* as the single line a failure report can show.

    Takes the exception or its text, since callers have one or the
    other and ``str()`` on a string is a no-op.

    Anything left after the trailer goes is kept: a message may carry
    GitHub's own body, which is usually the most specific thing a run
    ever learns about a rejection, and this function is not the place to
    decide it is too long.  Newlines inside it become spaces, because
    the report has one line to give it.

    Idempotent, so a reason that has already passed through here --- or
    one composed from an exception that had --- is not altered again.
    """
    text = _HTTPX_TRAILER_RE.sub("", str(error))
    return _NEWLINE_RUN_RE.sub(" ", text).strip()
