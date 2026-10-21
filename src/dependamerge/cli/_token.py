# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The GitHub token every GitHub command needs, resolved one way.

A missing token is an operator's configuration error, so it is reported
as one, with what to do about it, before anything tries to use it.
"""

import os

from ..error_codes import exit_for_configuration_error

__all__ = ["_require_github_token"]


def _require_github_token(token: str | None) -> str:
    """Return ``token``, or ``GITHUB_TOKEN`` when it is not given.

    Exits with a configuration error when neither is set.  The clients
    raise a bare ``ValueError`` for the same condition, which suits a
    library caller but not someone at a command line.
    """
    resolved = token or os.getenv("GITHUB_TOKEN")
    if not resolved:
        exit_for_configuration_error(
            message="❌ GitHub token required",
            details="Provide --token or set the GITHUB_TOKEN environment variable",
        )
    return resolved
