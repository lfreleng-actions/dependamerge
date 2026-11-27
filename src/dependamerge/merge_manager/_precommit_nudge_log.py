# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

# A declaration only, as in ``_base.py``: ``_LifecycleMixin.__init__``
# establishes the attribute on the assembled ``AsyncMergeManager``.
# pyright: reportUninitializedInstanceVariable=false

"""
The newest ``pre-commit.ci run`` comment on each pull request, as known here.

The record holds the nudges this manager posted and the ones it saw
when it read the comments, whoever posted them.  The cooldown
(:mod:`_precommit_ci`) and the end-of-run reason (:mod:`_live_waiting`)
read the comments again, and that reading can fail or stop before the
nudge; the record still holds it.
"""

from __future__ import annotations

from datetime import datetime

from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _PrecommitNudgeLogMixin(_MergeManagerBase):
    """The newest ``pre-commit.ci run`` known on each pull request."""

    _precommit_nudged: dict[str, datetime]

    def _log_precommit_nudge(self, pr_info: PullRequestInfo, at: datetime) -> None:
        """Record a ``pre-commit.ci run`` on *pr_info* at *at*, posted or seen.

        An older time never replaces a newer one, so a comment read late
        cannot hide a nudge this manager posted after it.
        """
        key = f"{pr_info.repository_full_name}#{pr_info.number}"
        known = self._precommit_nudged.get(key)
        if known is None or at > known:
            self._precommit_nudged[key] = at

    def _logged_precommit_nudge(self, pr_info: PullRequestInfo) -> datetime | None:
        """The newest ``pre-commit.ci run`` on *pr_info* posted or seen here."""
        return self._precommit_nudged.get(
            f"{pr_info.repository_full_name}#{pr_info.number}"
        )
