# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Naming the conditions that are blocking a merge, read live.

A merge rejection names the rules GitHub evaluated at the instant the
merge was attempted, and it names them as *prose*.  That prose is not a
reliable account of what is wrong now: it lists conditions that had not
finished as readily as ones that failed, and it can omit the condition
that is actually holding the merge.

Reading the state directly avoids both problems.  Three sources are
needed, because GitHub reports through namespaces that do not overlap,
and they carry different weight as evidence:

- **commit status contexts**, where pre-commit.ci, DCO and other
  integrations report.  A failing one that the branch also *requires*,
  of any App, is a proven blocker.  One bound to an App is not: that
  App's check run decides it, which :mod:`_live_waiting` reads.
- **Actions workflow runs**, which carry the workflow's ``name:``.  That
  is the namespace a rejection quotes, so a failing run GitHub named as
  required is a proven blocker too.
- **check runs**, which carry the *job* name.  Nothing available here
  shows a job to be required, so these are reported as failing without
  being called the cause.

A view built from check runs alone cannot see ``pre-commit.ci - pr`` at
all, which is how a rejection came to list three workflows that had
passed while saying nothing about the status context that was failing.
A view built from check runs and statuses can see it but still cannot
name a required *workflow*, because ``.github/workflows/codeql.yml``
declares the workflow ``CodeQL`` and the job ``Audit Repository``, and
only the first appears in the rejection.

Enumerating rulesets to prove the remainder is deliberately not
attempted.  A ruleset names a workflow by *file path*, while a run
carries its ``name:``, so the two only join by fetching and parsing each
file --- and ``GET /orgs/{org}/rulesets`` returned 403 throughout the
503-PR run analysed in ``docs/BULK_RUN_PERFORMANCE_AUDIT.md``, which is
why :mod:`dependamerge.rule_violations` already treats the rejection
message as the more dependable source for those names.
"""

from __future__ import annotations

import asyncio

from ..check_runs import failing_check_names
from ..github_async._required_checks import required_check_apps
from ..models import PullRequestInfo
from ..rule_violations import required_workflow_names
from ._base import _MergeManagerBase
from ._types import MergeResult

# Prefixes the note that keeps GitHub's rejection once a live reading
# has replaced it as the reason.
REFUSAL_NOTE = "the merge was refused as: "


class _LiveBlockerMixin(_MergeManagerBase):
    """Deriving the blocking set from live check and status state."""

    def _with_rebase_note(self, pr_info: PullRequestInfo, reason: str) -> str:
        """*reason*, followed by the note of a rebase the cooldown held back."""
        note = self._rebase_suppressed.get(
            f"{pr_info.repository_full_name}#{pr_info.number}"
        )
        return f"{reason}; {note}" if note else reason

    @staticmethod
    def _original_rejection(result: MergeResult) -> str:
        """GitHub's rejection, even after a live reading replaced the reason.

        The end-of-run pass confirms a failure a second time, when
        ``error`` may already be the first pass's live reading.  That
        wording drops the evidence only the rejection carries, such as
        the workflows it quotes as required, so the note that kept the
        rejection is read instead.
        """
        warning = result.warning or ""
        if warning.startswith(REFUSAL_NOTE):
            return warning[len(REFUSAL_NOTE) :]
        return result.error or ""

    async def _live_blocking_conditions(
        self,
        pr_info: PullRequestInfo,
        *,
        head_sha: str,
        base_branch: str,
        rejection: str,
    ) -> tuple[list[str], list[str], bool]:
        """Split what is failing into proven blockers and the rest.

        Returns ``(blocking, also_failing, complete)``.  Entries in
        ``blocking`` arrive labelled with the kind of rule that proves
        them, and are safe to present as the reason the merge was
        refused.  Entries in ``also_failing`` are bare check names that
        are genuinely failing but that nothing available here shows to be
        required.

        ``complete`` says every probe answered.  It matters because a
        failed probe is silent, not loud: a 403 on the required-checks
        lookup --- which the 503-PR audit recorded happening --- leaves
        the failing status contexts unprovable, so a reading that saw
        only an optional check would otherwise compose a confident
        "failing checks: …" from what amounts to half the evidence, and
        the real rejection would be discarded.

        *head_sha* and *base_branch* are passed in rather than read off
        *pr_info*, whose snapshot predates the merge attempt.  A
        dependabot rebase --- which this tool requests --- moves the head,
        and reading checks for the commit the PR has left would report
        conditions belonging to a commit nobody is trying to merge.
        """
        if self._github_client is None:
            return [], [], False
        try:
            owner, repo = pr_info.repository_full_name.split("/", 1)
        except ValueError:
            return [], [], False

        required, required_ok = await self._required_context_apps(
            owner, repo, base_branch, pr_info
        )
        contexts, contexts_ok = await self._failing_context_names(
            owner, repo, head_sha, pr_info
        )
        checks, checks_ok = await self._failing_check_run_names(
            owner, repo, head_sha, pr_info
        )
        (
            workflows,
            workflows_ok,
        ) = await self._failing_workflow_names(owner, repo, head_sha, pr_info)

        # GitHub quoted these as required when it refused the merge, so
        # they are required whatever the check-runs API omits.
        #
        # Comparison is case-*sensitive* throughout, because GitHub's own
        # matching is: a branch requiring ``Build`` is not satisfied by a
        # check called ``build``, and folding case here would let an
        # optional check be proven required by a name it merely
        # resembles --- the false diagnosis this whole reading exists to
        # prevent.
        named = {name.strip() for name in required_workflow_names(rejection)}

        blocking: list[str] = []
        promoted: list[str] = []
        for name in contexts:
            # A context bound to an App is satisfied by that App's check
            # run, which a name alone cannot show: a status of the same
            # name may stand for it or not.  Those are left to the
            # App-aware reading of required checks (``_live_waiting``).
            key = name.strip()
            if key in required and required[key] is None:
                promoted.append(name)
                blocking.append(f"required status check: {name}")

        # Compared against *workflow run* names, the namespace a
        # rejection quotes.  A check run carries its job name instead, so
        # matching there compares two vocabularies that need not agree.
        proven = [name for name in workflows if name.strip() in named]
        promoted += proven
        blocking += [f"required workflow: {name}" for name in proven]

        # Everything failing that nothing here proves to be required.
        # Promoted names are excluded so nothing is reported as both the
        # blocker and merely failing, and a failing workflow with no
        # check run to surface it --- ``startup_failure`` never gets far
        # enough to report a job --- still gets a mention.
        accounted = {name.strip() for name in promoted}
        also_failing: list[str] = []
        for name in (*contexts, *checks, *workflows):
            if name.strip() in accounted or name in also_failing:
                continue
            also_failing.append(name)

        complete = required_ok and contexts_ok and checks_ok and workflows_ok
        return blocking, also_failing, complete

    async def _required_context_apps(
        self, owner: str, repo: str, base_branch: str, pr_info: PullRequestInfo
    ) -> tuple[dict[str, frozenset[int] | None], bool]:
        """Required contexts, each with the Apps bound to it or ``None``.

        Returns ``(contexts, answered)``.  A lookup that failed yields
        nothing, which is indistinguishable from "nothing is required"
        --- so the caller is told which it was.  A context two
        requirements declare is bound to every App either names, and to
        none when either accepts any App.

        Case is preserved: GitHub matches required contexts exactly, so
        folding it here would prove a context required by a name that
        only resembles it.
        """
        if self._github_client is None:
            return {}, False
        try:
            (
                required,
                reliable,
            ) = await self._github_client.get_required_status_checks_reliable(
                owner, repo, base_branch
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug(
                "Could not read required checks for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return {}, False
        if not isinstance(required, list) or not reliable:
            # An unreliable read returns the same empty list a branch
            # with no requirements does, so the lookup's own verdict is
            # the only thing that separates them.
            return {}, False
        contexts: dict[str, frozenset[int] | None] = {}
        for entry in required:
            if not isinstance(entry, dict) or not entry.get("context"):
                continue
            name = str(entry["context"]).strip()
            apps = required_check_apps(entry)
            if name in contexts:
                known = contexts[name]
                apps = None if known is None or apps is None else known | apps
            contexts[name] = apps
        return contexts, True

    async def _failing_context_names(
        self, owner: str, repo: str, head_sha: str, pr_info: PullRequestInfo
    ) -> tuple[list[str], bool]:
        """Commit status contexts whose latest state is failing."""
        if self._github_client is None:
            return [], False
        try:
            contexts = await self._github_client.get_failing_status_contexts(
                owner, repo, head_sha
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug(
                "Could not read status contexts for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return [], False
        if not isinstance(contexts, list):
            return [], False
        return [
            name for name in contexts if isinstance(name, str) and name.strip()
        ], True

    async def _failing_check_run_names(
        self, owner: str, repo: str, head_sha: str, pr_info: PullRequestInfo
    ) -> tuple[list[str], bool]:
        """Check names whose *latest* run did not succeed.

        Deduplicated through :func:`check_runs.failing_check_names`, so a
        cancelled run superseded by a successful one of the same name is
        not reported as a blocker.
        """
        if self._github_client is None:
            return [], False
        try:
            runs = await self._github_client.get_check_runs_for_ref(
                owner, repo, head_sha
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug(
                "Could not read check runs for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return [], False
        if not isinstance(runs, list):
            return [], False
        return failing_check_names(runs), True

    async def _failing_workflow_names(
        self, owner: str, repo: str, head_sha: str, pr_info: PullRequestInfo
    ) -> tuple[list[str], bool]:
        """Actions workflow runs on *head_sha* that did not pass.

        Read in the **workflow** namespace, which is the one a ruleset
        rejection quotes, so a name from the message can be matched
        against something comparable.
        """
        if self._github_client is None:
            return [], False
        try:
            names = await self._github_client.get_failing_workflow_run_names_for_sha(
                owner, repo, head_sha
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug(
                "Could not read workflow runs for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return [], False
        if not isinstance(names, list):
            return [], False
        return [name for name in names if isinstance(name, str) and name.strip()], True
