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
  integrations report.  A failing one that the branch also *requires* is
  a proven blocker.
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
from ..models import PullRequestInfo
from ..rule_violations import workflow_name_fragments, workflow_name_spans
from ._base import _MergeManagerBase


class _LiveBlockerMixin(_MergeManagerBase):
    """Deriving the blocking set from live check and status state."""

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

        required, required_ok = await self._required_context_names(
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
            ambiguous,
            observed,
            workflows_ok,
        ) = await self._failing_workflow_names(owner, repo, head_sha, pr_info)

        # GitHub quoted these as required when it refused the merge, so
        # they are required whatever the check-runs API omits.  Matched
        # through the span rule rather than the display-oriented comma
        # split: a workflow genuinely called ``Build, Build`` splits into
        # two fragments, and testing those against the run's real name
        # would miss it.
        #
        # Comparison is case-*sensitive* throughout, because GitHub's own
        # matching is: a branch requiring ``Build`` is not satisfied by a
        # check called ``build``, and folding case here would let an
        # optional check be proven required by a name it merely
        # resembles --- the false diagnosis this whole reading exists to
        # prevent.
        named = {
            joined.strip()
            for _start, _width, joined in workflow_name_spans(rejection)
            if joined.strip()
        }

        blocking: list[str] = []
        promoted_contexts: list[str] = []
        for name in contexts:
            if name.strip() in required:
                promoted_contexts.append(name)
                blocking.append(f"required status check: {name}")

        # Compared against *workflow run* names, the namespace a
        # rejection quotes.  A check run carries its job name instead, so
        # matching there compares two vocabularies that need not agree.
        #
        # A name two workflow *files* share is not promoted, nor is a
        # span the observed runs do not corroborate.
        proven = [
            name
            for name in workflows
            if name.strip() in named
            and name not in ambiguous
            and self._span_is_trustworthy(rejection, name, observed)
        ]
        blocking += [f"required workflow: {name}" for name in proven]

        # Everything failing that nothing here proves to be required.
        # Contexts belong in this list too: a requirement pinned to one
        # GitHub App is dropped from ``required`` deliberately, and
        # dropping the failing context along with it would lose the only
        # mention of it.  Promoted names are excluded so nothing is
        # reported as both the blocker and merely failing.
        accounted = {name.strip() for name in proven}
        accounted |= {name.strip() for name in promoted_contexts}
        also_failing: list[str] = []
        for name in (*contexts, *checks, *workflows):
            if name.strip() in accounted or name in also_failing:
                continue
            also_failing.append(name)

        complete = required_ok and contexts_ok and checks_ok and workflows_ok
        return blocking, also_failing, complete

    @staticmethod
    def _span_is_trustworthy(rejection: str, name: str, observed: set[str]) -> bool:
        """Whether a span match proves *name* is one the rejection quoted.

        The comma-separated list a rejection quotes may be fewer names
        than pieces --- nothing in the message says whether ``'Build,
        Test'`` is one workflow or two --- so every contiguous span is
        offered as a candidate.  Treating each candidate as *proof* is
        the mistake: with the list naming one workflow called ``Build,
        Test`` while an unrelated optional workflow called ``Test``
        fails, the single-fragment span matches and the optional one is
        blamed.

        Three readings are trustworthy:

        - a list of one fragment, which carries no ambiguity;
        - the **whole** list read as a single name, which is what a
          workflow genuinely called ``Build, Build`` matches;
        - a *sub*-span, but only when every fragment names a workflow
          that actually ran --- which is what confirms the finest reading
          is the real one rather than a coincidence.

        Anything else stays unproven, and the workflow is reported as
        failing without being called the blocker.
        """
        fragments = workflow_name_fragments(rejection)
        if len(fragments) <= 1:
            return True
        if name.strip() in {sep.join(fragments) for sep in (", ", ",")}:
            return True
        return all(fragment.strip() in observed for fragment in fragments)

    async def _required_context_names(
        self, owner: str, repo: str, base_branch: str, pr_info: PullRequestInfo
    ) -> tuple[set[str], bool]:
        """Status contexts the PR's base branch requires, as GitHub spells them.

        Returns ``(names, answered)``.  A lookup that failed yields an
        empty set, which is indistinguishable from "nothing is required"
        --- so the caller is told which it was.

        Case is preserved: GitHub matches required contexts exactly, so
        folding it here would prove a context required by a name that
        only resembles it.  Requirements pinned to a specific GitHub App
        are omitted, for the reason given below.
        """
        if self._github_client is None:
            return set(), False
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
            return set(), False
        if not isinstance(required, list) or not reliable:
            # An unreliable read returns the same empty list a branch
            # with no requirements does, so the lookup's own verdict is
            # the only thing that separates them.
            return set(), False
        # A requirement pinned to one GitHub App is satisfied only by
        # that app's status.  The failing-status probe reports context
        # names without their source, so a same-named status from
        # somewhere else cannot be told apart --- and naming it as the
        # blocker would blame a check the branch never asked for.  Such
        # requirements are dropped rather than guessed at; the context
        # still surfaces under "also failing" if it is failing.
        return {
            str(entry.get("context", "")).strip()
            for entry in required
            if isinstance(entry, dict)
            and entry.get("context")
            and entry.get("integration_id") is None
        }, True

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
    ) -> tuple[list[str], set[str], set[str], bool]:
        """Actions workflow runs on *head_sha* that did not pass.

        Returns ``(failing, ambiguous, observed, answered)``.  Read in
        the **workflow** namespace, which is the one a ruleset rejection
        quotes, so a name from the message can be matched against
        something comparable.  ``ambiguous`` names are carried by more
        than one workflow file; ``observed`` is every workflow that ran,
        which is what lets the quoted list's reading be corroborated.
        """
        if self._github_client is None:
            return [], set(), set(), False
        try:
            (
                names,
                ambiguous,
                observed,
            ) = await self._github_client.get_failing_workflow_run_names_for_sha(
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
            return [], set(), set(), False
        if (
            not isinstance(names, list)
            or not isinstance(ambiguous, set)
            or not isinstance(observed, set)
        ):
            return [], set(), set(), False
        return (
            [name for name in names if isinstance(name, str) and name.strip()],
            ambiguous,
            observed,
            True,
        )
