# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Authorship gate for the source pull request.

An App's PR proceeds when the bot policy admits the App, and is refused
otherwise.  A person's PR needs ``--include-human-prs`` and a matching
``--override``, the rule every command applies (see ``_human_gate``).
"""

from ..error_codes import (
    ExitCode,
    exit_with_error,
)
from ._bot_scope import _out_of_scope_hint
from ._context import _MergeContext
from ._human_gate import (
    _confirm_human_override,
    _given_override,
    _refuse_human_source,
)
from ._sha import _generate_override_sha


def _require_source_bot_in_scope(ctx: _MergeContext) -> None:
    """Stop the run when the source PR's App is outside the bot policy.

    Being a bot is not enough: an App the policy does not admit needs a
    human to look at its PRs, so no override is offered.  The App can be
    brought into scope, which is what the message says.

    Raises:
        SystemExit: With ``VALIDATION_ERROR`` when the App is out of scope.
    """
    assert ctx.source_pr is not None
    author = ctx.source_pr.author
    decision = ctx.bot_policy.decide(author, ctx.source_pr.repository_full_name)
    if decision.in_scope:
        return
    exit_with_error(
        ExitCode.VALIDATION_ERROR,
        message=(
            f"❌ Source PR author {author} is not in this run's scope "
            f"({decision.reason})"
        ),
        details=_out_of_scope_hint(author),
    )


def _validate_automation_author(ctx: _MergeContext) -> None:
    """Gate the source PR on who wrote it.

    An App the bot policy admits passes straight through, without the
    API call an override costs.  A person's PR needs both
    ``--include-human-prs``, which also brings the author's similar PRs
    into scope, and the ``--override`` that confirms them.  Without the
    flag the run stops naming it; with the flag alone it stops printing
    the value to supply.

    Raises:
        SystemExit: ``VALIDATION_ERROR`` for an App outside the policy, a
            human source without the flag, or a wrong override;
            ``OVERRIDE_REQUIRED`` when only the override is missing.
    """
    assert ctx.github_client is not None
    assert ctx.source_pr is not None

    if ctx.github_client.is_automation_author(ctx.source_pr.author):
        _require_source_bot_in_scope(ctx)
        return

    if not ctx.include_human_prs:
        _refuse_human_source(ctx.source_pr.author)

    commit_messages = ctx.github_client.get_pull_request_commits(
        ctx.owner, ctx.repo_name, ctx.pr_number
    )
    first_commit_line = commit_messages[0].split("\n")[0] if commit_messages else ""
    _confirm_human_override(
        _given_override(ctx.override),
        _generate_override_sha(ctx.source_pr, first_commit_line),
        dry_run=ctx.dry_run,
        subject=(
            f"{ctx.source_pr.html_url} by {ctx.source_pr.author} "
            "and the PRs similar to it"
        ),
        derivation="the PR's author and first commit message",
    )
