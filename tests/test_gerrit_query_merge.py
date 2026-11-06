# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
A Gerrit query URL acts on exactly the open changes its query matches.

Unlike a change URL, no change anchors the batch and no similarity
scoring narrows it; blocked changes stay listed but are skipped.
"""

import re
from unittest.mock import Mock, patch

import pytest
import typer
from rich.console import Console

from dependamerge.cli import _handle_gerrit_merge
from dependamerge.error_codes import ExitCode
from dependamerge.gerrit import (
    GerritChangeInfo,
    GerritLabelInfo,
    GerritServiceError,
    GerritSubmitResult,
)
from dependamerge.url_parser import ParsedGerritTopicUrl, parse_gerrit_topic_url

pytestmark = pytest.mark.xfail(strict=True, reason="query URL exact scope: issue #613")

HOST = "gerrit.example.org"
FULL_ACCESS = {"Code-Review": ["-2", "-1", "0", "+1", "+2"]}


def _query_url(query: str = "topic:reusable-clm-workflow") -> ParsedGerritTopicUrl:
    return parse_gerrit_topic_url(f"https://{HOST}/r/q/{query.replace(' ', '+')}")


def _change(
    number: int,
    project: str,
    subject: str,
    owner: str = "lf-automation",
    revision: str | None = None,
    **extra,
) -> GerritChangeInfo:
    return GerritChangeInfo(
        number=number,
        change_id=f"I{number:040d}",
        project=project,
        subject=subject,
        owner=owner,
        branch="master",
        status="NEW",
        current_revision=revision or f"rev{number}",
        topic="reusable-clm-workflow",
        permitted_labels=FULL_ACCESS,
        **extra,
    )


def _clm_topic() -> list[GerritChangeInfo]:
    """Two dissimilar subject groups sharing one topic, as on ONAP."""
    return [
        _change(149213, "ccsdk/features", "Add Nexus IQ CLM scan"),
        _change(149214, "sdc/sdc-distribution-client", "Add Nexus IQ CLM scan"),
        _change(
            149215, "aai/rest-client", "Move CLM lane to security-workflows reusable"
        ),
        _change(
            149216, "so/so-etsi-nfvo", "Move CLM lane to security-workflows reusable"
        ),
    ]


def _submit_all(changes, *args, **kwargs):
    return [
        GerritSubmitResult.success_result(change_number=c.number, project=c.project)
        for c, _ in changes
    ]


@pytest.fixture
def gerrit():
    """Patch credentials, service, comparator and submit manager."""
    credentials = Mock(is_valid=True, username="user", password="pass")
    credentials.auth_method_display.return_value = ".netrc file"
    service = Mock(is_authenticated=True)
    service.submitted_together_problems.return_value = []
    comparator = Mock()
    comparator.is_automation_change.return_value = True
    submit_manager = Mock()
    submit_manager.submit_changes_parallel.side_effect = _submit_all
    with (
        patch("dependamerge.cli.resolve_gerrit_credentials", return_value=credentials),
        patch("dependamerge.cli.create_gerrit_service", return_value=service),
        patch("dependamerge.cli.create_gerrit_comparator", return_value=comparator),
        patch(
            "dependamerge.cli.create_submit_manager", return_value=submit_manager
        ) as create_submit_manager,
    ):
        yield Mock(
            service=service,
            comparator=comparator,
            submit_manager=submit_manager,
            create_submit_manager=create_submit_manager,
        )


def _run(
    parsed_url: ParsedGerritTopicUrl | None = None,
    *,
    similarity_threshold: float = 0.8,
    dry_run: bool = False,
    topic: str | None = None,
    include_human_prs: bool = False,
    override: str | None = None,
    console: Console | None = None,
) -> Console:
    console = console or Console(record=True, width=200)
    _handle_gerrit_merge(
        parsed_url=parsed_url or _query_url(),
        no_confirm=True,
        similarity_threshold=similarity_threshold,
        verbose=False,
        console=console,
        dry_run=dry_run,
        topic=topic,
        include_human_prs=include_human_prs,
        override=override,
        show_progress=False,
    )
    return console


def _submitted_numbers(submit_manager) -> list[int]:
    (changes,), _ = submit_manager.submit_changes_parallel.call_args
    return [change.number for change, _ in changes]


class TestQueryUrlScope:
    """The batch is the query's result set, and nothing else."""

    def test_submits_every_dissimilar_change_without_scoring(self, gerrit):
        """Both subject groups of the topic are submitted."""
        gerrit.service.get_open_changes_for_query.return_value = _clm_topic()

        console = _run()

        assert _submitted_numbers(gerrit.submit_manager) == [
            149213,
            149214,
            149215,
            149216,
        ]
        gerrit.service.find_similar_changes.assert_not_called()
        gerrit.service.get_changes_by_topic.assert_not_called()
        gerrit.service.get_change_info.assert_not_called()
        gerrit.service.rebase_change.assert_not_called()
        output = console.export_text()
        assert "Query matched 4 open changes:" in output
        assert "Found" not in output
        assert "Final Results: 4 submitted, 0 failed, 0 skipped" in output

    def test_submission_binds_to_previewed_revisions(self, gerrit):
        """Each change is reviewed and submitted at the revision listed."""
        gerrit.service.get_open_changes_for_query.return_value = _clm_topic()

        _run()

        _, kwargs = gerrit.submit_manager.submit_changes_parallel.call_args
        assert kwargs.get("bind_revisions") is True

    def test_whole_query_is_sent(self, gerrit):
        """Extra terms in the query reach the service."""
        gerrit.service.get_open_changes_for_query.return_value = _clm_topic()[:1]

        _run(_query_url("topic:x project:a/b"), dry_run=True)

        gerrit.service.get_open_changes_for_query.assert_called_once_with(
            "topic:x project:a/b"
        )

    def test_dry_run_lists_batch_and_submits_nothing(self, gerrit):
        """A dry run lists each change with project, number, subject, owner."""
        gerrit.service.get_open_changes_for_query.return_value = _clm_topic()

        output = _run(dry_run=True).export_text()

        assert (
            "aai/rest-client #149215: Move CLM lane to security-workflows "
            "reusable (lf-automation)"
        ) in output
        assert "4 changes would be reviewed and submitted" in output
        gerrit.create_submit_manager.assert_not_called()

    def test_no_open_changes_exits_1(self, gerrit):
        """A query matching nothing stops without submitting."""
        gerrit.service.get_open_changes_for_query.return_value = []
        console = Console(record=True, width=200)

        with pytest.raises(typer.Exit) as exc_info:
            _run(dry_run=True, console=console)

        assert exc_info.value.exit_code == 1
        assert "No open changes match query 'topic:reusable-clm-workflow'" in (
            console.export_text()
        )
        gerrit.create_submit_manager.assert_not_called()

    @pytest.mark.parametrize(
        "error",
        [
            "Query '(topic:x) status:open' matches more than 500 open changes",
            "Gerrit returned a change (7) that could not be parsed",
        ],
    )
    def test_incomplete_result_set_submits_nothing(self, gerrit, error):
        """An over-ceiling or partly parsed result refuses the run."""
        gerrit.service.get_open_changes_for_query.side_effect = GerritServiceError(
            error
        )

        console = Console(record=True, width=200)

        with pytest.raises(typer.Exit) as exc_info:
            _run(console=console)

        assert exc_info.value.exit_code != 0
        assert error in console.export_text()
        gerrit.create_submit_manager.assert_not_called()


class TestQueryUrlOptions:
    """Options that belong to a change URL."""

    def test_topic_flag_is_refused(self, gerrit):
        """--topic would replace the URL's query, so it is refused."""
        with pytest.raises(SystemExit) as exc_info:
            _run(topic="other-topic")

        assert exc_info.value.code == ExitCode.VALIDATION_ERROR
        gerrit.service.get_open_changes_for_query.assert_not_called()

    def test_threshold_is_noted_not_refused(self, gerrit):
        """A non-default --threshold prints a note and the run goes on."""
        gerrit.service.get_open_changes_for_query.return_value = _clm_topic()

        output = _run(similarity_threshold=0.5, dry_run=True).export_text()

        assert "--threshold does not apply" in output


class TestQueryUrlHumanGate:
    """The human gate covers the whole batch and binds to it."""

    @staticmethod
    def _override_from(capsys) -> str:
        captured = capsys.readouterr()
        found = re.findall(r"--override ([0-9a-f]{16})", captured.out + captured.err)
        assert found
        return str(found[-1])

    def _override_for(self, gerrit, capsys, changes) -> str:
        gerrit.service.get_open_changes_for_query.return_value = changes
        with pytest.raises(SystemExit) as exc_info:
            _run(include_human_prs=True)
        assert exc_info.value.code == ExitCode.OVERRIDE_REQUIRED
        return self._override_from(capsys)

    def test_mixed_batch_needs_include_human_prs(self, gerrit):
        """One human change among automation needs the flag."""
        changes = _clm_topic()
        gerrit.comparator.is_automation_change.side_effect = lambda c: (
            c.number != 149215
        )
        gerrit.service.get_open_changes_for_query.return_value = changes

        with pytest.raises(SystemExit) as exc_info:
            _run()

        assert exc_info.value.code == ExitCode.VALIDATION_ERROR
        gerrit.create_submit_manager.assert_not_called()

    def test_mixed_batch_needs_override(self, gerrit, capsys):
        """With the flag, the override is still required."""
        gerrit.comparator.is_automation_change.side_effect = lambda c: (
            c.number != 149215
        )
        override = self._override_for(gerrit, capsys, _clm_topic())

        assert len(override) == 16
        gerrit.create_submit_manager.assert_not_called()

    def test_override_submits_the_batch(self, gerrit, capsys):
        """The printed value confirms exactly that batch."""
        gerrit.comparator.is_automation_change.return_value = False
        override = self._override_for(gerrit, capsys, _clm_topic())

        _run(include_human_prs=True, override=override)

        assert len(_submitted_numbers(gerrit.submit_manager)) == 4

    def test_override_binds_to_revisions_and_membership(self, gerrit, capsys):
        """A new patchset or another change yields a different value."""
        gerrit.comparator.is_automation_change.return_value = False
        base = self._override_for(gerrit, capsys, _clm_topic())

        repatched = _clm_topic()
        repatched[2].current_revision = "newrev"
        assert self._override_for(gerrit, capsys, repatched) != base

        grown = [*_clm_topic(), _change(149217, "x/y", "Add Nexus IQ CLM scan")]
        assert self._override_for(gerrit, capsys, grown) != base

        assert self._override_for(gerrit, capsys, _clm_topic()[::-1]) == base

    def test_stale_override_is_refused(self, gerrit, capsys):
        """An override from a previous batch no longer confirms it."""
        gerrit.comparator.is_automation_change.return_value = False
        override = self._override_for(gerrit, capsys, _clm_topic())
        repatched = _clm_topic()
        repatched[0].current_revision = "newrev"
        gerrit.service.get_open_changes_for_query.return_value = repatched

        with pytest.raises(SystemExit) as exc_info:
            _run(include_human_prs=True, override=override)

        assert exc_info.value.code == ExitCode.VALIDATION_ERROR
        gerrit.create_submit_manager.assert_not_called()

    def test_dry_run_names_override_and_batch(self, gerrit, capsys):
        """A dry run prints the value and what it would act on."""
        gerrit.comparator.is_automation_change.return_value = False
        gerrit.service.get_open_changes_for_query.return_value = _clm_topic()

        _run(include_human_prs=True, dry_run=True)

        output = " ".join(capsys.readouterr().out.split())
        assert "a real run needs --override" in output
        assert (
            "to act on the 4 changes matched by query 'topic:reusable-clm-workflow'"
        ) in output


class TestQueryUrlBlockedChanges:
    """Blocked changes are listed and counted, never voted on."""

    @staticmethod
    def _with_verified_minus_one() -> list[GerritChangeInfo]:
        changes = _clm_topic()[:3]
        changes[1].labels = [
            GerritLabelInfo(name="Verified", rejected=True, value=-1),
        ]
        return changes

    def test_rejected_change_is_skipped_counted_and_fails_run(self, gerrit):
        """A Verified -1 change is skipped without a vote."""
        gerrit.service.get_open_changes_for_query.return_value = (
            self._with_verified_minus_one()
        )

        with pytest.raises(typer.Exit) as exc_info:
            _run()

        assert exc_info.value.exit_code == ExitCode.MERGE_ERROR
        assert _submitted_numbers(gerrit.submit_manager) == [149213, 149215]

    def test_rejected_change_is_reported(self, gerrit):
        """The preview marks it and the tally counts it."""
        gerrit.service.get_open_changes_for_query.return_value = (
            self._with_verified_minus_one()
        )
        console = Console(record=True, width=200)

        with pytest.raises(typer.Exit):
            _handle_gerrit_merge(
                parsed_url=_query_url(),
                no_confirm=True,
                similarity_threshold=0.8,
                verbose=False,
                console=console,
                show_progress=False,
            )

        output = console.export_text()
        assert "Blocked (Verified -1)" in output
        assert "Final Results: 2 submitted, 0 failed, 1 skipped" in output
        assert "sdc/sdc-distribution-client" in output.split("Skipped changes:")[1]

    @pytest.mark.parametrize(
        ("extra", "reason"),
        [
            ({"work_in_progress": True}, "work in progress"),
            ({"mergeable": False}, "merge conflict"),
        ],
    )
    def test_other_blocks_are_skipped(self, gerrit, extra, reason):
        """WIP and conflicted changes are skipped, never rebased."""
        changes = _clm_topic()[:2]
        changes[0] = _change(149213, "ccsdk/features", "Add Nexus IQ CLM scan", **extra)
        gerrit.service.get_open_changes_for_query.return_value = changes
        console = Console(record=True, width=200)

        with pytest.raises(typer.Exit) as exc_info:
            _handle_gerrit_merge(
                parsed_url=_query_url(),
                no_confirm=True,
                similarity_threshold=0.8,
                verbose=False,
                console=console,
                show_progress=False,
            )

        assert exc_info.value.exit_code == ExitCode.MERGE_ERROR
        assert _submitted_numbers(gerrit.submit_manager) == [149214]
        gerrit.service.rebase_change.assert_not_called()
        assert f"Blocked ({reason})" in console.export_text()


class TestQueryUrlPermissions:
    """Permissions are reported per change, not from one anchor."""

    def test_warns_on_the_change_lacking_permission(self, gerrit):
        changes = _clm_topic()[:2]
        changes[1].permitted_labels = {}
        gerrit.service.get_open_changes_for_query.return_value = changes

        output = _run(dry_run=True).export_text()

        listing = output.split("sdc/sdc-distribution-client")[1]
        assert "permission to give +2 Code-Review" in listing
        assert "lack permissions on 1 changes" in output

    def test_submit_permission_unverified_before_submittable(self, gerrit):
        """A change awaiting its +2 cannot show whether the caller may submit."""
        gerrit.service.get_open_changes_for_query.return_value = _clm_topic()[:2]

        output = _run(dry_run=True).export_text()

        assert "(+2 Code-Review, submit) on every change" not in output
        assert "+2 Code-Review permission on every change" in output
        assert "Submit permission not verified on 2 changes" in output

    def test_submittable_change_with_submit_action_confirms_both(self, gerrit):
        """Submit permission is confirmed only where Gerrit offers the action."""
        changes = [
            c.model_copy(update={"submittable": True, "actions": {"submit": {}}})
            for c in _clm_topic()[:2]
        ]
        gerrit.service.get_open_changes_for_query.return_value = changes

        output = _run(dry_run=True).export_text()

        assert "(+2 Code-Review, submit) on every change" in output
        assert "not verified" not in output


def test_wholly_blocked_batch_submits_nothing_and_fails(gerrit):
    """When every change is blocked, no submit manager is created."""
    changes = _clm_topic()[:1]
    changes[0].work_in_progress = True
    gerrit.service.get_open_changes_for_query.return_value = changes

    with pytest.raises(typer.Exit) as exc_info:
        _run()

    assert exc_info.value.exit_code == ExitCode.MERGE_ERROR
    gerrit.create_submit_manager.assert_not_called()


_VERIFIED_VALUES = {"-1": "Fails", " 0": "No score", "+1": "Verified"}
_REVIEW_VALUES = {
    "-2": "Do not submit",
    "-1": "I would prefer not",
    " 0": "No score",
    "+1": "Looks good to me",
    "+2": "Approved",
}


def _gerrit37_change(
    number: int,
    project: str,
    verified: list[int],
    requirements: dict[str, str],
    code_review: tuple[int, ...] = (),
) -> GerritChangeInfo:
    """A change as Gerrit 3.7 lists it with submit requirements.

    The label summaries carry no ``approved`` or ``rejected`` key; the
    votes sit under ``all`` and the range under ``values``.
    """
    return GerritChangeInfo.from_api_response(
        {
            "_number": number,
            "change_id": f"I{number:040d}",
            "project": project,
            "subject": "CI: Add SonarCloud master scan on GitHub Actions",
            "branch": "master",
            "status": "NEW",
            "owner": {"username": "lf-automation"},
            "current_revision": f"rev{number}",
            "permitted_labels": FULL_ACCESS,
            "labels": {
                "Verified": {
                    "all": [
                        {"value": v, "_account_id": i} for i, v in enumerate(verified)
                    ],
                    "values": _VERIFIED_VALUES,
                },
                "Code-Review": {
                    "all": [
                        {"value": v, "_account_id": 10 + i}
                        for i, v in enumerate(code_review)
                    ],
                    "values": _REVIEW_VALUES,
                },
            },
            "submit_requirements": [
                {"name": name, "status": status}
                for name, status in requirements.items()
            ],
        },
        host=HOST,
    )


_UNREVIEWED = {"Code-Review": "UNSATISFIED"}


class TestQueryUrlBlockedOnGerrit37:
    """Blockers read from votes and submit requirements, not summaries."""

    @staticmethod
    def _ready(number: int) -> GerritChangeInfo:
        return _gerrit37_change(
            number,
            f"proj/{number}",
            [1, 1, 1],
            {"Verified": "SATISFIED", **_UNREVIEWED},
        )

    @pytest.mark.parametrize(
        ("blocked", "reason"),
        [
            (
                _gerrit37_change(
                    149203,
                    "sdnc/apps",
                    [1, 1, -1],
                    {"Verified": "UNSATISFIED", **_UNREVIEWED},
                ),
                "Verified -1",
            ),
            (
                _gerrit37_change(
                    149208,
                    "sdc/onap-ui-common",
                    [1, 1],
                    {"Verified": "UNSATISFIED", **_UNREVIEWED},
                ),
                "Verified requirement unsatisfied",
            ),
            (
                _gerrit37_change(
                    149209,
                    "a/b",
                    [1, 1, 1],
                    {"Verified": "SATISFIED", **_UNREVIEWED},
                    code_review=(-2,),
                ),
                "Code-Review -2",
            ),
            (
                _gerrit37_change(
                    149210,
                    "a/c",
                    [1, 1, 1],
                    {"Verified": "SATISFIED", "No-Bypass": "ERROR", **_UNREVIEWED},
                ),
                "No-Bypass requirement in error",
            ),
            (
                _gerrit37_change(
                    149213,
                    "a/d",
                    [1, 1, 1],
                    {"Verified": "SATISFIED", "Code-Review": "ERROR"},
                ),
                "Code-Review requirement in error",
            ),
        ],
    )
    def test_blocked_change_is_marked_and_skipped(self, gerrit, blocked, reason):
        """The blocked change is listed, not voted on, and fails the run."""
        changes = [self._ready(149211), blocked, self._ready(149212)]
        gerrit.service.get_open_changes_for_query.return_value = changes
        console = Console(record=True, width=200)

        with pytest.raises(typer.Exit) as exc_info:
            _run(console=console)

        assert exc_info.value.exit_code == ExitCode.MERGE_ERROR
        assert _submitted_numbers(gerrit.submit_manager) == [149211, 149212]
        output = console.export_text()
        assert f"Blocked ({reason})" in output
        assert "Final Results: 2 submitted, 0 failed, 1 skipped" in output


def test_gerrit37_change_awaiting_only_review_is_not_blocked(gerrit):
    """Code-Review's own unmet requirement is what this run's +2 meets."""
    changes = [
        TestQueryUrlBlockedOnGerrit37._ready(149211),
        _gerrit37_change(
            149212, "a/b", [1, 1, 1], {"Verified": "SATISFIED", **_UNREVIEWED}, (-1,)
        ),
    ]
    gerrit.service.get_open_changes_for_query.return_value = changes

    output = _run(dry_run=True).export_text()

    assert "Blocked" not in output
    assert "2 changes would be reviewed and submitted; 0 blocked" in output


class TestQueryUrlSubmittedTogether:
    """Nothing outside the batch may be submitted along with it."""

    def test_submission_is_bound_to_the_batch_scope(self, gerrit):
        """The submit manager gets each change to submit and its revision."""
        changes = TestQueryUrlBlockedChanges._with_verified_minus_one()
        gerrit.service.get_open_changes_for_query.return_value = changes

        with pytest.raises(typer.Exit):
            _run()

        _, kwargs = gerrit.submit_manager.submit_changes_parallel.call_args
        assert kwargs.get("scope") == {149213: "rev149213", 149215: "rev149215"}
        checked = [
            c.args for c in gerrit.service.submitted_together_problems.call_args_list
        ]
        assert checked == [(n, kwargs["scope"]) for n in (149213, 149215)]

    @pytest.mark.parametrize("dry_run", [False, True])
    def test_change_submitting_outside_the_batch_refuses_run(self, gerrit, dry_run):
        """One change Gerrit would submit with others refuses every vote."""
        gerrit.service.get_open_changes_for_query.return_value = _clm_topic()
        gerrit.service.submitted_together_problems.side_effect = lambda n, _: (
            ["#149214 would also submit #5 (other/project), which is not among"]
            if n == 149214
            else []
        )
        console = Console(record=True, width=200)

        with pytest.raises(SystemExit) as exc_info:
            _run(dry_run=dry_run, console=console)

        assert exc_info.value.code == ExitCode.VALIDATION_ERROR
        assert "#149214 would also submit #5 (other/project)" in (console.export_text())
        gerrit.create_submit_manager.assert_not_called()

    def test_refusal_advice_does_not_suggest_narrowing(self, gerrit, caplog):
        """A filter that leaves a group member out cannot be fixed by narrowing."""
        gerrit.service.get_open_changes_for_query.return_value = _clm_topic()
        gerrit.service.submitted_together_problems.return_value = [
            "#149214 would also submit #5 (other/project), which is not among"
        ]

        with pytest.raises(SystemExit):
            _run()

        advice = " ".join(r.getMessage() for r in caplog.records)
        assert "Narrow the query" not in advice
        assert "matches every change Gerrit submits together" in advice
        assert "project: or branch:" in advice
        assert "resolve any blocked member" in advice
