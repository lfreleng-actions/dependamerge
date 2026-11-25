# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Major version bumps pass a gate on both Gerrit paths.

A query URL's batch and a change URL's similar changes are each held
back, prompted for or submitted according to ``--no-confirm``,
``--allow-major`` and ``--gate-zero-minor``.  A held-back change gets no
vote, is never submitted as a side effect, and leaves the exit code
alone.
"""

import re
from unittest.mock import Mock, patch

import pytest
import typer
from rich.console import Console

from dependamerge.cli import _generate_gerrit_query_sha, _handle_gerrit_merge
from dependamerge.error_codes import ExitCode
from dependamerge.gerrit import (
    GerritChangeInfo,
    GerritComparisonResult,
    GerritRestError,
    GerritService,
    GerritSubmitManager,
    GerritSubmitResult,
    SubmissionGuard,
)
from dependamerge.gerrit._submitted_together import submitted_together_problems
from dependamerge.url_parser import (
    ChangeSource,
    ParsedGerritTopicUrl,
    ParsedUrl,
    parse_gerrit_topic_url,
)

HOST = "gerrit.example.org"
EXPRESSION = "topic:deps"
FULL_ACCESS = {"Code-Review": ["-2", "-1", "0", "+1", "+2"]}
MAJOR = "Chore: Bump python-test-action from 1.3.3 to 2.0.0"
MINOR = "Chore: Bump actions/checkout from 4.1.0 to 4.2.0"
ZERO_MINOR = "Chore: Bump lib from 0.7.0 to 0.8.0"
SEMVER_MAJOR_MESSAGE = (
    "Chore: Update CI workflows\n\n"
    "updated-dependencies:\n"
    "- dependency-name: actions/checkout\n"
    "  dependency-type: direct:production\n"
    "  update-type: version-update:semver-major\n"
)


def _change(
    number: int,
    subject: str = MINOR,
    revision: str | None = None,
    **extra,
) -> GerritChangeInfo:
    return GerritChangeInfo(
        number=number,
        change_id=f"I{number:040d}",
        project=f"proj-{number}",
        subject=subject,
        owner="dependabot",
        branch="master",
        status="NEW",
        current_revision=revision or f"rev{number}",
        topic="deps",
        permitted_labels=FULL_ACCESS,
        url=f"https://{HOST}/c/proj-{number}/+/{number}",
        **extra,
    )


def _submit_all(changes, *args, **kwargs):
    return [
        GerritSubmitResult.success_result(change_number=c.number, project=c.project)
        for c, _ in changes
    ]


def _query_url() -> ParsedGerritTopicUrl:
    return parse_gerrit_topic_url(f"https://{HOST}/q/{EXPRESSION}")


def _change_url(source: GerritChangeInfo) -> ParsedUrl:
    return ParsedUrl(
        source=ChangeSource.GERRIT,
        host=HOST,
        base_path=None,
        project=source.project,
        change_number=source.number,
        original_url=source.url,
    )


@pytest.fixture
def gerrit():
    """Patch credentials, service, comparator and submit manager.

    A change URL's source and similar changes come from ``gerrit.source``
    and ``gerrit.similar``; the steps that read them from Gerrit are
    replaced.
    """
    credentials = Mock(is_valid=True, username="user", password="pass")
    credentials.auth_method_display.return_value = ".netrc file"
    service = Mock(is_authenticated=True)
    service.submitted_together_problems.return_value = []
    service.submitted_together_numbers.return_value = set()
    service.get_open_changes_by_number.return_value = []
    comparator = Mock()
    comparator.is_automation_change.return_value = True
    submit_manager = Mock()
    submit_manager.submit_changes_parallel.side_effect = _submit_all
    state = Mock(
        service=service,
        submit_manager=submit_manager,
        source=None,
        similar=[],
    )
    module = "dependamerge.cli._gerrit_merge"
    with (
        patch("dependamerge.cli.resolve_gerrit_credentials", return_value=credentials),
        patch("dependamerge.cli.create_gerrit_service", return_value=service),
        patch("dependamerge.cli.create_gerrit_comparator", return_value=comparator),
        patch(
            "dependamerge.cli.create_submit_manager", return_value=submit_manager
        ) as create_submit_manager,
        patch(
            f"{module}._resolve_gerrit_source_change",
            side_effect=lambda *a, **k: state.source,
        ),
        patch(f"{module}._resolve_gerrit_only_automation", return_value=True),
        patch(
            f"{module}._maybe_rebase_gerrit_change",
            side_effect=lambda service, change, *a, **k: change,
        ) as rebase,
        patch(f"{module}._gerrit_candidate_scope"),
        patch(f"{module}._resolve_gerrit_candidates", return_value=[]),
        patch(
            f"{module}._find_and_print_similar_changes",
            side_effect=lambda *a, **k: [(c, None) for c in state.similar],
        ),
    ):
        state.create_submit_manager = create_submit_manager
        state.rebase = rebase
        yield state


def _run(
    parsed_url: ParsedUrl | ParsedGerritTopicUrl,
    *,
    no_confirm: bool = True,
    dry_run: bool = False,
    allow_major: bool = False,
    gate_zero_minor: bool = False,
) -> Console:
    console = Console(record=True, width=200)
    _handle_gerrit_merge(
        parsed_url=parsed_url,
        no_confirm=no_confirm,
        similarity_threshold=0.8,
        verbose=False,
        console=console,
        dry_run=dry_run,
        show_progress=False,
        allow_major=allow_major,
        gate_zero_minor=gate_zero_minor,
    )
    return console


def _run_query(gerrit, changes: list[GerritChangeInfo], **kwargs) -> Console:
    gerrit.service.get_open_changes_for_query.return_value = changes
    return _run(_query_url(), **kwargs)


def _run_change(
    gerrit,
    source: GerritChangeInfo,
    similar: list[GerritChangeInfo],
    **kwargs,
) -> Console:
    gerrit.source = source
    gerrit.similar = similar
    return _run(_change_url(source), **kwargs)


def _submitted(gerrit) -> list[int]:
    (changes,), _ = gerrit.submit_manager.submit_changes_parallel.call_args
    return sorted(change.number for change, _ in changes)


def _submit_kwargs(gerrit) -> dict[str, object]:
    kwargs: dict[str, object] = (
        gerrit.submit_manager.submit_changes_parallel.call_args.kwargs
    )
    return kwargs


def _entered_value(console: Console) -> str:
    """The value the gate printed for the operator to enter."""
    text = console.export_text(clear=False)
    match = re.search(r"To submit them too enter: (\w+)", text)
    assert match, text
    return match.group(1)


def _prompt_entering(console_holder: list[Console], entry: str | None):
    """A ``typer.prompt`` replacement entering ``entry``, or the right value."""

    def prompt(*args, **kwargs):
        if entry is None:
            return _entered_value(console_holder[0])
        return entry

    return prompt


BOTH_MODES = pytest.mark.parametrize("mode", ["query", "change"])


def _run_mode(gerrit, mode: str, changes: list[GerritChangeInfo], **kwargs):
    """Run ``changes`` as a query batch, or as a change URL's similar set.

    In change mode the last change is the source; tests that gate a
    change other than the source put a plain bump last.
    """
    if mode == "query":
        return _run_query(gerrit, changes, **kwargs)
    return _run_change(gerrit, changes[-1], changes[:-1], **kwargs)


class TestUnattended:
    """--no-confirm holds gated changes back without failing the run."""

    @BOTH_MODES
    def test_major_is_held_back_and_the_rest_submitted(self, gerrit, mode):
        """No vote for the bump, exit 0, the hint printed."""
        output = _run_mode(
            gerrit, mode, [_change(1, MAJOR), _change(2), _change(3)]
        ).export_text()

        assert _submitted(gerrit) == [2, 3]
        assert "Major version bumps held back: 1" in output
        assert "held back" in output.split("Final Results:")[1].splitlines()[0]
        assert "major version bump: python-test-action 1.3.3 → 2.0.0" in output
        assert "re-run with --allow-major" in output

    def test_query_blocked_change_still_fails_the_run(self, gerrit):
        """MERGE_ERROR from the blocked change; both counts in the tally."""
        changes = [_change(1, MAJOR), _change(2), _change(3, work_in_progress=True)]
        console = Console(record=True, width=200)
        gerrit.service.get_open_changes_for_query.return_value = changes

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
        output = console.export_text()
        assert "Final Results: 1 submitted, 0 failed, 1 skipped, 1 held back" in output
        assert "1 changes not submitted (0 failed, 1 skipped)" in output

    @BOTH_MODES
    def test_allow_major_submits_and_lists_the_bump(self, gerrit, mode):
        output = _run_mode(
            gerrit, mode, [_change(1, MAJOR), _change(2)], allow_major=True
        ).export_text()

        assert _submitted(gerrit) == [1, 2]
        assert "held back" not in output
        report = output.split("Major version bumps submitted: 1")[1]
        assert f"https://{HOST}/c/proj-1/+/1" in report
        assert "python-test-action 1.3.3 → 2.0.0" in report

    @BOTH_MODES
    @pytest.mark.parametrize(
        ("gate_zero_minor", "submitted"), [(True, [2]), (False, [1, 2])]
    )
    def test_gate_zero_minor(self, gerrit, mode, gate_zero_minor, submitted):
        _run_mode(
            gerrit,
            mode,
            [_change(1, ZERO_MINOR), _change(2)],
            gate_zero_minor=gate_zero_minor,
        )

        assert _submitted(gerrit) == submitted

    @BOTH_MODES
    def test_semver_major_message_is_gated(self, gerrit, mode):
        """No versions in the subject; the commit message says major."""
        _run_mode(
            gerrit,
            mode,
            [
                _change(1, "Chore: Update CI workflows", message=SEMVER_MAJOR_MESSAGE),
                _change(2),
            ],
        )

        assert _submitted(gerrit) == [2]

    @BOTH_MODES
    def test_unknown_is_never_gated(self, gerrit, mode):
        _run_mode(
            gerrit,
            mode,
            [_change(1, "Add Nexus IQ CLM scan"), _change(2)],
            gate_zero_minor=True,
        )

        assert _submitted(gerrit) == [1, 2]

    def test_query_scope_refusal_recaps_the_held_back_changes(self, gerrit):
        """#2 sits on held-back #1: the refusal names it and --allow-major."""
        client = Mock()
        client.get.return_value = {
            "changes": [
                {"_number": 2, "current_revision": "rev2"},
                {"_number": 1, "current_revision": "rev1", "project": "proj-1"},
            ],
            "non_visible_changes": 0,
        }
        gerrit.service.submitted_together_problems.side_effect = lambda number, scope: (
            submitted_together_problems(client, number, scope)
        )
        console = Console(record=True, width=200)
        gerrit.service.get_open_changes_for_query.return_value = [
            _change(1, MAJOR),
            _change(2),
        ]

        with pytest.raises(SystemExit) as exc_info:
            _handle_gerrit_merge(
                parsed_url=_query_url(),
                no_confirm=True,
                similarity_threshold=0.8,
                verbose=False,
                console=console,
                show_progress=False,
            )

        assert exc_info.value.code == ExitCode.VALIDATION_ERROR
        gerrit.create_submit_manager.assert_not_called()
        output = console.export_text()
        recap = output.split("Held back changes:", 1)[1]
        assert "major version bump: python-test-action 1.3.3 → 2.0.0" in recap
        assert "--allow-major" in recap


class TestPrecedence:
    """A change Gerrit reports as blocked never reaches the gate."""

    def test_query_blocked_major_is_skipped_as_blocked(self, gerrit):
        """Blocked comes first: the bump is skipped, not held back."""
        console = Console(record=True, width=200)
        gerrit.service.get_open_changes_for_query.return_value = [
            _change(1, MAJOR, work_in_progress=True),
            _change(2),
        ]

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
        assert "Final Results: 1 submitted, 0 failed, 1 skipped" in output
        assert "held back" not in output

    @pytest.mark.parametrize("blocked_source", [False, True])
    def test_change_blocked_major_is_skipped_as_blocked(self, gerrit, blocked_source):
        """Unattended: no vote, not held back, and the run still fails."""
        blocked = _change(1, MAJOR, work_in_progress=True)
        source, similar = (
            (blocked, [_change(2)]) if blocked_source else (_change(2), [blocked])
        )

        gerrit.source = source
        gerrit.similar = similar
        console = Console(record=True, width=200)

        with pytest.raises(typer.Exit) as exc_info:
            _handle_gerrit_merge(
                parsed_url=_change_url(source),
                no_confirm=True,
                similarity_threshold=0.8,
                verbose=False,
                console=console,
                show_progress=False,
            )

        assert exc_info.value.exit_code == ExitCode.MERGE_ERROR
        assert _submitted(gerrit) == [2]
        output = console.export_text()
        assert "Final Results: 1 submitted, 0 failed, 1 skipped" in output
        assert "Blocked: work in progress" in output
        assert "held back" not in output

    def test_change_blocked_ungated_change_reaches_the_submit_manager(self, gerrit):
        """The gate leaves a blocked change it would not take as it was."""
        _run_change(gerrit, _change(2), [_change(1, work_in_progress=True)])

        assert _submitted(gerrit) == [1, 2]

    def test_change_blocked_dependent_is_not_held_back(self, gerrit):
        """WIP #2 would take held-back #1 along: it fails, not held back."""
        gerrit.service.submitted_together_numbers.side_effect = {
            2: {1},
            3: set(),
        }.get

        def submit(changes, *args, **kwargs):
            return [
                GerritSubmitResult.failure_result(
                    change_number=c.number, project=c.project, error="WIP"
                )
                if c.work_in_progress
                else GerritSubmitResult.success_result(
                    change_number=c.number, project=c.project
                )
                for c, _ in changes
            ]

        gerrit.submit_manager.submit_changes_parallel.side_effect = submit

        with pytest.raises(typer.Exit) as exc_info:
            _run_change(
                gerrit,
                _change(3),
                [_change(1, MAJOR), _change(2, work_in_progress=True)],
            )

        assert exc_info.value.exit_code == ExitCode.MERGE_ERROR
        assert _submitted(gerrit) == [2, 3]
        assert _submit_kwargs(gerrit)["scope"] == SubmissionGuard(
            confirmed={2: "rev2", 3: "rev3"}, held=frozenset({1})
        )

    @pytest.mark.parametrize("blocked_source", [False, True])
    def test_change_blocker_details_are_read_before_the_gate(
        self, gerrit, blocked_source
    ):
        """An unsatisfied Verified requirement the first read lacked."""
        major = _change(1, MAJOR)
        source, similar = (
            (major, [_change(2)]) if blocked_source else (_change(2), [major])
        )
        gerrit.service.get_open_changes_by_number.return_value = [
            _change(1, MAJOR, submit_requirement_status={"Verified": "UNSATISFIED"}),
            _change(2),
        ]

        with pytest.raises(typer.Exit) as exc_info:
            _run_change(gerrit, source, similar)

        assert exc_info.value.exit_code == ExitCode.MERGE_ERROR
        gerrit.service.get_open_changes_by_number.assert_called_once()
        assert sorted(gerrit.service.get_open_changes_by_number.call_args.args[0]) == [
            1,
            2,
        ]
        assert _submitted(gerrit) == [2]

    def test_change_without_gated_changes_reads_nothing_more(self, gerrit):
        _run_change(gerrit, _change(1), [_change(2)])

        gerrit.service.get_open_changes_by_number.assert_not_called()

    @patch("dependamerge.gerrit.service.create_url_builder")
    @patch("dependamerge.gerrit.service.build_client")
    def test_blocker_read_asks_for_requirements_and_falls_back(
        self, build_client, url_builder
    ):
        """Detailed labels and submit requirements, or neither on an old server."""
        from urllib.parse import parse_qs, urlsplit

        client = Mock()
        client.get.side_effect = [
            GerritRestError(
                "HTTP 400",
                status_code=400,
                response_body='"SUBMIT_REQUIREMENTS" is not a valid value for "-o"',
            ),
            [
                {
                    "_number": 2,
                    "change_id": "I2",
                    "project": "p",
                    "current_revision": "rev2",
                }
            ],
        ]
        build_client.return_value = client

        changes = GerritService(host=HOST).get_open_changes_by_number([2, 1, 2])

        assert [c.number for c in changes] == [2]
        first, second = (
            parse_qs(urlsplit(c.args[0]).query) for c in client.get.call_args_list
        )
        assert first["q"] == ["(change:1 OR change:2) status:open"]
        assert {"DETAILED_LABELS", "SUBMIT_REQUIREMENTS"} <= set(first["o"])
        assert "DETAILED_LABELS" in second["o"]
        assert "SUBMIT_REQUIREMENTS" not in second["o"]


class TestQueryInteractive:
    """A query batch asks for a value of the gated changes' own."""

    def _run(self, gerrit, entry, changes=None):
        gerrit.service.get_open_changes_for_query.return_value = changes or [
            _change(1, MAJOR),
            _change(2),
        ]
        console = Console(record=True, width=200)
        with (
            patch(
                "typer.prompt",
                side_effect=entry
                if isinstance(entry, BaseException)
                else _prompt_entering([console], entry),
            ),
            patch(
                "dependamerge.cli._confirm_continue_sha", return_value=True
            ) as confirm,
        ):
            _handle_gerrit_merge(
                parsed_url=_query_url(),
                no_confirm=False,
                similarity_threshold=0.8,
                verbose=False,
                console=console,
                show_progress=False,
            )
        return console.export_text(), confirm

    def test_the_value_includes_the_gated_set(self, gerrit):
        changes = [_change(1, MAJOR), _change(2)]
        output, confirm = self._run(gerrit, None, changes)

        assert "Major version bumps included" in output
        assert _submitted(gerrit) == [1, 2]
        (value, _), _ = confirm.call_args
        assert value == _generate_gerrit_query_sha(
            "continue", HOST, EXPRESSION, changes
        )

    @pytest.mark.parametrize(
        ("entry", "said"),
        [
            ("", "Major version bumps held back."),
            ("wrong", "The value does not match"),
            (EOFError(), "No value entered"),
        ],
        ids=["enter", "wrong", "eof"],
    )
    def test_anything_else_holds_the_set_back(self, gerrit, entry, said):
        changes = [_change(1, MAJOR), _change(2)]
        output, confirm = self._run(gerrit, entry, changes)

        assert said in output
        assert _submitted(gerrit) == [2]
        assert "1 held back" in output
        (value, _), _ = confirm.call_args
        assert value == _generate_gerrit_query_sha(
            "continue", HOST, EXPRESSION, changes[1:]
        )

    def test_held_back_change_is_outside_the_scope(self, gerrit):
        self._run(gerrit, "")

        assert _submit_kwargs(gerrit)["scope"] == {2: "rev2"}
        for call in gerrit.service.submitted_together_problems.call_args_list:
            number, scope = call.args
            assert number == 2
            assert 1 not in scope

    def test_submission_taking_a_held_back_change_is_refused(self, gerrit):
        """#2 sits on #1: submitting it would submit the held-back bump."""
        client = Mock()
        client.get.return_value = {
            "changes": [
                {"_number": 2, "current_revision": "rev2"},
                {"_number": 1, "current_revision": "rev1", "project": "proj-1"},
            ],
            "non_visible_changes": 0,
        }
        gerrit.service.submitted_together_problems.side_effect = lambda number, scope: (
            submitted_together_problems(client, number, scope)
        )

        with pytest.raises(SystemExit) as exc_info:
            self._run(gerrit, "")

        assert exc_info.value.code == ExitCode.VALIDATION_ERROR
        gerrit.create_submit_manager.assert_not_called()


class TestMajorsValue:
    """The value binds to the gated changes and their revisions."""

    def test_new_patchset_or_new_member_changes_it(self):
        from dependamerge.cli._sha import _generate_gerrit_majors_sha

        base = _generate_gerrit_majors_sha(HOST, [_change(1, MAJOR)])

        assert base == _generate_gerrit_majors_sha(HOST, [_change(1, MAJOR)])
        assert base != _generate_gerrit_majors_sha(
            HOST, [_change(1, MAJOR, revision="rev1b")]
        )
        assert base != _generate_gerrit_majors_sha(
            HOST, [_change(1, MAJOR), _change(3, MAJOR)]
        )

    def test_it_differs_from_the_continue_value(self):
        from dependamerge.cli._sha import _generate_gerrit_majors_sha

        changes = [_change(1, MAJOR)]
        assert _generate_gerrit_majors_sha(HOST, changes) not in {
            _generate_gerrit_query_sha(purpose, HOST, "", changes)
            for purpose in ("continue", "override")
        }


class TestChangeMode:
    """A change URL's source and similar changes."""

    def _interactive(self, gerrit, source, similar, entry):
        gerrit.source = source
        gerrit.similar = similar
        console = Console(record=True, width=200)
        with (
            patch(
                "typer.prompt", side_effect=_prompt_entering([console], entry)
            ) as prompt,
            patch(
                "dependamerge.cli._confirm_gerrit_submission", return_value=True
            ) as confirm,
        ):
            _handle_gerrit_merge(
                parsed_url=_change_url(source),
                no_confirm=False,
                similarity_threshold=0.8,
                verbose=False,
                console=console,
                show_progress=False,
            )
        return console.export_text(), prompt, confirm

    def test_gated_source_warns_and_asks_for_no_value(self, gerrit):
        output, prompt, confirm = self._interactive(
            gerrit, _change(1, MAJOR), [_change(2, MAJOR), _change(3)], ""
        )

        prompt.assert_not_called()
        confirm.assert_called_once()
        assert "The source change is a major version bump" in output
        assert _submitted(gerrit) == [1, 2, 3]
        assert _submit_kwargs(gerrit)["bind_revisions"] is True

    def test_unattended_holds_the_gated_source_back(self, gerrit):
        output = _run_change(gerrit, _change(1, MAJOR), [_change(2)]).export_text()

        assert _submitted(gerrit) == [2]
        assert "proj-1 #1: major version bump" in output
        gerrit.rebase.assert_not_called()

    def test_included_changes_are_bound_to_their_revisions(self, gerrit):
        output, prompt, _ = self._interactive(
            gerrit, _change(1), [_change(2, MAJOR)], None
        )

        prompt.assert_called_once()
        assert "Major version bumps included" in output
        assert _submitted(gerrit) == [1, 2]
        assert _submit_kwargs(gerrit)["bind_revisions"] is True

    def test_without_gated_changes_revisions_stay_unbound(self, gerrit):
        _run_change(gerrit, _change(1), [_change(2)])

        assert _submit_kwargs(gerrit)["bind_revisions"] is False

    def test_held_back_value_submits_the_rest(self, gerrit):
        output, _, _ = self._interactive(
            gerrit, _change(1), [_change(2, MAJOR), _change(3)], ""
        )

        assert _submitted(gerrit) == [1, 3]
        assert "Final Results: 2 submitted, 0 failed, 1 held back" in output

    def test_change_taking_a_held_back_change_is_held_back(self, gerrit):
        """#3 sits on the held-back #2, and #1 on #3."""
        together = {1: {3}, 3: {2}, 4: set()}
        gerrit.service.submitted_together_numbers.side_effect = together.get

        output = _run_change(
            gerrit, _change(1), [_change(2, MAJOR), _change(3), _change(4)]
        ).export_text()

        assert _submitted(gerrit) == [4]
        assert "submitting it would also submit held-back #2" in output
        assert "submitting it would also submit held-back #3" in output
        assert "3 held back" in output

    def test_change_gerrit_cannot_place_is_held_back(self, gerrit):
        gerrit.service.submitted_together_numbers.return_value = None

        output = _run_change(gerrit, _change(1), [_change(2, MAJOR)]).export_text()

        assert "Gerrit did not report which changes it submits" in output
        gerrit.submit_manager.submit_changes_parallel.assert_not_called()


_Batch = list[tuple[GerritChangeInfo, GerritComparisonResult | None]]


class TestChangeSubmissionGuard:
    """A change URL's gated batch rechecks each group before vote and submit."""

    def test_held_back_changes_reach_the_submit_manager(self, gerrit):
        _run_change(gerrit, _change(1), [_change(2, MAJOR), _change(3)])

        kwargs = _submit_kwargs(gerrit)
        assert kwargs["bind_revisions"] is True
        assert kwargs["scope"] == SubmissionGuard(
            confirmed={3: "rev3", 1: "rev1"}, held=frozenset({2})
        )

    def test_confirmed_changes_reach_the_submit_manager(self, gerrit):
        console = Console(record=True, width=200)
        gerrit.source = _change(1)
        gerrit.similar = [_change(2, MAJOR)]
        with (
            patch("typer.prompt", side_effect=_prompt_entering([console], None)),
            patch("dependamerge.cli._confirm_gerrit_submission", return_value=True),
        ):
            _handle_gerrit_merge(
                parsed_url=_change_url(gerrit.source),
                no_confirm=False,
                similarity_threshold=0.8,
                verbose=False,
                console=console,
                show_progress=False,
            )

        assert _submit_kwargs(gerrit)["scope"] == SubmissionGuard(
            confirmed={2: "rev2", 1: "rev1"}, held=frozenset()
        )

    def test_without_gated_changes_there_is_no_guard(self, gerrit):
        _run_change(gerrit, _change(1), [_change(2)])

        assert _submit_kwargs(gerrit)["scope"] is None

    @staticmethod
    def _manager(client) -> GerritSubmitManager:
        with patch(
            "dependamerge.gerrit.submit_manager.build_client", return_value=client
        ):
            return GerritSubmitManager(host=HOST, username="user")

    @staticmethod
    def _client(revisions, group_before, group_after, hidden=None):
        """Answer as Gerrit does; groups move once every vote is in.

        ``hidden`` maps a change to the ``non_visible_changes`` value its
        group reports; ``...`` leaves the count out.
        """
        state = {"votes": 0}
        client = Mock()

        def get(endpoint: str):
            number = int(endpoint.split("/")[2].split("?")[0])
            if "submitted_together" in endpoint:
                group = group_after if state["votes"] >= 2 else group_before
                answer: dict[str, object] = {
                    "changes": [
                        {"_number": n, "current_revision": revisions[n]}
                        for n in group.get(number, [])
                    ],
                }
                count = (hidden or {}).get(number, 0)
                if count is not ...:
                    answer["non_visible_changes"] = count
                return answer
            return {"status": "NEW", "current_revision": revisions[number]}

        def post(endpoint: str, data=None):
            if endpoint.endswith("/review"):
                state["votes"] += 1
            return {}

        client.get.side_effect = get
        client.post.side_effect = post
        return client

    def test_sibling_taking_a_confirmed_major_at_a_new_patchset_is_refused(self):
        """#2 was confirmed at rev2; its topic sibling #1 would submit rev2-new."""
        revisions = {1: "rev1", 2: "rev2-new"}
        topic = {1: [1, 2], 2: [1, 2]}
        client = self._client(revisions, topic, topic)
        batch: _Batch = [(_change(1), None), (_change(2, MAJOR), None)]
        guard = SubmissionGuard(confirmed={1: "rev1", 2: "rev2"}, held=frozenset())

        results = self._manager(client).submit_changes_parallel(
            batch, bind_revisions=True, scope=guard
        )

        assert [r.success for r in results] == [False, False]
        assert "#2 at a revision other than the one confirmed" in (
            results[0].error or ""
        )
        assert "new patchset" in (results[1].error or "")
        client.post.assert_not_called()

    def test_held_back_major_joining_the_topic_after_the_votes_is_refused(self):
        """#9 was held back; it joins the topic of #1 and #3 once they are voted."""
        revisions = {1: "rev1", 3: "rev3", 9: "rev9"}
        before = {1: [1, 3], 3: [1, 3]}
        after = {1: [1, 3, 9], 3: [1, 3, 9]}
        client = self._client(revisions, before, after)
        batch: _Batch = [(_change(1), None), (_change(3), None)]
        guard = SubmissionGuard(confirmed={1: "rev1", 3: "rev3"}, held=frozenset({9}))

        results = self._manager(client).submit_changes_parallel(
            batch, bind_revisions=True, scope=guard
        )

        kinds = [c.args[0].rsplit("/", 1)[1] for c in client.post.call_args_list]
        assert kinds == ["review", "review"]
        assert [r.success for r in results] == [False, False]
        assert all("held-back #9" in (r.error or "") for r in results)

    def test_skipped_major_is_refused_once_its_blocker_clears(self, gerrit):
        """WIP major #1 is skipped; cleared later, sibling #2 would take it."""
        with pytest.raises(typer.Exit):
            _run_change(gerrit, _change(2), [_change(1, MAJOR, work_in_progress=True)])

        guard = SubmissionGuard(confirmed={2: "rev2"}, held=frozenset({1}))
        assert _submit_kwargs(gerrit)["scope"] == guard

        revisions = {1: "rev1", 2: "rev2"}
        topic = {2: [1, 2]}
        client = self._client(revisions, topic, topic)
        results = self._manager(client).submit_changes_parallel(
            [(_change(2), None)], bind_revisions=True, scope=guard
        )

        assert results[0].success is False
        assert "held-back #1" in (results[0].error or "")
        client.post.assert_not_called()

    @pytest.mark.parametrize(
        ("count", "expected"),
        [(1, "1 changes you cannot see"), (..., "whether it submits changes")],
    )
    def test_group_with_hidden_members_is_refused(self, count, expected):
        """Held-back #9 turned private: Gerrit counts it but does not list it."""
        revisions = {1: "rev1"}
        group = {1: [1]}
        client = self._client(revisions, group, group, hidden={1: count})
        guard = SubmissionGuard(confirmed={1: "rev1"}, held=frozenset({9}))

        results = self._manager(client).submit_changes_parallel(
            [(_change(1), None)], bind_revisions=True, scope=guard
        )

        assert results[0].success is False
        assert expected in (results[0].error or "")
        client.post.assert_not_called()

    def test_group_reaching_past_the_batch_is_otherwise_submitted(self):
        """Unlike a query scope, the guard lets an unheld ancestor along."""
        revisions = {1: "rev1", 5: "rev5"}
        group = {1: [1, 5]}
        client = self._client(revisions, group, group)
        guard = SubmissionGuard(confirmed={1: "rev1"}, held=frozenset({9}))

        batch: _Batch = [(_change(1), None)]

        results = self._manager(client).submit_changes_parallel(
            batch, bind_revisions=True, scope=guard
        )

        assert results[0].success is True


class TestDryRun:
    """A dry run prompts for nothing and submits nothing."""

    @BOTH_MODES
    @pytest.mark.parametrize("no_confirm", [True, False])
    def test_prompts_and_submits_nothing(self, gerrit, mode, no_confirm):
        with (
            patch("typer.prompt") as prompt,
            patch("dependamerge.cli._confirm_continue_sha") as confirm_query,
            patch("dependamerge.cli._confirm_gerrit_submission") as confirm_change,
        ):
            output = _run_mode(
                gerrit,
                mode,
                [_change(1, MAJOR), _change(2)],
                no_confirm=no_confirm,
                dry_run=True,
            ).export_text()

        prompt.assert_not_called()
        confirm_query.assert_not_called()
        confirm_change.assert_not_called()
        gerrit.create_submit_manager.assert_not_called()
        expected = (
            "Major version bumps a real run holds back: 1"
            if no_confirm
            else "a real run asks for a value"
        )
        assert expected in output
