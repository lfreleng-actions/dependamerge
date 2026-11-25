# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
A Gerrit change URL compares its source against a complete scope.

The candidates are every open change with the topic, every open change
of the source change's owner, or every open change on the server; each
is read in full or the run stops before anything is submitted.
"""

from typing import Any
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import pytest
import typer
from rich.console import Console

from dependamerge.cli import _generate_gerrit_override_sha, _handle_gerrit_merge
from dependamerge.gerrit import (
    QUERY_BATCH_CEILING,
    SIMILAR_SCAN_CEILING,
    GerritChangeComparator,
    GerritChangeInfo,
    GerritService,
)
from dependamerge.url_parser import ChangeSource, ParsedUrl

HOST = "gerrit.example.org"
SOURCE = 1
ACCOUNT_ID = 1000001
WORKFLOW = ".github/workflows/ci.yaml"
BUMP = "Chore: Bump actions/checkout from 4.1.0 to 4.2.0"
WHOLE_SERVER = "(status:open) status:open"
OWNER_QUERY = f"(owner:{ACCOUNT_ID}) status:open"


def _change(
    number: int,
    owner: str = "dependabot",
    account_id: int | None = ACCOUNT_ID,
    subject: str = BUMP,
    topic: str | None = None,
) -> dict[str, Any]:
    """A change as Gerrit lists it with DEFAULT_LIST_OPTIONS."""
    owner_data: dict[str, Any] = {"username": owner}
    if account_id is not None:
        owner_data["_account_id"] = account_id
    revision = f"rev{number}"
    data: dict[str, Any] = {
        "_number": number,
        "change_id": f"I{number:040d}",
        "project": f"proj-{number}",
        "subject": subject,
        "branch": "main",
        "status": "NEW",
        "owner": owner_data,
        "current_revision": revision,
        "revisions": {
            revision: {
                "commit": {"message": subject},
                "files": {WORKFLOW: {"status": "M"}},
            }
        },
        "permitted_labels": {"Code-Review": ["-2", "-1", "0", "+1", "+2"]},
    }
    if topic:
        data["topic"] = topic
    return data


class FakeGerrit:
    """Answers change reads and paged queries from fixed result sets."""

    def __init__(self, source: dict[str, Any], results: dict[str, list[Any]]):
        self.source = source
        self.results = results
        self.queries: list[str] = []
        self.is_authenticated = True

    def get(self, endpoint: str) -> Any:
        parts = urlsplit(endpoint)
        if parts.path != "/changes/":
            if parts.path.endswith("/mergeable"):
                return {"mergeable": True}
            return self.source
        params = parse_qs(parts.query)
        query = params["q"][0]
        offset, size = int(params["S"][0]), int(params["n"][0])
        self.queries.append(query)
        page = [dict(item) for item in self.results[query][offset : offset + size]]
        if page and offset + size < len(self.results[query]):
            page[-1]["_more_changes"] = True
        return page


class Runner:
    """Run ``merge`` on a change URL against a FakeGerrit."""

    def __init__(self) -> None:
        self.output = ""
        self.submit_manager = Mock()
        self.submit_manager.submit_changes_parallel.return_value = []
        self.credentials = Mock(is_valid=True, username="user", password="pass")
        self.credentials.auth_method_display.return_value = ".netrc file"

    def __call__(
        self,
        source: dict[str, Any],
        results: dict[str, list[Any]],
        *,
        threshold: float = 0.8,
        topic: str | None = None,
        include_human_prs: bool = False,
        override: str | None = None,
    ) -> tuple[FakeGerrit, str]:
        fake = FakeGerrit(source, results)
        with patch("dependamerge.gerrit.service.build_client", return_value=fake):
            service = GerritService(host=HOST)
        console = Console(record=True, width=200)
        parsed_url = ParsedUrl(
            source=ChangeSource.GERRIT,
            host=HOST,
            base_path=None,
            project=source["project"],
            change_number=source["_number"],
            original_url=f"https://{HOST}/c/{source['project']}/+/{source['_number']}",
        )
        with (
            patch(
                "dependamerge.cli.resolve_gerrit_credentials",
                return_value=self.credentials,
            ),
            patch("dependamerge.cli.create_gerrit_service", return_value=service),
            patch(
                "dependamerge.cli.create_gerrit_comparator",
                side_effect=lambda similarity_threshold: GerritChangeComparator(
                    similarity_threshold
                ),
            ),
            patch(
                "dependamerge.cli.create_submit_manager",
                return_value=self.submit_manager,
            ),
        ):
            try:
                _handle_gerrit_merge(
                    parsed_url=parsed_url,
                    no_confirm=True,
                    similarity_threshold=threshold,
                    verbose=False,
                    console=console,
                    topic=topic,
                    include_human_prs=include_human_prs,
                    override=override,
                    show_progress=False,
                )
            finally:
                self.output = console.export_text()
        return fake, self.output


@pytest.fixture
def run() -> Runner:
    """A fresh Runner."""
    return Runner()


def _many(start: int, count: int, **kwargs: Any) -> list[dict[str, Any]]:
    return [_change(number, **kwargs) for number in range(start, start + count)]


class TestTopicScope:
    """A topic scopes the candidates to every open change carrying it."""

    def test_every_change_of_a_large_topic_is_compared(self, run):
        """150 changes span two pages; all of them are compared."""
        topic_changes = _many(SOURCE, 150, topic="bump")
        fake, output = run(
            topic_changes[0], {"(topic:bump) status:open": topic_changes}
        )

        assert fake.queries.count("(topic:bump) status:open") >= 2
        assert "Compared against 150 open changes with topic 'bump'" in output
        assert "Found 149 similar changes" in output
        (submitted,), _ = run.submit_manager.submit_changes_parallel.call_args
        assert len(submitted) == 150

    def test_topic_above_the_ceiling_submits_nothing(self, run):
        """A topic over QUERY_BATCH_CEILING stops the run."""
        topic_changes = _many(SOURCE, QUERY_BATCH_CEILING + 1, topic="bump")

        with pytest.raises(typer.Exit) as exc_info:
            run(topic_changes[0], {"(topic:bump) status:open": topic_changes})

        assert exc_info.value.exit_code == 1
        assert "Could not read every open change with topic 'bump'" in run.output
        assert "--topic" in run.output
        run.submit_manager.submit_changes_parallel.assert_not_called()

    def test_unparsable_change_submits_nothing(self, run):
        """A listed change that does not parse stops the run."""
        topic_changes = [*_many(SOURCE, 3, topic="bump"), {"_number": "x"}]

        with pytest.raises(typer.Exit):
            run(topic_changes[0], {"(topic:bump) status:open": topic_changes})

        assert "could not be parsed" in run.output
        run.submit_manager.submit_changes_parallel.assert_not_called()

    def test_topic_flag_is_quoted(self, run):
        """--topic wins over the source's topic and is quoted as needed."""
        source = _change(SOURCE, topic="own")
        query = '(topic:"two words") status:open'
        fake, _ = run(source, {query: [source]}, topic="two words")

        assert fake.queries == [query]


class TestOwnerAndServerScope:
    """Without a topic, the owner scope holds whenever it suffices."""

    def test_default_threshold_reads_the_owner_by_account_id(self, run):
        """Above 0.75 only same-owner changes can match: owner scope."""
        owned = _many(SOURCE, 120)
        fake, output = run(owned[0], {OWNER_QUERY: owned})

        assert set(fake.queries) == {OWNER_QUERY}
        assert len(fake.queries) >= 2
        assert "Compared against 120 open changes owned by dependabot" in output

    def test_include_human_prs_takes_the_owner_scope(self, run):
        """A person's change matches only their own changes."""
        source = _change(SOURCE, owner="alice", subject="Fix: Tidy the CI job")
        override = _generate_gerrit_override_sha(
            GerritChangeInfo.from_api_response(source)
        )
        fake, _ = run(
            source,
            {OWNER_QUERY: [source]},
            threshold=0.5,
            include_human_prs=True,
            override=override,
        )

        assert fake.queries == [OWNER_QUERY]

    @pytest.mark.parametrize(
        ("threshold", "query"),
        [(0.7, WHOLE_SERVER), (0.75, WHOLE_SERVER), (0.76, OWNER_QUERY)],
    )
    def test_threshold_bound_chooses_the_scope(self, run, threshold, query):
        """At or below the bound another owner can match: whole server."""
        source = _change(SOURCE)
        fake, _ = run(source, {query: [source]}, threshold=threshold)

        assert fake.queries == [query]

    def test_source_without_account_id_reads_the_whole_server(self, run):
        """No account id to scope by: every open change is read."""
        source = _change(SOURCE, account_id=None)
        fake, output = run(source, {WHOLE_SERVER: [source, _change(2)]})

        assert fake.queries == [WHOLE_SERVER]
        assert f"Compared against 2 open changes on {HOST}" in output

    def test_whole_server_above_the_ceiling_submits_nothing(self, run):
        """More than SIMILAR_SCAN_CEILING open changes stops the run."""
        changes = _many(SOURCE, SIMILAR_SCAN_CEILING + 1, account_id=None)

        with pytest.raises(typer.Exit):
            run(changes[0], {WHOLE_SERVER: changes}, threshold=0.7)

        assert f"Could not read every open change on {HOST}" in run.output
        assert "query URL" in run.output
        run.submit_manager.submit_changes_parallel.assert_not_called()
