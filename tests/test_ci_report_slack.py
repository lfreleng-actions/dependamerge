# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the Slack digest built from a results document."""

import json
from typing import Any

from dependamerge.ci_report import SCHEMA_VERSION
from dependamerge.ci_report.__main__ import main
from dependamerge.ci_report.slack import (
    MAX_TEXT_CHARS,
    escape,
    fit_lines,
    render_payload,
    text_length,
)

RUN_URL = "https://github.com/acme/.github/actions/runs/1"


def _entry(number: int, status: str, **overrides: Any) -> dict[str, Any]:
    entry = {
        "repository": "acme/widget",
        "number": number,
        "title": f"Chore: Bump foo to {number}",
        "url": f"https://github.com/acme/widget/pull/{number}",
        "author": "dependabot[bot]",
        "status": status,
        "reason": "boom" if status in ("failed", "blocked", "skipped") else None,
        "details": [],
        "warning": None,
    }
    entry.update(overrides)
    return entry


def _document(entries: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    counts = dict.fromkeys(
        (
            "merged",
            "auto_merge_pending",
            "failed",
            "blocked",
            "unsettled",
            "skipped",
            "closed",
        ),
        0,
    )
    for entry in entries:
        counts[entry["status"]] += 1
    document = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "dependamerge", "version": "0.15.0"},
        "target": "https://github.com/acme",
        "scope": "owner",
        "preview": False,
        "dry_run": False,
        "selection": None,
        "counts": counts,
        "results": entries,
        "scan_errors": [],
    }
    document.update(overrides)
    return document


def _texts(payload: dict[str, Any]) -> list[str]:
    texts = []
    for block in payload["blocks"]:
        if "text" in block:
            texts.append(block["text"]["text"])
        for element in block.get("elements", []):
            texts.append(element["text"])
    return texts


class TestEscape:
    def test_mentions_and_links_cannot_be_injected(self):
        assert escape("<!channel> & <https://x|y>") == (
            "&lt;!channel&gt; &amp; &lt;https://x|y&gt;"
        )

    def test_line_breaks_are_flattened(self):
        assert escape("a\nb\t c") == "a b c"


class TestFitLines:
    def test_everything_fits_when_it_can(self):
        assert fit_lines("*H*", ["a", "b"]) == "*H*\na\nb"

    def test_lines_are_shed_with_a_count(self):
        text = fit_lines("*H*", ["x" * 10] * 10, budget=40)
        assert text_length(text) <= 40
        assert text.endswith("more")
        assert text.startswith("*H*\nxxxxxxxxxx")

    def test_utf16_counting_includes_astral_emoji(self):
        # Each U+1F916 is two UTF-16 units though one code point.
        text = fit_lines("H", ["\U0001f916" * 10] * 5, budget=50)
        assert text_length(text) <= 50

    def test_a_huge_list_keeps_a_prefix_and_counts_the_rest(self):
        lines = [f"line {n}" for n in range(100_000)]
        text = fit_lines("*H*", lines)
        kept = text.splitlines()[1:-1]
        assert kept == lines[: len(kept)]
        assert text.endswith(f"\u2026 and {len(lines) - len(kept)} more")
        assert text_length(text) <= MAX_TEXT_CHARS


class TestRenderPayload:
    def test_structure(self):
        payload = render_payload(
            _document([_entry(1, "merged"), _entry(2, "failed"), _entry(3, "blocked")]),
            channel="C123",
            run_url=RUN_URL,
        )
        assert payload["channel"] == "C123"
        assert payload["unfurl_links"] is False
        assert payload["blocks"][0]["type"] == "header"
        assert payload["blocks"][0]["text"]["text"].endswith("acme")
        texts = _texts(payload)
        joined = "\n".join(texts)
        # Failures before merges, and the counts line names each.
        assert joined.index("*Failed*") < joined.index("*Merged*")
        assert "1 merged" in texts[1] and "1 failed" in texts[1]
        assert "<https://github.com/acme/widget/pull/2|acme/widget#2>" in joined
        assert "\u2014 _boom_" in joined
        assert f"<{RUN_URL}|View workflow run>" in texts[-1]
        assert "dependamerge 0.15.0" in texts[-1]

    def test_dry_run_is_labelled(self):
        payload = render_payload(
            _document([], dry_run=True, preview=True), channel="C1"
        )
        assert "(dry run)" in payload["blocks"][0]["text"]["text"]
        assert "No pull requests to process." in _texts(payload)[1]

    def test_preview_outcomes_are_predictions(self):
        payload = render_payload(
            _document(
                [_entry(1, "merged"), _entry(2, "failed")],
                dry_run=True,
                preview=True,
            ),
            channel="C1",
        )
        joined = "\n".join(_texts(payload))
        assert "1 mergeable" in joined and "1 would fail" in joined
        assert "*Mergeable*" in joined and "*Would fail*" in joined
        assert "merged" not in joined.lower().replace("mergeable", "")

    def test_sections_are_verbatim(self):
        # Without verbatim Slack would auto-link bare URLs in titles.
        payload = render_payload(
            _document([_entry(1, "failed", title="see https://evil.example")]),
            channel="C1",
        )
        sections = [b for b in payload["blocks"] if b["type"] == "section"]
        assert sections
        assert all(block["text"]["verbatim"] is True for block in sections)

    def test_counted_only_outcomes_have_no_listing(self):
        payload = render_payload(
            _document([_entry(1, "auto_merge_pending"), _entry(2, "unsettled")]),
            channel="C1",
        )
        joined = "\n".join(_texts(payload))
        assert "1 auto-merge pending" in joined
        assert "acme/widget#1" not in joined

    def test_a_huge_run_stays_within_slack_limits(self):
        entries = [
            _entry(n, "merged", title="\U0001f916 " + "long title " * 20)
            for n in range(500)
        ] + [_entry(n, "failed", reason="r " * 400) for n in range(500, 700)]
        payload = render_payload(_document(entries), channel="C1", run_url=RUN_URL)
        assert len(payload["blocks"]) <= 50
        for text in _texts(payload):
            assert text_length(text) <= MAX_TEXT_CHARS
        assert any("more" in text for text in _texts(payload))

    def test_hostile_titles_cannot_mention_or_link(self):
        payload = render_payload(
            _document([_entry(1, "merged", title="<!channel> see <https://evil|x>")]),
            channel="C1",
        )
        joined = "\n".join(_texts(payload))
        assert "<!channel>" not in joined
        assert "<https://evil" not in joined

    def test_a_malformed_run_url_is_left_out(self):
        payload = render_payload(
            _document([]), channel="C1", run_url="https://x|<!here>"
        )
        assert "View workflow run" not in _texts(payload)[-1]

    def test_selection_and_scan_errors_are_shown(self):
        payload = render_payload(
            _document(
                [],
                selection="Excluding: beta",
                scan_errors=["Error scanning repository acme/x: boom"],
            ),
            channel="C1",
        )
        joined = "\n".join(_texts(payload))
        assert "_Excluding: beta_" in joined
        assert "Repositories not scanned" in joined


class TestSlackCommand:
    def test_prints_one_line_of_json(self, tmp_path, capsys):
        path = tmp_path / "results.json"
        path.write_text(json.dumps(_document([_entry(1, "failed")])))
        code = main(["slack", str(path), "--channel", "C9", "--run-url", RUN_URL])
        assert code == 0
        out = capsys.readouterr().out
        assert out.count("\n") == 1
        assert json.loads(out)["channel"] == "C9"

    def test_channel_is_required(self, tmp_path, capsys):
        path = tmp_path / "results.json"
        path.write_text(json.dumps(_document([])))
        assert main(["slack", str(path)]) == 2
