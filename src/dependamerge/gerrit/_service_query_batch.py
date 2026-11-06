# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
The query behind a Gerrit query URL's batch.

:class:`_GerritQueryBatchMixin` fetches every open change a query
expression matches, or refuses.  Unlike the searches in
``_service_queries``, which cap their results and skip a change that
does not parse, it never returns part of a result set: a query URL acts
on exactly what Gerrit shows for that query, so a partial set would act
on changes the user did not preview, or leave out ones they did.

As for the sibling mixins, every attribute read here is established by
``GerritService.__init__``, and the logger name is unchanged so records
keep reporting as ``dependamerge.gerrit.service``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import logging
from collections.abc import Mapping
from urllib.parse import quote

from dependamerge.gerrit.client import GerritRestClient, GerritRestError
from dependamerge.gerrit.models import GerritChangeInfo

from ._service_errors import GerritServiceError
from ._submitted_together import submitted_together_problems

log = logging.getLogger("dependamerge.gerrit.service")

# The most open changes a query URL may act on.  A query matching more
# is refused rather than truncated: acting on part of a result set the
# user previewed in the Gerrit UI is the failure this guards against.
QUERY_BATCH_CEILING = 500

# Page size for a query batch, and the options it requests.  Wider than
# DEFAULT_LIST_OPTIONS where the preview needs it: DETAILED_LABELS
# carries each change's permitted votes and every reviewer's vote with
# the label's range, SUBMITTABLE and CURRENT_ACTIONS its submit state, so
# permissions can be reported per change, and SUBMIT_REQUIREMENTS what
# Gerrit 3.x reports as blocking it.  The file list is dropped, since no
# similarity scoring runs over a query.
_QUERY_BATCH_PAGE_SIZE = 100
_SUBMIT_REQUIREMENTS = "SUBMIT_REQUIREMENTS"
QUERY_BATCH_OPTIONS: list[str] = [
    "CURRENT_REVISION",
    "CURRENT_COMMIT",
    "CURRENT_ACTIONS",
    "LABELS",
    "DETAILED_LABELS",
    "DETAILED_ACCOUNTS",
    "SUBMITTABLE",
    _SUBMIT_REQUIREMENTS,
]


# How often a multi-page result is read before refusing.  Two reads
# that agree accept it; a third allows for one change landing mid-read.
_QUERY_BATCH_READS = 3

# The fields a listed change must carry for the batch to act on it: the
# shared parser defaults each, which would hide a malformed entry.
_QUERY_BATCH_REQUIRED = ("_number", "change_id", "project", "current_revision")


class _GerritQueryBatchMixin:
    """The complete-or-refuse query shared into ``GerritService``."""

    # Established by GerritService.__init__.
    _client: GerritRestClient
    host: str
    base_path: str | None

    def get_open_changes_for_query(
        self,
        expression: str,
        ceiling: int = QUERY_BATCH_CEILING,
        options: list[str] | None = None,
    ) -> list[GerritChangeInfo]:
        """
        Get every open change a query expression matches, or refuse.

        The expression is sent as ``(<expression>) status:open`` and every
        page is fetched.  Unlike the searches in ``_service_queries``, nothing is
        dropped quietly: a result over ``ceiling`` or a change that does
        not parse raises, so a caller never acts on part of the set.

        Offset pages are not a snapshot: a change updated or closed while
        they are read shifts the later pages, so one change can be read
        twice and another skipped.  A result spanning several pages is therefore read
        again and accepted only once two reads agree and hold no repeat.

        Args:
            expression: A Gerrit query expression, as a query URL holds.
            ceiling: The most changes to accept.
            options: Optional list of query options.  Defaults to
                QUERY_BATCH_OPTIONS, less SUBMIT_REQUIREMENTS on a server
                older than Gerrit 3.5, which refuses that option.

        Returns:
            Every open change the query matches, in Gerrit's order.

        Raises:
            GerritRestError: If a page cannot be fetched.
            GerritServiceError: If the query matches more than
                ``ceiling`` changes, any returned change cannot be
                parsed, or the result set keeps changing while read.
        """
        if options is None:
            try:
                return self.get_open_changes_for_query(
                    expression, ceiling, QUERY_BATCH_OPTIONS
                )
            except GerritRestError as exc:
                # The body names the option refused; the message, which
                # carries the request path, always would.
                body = exc.response_body or ""
                if exc.status_code != 400 or _SUBMIT_REQUIREMENTS not in body:
                    raise
                log.debug(
                    "Gerrit refused %s; querying without it", _SUBMIT_REQUIREMENTS
                )
                options = [o for o in QUERY_BATCH_OPTIONS if o != _SUBMIT_REQUIREMENTS]

        query = f"({expression}) status:open"
        previous: list[int] | None = None
        for _attempt in range(_QUERY_BATCH_READS):
            changes, pages = self._read_query_pages(query, ceiling, options)
            numbers = [change.number for change in changes]
            if len(set(numbers)) != len(numbers):
                log.debug("Query %r repeated a change across pages; rereading", query)
                previous = None
                continue
            if pages == 1 or (
                previous is not None and sorted(previous) == sorted(numbers)
            ):
                return changes
            previous = numbers

        raise GerritServiceError(
            f"The open changes matching query {query!r} kept changing while "
            f"being read ({_QUERY_BATCH_READS} reads disagreed); refusing to "
            "act on a result set that may be incomplete. Try again once the "
            "changes settle"
        )

    def submitted_together_problems(
        self, change_number: int, scope: Mapping[int, str]
    ) -> list[str]:
        """Why submitting a change could act outside *scope*, if it could.

        See :func:`~._submitted_together.submitted_together_problems`.
        """
        return submitted_together_problems(self._client, change_number, scope)

    def _read_query_pages(
        self,
        query: str,
        ceiling: int,
        options: list[str],
    ) -> tuple[list[GerritChangeInfo], int]:
        """Read every page of *query* once; return its changes and page count."""
        # Encoded so a '&' or '#' in the expression stays inside q=.
        encoded = quote(query, safe=":/")
        changes: list[GerritChangeInfo] = []
        offset = 0
        pages = 0

        while True:
            params = [
                f"q={encoded}",
                f"n={_QUERY_BATCH_PAGE_SIZE}",
                f"S={offset}",
                *(f"o={opt}" for opt in options),
            ]
            endpoint = "/changes/?" + "&".join(params)
            log.debug("Querying changes: %s", endpoint)

            data = self._client.get(endpoint)
            pages += 1
            if not isinstance(data, list):
                raise GerritServiceError(
                    f"Gerrit returned {type(data).__name__} rather than a "
                    f"list of changes for query {query!r}; the response "
                    "schema may have changed"
                )

            changes.extend(self._parse_query_item(item, query) for item in data)

            if len(changes) > ceiling:
                raise GerritServiceError(
                    f"Query {query!r} matches more than {ceiling} open "
                    "changes; narrow it (for example with project: or "
                    "branch:) rather than act on part of the result set"
                )

            more = (
                bool(data)
                and isinstance(data[-1], dict)
                and bool(data[-1].get("_more_changes"))
            )
            if not data or (len(data) < _QUERY_BATCH_PAGE_SIZE and not more):
                break
            offset += len(data)

        return changes, pages

    def _parse_query_item(self, item: object, query: str) -> GerritChangeInfo:
        """Parse one listed change strictly, or refuse the whole batch.

        The shared parser fills a missing number, project or revision with
        a default, which would let a malformed entry through as a change
        that cannot be submitted, or whose override binds to no patchset.
        """
        number = item.get("_number") if isinstance(item, dict) else None
        try:
            if not isinstance(item, dict):
                raise ValueError(f"expected an object, got {type(item).__name__}")
            missing = [key for key in _QUERY_BATCH_REQUIRED if not item.get(key)]
            if missing:
                raise ValueError(f"missing or empty {', '.join(missing)}")
            if not _is_number(item["_number"]):
                raise ValueError(f"_number {item['_number']!r} is not a change number")
            return GerritChangeInfo.from_api_response(
                item, host=self.host, base_path=self.base_path
            )
        except Exception as exc:
            raise GerritServiceError(
                f"Gerrit returned a change ({number or 'unnumbered'}) "
                f"for query {query!r} that could not be parsed: {exc}; "
                "refusing to act on part of the result set"
            ) from exc


def _is_number(value: object) -> bool:
    """Whether *value* is a positive change number (bool excluded)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
