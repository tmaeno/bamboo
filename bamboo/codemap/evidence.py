"""Production evidence for the gates the source alone cannot answer.

A Code Map is built offline from source, and most of its gates check the code
against itself.  Two questions survive that and need production: whether what
the map calls an observable is actually emitted, and whether the transitions
the branch tables predict are the ones that happen.

The dev machine has no access to the deployment's log files, so the evidence
comes over PanDA's async grep API -- ``submit_grep_request`` puts a request in
a table, the target service's daemon runs ``rg`` against a file under its own
``logdir``, and ``get_result`` returns the output per machine.  That makes the
whole check automatable from a laptop, which is the difference between a
periodic check and one nobody runs.

**One distribution, two machine groups.**  ``panda-server-source`` ships as a
single version, but it is deployed as JEDI and as the httpd/wsgi server, and
they share only the database.  Task transitions and brokerage land in JEDI's
logs; job transitions, the pilot boundary and the error codes land in the
server's.  The package a junction lives in does not decide which: JEDI opens
its own TaskBuffer straight to the database, so ``db_proxy_mods`` code invoked
by a knight runs inside the JEDI process and logs there, while the same
package's ``api/v1`` runs under httpd.  Most junctions are on that crossing
side, so evidence is gathered from both services and unioned, and every
observation records where it came from.  That record is a deliverable rather
than bookkeeping -- "which component's log holds this" is what the map is for.

**Three ways a grep comes back empty, and only one of them is an answer.**
``rg`` exits 1 when a file has no match, 2 when it could not read the file at
all, and the processor truncates any result over a megabyte.  A missing file
and a truncated sample both look exactly like "production never emits this",
so they are kept distinct all the way into the gates: only exit 1 on an
untruncated result licenses the conclusion that something is absent.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# The two machine groups one PanDA distribution is deployed as.  These are the
# service names the API resolves to a set of live machines.
JEDI = "jedi"
SERVER = "server"
SERVICES = (JEDI, SERVER)

_SUBMIT_ENDPOINT = "async_process/submit_grep_request"
_RESULT_ENDPOINT = "async_process/get_result"

# The processor runs the grep on its own daemon cycle, so the first poll almost
# never finds it done.  Its subprocess timeout is 240s; leaving headroom above
# that keeps a slow-but-succeeding grep from being abandoned here.
POLL_INTERVAL_SECONDS = 5.0
POLL_TIMEOUT_SECONDS = 330.0

# ``rg`` exit codes.  The difference between 1 and 2 is the whole basis for
# distinguishing "not in production" from "we did not look".
_RG_NO_MATCH = 1

# PandaLogger formats every record as
# ``"%(asctime)s %(name)-12s: %(levelname)-8s %(message)s"``, so the level is
# the token after the first ``": "``.  Anchoring on that separator rather than
# on the bare word keeps a message that happens to contain "DEBUG" from being
# read as a DEBUG line.
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_LEVEL_ALTERNATION = "|".join(LEVELS)
ANY_LINE_PATTERN = rf": ({_LEVEL_ALTERNATION}) "
_LEVEL_IN_LINE = re.compile(rf": ({_LEVEL_ALTERNATION})\s")

# Ascending severity, so the lowest level actually present is the effective
# threshold production is running at.
_SEVERITY = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


class GrepQuery(BaseModel):
    """One question put to one service's logs."""

    pattern: str
    log_filename: str
    service: str

    def key(self) -> tuple[str, str, str]:
        return (self.pattern, self.log_filename, self.service)


class GrepResult(BaseModel):
    """What one machine answered.

    ``lines`` is empty for three different reasons and the gates must not
    conflate them, so the reason is carried rather than reduced to a count.
    """

    query: GrepQuery
    machine: str
    lines: list[str] = Field(default_factory=list)
    truncated: bool = False
    return_code: Optional[int] = None
    error: Optional[str] = None

    @property
    def conclusive(self) -> bool:
        """Whether an empty result may be read as "production does not emit this".

        Only when the tool ran, read the file, and returned everything it
        found.  A truncated result is a sample: something absent from it may
        still be in the part that was cut.
        """
        return self.error is None and not self.truncated and self.return_code in (0, _RG_NO_MATCH)


class Evidence(BaseModel):
    """Everything read from production for one ``check-map`` run.

    Saved to a file so that fetching and checking are separate steps: the
    gates then re-run offline against a fixed record, tests can use a fixture
    instead of the network, and the checking half stays developable without
    the API allowlist.
    """

    fetched_at: str
    results: list[GrepResult] = Field(default_factory=list)

    def matching(self, pattern: str, service: Optional[str] = None) -> list[GrepResult]:
        return [
            r
            for r in self.results
            if r.query.pattern == pattern and (service is None or r.query.service == service)
        ]

    def lines(self, pattern: str, service: Optional[str] = None) -> list[str]:
        """Every line any machine returned, in the order the machines answered.

        A union, not an intersection: the services run different knights, so a
        tag emitted on one machine and not another is normal.
        """
        return [line for result in self.matching(pattern, service) for line in result.lines]

    def conclusive(self, pattern: str, service: Optional[str] = None) -> bool:
        """Whether every machine's answer to this query can be read negatively."""
        results = self.matching(pattern, service)
        return bool(results) and all(r.conclusive for r in results)

    def services(self) -> set[str]:
        return {r.query.service for r in self.results}

    def failures(self) -> list[GrepResult]:
        """Queries that did not run -- a missing file, or a tool error."""
        return [r for r in self.results if r.error is not None]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "Evidence":
        return cls.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _call(method: str, endpoint: str, data: dict):
    from bamboo.utils.panda_client import _call_api  # noqa: PLC0415

    return _call_api(method, endpoint, data)


async def _submit(query: GrepQuery) -> str:
    """Queue one grep and return its request id."""
    data = {
        "pattern": query.pattern,
        "log_filename": query.log_filename,
        "service_name": query.service,
    }
    payload = await asyncio.to_thread(_call, "post", _SUBMIT_ENDPOINT, data)
    request_id = (payload or {}).get("request_id")
    if not request_id:
        raise RuntimeError(f"no request_id returned for {query.key()}")
    return request_id


async def _await_result(
    request_id: str,
    timeout: float = POLL_TIMEOUT_SECONDS,
    interval: float = POLL_INTERVAL_SECONDS,
) -> dict:
    """Poll until every expected machine has answered, or give up.

    Returning the last partial payload on timeout rather than raising is
    deliberate: one unresponsive machine should not discard the answers the
    others already gave, and the per-machine record shows which is which.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    payload: dict = {}
    while True:
        payload = await asyncio.to_thread(
            _call, "get", _RESULT_ENDPOINT, {"request_id": request_id}
        ) or {}
        if payload.get("overall_status") == "complete":
            return payload
        if asyncio.get_running_loop().time() >= deadline:
            logger.warning(
                "evidence: request %s still pending after %.0fs; using partial results",
                request_id, timeout,
            )
            return payload
        await asyncio.sleep(interval)


def _results_from(query: GrepQuery, payload: dict) -> list[GrepResult]:
    """Turn one API payload into per-machine results.

    A machine that never answered becomes an explicit error rather than an
    empty result, because silence and "found nothing" are different facts.
    """
    answered = {}
    for row in payload.get("results") or []:
        machine = row.get("machine_name") or "unknown"
        stdout = row.get("result") or ""
        error = row.get("error_msg") or None
        stderr = (row.get("stderr") or "").strip()
        return_code = row.get("return_code")
        if error is None and return_code not in (0, _RG_NO_MATCH):
            # Exit 2 is the log file not being readable -- a wrong filename,
            # usually.  Surfacing stderr turns that into a one-step fix.
            error = stderr or f"grep exited {return_code}"
        answered[machine] = GrepResult(
            query=query,
            machine=machine,
            lines=stdout.splitlines(),
            truncated=bool(row.get("truncated")),
            return_code=return_code,
            error=error,
        )
    for machine in payload.get("expected_machines") or []:
        if machine not in answered:
            answered[machine] = GrepResult(
                query=query, machine=machine, error="no result returned"
            )
    return list(answered.values())


async def run_query(query: GrepQuery, timeout: float = POLL_TIMEOUT_SECONDS) -> list[GrepResult]:
    """Submit one grep and collect what every machine in the service returned."""
    logger.info("evidence: %s on %s for %r", query.service, query.log_filename, query.pattern)
    request_id = await _submit(query)
    payload = await _await_result(request_id, timeout=timeout)
    return _results_from(query, payload)


async def collect(queries: list[GrepQuery], timeout: float = POLL_TIMEOUT_SECONDS) -> Evidence:
    """Run every query and record the answers.

    Queries go out concurrently: each spends most of its time waiting for a
    daemon cycle on the far side, and they are independent.  A query that
    raises is recorded as a failed result instead of aborting the run, so one
    bad filename does not cost the rest of the evidence.
    """
    async def one(query: GrepQuery) -> list[GrepResult]:
        try:
            return await run_query(query, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            logger.warning("evidence: %s failed: %s", query.key(), exc)
            return [GrepResult(query=query, machine="-", error=str(exc))]

    gathered = await asyncio.gather(*(one(q) for q in queries))
    return Evidence(
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        results=[result for batch in gathered for result in batch],
    )


def sample_queries(log_files: dict[str, str]) -> list[GrepQuery]:
    """The queries that establish what each service's logs look like.

    One per service, matching any well-formed log line.  The result is the
    first megabyte of that service's log, which is enough to see which levels
    are being emitted -- and, because it matches on the format itself, it
    fails loudly if the format is not what the map assumed.
    """
    return [
        GrepQuery(pattern=ANY_LINE_PATTERN, log_filename=filename, service=service)
        for service, filename in sorted(log_files.items())
    ]


# ---------------------------------------------------------------------------
# Reading the sample
# ---------------------------------------------------------------------------


def service_for_module(rel_path: str) -> str:
    """Which machine group's logs a module's output lands in -- approximately.

    The package is only a first approximation, and it is wrong exactly where
    most junctions are: JEDI opens its own TaskBuffer, so ``pandaserver``'s
    ``db_proxy_mods`` code runs inside the JEDI process when a knight calls it
    and logs to JEDI's files, while the same package's ``api/v1`` runs under
    httpd.  Getting that right needs the observation, not the path.

    It is sound for the slices that use it here: brokerage's filter chains
    live in ``pandajedi/jedibrokerage`` and run in JEDI, with no caller from
    the other group.  Anything reaching past those should query both services
    and union rather than trust this.
    """
    return SERVER if rel_path.startswith("pandaserver") else JEDI


def level_histogram(evidence: Evidence, service: Optional[str] = None) -> Counter:
    """Count log lines by level in the sample taken from *service*."""
    counts: Counter = Counter()
    for line in evidence.lines(ANY_LINE_PATTERN, service):
        found = _LEVEL_IN_LINE.search(line)
        if found:
            counts[found.group(1)] += 1
    return counts


def effective_level(evidence: Evidence, service: Optional[str] = None) -> Optional[str]:
    """The lowest level actually present, or None when nothing was sampled.

    This is a lower bound in the safe direction.  Seeing a DEBUG line proves
    DEBUG is enabled; not seeing one in a sample only suggests it is not, so
    callers treat the absence as evidence rather than proof and say so.
    """
    counts = level_histogram(evidence, service)
    if not counts:
        return None
    return min(counts, key=lambda level: _SEVERITY[level])


def below_threshold(level: Optional[str], threshold: Optional[str]) -> bool:
    """Whether a recorded emit level would be suppressed at *threshold*."""
    if level is None or threshold is None:
        return False
    if level.upper() not in _SEVERITY or threshold.upper() not in _SEVERITY:
        return False
    return _SEVERITY[level.upper()] < _SEVERITY[threshold.upper()]
