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
``rg`` exits 1 when a file has no match, the file may not be there at all, and
a result can be cut short -- by the megabyte the processor stores, or by the
match cap the query carries.  A missing file and a truncated sample both look
exactly like "production never emits this", so they are kept distinct all the
way into the gates: only exit 1 on an untruncated result licenses the
conclusion that something is absent.

**Every query is bounded.**  ``panda-DBProxy.log`` is six gigabytes and the
processor buffers a matcher's whole output before storing a slice of it, so an
unbounded question is one the daemon pays for in memory.  Each query carries a
match cap and a tail window; the window also makes the answer recent, which is
what a check reported over a time window wants anyway.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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

# A log file that is not there is a fact, not a failure: PandaLogger opens the
# file when its logger first emits, so its absence says the code never ran.
_MISSING_FILE = re.compile(r"No such file or directory", re.IGNORECASE)

# Bounds sent with every query.  A sample is all any of these gates needs, and
# the alternative is asking a six-gigabyte file for all of itself.
DEFAULT_MAX_MATCHES = 5000
DEFAULT_TAIL_BYTES = 64 * 1024 * 1024

# How many matched lines a result keeps.  Separate from the match cap because
# the two answer different questions: the cap bounds what the server reads, and
# this bounds what is written down.  The level query needs no lines at all --
# its answer is a histogram -- and storing them made a 112 MB evidence file out
# of what fits in twenty rows.
DEFAULT_KEEP_LINES = 2000

# The level query is deliberately unselective: it matches every well-formed log
# line, because what it measures is which levels appear at all.  That makes the
# window, not the cap, the thing that has to be small -- ``tail -c N | rg -m M``
# returns the *first* M matches inside the window, so a wide window with a cap
# yields the oldest lines in it and comes back truncated, which is worse than
# useless: a truncated sample cannot license "production does not emit this".
# A window this size holds a few thousand lines, so the cap is never reached,
# the sample runs up to the present, and the result is conclusive.
LEVEL_TAIL_BYTES = 256 * 1024
LEVEL_MAX_MATCHES = 20000

# Patterns for the questions that need to read lines.  Both mirror what the
# selection recognizer reads out of the source, which is the point: the same
# convention seen from the other side.
TAG_PATTERN = r"criteria=-[A-Za-z0-9_.]+"
FUNNEL_PATTERN = r"candidates passed"

# ``set task_status=running`` -- what the knights log when they move a task.
# The database keeps only the current status and the one before it, so this is
# the only place a *sequence* of transitions can be recovered from, and it is
# written at the very sites the map already holds as junctions.
TRANSITION_PATTERN = r"set task_status="

# Bounds for the transition query.  Narrower window and, unlike the other
# reading queries, ``keep_lines`` equal to the cap: the answer is a sequence, so
# a trimmed result is not a smaller sample of it but a different one -- dropping
# lines from the middle invents transitions that never happened.
TRANSITION_TAIL_BYTES = 8 * 1024 * 1024
TRANSITION_MAX_MATCHES = 2000

# Which subject those lines are about.  ``task_status`` in the message and
# ``status`` on the spec are the same field under two names -- a convention of
# this corpus, like the patterns above, and the join between what production
# logged and what the map extracted.
TRANSITION_SUBJECT = "JediTaskSpec.status"

_TAG_IN_LINE = re.compile(r"\bcriteria=(-[\w.]+)")
_FUNNEL_IN_LINE = re.compile(r"candidates passed(?:\s+for)?\s+(.+?)\s*$")

# ``<jediTaskID=52266181 datasetID=685030095>`` -- what the log wrapper puts in
# front of every line of one chain run, and so the only trustworthy way to tell
# where one run ends and the next begins.
_RUN_KEY = re.compile(r"<([^>]*)>")

# The three things a transition line has to yield.  The timestamp is compared as
# text: PandaLogger writes ``asctime`` as ``%Y-%m-%d %H:%M:%S,%f``, which sorts
# lexicographically, so ordering needs no parsing and cannot fail on a locale.
_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
_TASK_ID = re.compile(r"jediTaskID=(\d+)")
_NEW_STATUS = re.compile(r"set task_status=([A-Za-z0-9_.]+)")

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


def missing_file(result: "GrepResult") -> bool:
    """Whether this machine's answer is "the log file is not here".

    Its own predicate because two readers want it and they want opposite things
    from it.  For a file every machine reports missing, the absence is the
    strongest statement production makes -- PandaLogger creates the file on the
    logger's first emit, so no file means that code has never run here.  For one
    machine out of a service that was asked speculatively, it is not an answer
    at all and has to be dropped before the rest are read, or a query that was
    always going to miss on one group would make every group inconclusive.
    """
    return result.error is not None and bool(_MISSING_FILE.search(result.error))


class GrepQuery(BaseModel):
    """One question put to one service's logs, bounded on both axes.

    Unbounded is not an option here.  ``panda-DBProxy.log`` is six gigabytes,
    and the processor buffers a matcher's whole output before capping what it
    stores, so a pattern matching most lines costs the daemon that much
    memory.  Both bounds come back as ``truncated``, which is what keeps a
    partial answer from being read as an absent one.
    """

    pattern: str
    log_filename: str
    service: str
    max_matches: int = DEFAULT_MAX_MATCHES
    tail_bytes: int = DEFAULT_TAIL_BYTES
    keep_lines: int = DEFAULT_KEEP_LINES

    def key(self) -> tuple[str, str, str]:
        return (self.pattern, self.log_filename, self.service)


class GrepResult(BaseModel):
    """What one machine answered.

    ``lines`` is empty for three different reasons and the gates must not
    conflate them, so the reason is carried rather than reduced to a count.
    """

    query: GrepQuery
    machine: str
    lines: list[str] = Field(
        default_factory=list,
        description="The matched lines that were kept -- see the query's keep_lines.",
    )
    matched: int = Field(
        default=0,
        description=(
            "How many lines matched, whether or not they were kept.  Read "
            "instead of len(lines) wherever the question is 'did anything "
            "match', so that a query keeping no lines still answers it."
        ),
    )
    level_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Lines per log level over everything that matched, computed before "
            "any were dropped.  The level question's whole answer, at a size "
            "that can be written down."
        ),
    )
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

    @property
    def complete(self) -> bool:
        """Whether the *lines* here are all of them, not just all the tool found.

        A second, stricter condition than :attr:`conclusive`, and the two are
        about different halves of the trip: conclusive says the server returned
        everything it matched, complete says nothing was dropped writing it
        down.  A gate that searches the lines for something and concludes it is
        missing needs this one -- with only conclusive, trimming to
        ``keep_lines`` would turn a kept sample into a false absence.
        """
        return self.conclusive and len(self.lines) == self.matched


class JobRecords(BaseModel):
    """The jobs of one task, reduced to the fields a gate reads.

    A second kind of evidence, because a record is not a log line and reading
    it needs none of the grep machinery: there is no pattern, no per-machine
    union, and no ``truncated`` -- the API answers with the rows or it errors.

    Compacted on the way in rather than on the way out.  The full descriptions
    of one busy task run to megabytes, and an evidence file has already once
    reached 112 MB by writing down everything it was handed; what any gate here
    needs is a handful of coded fields per job.

    What bounds this sample is how many tasks were asked about, which is a
    number rather than a flag, so :class:`Evidence` records it alongside how
    many were available to ask.
    """

    task_id: str
    jobs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="One dict per job, holding only KEPT_JOB_FIELDS.",
    )
    error: Optional[str] = None


class TaskRecord(BaseModel):
    """One task's row, reduced to the fields an investigation reads.

    The cheapest evidence in the system, and the only kind with no window: a
    column is returned whole by one API call, where a log line has to be found
    in a rotation under a byte cap and its absence proves nothing.  What it
    carries that a log line does not is ``errordialog`` -- the message the
    deciding branch wrote about itself, still in the record hours later.

    Not free of its own asymmetry.  The field holds the *last* message written
    to it, so naming a branch is proof and naming none is not, which is why
    this is used to confirm and never to eliminate.
    """

    task_id: str
    fields: dict[str, Any] = Field(
        default_factory=dict, description="Only KEPT_TASK_FIELDS."
    )
    error: Optional[str] = None


class Evidence(BaseModel):
    """Everything read from production for one ``check-map`` run.

    Saved to a file so that fetching and checking are separate steps: the
    gates then re-run offline against a fixed record, tests can use a fixture
    instead of the network, and the checking half stays developable without
    the API allowlist.
    """

    fetched_at: str
    results: list[GrepResult] = Field(default_factory=list)
    records: list[JobRecords] = Field(
        default_factory=list,
        description=(
            "Job rows, one entry per task asked about.  Defaulted so that an "
            "evidence file written before records existed still loads."
        ),
    )
    tasks_available: int = Field(
        default=0,
        description=(
            "How many task ids the log evidence offered when the records were "
            "fetched.  With ``len(records)`` this is the whole sample story -- "
            "a record query has no ``truncated`` to carry it."
        ),
    )
    tasks: list[TaskRecord] = Field(
        default_factory=list,
        description=(
            "Task rows, one per task asked about.  Kept apart from ``records`` "
            "rather than generalised into it: the two answer different "
            "questions of different endpoints, and a list of jobs and a single "
            "row would have to be told apart by shape."
        ),
    )

    def matching(
        self,
        pattern: str,
        service: Optional[str] = None,
        log_filename: Optional[str] = None,
    ) -> list[GrepResult]:
        return [
            r
            for r in self.results
            if r.query.pattern == pattern
            and (service is None or r.query.service == service)
            and (log_filename is None or r.query.log_filename == log_filename)
        ]

    def lines(self, pattern: str, **where) -> list[str]:
        """Every line any machine returned, in the order the machines answered.

        A union, not an intersection: the services run different knights, so a
        tag emitted on one machine and not another is normal.
        """
        return [line for result in self.matching(pattern, **where) for line in result.lines]

    def conclusive(self, pattern: str, **where) -> bool:
        """Whether every machine's answer to this query can be read negatively."""
        results = self.matching(pattern, **where)
        return bool(results) and all(r.conclusive for r in results)

    def complete(self, pattern: str, **where) -> bool:
        """Whether every matched line is here to be searched.

        What a gate needs before reporting that something is *not* in the log:
        conclusive alone allows a sample that was trimmed on the way in.
        """
        results = self.matching(pattern, **where)
        return bool(results) and all(r.complete for r in results)

    def services(self) -> set[str]:
        return {r.query.service for r in self.results}

    def log_filenames(self) -> set[str]:
        return {r.query.log_filename for r in self.results}

    def file_status(self, log_filename: str) -> str:
        """``present`` | ``absent`` | ``unknown`` for one log file.

        ``absent`` is the strongest thing production can say about a piece of
        the map: PandaLogger creates a file the first time its logger emits,
        so no file means that logger has never emitted on any machine asked --
        the code path is not merely quiet, it has not run in this deployment.
        Distinguishing it from ``unknown`` matters because a query that failed
        for some other reason must not be read as that.
        """
        results = self.matching(ANY_LINE_PATTERN, log_filename=log_filename)
        if not results:
            return "unknown"
        if any(r.error is None for r in results):
            return "present"
        if all(missing_file(r) for r in results):
            return "absent"
        return "unknown"

    def failures(self) -> list[GrepResult]:
        """Queries that did not run, excluding a file simply not being there.

        A missing log file is an answer -- ``code-paths-are-live`` reports it
        as one -- so listing it here alongside "not authorized" would bury the
        errors that mean the check itself could not be trusted.
        """
        return [
            r
            for r in self.results
            if r.error is not None and not _MISSING_FILE.search(r.error)
        ]

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
        "max_matches": query.max_matches,
        "tail_bytes": query.tail_bytes,
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
        lines = stdout.splitlines()
        answered[machine] = GrepResult(
            query=query,
            machine=machine,
            # Counted and tallied over everything that came back, then trimmed:
            # the histogram has to describe the whole sample, not the part that
            # happened to be kept.
            matched=len(lines),
            level_counts=dict(_levels_in(lines)),
            lines=lines[: query.keep_lines],
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


# What a job row is reduced to before it is written down.  The coded fields the
# index has anything to say about, plus enough to tell one job from another when
# a finding has to be followed up by hand.
KEPT_JOB_FIELDS = (
    "PandaID",
    "jobStatus",
    "computingSite",
    "pilotErrorCode",
    "exeErrorCode",
    "supErrorCode",
    "ddmErrorCode",
    "brokerageErrorCode",
    "jobDispatcherErrorCode",
    "taskBufferErrorCode",
)

# How many tasks the record query asks about by default.  One round trip per
# task against the production API, and the direction these records support is
# positive -- a code seen is a code used -- so a sample costs coverage and
# nothing else.
DEFAULT_TASK_SAMPLE = 50


async def collect_job_records(
    task_ids: list[str], sample: int = DEFAULT_TASK_SAMPLE
) -> tuple[list[JobRecords], int]:
    """Fetch the jobs of up to *sample* tasks, compacted.

    The task ids are not asked of the API: they come from the log evidence,
    which parses ``jediTaskID`` out of the transition lines.  That matters for
    more than convenience.  Every endpoint that returns a *population* of tasks
    scopes it to one ``userName`` -- ``get_tasks_detailed_info_since`` seeds its
    criteria with the caller's DN and only a plain filter value can displace it,
    and ``get_tasks_modified_since`` pins it in SQL -- so there is no query for
    "the tasks production ran".  The per-id endpoints have no such check, and
    the logs hand over thousands of ids with no user scoping at all.

    Taken from the front of the list rather than at random: the caller passes
    them in a fixed order, and a reproducible sample is worth more here than an
    unbiased one, given the direction these records support.
    """
    from bamboo.utils.panda_client import get_job_descriptions  # noqa: PLC0415

    async def one(task_id: str) -> JobRecords:
        try:
            jobs = await get_job_descriptions(int(task_id), unsuccessful_only=True)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            logger.warning("evidence: job records for task %s failed: %s", task_id, exc)
            return JobRecords(task_id=task_id, error=str(exc))
        return JobRecords(
            task_id=task_id,
            jobs=[{k: job.get(k) for k in KEPT_JOB_FIELDS} for job in jobs],
        )

    chosen = sorted(task_ids)[:sample]
    return list(await asyncio.gather(*(one(t) for t in chosen))), len(task_ids)


#: What a task row is kept down to.  The status pair says where the row is and
#: where it came from, ``errordialog`` is the message the deciding branch left
#: about itself, and the rest place the task without pulling in the parameter
#: blob -- a full task description is large and nothing here reads it.
KEPT_TASK_FIELDS = (
    "jeditaskid",
    "status",
    "oldstatus",
    "errordialog",
    "modificationtime",
    "prodsourcelabel",
    "username",
)


async def collect_task_records(task_ids: list[str]) -> list[TaskRecord]:
    """Fetch one row per task, compacted.

    ``task/get_detailed_info`` takes an id and checks no owner, which is what
    makes this usable at all -- every endpoint returning a *population* scopes
    it to the caller's DN.  The keys PanDA returns are lower-cased column names,
    so they are matched case-insensitively rather than assumed.
    """
    from bamboo.utils.panda_client import fetch_task_data  # noqa: PLC0415

    async def one(task_id: str) -> TaskRecord:
        try:
            data = await fetch_task_data(task_id)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            logger.warning("evidence: task record for %s failed: %s", task_id, exc)
            return TaskRecord(task_id=task_id, error=str(exc))
        lowered = {str(k).lower(): v for k, v in (data or {}).items()}
        return TaskRecord(
            task_id=task_id,
            fields={k: lowered.get(k) for k in KEPT_TASK_FIELDS if lowered.get(k) is not None},
        )

    return list(await asyncio.gather(*(one(t) for t in task_ids)))


def observed_codes(evidence: Evidence) -> dict[str, Counter]:
    """``{job field: Counter(code)}`` over every job row on record.

    Zero and null are dropped: an error-code field is zero when there was no
    error, so counting it would make every index look as though it were missing
    an entry for the ordinary case.
    """
    seen: dict[str, Counter] = {}
    for record in evidence.records:
        for job in record.jobs:
            for field, value in job.items():
                if field in ("PandaID", "jobStatus", "computingSite"):
                    continue
                if value in (None, 0, "0", ""):
                    continue
                seen.setdefault(field, Counter())[value] += 1
    return seen


def sample_queries(targets: dict[str, str]) -> list[GrepQuery]:
    """The queries that establish what each log file contains.

    One per file, matching any well-formed log line.  Per file rather than per
    service because PanDA writes one file per logger: a question about
    brokerage goes to ``panda-AtlasProdJobBroker.log`` and comes back narrow,
    where the same question against a whole service's output would arrive as a
    truncated slice of everything.  Matching on the log format itself also
    makes the query fail loudly if the format is not what the map assumed.

    *targets* maps log filename to the service that writes it.
    """
    return [
        GrepQuery(
            pattern=ANY_LINE_PATTERN,
            log_filename=filename,
            service=service,
            tail_bytes=LEVEL_TAIL_BYTES,
            max_matches=LEVEL_MAX_MATCHES,
            keep_lines=0,
        )
        for filename, service in sorted(targets.items())
    ]


def reading_queries(targets: dict[str, str]) -> list[GrepQuery]:
    """The queries whose answers are the lines themselves.

    Rejection tags and funnel steps, both selective enough that a whole window
    fits under the cap -- which is what lets their absence mean something.  The
    level query cannot make that claim and these can.
    """
    return [
        GrepQuery(pattern=pattern, log_filename=filename, service=service)
        for pattern in (TAG_PATTERN, FUNNEL_PATTERN)
        for filename, service in sorted(targets.items())
    ]


def sample_state(
    evidence: Evidence,
    pattern: str,
    log_filenames,
    needs_lines: bool = True,
) -> tuple[int, int]:
    """``(files whose answer is whole, files that answered at all)``.

    How much of what was asked actually came back, per query.  The gates
    already establish this to decide what they may conclude, and kept it to
    themselves; a reader cannot tell which verdicts are load-bearing without
    it, so it is computed once here and reported.

    *needs_lines* picks which standard applies.  A gate searching the text for
    something needs :attr:`GrepResult.complete` -- every matched line written
    down.  The level histogram needs only :attr:`GrepResult.conclusive`,
    because its answer is the tally, which is computed over everything that
    matched before any line is dropped.
    """
    asked = [f for f in log_filenames if evidence.matching(pattern, log_filename=f)]
    test = evidence.complete if needs_lines else evidence.conclusive
    return sum(1 for f in asked if test(pattern, log_filename=f)), len(asked)


def bounds_hit(evidence: Evidence) -> tuple[int, int, set[tuple[int, int]]]:
    """``(answers that hit a bound, answers in all, the bounds involved)``.

    The bounds are reported with the count because they are the reason: a
    reader deciding whether to trust an absence needs to know the window and
    the cap that produced it, not just that something was cut.
    """
    capped = [r for r in evidence.results if r.truncated]
    return (
        len(capped),
        len(evidence.results),
        {(r.query.tail_bytes, r.query.max_matches) for r in capped},
    )


def transition_queries(targets: dict[str, str]) -> list[GrepQuery]:
    """The queries that recover which task statuses production actually set.

    Put to every log file the map names rather than to the ones whose modules
    are known to write status.  Two reasons, and the second is the point of the
    exercise: the pattern is selective enough that asking widely costs little,
    and "which component's log holds this" is the answer the map exists to give
    -- deriving the file list from what the map already believes would only
    confirm the belief.
    """
    return [
        GrepQuery(
            pattern=TRANSITION_PATTERN,
            log_filename=filename,
            service=service,
            tail_bytes=TRANSITION_TAIL_BYTES,
            max_matches=TRANSITION_MAX_MATCHES,
            keep_lines=TRANSITION_MAX_MATCHES,
        )
        for filename, service in sorted(targets.items())
    ]


def observed_task_status(evidence: Evidence) -> dict[str, list[tuple[str, str, str]]]:
    """``{jediTaskID: [(timestamp, status, log file), ...]}``, in time order.

    Merged across files and machines, because one task's history is spread over
    both: the refiner, the generator, the post-processor and the watchdog each
    write to their own file, and a knight runs on whichever machine picked the
    task up.  Ordering therefore has to come from the timestamps rather than
    from the order the lines were returned in.

    Runs of one repeated value are collapsed.  Two knights logging ``running``
    in sequence is not a transition, and counting it as one would put a
    self-loop in every task's history.
    """
    seen: dict[str, list[tuple[str, str, str]]] = {}
    for result in evidence.matching(TRANSITION_PATTERN):
        for line in result.lines:
            stamp = _STAMP.match(line)
            task = _TASK_ID.search(line)
            status = _NEW_STATUS.search(line)
            if not (stamp and task and status):
                continue
            seen.setdefault(task.group(1), []).append(
                (stamp.group(1), status.group(1), result.query.log_filename)
            )
    histories: dict[str, list[tuple[str, str, str]]] = {}
    for task, rows in seen.items():
        ordered: list[tuple[str, str, str]] = []
        for row in sorted(rows):
            if not ordered or ordered[-1][1] != row[1]:
                ordered.append(row)
        histories[task] = ordered
    return histories


def observed_departures(histories: dict[str, list[tuple[str, str, str]]]) -> Counter:
    """How often a task was seen to leave each status.

    Sound under an incomplete sample, which is why it is separated from the
    adjacent pairs.  Seeing a task in ``finishing`` and later in anything else
    proves it left ``finishing``, whether or not the step in between was
    sampled; the *pair* is what a gap can invent.
    """
    counts: Counter = Counter()
    for rows in histories.values():
        # Every status but the last: something moved the task on from each of
        # them, and the last one is where the sample stops rather than where
        # the task stopped.
        for _stamp, status, _file in rows[:-1]:
            counts[status] += 1
    return counts


def observed_pairs(histories: dict[str, list[tuple[str, str, str]]]) -> Counter:
    """How often each adjacent ``(from, to)`` was observed.

    **A report, not evidence for a gate.**  A transition the sample did not
    catch -- a file that was not queried, a window that cut mid-history --
    leaves its neighbours next to each other, and ``a -> c`` then looks like a
    step the code takes.  Nothing in the log marks the gap.
    """
    counts: Counter = Counter()
    for rows in histories.values():
        for (_, before, _), (_, after, _) in zip(rows, rows[1:], strict=False):
            counts[(before, after)] += 1
    return counts


def observed_tags(evidence: Evidence, log_filename: Optional[str] = None) -> Counter:
    """Rejection tags production actually emitted, and how often."""
    counts: Counter = Counter()
    for line in evidence.lines(TAG_PATTERN, log_filename=log_filename):
        for tag in _TAG_IN_LINE.findall(line):
            counts[tag] += 1
    return counts


def observed_runs(evidence: Evidence, log_filename: str) -> list[list[str]]:
    """Funnel step labels per chain run, in the order production logged them.

    Splitting by run is not a refinement, it is the difference between a
    working check and noise.  The chain runs once per task and dataset, and a
    sample spans thousands of them; read as one sequence, the last step of one
    run followed by the first step of the next is indistinguishable from a
    transposition.  Read that way against real logs this reported 2416
    disagreements, none of them real.

    The run key is the log wrapper's own prefix --
    ``<jediTaskID=52266181 datasetID=685030095>`` -- so the segmentation comes
    from the code's own idea of what one run is rather than from a guess about
    timing.
    """
    runs: dict[str, list[str]] = {}
    for line in evidence.lines(FUNNEL_PATTERN, log_filename=log_filename):
        found = _FUNNEL_IN_LINE.search(line)
        if not found:
            continue
        key = _RUN_KEY.search(line)
        runs.setdefault(key.group(1) if key else "", []).append(found.group(1).strip())
    return list(runs.values())


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


def _levels_in(lines: list[str]) -> Counter:
    """Tally log levels over *lines*."""
    counts: Counter = Counter()
    for line in lines:
        found = _LEVEL_IN_LINE.search(line)
        if found:
            counts[found.group(1)] += 1
    return counts


def level_histogram(evidence: Evidence, **where) -> Counter:
    """Count log lines by level in the sample, filtered by service or file.

    Read from each result's stored tally rather than recounted from its lines,
    because the level query keeps no lines: the tally is what survives, and it
    covers the whole sample instead of the part that fit.
    """
    counts: Counter = Counter()
    for result in evidence.matching(ANY_LINE_PATTERN, **where):
        counts.update(result.level_counts)
    return counts


def effective_level(evidence: Evidence, **where) -> Optional[str]:
    """The lowest level actually present, or None when nothing was sampled.

    Measured per log file, because that is the grain PanDA configures: two
    loggers in the same service can sit at different levels, so a threshold
    taken from one and applied to the other would drop observables on a
    number that was never established for them.

    A lower bound in the safe direction.  Seeing a DEBUG line proves DEBUG is
    enabled; not seeing one in a sample only suggests it is not, so callers
    treat the absence as evidence rather than proof and say so.
    """
    counts = level_histogram(evidence, **where)
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
