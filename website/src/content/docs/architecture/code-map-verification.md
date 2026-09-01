---
title: "Verifying the Code Map"
description: How the Code Map checks itself offline and against production, and how to read what check-map reports.
---

A map extracted by a machine from source it does not control needs to be able to say
where it is wrong. This page covers how it does that, and — the part an operator needs —
how to read the report without over-reading it.

There is no corpus of confirmed root causes to grade against, so verification cannot
work by comparing answers. It works by a different principle:

> **Find the places where the same fact is expressed twice, and compare the expressions.**

PanDA states most things twice, which is what makes this possible at all. A gate is one
such comparison. A failing gate means one of the two expressions is wrong — **not
necessarily the extraction**, and several gates have found real problems in PanDA itself.

## Two kinds of gate

**Code-internal (i)** needs only the source, so it runs inside `build-map` and every
build is verified before anything is stored.

**Production comparison (ii)** needs logs or records, and runs in `check-map`. It catches
what the source cannot reveal: version skew, and things that are true of the deployment
rather than of the code.

```bash
bamboo build-map --dry-run                    # the (i) gates
bamboo check-map                              # the (ii) gates, against stored evidence
bamboo check-map --fetch                      # …after querying production first
bamboo check-map --strict                     # non-zero exit if something asks for a change
bamboo diff-map --source-root ~/panda-server  # neither: compare two source trees
```

## The asymmetry of absence

This is the single idea to hold on to when reading any production result, and it is not a
limitation of the sample size.

**Seeing a line proves the code emitted it**, whatever fraction of the log was read. That
direction is safe and it is where all the value is.

**Not seeing a line proves nothing** — and a *complete* sample does not fix it, because
writing the line requires the branch to fire. A rejection that happens twice a month is
absent from a perfect reading of a one-hour window. So "the map has X and production does
not" has two causes that no amount of reading separates: the map is stale, or that branch
simply did not run.

Exactly one negative claim survives this, and it survives for a structural reason:

:::tip[Why `code-paths-are-live` is the strongest check here]
A log file is created the first time its logger emits. Its **absence is not conditioned
on any branch** — it means that logger has never emitted, on any machine asked. Every
other absence depends on a branch firing; this one does not.

It has already been confirmed in the field. The map concluded from source and evidence
alone that `GenJobBroker` and `SimpleTaskSetupper` were not running in this deployment,
and both turned out to be for non-ATLAS instances.
:::

A consequence worth stating because the opposite is tempting: **narrowing the query window
until the sample is complete does not strengthen these checks.** It manufactures false
findings in two of them, and it costs the positive direction — the rejection tag that
exposed a real extraction blind spot appeared six times in a wide window and would have
been missed in a narrow one.

## The gate catalogue

### In `build-map` — source only

| Gate | Question |
|---|---|
| `value-enum-referenced` | Is every constant in the index read by something? |
| `namespace-disambiguates` | Does one value identify one constant inside its namespace? |
| `boundary-ownership-param` | Does an ownership check name a parameter the endpoint declares? |
| `structural-attribution-agrees` | Do the declared type and the object's usage name one class? |
| `declared-status-is-written` | Does some writer produce every value the code declares? |
| `map-references-resolve` | Does every edge land on a node the map contains? |

One more check runs alongside them without being one of them:
`selection-steps-explained` asks whether every cut the funnel counts has a reason the map
can read. It is printed on its own line rather than in the gate table because it is
computed by the selection slice, not by `gates.py` — and it is silent when it passes.

### In `check-map` — against production

| Gate | Question | Direction |
|---|---|---|
| `log-format-recognised` | Is the log in the format the map assumed? | both |
| `code-paths-are-live` | Are the map's code paths running here? | **negative, and the only one** |
| `observables-are-emitted` | Do the promised lines survive the log level? | positive |
| `tags-are-known` | Is every cut production makes in the map? | positive |
| `funnel-steps-are-known` | Is every step production counts a cut at in the map? | positive |
| `funnel-order-matches` | Does production run the chain in map order? | majority |
| `transitions-are-explained` | Can the map produce every status production set? | positive |
| `error-codes-are-known` | Is every code production records in the index? | positive |

`transitions-are-explained` is conformance checking in the process-mining sense: the map
is the model, the knights' own `set task_status=` lines are the trace. It is independent
of how the code is written, which makes it the strongest check on completeness — a status
the system was observed to enter and no branch can produce is a blind spot the system
named itself.

## Reading a report

A failing gate asks for one of three different things, and printing them all as "FAIL"
is what made the report unreadable before they were separated.

| Verdict | Means | What to do |
|---|---|---|
| `FAIL` | **The map is wrong or incomplete** | Fix the extraction |
| `DIFFERS` | The map is right; **this deployment differs** | Nothing. Leave it out of a strategy |
| `BROKEN` | **The check did not run** | Fix the query. It is not a statement about the target system |

`--strict` exits non-zero on `FAIL` and `BROKEN` only. A deployment difference is not a
reason to fail a pipeline: the map and the source agree, and the deployment simply does
not exercise that code.

The report opens with the verdict and the findings, so the first screen answers "is
anything wrong":

```
sample     levels  complete  30/30 file(s) answered whole
           tags    PARTIAL   0/3 file(s) answered whole
           records PARTIAL   50/8687 task(s) asked, 544 job row(s)
           55 of 666 answer(s) hit a bound (8MB window / 2000 matches per machine, …)
           absence in a bounded sample proves nothing -- and for a line a
           branch has to fire to write, neither does a complete one.

verdict    0 broken checks · 0 map defects · 1 deployment difference
           the map describes code this deployment never runs.
```

**Read the `sample` block before believing any negative.** A gate whose sample is
`PARTIAL` could not have concluded an absence even if it wanted to, and the gates
themselves enforce that — the block exists so that a reader can see which conclusions are
load-bearing without having to know which gates guard themselves.

Add `--full` to expand every folded row.

## Where the evidence comes from

The dev machine cannot read the deployment's log files, but it does not need to: PanDA's
async grep API runs `rg` against a file under the target service's own log directory and
returns the output per machine. That makes the whole check runnable from a laptop, which
is the difference between a periodic check and one nobody runs.

**One distribution, two machine groups.** `panda-server-source` ships as a single version
but is deployed as JEDI and as the httpd/wsgi server, sharing only the database. Which
package a junction lives in does *not* decide which group runs it — JEDI opens its own
TaskBuffer, so `db_proxy_mods` code invoked by a knight runs inside the JEDI process and
logs there. Most junctions are on that crossing, so both services are asked and the
answers are unioned, with the origin recorded. That record is a deliverable rather than
bookkeeping: *which component's log holds this* is the question the map exists to answer.

**Every query is bounded**, and the bound is reported. `panda-DBProxy.log` is six
gigabytes and the service buffers a matcher's whole output before storing a slice, so an
unbounded question is one the daemon pays for in memory. Finding that led to a fix in
PanDA itself — the API now takes a match cap and a tail window, which cost callers
nothing because the result was already being truncated on storage.

**Job records are a second round, and have to be.** The task ids come from the transition
lines the first round returned. That is not a convenience: every PanDA endpoint that
returns a *population* of tasks scopes it to a single `userName`, so there is no query
for "the tasks production ran" — while the per-id endpoints have no such restriction and
the logs hand over thousands of ids with no scoping at all.

Evidence is always written to a file before it is checked. Fetching is slow and needs an
allowlist; checking should not be. With the two separated, gates re-run offline against a
fixed record, tests use a fixture instead of the network, and the checking half stays
developable in an environment that cannot reach production at all.

## What the gates have actually caught

Worth knowing, because it calibrates how much to trust them.

- **A cut the map could not explain.** Production logged `criteria=-link_unusable` and the
  map had no stage for it: the tag was assigned to a variable and interpolated later, a
  shape the recognizer did not read. Two more of the same form were in the same file.
- **A status nothing on the map could produce.** `aborted`, set 104 times in the window,
  came from a helper's return value — which showed that the "follow a helper one level"
  rule the design had for path conditions was needed on the outcome side too.
- **Two thresholds that had already drifted**, found by `diff-map` between two source
  trees. Both were on `-disk`, one of brokerage's most frequently cited rejection reasons,
  and one removed a documented bypass — so a map built from the older tree would have
  offered a no-longer-existent escape hatch as the reason a site survived.
- **Problems in PanDA rather than in the map**: unreferenced constants, an `int`/`str`
  double comparison, a copy-pasted comment, and the unbounded grep output above.
- **Its own false positives.** Three separate over-readings in the funnel-order check were
  found and removed — chains sharing a log file, a step the map places at two points, and
  a lone reversed observation being read as a majority.

## Keeping this page honest

Every figure here comes from `build-map` or `check-map` output at a stated version, and
the commands are the authority. Where a page enumerates something the code also
enumerates, `tests/test_docs.py` checks that the list is complete — added after the graph
schema page spent months claiming eighteen node types when there were twenty-three.
