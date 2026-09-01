---
title: "Building the Code Map"
description: How build-map extracts a Code Map from source — the slices, attribution, promotion, and the two indexes that sit outside it.
---

This page is for contributors changing the extraction or mapping a new system. It
assumes the vocabulary from the [Code Map overview](/bamboo/architecture/code-map/) and
teaches the shape of the extraction; the reasoning behind each individual rule is in the
module docstrings, which are unusually long on purpose — 28% of
[`bamboo/codemap/`](https://github.com/tmaeno/bamboo/tree/master/bamboo/codemap) is
docstring, and it is where the measurements that justified each rule live. This page
links to them rather than restating them.

## The command

```bash
bamboo build-map --dry-run                     # extract, run the offline gates, write nothing
bamboo build-map                               # …and store it under its own labels
bamboo build-map --source-root ~/panda-server  # map a checkout instead of the installed release
```

`--dry-run` is the one to reach for while working on a recognizer: it prints the whole
report — coverage per slice and file, attribution, promotion, the indexes, the gates —
without touching the database.

Every node is stamped with the version it came from. For an installed distribution that
is its version; for a git checkout it is `git describe --tags --always --dirty`, because
a path does not identify content and a map built from a dirty tree cannot be reproduced.

## Extraction is recognizer-driven, not file-driven

Every module in the target packages is parsed, and a recognizer contributes wherever it
matches. There is no list of interesting files to keep up to date, so a new file is
picked up with no configuration — which matters, because a file list would have missed
five of PanDA's 102 rejection tags that live in a shared utility module rather than in
the four broker files anyone would have listed.

The tool is the standard library's `ast`. No external analyser is involved, and the
reason is measured rather than assumed: for the hardest slice, 92% of write sites resolve
with no interprocedural data flow at all, and the ratio was identical across two source
versions.

## The slices

Coverage below is from `build-map --dry-run` against `panda-server-source` 1.0.2. It will
move; the command is the authority.

| Slice | What it recognizes | Coverage |
|---|---|---|
| [`progress`](https://github.com/tmaeno/bamboo/blob/master/bamboo/codemap/panda/recognizers/progress.py) | `spec.attr = <value>` — the junctions the backward walk starts from | 1461/1534 |
| [`sqlwrite`](https://github.com/tmaeno/bamboo/blob/master/bamboo/codemap/panda/recognizers/sqlwrite.py) | `UPDATE … SET col=:bind` — the dominant write form; a knight decides and a proxy writes | 808/1162 |
| [`selection`](https://github.com/tmaeno/bamboo/blob/master/bamboo/codemap/panda/recognizers/selection.py) | `criteria=-diskIO` and `candidates passed <step>` — the filter chain | 109/109 |
| [`boundary`](https://github.com/tmaeno/bamboo/blob/master/bamboo/codemap/panda/recognizers/boundary.py) | `@request_validation` endpoints, and shared tables PanDA talks to DEFT through | 99/108 and 31/31 |
| [`errorcode`](https://github.com/tmaeno/bamboo/blob/master/bamboo/codemap/panda/recognizers/errorcode.py) | `EC_Kill = 100` — the constants that decode a value seen in a record | 178/194 |
| [`alias`](https://github.com/tmaeno/bamboo/blob/master/bamboo/codemap/panda/recognizers/alias.py) | `setOnHold()` seen from its callers, and helpers that *return* a value | 33/33 and 2/7 |
| [`trigger`](https://github.com/tmaeno/bamboo/blob/master/bamboo/codemap/panda/recognizers/trigger.py) | What makes a junction run, and whether missing it repairs itself | 273/498 reached |
| [`logfile`](https://github.com/tmaeno/bamboo/blob/master/bamboo/codemap/panda/recognizers/logfile.py) | Which log file a module's diagnostics land in | 315/368 resolved |

A slice reporting less than 100% is reporting honestly, not failing. `return-alias` at
2/7 is the clearest case: the other five helpers return computed values, and *"the map
does not decide this"* is the correct answer for them.

## Recognizers sit in three layers, and the fragile ones are not the ones you would guess

| Layer | Depends on | Example |
|---|---|---|
| 1 | Python itself | Path conditions, tracing an identifier to its assignment, function signatures |
| 2 | The project's conventions | `criteria=-<tag>`, `varMap[":status"]`, `*_JEDI` naming |
| 3 | One file's habits | `newScanSiteList = []` … `scanSiteList = newScanSiteList` |

**The value is concentrated in layers 2 and 3.** A bare `if` says nothing;
`criteria=-diskIO` *is* the identity of a filter stage and is also what production logs
carry per rejected site. Layer 1 is universal and nearly contentless.

Fragility runs inversely to how much the code declares about itself, and the ranking held
up against a real version change: `EC_Kill = 100` (an explicit enumeration) survives
anything; `varMap[":status"]` (a consistent idiom) is safe; `criteria=-tag` (a string
with a tag in it) is readable; `newScanSiteList` (a structural habit with no name) is the
weak one.

Which is why the biggest surprise of the build was that the slice everyone expected to be
fragile — brokerage, where that unnamed habit appears 26, 21, 9 and **0** times across
four sibling files — turned out not to need it. Measuring first found two declarations
that cover all four files between them, and the chain matcher the design had budgeted for
was never written.

## Attribution: which class does this write belong to?

`x.status = "cached"` settles nothing until you know what `x` is — eight PanDA classes
declare `status`.
[`attribution.py`](https://github.com/tmaeno/bamboo/blob/master/bamboo/codemap/panda/attribution.py)
answers that, and it lives outside the recognizers because two slices need the same
answer and would otherwise each implement it.

| Basis | Evidence | Share |
|---|---|---|
| `certain` | A single declaring class, `self` through the MRO, a constructor, a type annotation | 78% |
| `container` | The element type of the container it came out of, learned from the adder | 2% |
| `structural` | The set of attributes touched has exactly one declaring class that is a superset | 10% |
| `unresolved` | Undecided — **and still on the map**, because the write is real | 8% |

**Every basis reads evidence; none guesses.** A criterion that inferred the class from
the variable's *name* was implemented, measured at 2 sites out of 278, found to be the
only rule that needed a "not trustworthy" mark, and deleted.

Structural attribution is the interesting one. A variable's name is *what someone called
it*; the set of attributes the code touches on it is *what the code requires of it*. If
`.lfn .type .GUID .checksum .fsize .md5sum` are all read, only one class declares all
six. And because a class declaration and its usage are two independent readings of the
same fact, they can be compared — which is what the `structural-attribution-agrees` gate
does, over 277 sites with no disagreement.

:::caution[The trap that cost two debugging sessions]
`_attributes` on a PanDA spec class is **the list of database columns**, not the object's
attribute surface. `__slots__ = _attributes + ("Files", …)` says so. Methods and
containers are not in it, and reading it as "everything this object has" breaks both
structural attribution and container element learning.
:::

## Promotion: which attributes become subjects?

The spec classes declare 421 attributes and most are identifiers, timestamps and
counters. Promotion keeps the ones a diagnosis would ask *why* about.
[`promotion.py`](https://github.com/tmaeno/bamboo/blob/master/bamboo/codemap/panda/promotion.py)
applies five criteria; **any one of them promotes**, so they are a disjunction and adding
one cannot invalidate an existing answer.

| Criterion | Reading | Subjects |
|---|---|---|
| `1:state-gate-in-where` | Appears as a literal predicate in a query, so it gates another component's progress | 34 |
| `2:declared-vocabulary` | The code declares its value set outright | 2 |
| `3:closed-literal-set` | A majority of its writes name a value from a small set | 35 |
| `4:carried-into-a-promoted-subject` | A promoted subject's value passes through it | 7 |
| `5:gates-a-filter-stage` | A filter stage's exclusion condition reads it | 58 |

Two criteria that look plausible are deliberately absent. "Written under a guard" fires
on 79% of attributes and "mentioned in a log line" on 69% — they discriminate nothing and
are kept as corroboration for ranking only.

Criterion 1 needed narrowing three times, each time because something obviously wrong
came top of the list: `WHERE PandaID=:PandaID` is a lookup key,
`WHERE t.jediTaskID=f.jediTaskID` is a join, and a literal inside a subquery belongs to
the subquery. Criterion 3 needs the *majority*, not merely the presence of two literals,
or `jediTaskID` qualifies on the handful of its 528 writes that happen to be constants.

:::tip[The rule that keeps criterion 3 honest]
Its denominator is **every** write, including the ones whose value is computed. When the
attribute slice recorded only string literals, free-text fields such as `ddmErrorDiag`
looked like closed vocabularies — 3 literal writes out of 3 — and were promoted. Recording
all 34 of its writes took the ratio to 0.09 and it dropped out, which is the criterion
working as designed rather than a regression.
:::

## Two indexes deliberately outside promotion

Some fields are worth indexing without being subjects, and forcing them through promotion
would mean inventing a classifier nobody can justify.

**Diagnostic templates.** `ddmErrorDiag` has no value set to enumerate, so *"why is it
this value?"* is the wrong question — but *"this message was seen, who wrote it?"* is
exactly right, and that is an index from a template to an anchor. An index makes no claim
about the field, so it needs no criterion and no threshold. 73 templates over 22 fields.

**Enumeration bindings.** A code seen in a record cannot be decoded from its value alone:
`100` is `EC_Kill` in `taskbuffer`, `EC_Setupper` in `dataservice` and `EC_Watcher` in
`jobdispatcher`. The binding from field to enumeration is recoverable from the write —
`jobSpec.taskBufferErrorCode = ErrorCode.EC_Kill` states both halves — and the corpus does
that 79 times over five fields. Promotion drops those fields, correctly; the index keeps
the binding.

## Two rules worth knowing before you add a recognizer

**`passthrough(X)` may only name a place a value lives** — a spec attribute or a table
column, never a local variable. `passthrough(err_msg)` is a type error in the map's
vocabulary: it claims the backward walk continues at a node that cannot exist. An
unresolvable local is `runtime(...)` and tier 2.

**Weigh a mechanism by its harvest, not by its correctness.** Several rules that were
correct were deleted for yielding too little: naming-based attribution (2 sites of 278),
and a general "known column compared to a literal in an `if`" promotion rule that would
have matched 72 names over 540 sites led by `key`, `name` and `value` to gain one
subject. The standing alternative, when a shape cannot be resolved from evidence, is a
standard type annotation in the target system — checked by its own CI, useful to its own
developers — rather than another inference mechanism here.

## Adding a map for another system

The plugin contract is output-based: `prepare(source_root)` then `run() -> MapFragment`.
What is inside is the plugin's business, so a recognizer written with a different tool
composes with these on equal terms, and node identity being a semantic signature means
two tools' fragments merge without duplicating.

There is one condition on adopting a new map, and it is not about the extraction:
**a new map must come with machine-checkable gates against its own system's output.** A
system that does not emit enough to verify a map about itself should stay a `boundary`.
See [Verifying the map](/bamboo/architecture/code-map-verification/).
