# Task Analysis

Diagnose a failing task against the accumulated knowledge base. Where `bamboo
investigate` co-drives a *live* investigation turn-by-turn, `bamboo analyze` is
the *automated, read-only* counterpart: give it a task and it queries the graph +
vector databases for matching causes and similar past incidents, reasons over
them with the LLM, and returns a root cause, confidence, and recommended
resolution — replaying stored investigation procedures along the way when a
known cause matches. Because it is automatic (no operator watching), `analyze` is
strictly **read-only**: it never executes state-changing code, even from a stored
procedure (those are surfaced as suggestions to run in `investigate`). See
[EXECUTION_TRUST.md](EXECUTION_TRUST.md).

## Quick start

```bash
# Fetch the task live from PanDA by jediTaskID
bamboo analyze --task-id <jediTaskID>

# …or analyze a local task_data JSON file (no PanDA fetch)
bamboo analyze --task-data path/to/task.json
```

One of `--task-id` / `--task-data` is required; they are mutually exclusive.

## What you get

```
================================================================================
TASK ANALYSIS RESULTS
================================================================================

Task ID: 12345
Root Cause: input dataset exceeds the per-job file limit
Confidence: 78.00%

Resolution: split the input dataset or raise nFilesPerJob, then resubmit

Explanation:
<full LLM reasoning narrative…>

--------------------------------------------------------------------------------
PRESCRIPTION
--------------------------------------------------------------------------------
  • Resubmit with a smaller nFilesPerJob
  • Verify the dataset is complete
  Suggested options: --nFilesPerJob 5
```

For a **known incident** the command then prints an **email draft** and asks
`Do you approve this analysis? (yes/no/edit)`. For a **novel incident** it instead
writes a seed draft for review (see below).

The structured result (`AnalysisResult`) carries: `root_cause`, `confidence`
(0–1), `resolution`, `explanation`, `supporting_evidence`, `capability_gaps`,
`unmatched_symptoms`, and `metadata`. Use `--output <file>` to save it as JSON.

## How it works

1. **Feature extraction** — pull symptoms, task features/context, environment
   factors, and components out of the task fields.
2. **Retrieval** — query the **graph DB** for causes matching those clues, and the
   **vector DB** for summaries of similar past incidents.
3. **Phase 1 — synthesis** — the LLM proposes a `root_cause` + `confidence` +
   `resolution` from the retrieved evidence and the task data.
4. **Exploratory enrichment** — triggered when confidence is below the threshold
   (0.5) **or** there are unmatched symptoms (a novel incident always explores,
   because the LLM can look confident from the raw error alone). bamboo runs MCP
   tools to gather more signals, recording any `capability_gaps` (useful
   investigation directions no available tool could address), then re-synthesizes.
5. **Phase 2 — procedure-driven** — if a matched cause has stored investigation
   **Procedures**, bamboo replays them (the exact code captured by a prior
   `investigate`/`populate`) and re-synthesizes with the results; otherwise it
   notes that manual investigation is recommended. This phase is **read-only**:
   a stored procedure whose code would call a state-changing (`read_only=False`) tool is *not* run —
   it is skipped and surfaced as a suggestion (the explorer's tools are filtered to
   read-only, with a `ToolProxy` allow-set as the runtime backstop). See
   [EXECUTION_TRUST.md](EXECUTION_TRUST.md).

## Known vs. novel incidents

- **Known incident** (`unmatched_symptoms` empty) — every symptom matched the
  knowledge base. bamboo composes a prescription (action hints, optional command
  template) and an **email draft**, then prompts you to approve / edit / reject.
- **Novel incident** (`unmatched_symptoms` non-empty) — at least one symptom has
  no precedent. Rather than emit an unvalidated answer, bamboo writes a **seed
  draft** JSON to `--drafts-dir` (default `drafts/`) tagged `reviewed: false` with
  a review hint. Promote it into the knowledge base with the standard curation
  flow — `bamboo review-drafts` then `bamboo batch-populate` (see
  [KNOWLEDGE_POPULATION.md](KNOWLEDGE_POPULATION.md)).

## Comparing tasks (pattern mode)

Instead of analyzing one task, surface what several failing tasks have in common:

```bash
bamboo analyze --task-id 12345 \
  --compare-task-id 23456 --compare-task-id 34567 \
  --min-occurrences 2
```

This prints the **common subgraph** — nodes and edges shared by at least
`--min-occurrences` tasks (default 2). All compared tasks must have been
**populated** beforehand (via `bamboo populate` / `batch-populate`); pattern mode
reads their stored graphs, it does not analyze them afresh.

## CLI options

| Option | Type | Default | Description |
|---|---|---|---|
| `--task-id` | int | — | PanDA jediTaskID; fetch task data live. Mutually exclusive with `--task-data` |
| `--task-data` | path | — | Task data JSON file. Mutually exclusive with `--task-id` |
| `--external-data` | path | — | Extra environmental data JSON to fold into the analysis |
| `--output` | path | — | Save the full `AnalysisResult` as JSON |
| `--compare-task-id` | int (repeatable) | — | Show the common subgraph across these tasks (pattern mode) |
| `--min-occurrences` | int ≥ 2 | 2 | Min tasks that must share an edge for it to appear in pattern output |
| `--post-to-mattermost` | channel id | — | Also post the result to a Mattermost channel |
| `--rebuild-docs` | flag | off | Force a full rebuild of the doc index cache |
| `--debug-report` | path | — | Write a JSON trace of every analysis step |
| `-v, --verbose` | flag | off | DEBUG logging |
| `--drafts-dir` | path | `drafts` | Where to write seed drafts for novel incidents |

## Inputs & prerequisites

- **Task input**: `--task-id` fetches live from PanDA (needs the PanDA
  environment set — see [PANDA_INTEGRATION.md](PANDA_INTEGRATION.md)), or
  `--task-data` reads a local JSON file.
- **Knowledge base**: analyze *queries* the graph + vector databases, so results
  are only as good as what's been populated. A single-task analyze runs against an
  empty KB but will simply find no precedent (→ novel-incident path). Populate the
  KB first with [`bamboo populate`](KNOWLEDGE_POPULATION.md) or capture live
  investigations with [`bamboo investigate`](INVESTIGATE.md).

## Posting to chat

Add `--post-to-mattermost <channelID>` to also publish the analysis as a
Mattermost card (colored by confidence). Requires the `bamboo[mattermost]` extra
and `MATTERMOST_URL` / `MATTERMOST_TOKEN` — see
[MATTERMOST.md](MATTERMOST.md).

## Debugging

- `--debug-report <file>` — writes a per-step JSON trace (extracted clues,
  per-symptom graph probes, vector hits, the root-cause analysis, and the
  novel-incident decision) — the fastest way to see *why* a task was or wasn't
  flagged as novel.
- `--rebuild-docs` — clears and rebuilds the documentation index cache.
- `-v` — DEBUG logging.

## How it relates to `populate` and `investigate`

Same knowledge graph, different drivers:
- `populate` / `investigate` — **write** knowledge (frozen email vs. live dialog).
- `analyze` — **reads** that knowledge to diagnose a new task; Phase 2 re-executes
  the procedures those commands captured, and novel incidents feed back into the
  curation pipeline.
