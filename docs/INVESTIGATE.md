# Co-Investigation Mode

Live, human-driven dialog for investigating an ongoing incident. Where `bamboo populate` ingests a *retrospective* (the investigation already happened, captured as an email), `bamboo investigate` co-drives a *live* investigation turn-by-turn with a human, capturing each step as executable orchestration code that future analyze runs can re-execute deterministically.

## Quick start

```bash
bamboo investigate --task-id <jediTaskID>
```

This:
1. Fetches the task's `task_data` from PanDA, displays its status + `errorDialog` + key signals.
2. Surfaces past similar incidents from the knowledge graph (proactive hypothesis).
3. Enters a turn-by-turn dialog loop. Each human turn becomes either a **tool** call (bamboo does something) or **narration** (you're sharing a finding) — a binary intent classifier routes per turn.
4. On `/done`, prompts you for the cause + resolution, shows a diff against the existing graph, and commits.

## A typical session

```
$ bamboo investigate --task-id 12345

╭─ task under investigation ─────────────────────────────────────╮
│ status: failed                                                 │
│ errorDialog: action=set_exhausted reason=low_efficiency...     │
│ signals:                                                       │
│   status: failed                                               │
│   gshare: production                                           │
│   ramCount: 4096                                               │
╰────────────────────────────────────────────────────────────────╯

╭─ past similar incidents ───────────────────────────────────────╮
│ most-similar past root cause: input dataset exceeds file limit │
│ confidence: 0.74                                               │
│ (this is a hypothesis from past incidents — confirm or chase   │
│ a different lead.)                                             │
╰────────────────────────────────────────────────────────────────╯

> show me the failed scout jobs

╭─ proposed orchestration ───────────────────────────────────────╮
│ strategy:  inspect_failed_scout_jobs                           │
│ summary:   Fetch scout-job details for this task.              │
│ trigger:                                                       │
│   - scout phase has failed jobs                                │
│   - no scout-job inspection yet                                │
╰────────────────────────────────────────────────────────────────╯
jobs = await tools.get_scout_job_details(task_id=task_id)
return {"jobs": jobs}

Proceed? [y/N/edit]: y
[... result displayed ...]

> the failed jobs all OOMed on site X
[... noted; extracted a tentative Cause + Task_Feature ...]

> /done
[procedure list shown; cause prompt; resolution prompt; commit diff; commit]
```

## Slash commands

| Command | What it does |
|---|---|
| `/done` | Finish the session and run the end-of-session form |
| `/abandon` | Quit without confirming a cause; commits Procedures as **tentative** |
| `/undo` | Roll back the last turn's mutation (single-level snapshot) |
| `/skip` | No-op turn — useful after a rejected tool call |
| `/tool <text>` | Force tool intent (skip the classifier — useful when it misreads) |
| `/show-graph` | Print the current partial graph |
| `/show-tools` | Print the unified tool registry (PanDA MCP + internal queries) |

## CLI options

| Option | Default | Description |
|---|---|---|
| `--task-id INT` | — | PanDA jediTaskID (typical entry) |
| `--task-data PATH` | — | JSON file (alternative to `--task-id`) |
| `--symptom TEXT` | — | Free-text symptom for non-PanDA scenarios |
| `--save PATH` | `~/.bamboo/investigations/<sid>.json` | Checkpoint after each turn |
| `--resume PATH` | — | Resume from a prior `--save` file |
| `--max-turns INT` | 30 | Safety cap |
| `--dry-run` | off | Walk through the session but never commit |
| `-v, --verbose` | off | DEBUG logging |

## What's captured (and why it's reusable)

Each tool turn produces **one orchestration block** — a small async Python function body that calls one or more MCP tools. When the session commits, each block becomes a `Procedure` node with the source code stored on `metadata.orchestration_code` along with `code_summary`, `has_side_effects`, and the per-incident `trigger_signals`.

When `bamboo analyze` later encounters a similar task and Phase 2 retrieves the Procedure for the matched Cause, it **prefers the stored code over regenerating new code** — the exact bytes that worked last time run this time. This is more reproducible than the prior approach of regenerating orchestration code from a procedure description every analyze run, and faster (skips one LLM call per procedure). Procedures captured by `bamboo populate` (no stored code) still work through the regenerate path — zero regression.

The full design rationale is in [the plan file](../.claude/plans/i-m-planning-to-evolve-bubbly-blossom.md).

## Safety model

- **Tool calls with side effects** (anything that hits PanDA) always show a pre-execution confirmation panel with the proposed code + summary + trigger signals. `y` runs, `N` aborts, `edit` opens it in `$EDITOR` for free-form editing (you can fix the strategy slug, summary, trigger signals, or code itself in one buffer).
- **Read-only internal queries** (graph DB lookups against bamboo's own state) auto-execute without confirmation — these are millisecond-scale and have no external impact.
- **Abandoned sessions** (`/abandon` without declaring a cause) commit Procedures tagged `metadata.status="tentative"`. Default analyze queries filter these out via `include_tentative=False`.

## Two ways to use it

1. **Live incident, unknown cause**: this is the canonical use case. You're staring at a failing task and don't know why. The dialog helps you investigate and captures the steps for next time.
2. **Walkthrough of a resolved incident**: even after the fact, narrating an investigation through `bamboo investigate` captures more reusable knowledge than a one-shot email — the LLM-generated orchestration code is replayable in a way that natural-language descriptions are not.

## How it relates to `populate` and `analyze`

Same knowledge graph, different drivers:
- `populate` — ingest a frozen narrative (email) in batch
- `investigate` — generate the same narrative via streaming dialog
- `analyze` — query the graph; Phase 2 re-executes stored procedures on a new task

All three share the same node-vs-edge schema and storage path. The differences are in the front door and the input modality, not the output.
