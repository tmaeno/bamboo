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

Review — [y] run once / [a] auto-run / [k] always ask / [edit] / [N] reject: a
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
| `/approvals` | List the code-execution policies set this session (`auto_run` / `always_ask`) |
| `/revoke <hash-prefix\|all>` | Clear one or all auto-run/always-ask policies for this session |

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
| `-v, --verbose` | off | DEBUG logging + behind-the-scenes narration (intent, strategy, per-tool calls) |

## What's captured (and why it's reusable)

Each tool turn produces **one orchestration block** — a small async Python function body that calls one or more MCP tools. When the session commits, each block becomes a `Procedure` node with the source code stored on `metadata.orchestration_code` along with `code_summary`, `external_access` (whether the code hits PanDA), and the per-incident `trigger_signals`.

A procedure is identified by a **stable tool-call signature** — the set of tools its code calls — rather than the free-text strategy name, so the *same investigation step* captured under different phrasings (or different causes) dedups to one node (the cause is carried by an edge, and reuse frequency accumulates across causes). Approved, **non-trivial (≥2-tool), read-only** procedures are then offered back to the planner as reusable `proc__…` tools: in a later session the planner can call one to **reuse** proven prior work instead of re-deriving the logic. (Reuse is still reviewed per turn like any code — no auto-run yet.) See [EXECUTION_TRUST.md](EXECUTION_TRUST.md).

When `bamboo analyze` later encounters a similar task and Phase 2 retrieves the Procedure for the matched Cause, it **replays the stored code** — the exact bytes that worked last time run this time, more reproducibly than regenerating from a description. But `analyze` is an **automatic, read-only** phase (no operator watching), so a stored procedure whose code would call a state-changing (`read_only=False`) tool is **not** replayed there — it is skipped and surfaced as a *suggestion* to run in the interactive `investigate` loop. External PanDA *reads* replay fine; state changes only ever happen inside the interactive loop. See [EXECUTION_TRUST.md](EXECUTION_TRUST.md).

The full design rationale is in [the plan file](../.claude/plans/i-m-planning-to-evolve-bubbly-blossom.md).

## Safety model

The full model — including the unattended (`analyze`/startup) read-only boundary — is in
[EXECUTION_TRUST.md](EXECUTION_TRUST.md). In the interactive loop:

- **Every new code block is reviewed before it runs** — the proposed code + summary + trigger signals are shown, regardless of whether it only reads or changes state (a read is *not* a free pass). You choose a per-code policy: `y` runs it once; `a` runs it and **auto-runs** the same code for the rest of the session without prompting; `k` runs it but keeps asking each time; `edit` opens it in `$EDITOR`; `N` rejects.
- **Auto-run is keyed on the exact code** (whitespace-normalized) and scoped to the session (persists across `--resume`; not shared with other sessions/users). `/approvals` lists the policies; `/revoke` clears them.
- **Abandoned sessions** (`/abandon` without declaring a cause) commit Procedures tagged `metadata.status="tentative"`. Default analyze queries filter these out via `include_tentative=False`.

## Two ways to use it

1. **Live incident, unknown cause**: this is the canonical use case. You're staring at a failing task and don't know why. The dialog helps you investigate and captures the steps for next time.
2. **Walkthrough of a resolved incident**: even after the fact, narrating an investigation through `bamboo investigate` captures more reusable knowledge than a one-shot email — the LLM-generated orchestration code is replayable in a way that natural-language descriptions are not.

## How it relates to `populate` and `analyze`

Same knowledge graph, different drivers:
- `populate` — ingest a frozen narrative (email) in batch
- `investigate` — generate the same narrative via streaming dialog
- `analyze` — query the graph; Phase 2 replays stored procedures on a new task (automatic = read-only; state-changing procedures are suggested, not run)

All three share the same node-vs-edge schema and storage path. The differences are in the front door and the input modality, not the output.
