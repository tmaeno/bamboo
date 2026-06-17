# Code-Execution Trust Model

Bamboo's agents run **LLM-generated** (and previously **captured**) Python orchestration code that
calls MCP tools — some of which mutate PanDA. This document is the single source of truth for *when
that code is allowed to run*. It is cross-cutting: `investigate`, `analyze`, `populate`, and the
Mattermost bot all go through the same rules.

> This is the **Phase 1** model. The persistent cross-session policy store, reuse-before-regenerate,
> and signature-based de-duplication are deferred to Phase 2 (see the plan); this document will be
> extended when they land.

## Two tool axes

Every `McpTool` carries two **orthogonal** flags (see `bamboo/mcp/base.py`):

|  | `external_access=False` (bamboo's own state) | `external_access=True` (PanDA) |
|---|---|---|
| `read_only=True`  | internal graph query | PanDA fetch / search / compare |
| `read_only=False` | writes bamboo's DB | mutates PanDA (kill / retry) |

- **`read_only`** — the **security** axis: does the tool change *any* state (internal *or* external)?
  A bamboo-DB writer is **not** read-only even though it never touches PanDA. The automatic-phase
  guard permits only `read_only=True` tools.
- **`external_access`** — informational: does it reach PanDA (cost / observable interaction) vs.
  operate on bamboo's own state? It is **not** a mutation flag — an external PanDA *read* is
  `read_only=True, external_access=True`. (Formerly the single, overloaded `has_side_effects`.)

## Tool selection ≠ execution authorization

For large MCP catalogues, `investigate`/`explore` may show the LLM only a
*relevance-filtered subset* of tools in the prompt (see [AGENTS.md](AGENTS.md),
"Bounding the tool list"). This is a **prompt-budget optimization only** and is
fully decoupled from this trust model: the `ToolProxy` allow-set — not what the
prompt happens to list — is the execution boundary. Selecting fewer tools never
grants new authorization, and it never *refuses* a contextually-valid call the LLM
names from a tool the prompt omitted. The automatic-phase read-only guard is
unchanged.

## Principles

1. **A captured procedure is *knowledge*, not *authorization*.** Re-running a state-changing
   procedure is a fresh real-world action whose effect depends on the *current* task and identity,
   and such effects are not idempotent. Permission is never inherited from the moment of capture.
2. **State changes happen only inside the interactive turn loop** (`investigate`'s `_tool_turn`),
   where the operator is present. The **automatic phases** — `analyze`, the investigate-startup
   hypothesis (`analyze_task`), and `populate` — are strictly **read-only** (they may *read* PanDA,
   but never run a `read_only=False` tool), whatever a procedure's history says.
3. **The enforcement boundary is runtime, not static analysis.** The `ToolProxy` checks the resolved
   tool name at call time, so it is alias-proof; static analysis is advisory (it decides what to show
   and what to warn about, not what is allowed).

Note that `external_access` is **not** the basis for trust: an external PanDA *read* is allowed in
automatic phases (fetching data is their whole job). Only `read_only` gates execution there; and in
the interactive loop, the auto-run decision is a human's per-code judgment (below), not an inference
from either flag.

## Interactive review-and-policy lifecycle (`investigate`)

Every tool turn generates a code block. Before it runs:

1. **New code is always reviewed.** The proposed code + summary + trigger signals are shown
   (regardless of whether it reads or changes state).
2. The operator picks a **per-code policy**:
   - `run-once` (`y`) — run this time only; remember nothing.
   - `auto-run` (`a`) — run, and skip the prompt for *this exact code* for the rest of the session.
   - `always-ask` (`k`) — run, but keep asking each time.
   - `edit` — open the code in `$EDITOR` (chat: reply with a replacement block), then re-review.
   - `reject` (`N`) — don't run.
3. `auto-run` code runs with **no prompt** on later turns; `always-ask` re-prompts each time.

**Identity.** The policy is keyed on `code_hash` — a SHA-256 of the **whitespace-normalized** code
(per-line trailing whitespace stripped, blank lines dropped; leading indentation preserved). Two
blocks that differ only in formatting share a policy; a logic change yields a new hash and is
re-reviewed — the safe direction.

**Scope.** Policies live on the investigation session (`InvestigationSession.code_policies`), so they
persist across `--resume` but are **not** shared with other sessions or users. `/approvals` lists the
current policies; `/revoke <hash-prefix|all>` clears them.

> **Honest limit.** Because the LLM regenerates different code per turn, `code_hash` matches mostly on
> stored-procedure replay or identical regeneration — so within a session most turns are reviewed
> (which is the intent). Broad, durable auto-run is Phase 2.

## Automatic phases are read-only (`analyze`, investigate-startup, `populate`)

These run with no operator watching turn-by-turn (`ReasoningNavigator.analyze_task` →
`ContextEnricher`), so they may never change state — but they **may read PanDA** (that is their whole
job: fetch logs, compare jobs). Enforced two ways, both keyed on `read_only` (not `external_access`):

- **Generation is read-only by construction** — the explorer's tool list (`ContextEnricher._filtered_tools`)
  drops `read_only=False` tools, so the planner can't generate calls to them. External PanDA *reads*
  are kept.
- **Runtime backstop** — execution passes `allowed_tools = <read_only tool names>` to the `ToolProxy`;
  any `read_only=False` call (aliased, hallucinated, or from a replayed stored procedure) is **refused
  at the call site** before dispatch.

A stored procedure whose code would call a `read_only=False` tool is **not** replayed in an automatic
phase — it is skipped and surfaced as a *suggested strategy* for the operator to run in the
interactive `investigate` loop. (Today no tool is `read_only=False`, so all current PanDA-read
procedures replay; the guard activates when a kill/retry-style tool is added.)

## The runtime boundary (`ToolProxy.allowed_tools`)

`bamboo/agents/orchestration.py` — `ToolProxy` takes an optional `allowed_tools` frozenset. When set,
`call()` refuses any tool whose resolved name is not in the set, *before* dispatch. This catches
aliasing that static analysis misses:

```python
m = tools.kill_job      # static scan of the source might call this "read-only"
await m()               # …but the proxy resolves name="kill_job" at call time → refused
```

`run_orchestration_code(..., allowed_tools=...)` threads the set through. Callers:

| Caller | `allowed_tools` |
|---|---|
| `investigate._tool_turn` (interactive, human-reviewed) | `None` (unrestricted — the human gated it) |
| `ContextEnricher` (automatic explorer / stored replay) | the `read_only=True` tool names |

`analyze_code_side_effects` is a generic *static* helper (does the code reference any name in a
caller-supplied set?), used only to decide whether to show a hint / to pre-screen a stored procedure
(against the `read_only=False` names) for the skip-and-suggest path. It is a syntactic
over-approximation (and misses aliasing) — which is why the runtime allow-set, not it, is the boundary.

## Reusable procedures as tools (Phase 2a)

A captured procedure is a **reusable unit** identified by its **tool-call signature** (the set of
`tools.<name>` its code calls) — a stable identity that replaces the free-text `strategy_type`, so
the same step dedups across phrasings and causes (the cause is carried by an edge; reuse frequency
accumulates across causes). Approved, **non-trivial (≥2-tool), read-only** procedures are exposed to
the `investigate` planner as callable `proc__…` tools (cause-agnostic, capped by frequency via
`find_all_procedures`); the planner *reuses* one by calling it, and it replays the stored,
already-reviewed code through the same sandbox (so the read-only boundary composes). Single-tool
blocks are still captured + replayable but not exposed as tools (the raw tool already covers them).

## Durable per-procedure auto-run (Phase 2b)

A procedure can carry a **durable, cross-session auto-run grant** — `ProcedureNode.metadata.auto_run`
(a per-deployment flag on the node; no schema change). When a `investigate` turn's code references
**only** durably-granted procedures, it runs **without a review prompt**; a turn that mixes in a raw
or un-granted tool is still reviewed (the lifecycle above). The grant is created by choosing `auto-run`
on a turn that *reuses* procedure-tool(s) (the `a` choice persists those procedures via
`set_procedure_auto_run`); ad-hoc/raw-tool code has no stable identity, so its `auto-run` stays
session-scoped (`code_hash`).

**Read-only by default.** Only read-only procedures are exposed/auto-runnable (each procedure's
`read_only` is **recomputed from its code**, not trusted from a stored flag — so a procedure that
becomes state-changing can't keep auto-running). The opt-in escape hatch — `bamboo investigate
--allow-mutating-autorun` (or the `allow_mutating_autorun` config setting, read by the Mattermost bot)
— also exposes + allows **state-changing** procedures, but **only in the interactive loop**; the
**automatic** `analyze` phase stays read-only-enforced regardless. Per-deployment sharing is safe
precisely because, by default, only reads can ever be auto-run.

`/approvals` lists session + durable grants; `/revoke <hash-prefix|proc-name|all>` clears them
(durable grants via `set_procedure_auto_run(..., False)`). Grant creation is narrated (audit).

## Verbose visibility (`-v` / `--verbose`)

Orthogonal to trust: CLI `bamboo investigate -v` and Mattermost `investigate <id> --verbose` stream
behind-the-scenes DEBUG narration — the intent decision, the chosen strategy, and each tool call with
its arguments and result shape — so an operator can watch how a turn was constructed. It changes
visibility only, never what is allowed to run.

(Distinct from the bot's *launch* flag `serve-mattermost -v`, which is a server-side knob — full
DEBUG to the console/log for the whole process, not the per-command Mattermost reply; see
[MATTERMOST.md](MATTERMOST.md). It too only affects visibility.)

## Where this is implemented

| Concern | Location |
|---|---|
| Runtime allow-set boundary | `ToolProxy` / `run_orchestration_code` (`bamboo/agents/orchestration.py`) |
| Review-and-policy lifecycle | `_tool_turn` / `_review_code` / `code_policies` (`bamboo/agents/investigation_session.py`) |
| Read-only automatic explorer | `_filtered_tools` / `_run_orchestration_code` (`bamboo/agents/context_enricher.py`) |
| Skip + suggest state-changing stored procedures | `_run_investigation` (`bamboo/agents/reasoning_navigator.py`) |
| Procedure identity + reusable procedure-tools | `procedure_signature` / `build_procedure_tools_registry` (`bamboo/agents/procedure_tools.py`); `find_all_procedures` (`bamboo/database/`) |
| Durable per-procedure auto-run | `ProcedureNode.metadata.auto_run` via `set_procedure_auto_run` (`bamboo/database/`); `_review_code` durable check/grant + `_durable_autorun_procs` (`bamboo/agents/investigation_session.py`); `--allow-mutating-autorun` / `allow_mutating_autorun` setting |
| Per-thread verbose | `Command.verbose` / `parse_command` / `stream_narration` (`bamboo/frontends/mattermost/`) |
