# 🤖 Agent Reference

Bamboo is built around two main pipelines, each composed of several cooperating agents.
The **Knowledge Accumulation** pipeline learns from resolved incidents and builds the knowledge
databases.  The **Reasoning Navigation** pipeline diagnoses new incidents by querying those
databases.  Both pipelines share the same extraction layer; the accumulation pipeline also
includes an optional quality-gate loop.

---

## Pipeline Overview

```
┌────────────────────────────────────────────────────────────────┐
│  Knowledge Accumulation                                        │
│                                                                │
│  incident data ──► KnowledgeAccumulator                        │
│                         │                                      │
│                         ├─ KnowledgeGraphExtractor             │
│                         │      └─ PandaKnowledgeExtractor      │
│                         │            └─ PandaJobDataAggregator │
│                         │                                      │
│                         ├─ KnowledgeReviewer                   │
│                         │                                      │
│                         └─ ExtraSourceExplorer                 │
│                                ├─ ExplorationPlanner           │
│                                └─ MCP client layer             │
│                                     ├─ PandaMcpClient          │
│                                     ├─ ExternalMcpClient (HTTP)│
│                                     └─ StdioMcpClient  (stdio) │
└────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Reasoning Navigation                                       │
│                                                             │
│  task data ──► ReasoningNavigator                           │
│                    └─ KnowledgeGraphExtractor  (read-only)  │
└─────────────────────────────────────────────────────────────┘
```

The **review–explore loop** inside `KnowledgeAccumulator` always runs:

```
extract → KnowledgeReviewer
              │ approved          → store
              │ rejected (pass 0) → ExplorationPlanner
              │                          │ plan OK  → ExtraSourceExplorer (step-by-step)
              │                          │ plan None → ExtraSourceExplorer (single-wave fallback)
              │                     → re-extract → KnowledgeReviewer
              │ rejected (pass N) → store best result (with warning)
```

---

## Knowledge Accumulation Pipeline

### `KnowledgeAccumulator`

**File:** `bamboo/agents/knowledge_accumulator.py`

The top-level orchestrator for knowledge accumulation.  Given the raw data for one resolved
incident it runs the full pipeline: extraction, optional review, database storage, and vector
indexing.

**Inputs** (all optional except at least one of `email_text` / `task_data`):

| Parameter | Type | Description |
|---|---|---|
| `email_text` | `str` | Full email thread from the incident |
| `task_data` | `dict` | Structured PanDA task fields |
| `external_data` | `dict` | Supplementary key→value metadata |
| `task_logs` | `dict[str, str]` | Task-level logs keyed by source name (e.g. `"jedi"`) |
| `dry_run` | `bool` | Skip all DB writes; useful for `bamboo extract` previews |

**Output:** `ExtractedKnowledge` — graph, narrative summary, vector key insights, metadata.

**Configuration:**

| Setting | Default | Effect |
|---|---|---|
| `--max-retries N` | `2` | Reviewer retry limit (`bamboo extract` CLI only) |

**Retry loop:** on each rejection the accumulator increments `attempt`.  The
`ExtraSourceExplorer` fires exactly once (at `attempt == 0`).  After
`max_review_retries` rejections the best result is stored with a warning.

---

### `KnowledgeGraphExtractor`

**File:** `bamboo/agents/extractors/knowledge_graph_extractor.py`

A thin dispatcher that selects the active extraction strategy, calls it, and assigns a stable
UUID to every returned node.  Neither the accumulator nor the reasoning navigator talk to an
extraction strategy directly — they always go through this class.

**Configuration:** `EXTRACTION_STRATEGY` env var (default: `"panda"`).  See
[Extraction Strategy Plugin System](EXTRACTION_PLUGIN_SYSTEM.md) for adding custom strategies.

---

### `PandaKnowledgeExtractor`

**File:** `bamboo/agents/extractors/panda_knowledge_extractor.py`

The default extraction strategy for PanDA task data.  Combines structured field parsing with
several LLM sub-calls to produce a complete `KnowledgeGraph`.

**Input routing:**

| Source | Node types produced |
|---|---|
| `task_data` discrete fields (site, processingType, …) | `TaskFeatureNode` |
| `task_data` continuous fields (ramCount, walltime, …) | `TaskFeatureNode` (bucketed) |
| `task_data.errorDialog` | `SymptomNode` (canonical category name; raw text in `description`) |
| `task_data.status` | `SymptomNode` (when `broken` / `failed`) |
| `task_data` free-form fields (taskName, …) | `TaskContextNode` (vector DB only) |
| `external_data` key→value pairs | `TaskFeatureNode` |
| `email_text` | `CauseNode`, `ResolutionNode`, `ProcedureNode`, `TaskContextNode` (LLM) |
| `task_logs` | `SymptomNode`, `ComponentNode`, `TaskContextNode` (LLM) |

The `errorDialog` field is also scanned for embedded HTML log links; each linked file is
downloaded and processed as a task-level log.

**Canonicalisation:** error categories, cause names, and resolution names are normalised via
a vector-DB + LLM round-trip so the same concept always maps to the same node name across
incidents.

---

### `KnowledgeReviewer`

**File:** `bamboo/agents/knowledge_reviewer.py`

An LLM-based quality gate that evaluates the extracted graph for completeness *before* it is
written to the databases.  It acts as a **gap analyzer** — it identifies information that
should be present but is missing — rather than cross-checking against source text.

**Gap categories:**

| Category | Example |
|---|---|
| Structural | `SymptomNode` with no `CauseNode` explaining it |
| Specificity | Node named `"error"` with no error code or detail |
| Contextual | `SymptomNode` present but no `CauseNode` reachable via graph traversal |

**Grounding rule:** the LLM may only flag a gap if it is implied by (a) the graph structure
itself or (b) the available task context fields.  Speculation beyond the provided data is
prohibited.

**MCP tool awareness:** the reviewer receives the full MCP tool catalogue at review time.
When a gap could be resolved by a specific tool, the reviewer annotates the issue string
accordingly — e.g. `"SymptomNode present but no Cause found → resolvable with get_task_logs"`.
This makes the explorer's downstream tool-selection more reliable without changing the
`ReviewResult` schema.  The reviewer only sees tools that are statically registered (PanDA
tools are always available; external server tools appear after the first `connect()`).

**Fail-open:** any LLM or parse error returns `approved=True` so a reviewer malfunction
never blocks the accumulation pipeline.

**Output:** `ReviewResult` — `approved`, `confidence`, `issues` (list of gap descriptions,
optionally annotated with `→ resolvable with <tool>`), `feedback` (actionable instruction
for the extractor on retry).

---

### `ExplorationPlanner`  *(optional, two-phase)*

**File:** `bamboo/agents/exploration_planner.py`

Sits between `KnowledgeReviewer` and `ExtraSourceExplorer`.  Converts the reviewer's raw
issue list into a structured `ExplorationPlan` via two sequential LLM calls, then hands
the plan to the explorer for execution.

**Phase 1 — Gap analysis** (`EXPLORATION_GAP_ANALYSIS_PROMPT`):  
Given the reviewer issues and the available tool catalogue, produce a list of precise,
tool-neutral gap descriptions — *what* specific information is missing and *why* it
matters.  Tool names are intentionally excluded; this phase only identifies the holes.
Output is shown as a `[planner: gap analysis]` panel in verbose mode.

**Phase 2 — Step planning** (`EXPLORATION_PLAN_PROMPT`):  
Map each resolvable gap to one or more MCP tool calls, grouped into sequential
`PlanStep` objects.  Tools within a step are independent and run concurrently;
steps run in order so a later step can rely on earlier steps having populated the
extraction context.

**Output:** `ExplorationPlan` — `gaps` (list of gap strings), `steps` (list of `PlanStep`).
Each `PlanStep` has a `reason` (human-readable) and `tool_calls` (list of tool + args dicts).

**Fail-open:** any error in either phase causes `plan()` to return `None`, signalling the
explorer to fall back to its single-LLM-call `_select_tools` path.

---

### `ExtraSourceExplorer`  *(fires once per run)*

**File:** `bamboo/agents/extra_source_explorer.py`

Sits between the first reviewer rejection and the second extraction attempt.  Executes
the `ExplorationPlan` produced by `ExplorationPlanner` (if available) or falls back to
a single LLM call that selects tools directly.

**Flow (plan-based, default):**

1. `connect()` the MCP client.
2. List available tools; pass them to `ExplorationPlanner.plan()`.
3. Execute plan steps **sequentially**; tools within each step run concurrently via `asyncio.gather`.
4. Log `[step N/M] <reason>` for each step in verbose mode.
5. Merge all results into `task_logs` and `external_data` for the next extraction pass.
6. `close()` the MCP client.

**Fallback flow (no planner, or planner returns `None`):**

Steps 1–2 as above, then a single LLM call (`EXPLORER_TOOL_SELECTION_PROMPT`) selects
all tools at once and they run concurrently in a single wave.

The explorer fires **at most once** per accumulation run (at `attempt == 0` only).
Any individual tool failure is logged and skipped — the pipeline never stalls.

Results from unrecognised tools (e.g. from external MCP servers) are stored in
`external_data` under `"tool:<name>"` and forwarded to the LLM extractor as additional context.

**Output:** `ExplorationResult` — `task_logs`, `external_data`, `tool_calls` (all calls across all steps, for observability).

---

### MCP Client Layer

The MCP client is built by `build_mcp_client()` in `bamboo/mcp/factory.py` and passed to
`ExtraSourceExplorer` at startup.  When no external servers are configured it returns a bare
`PandaMcpClient`; otherwise it returns a `CompositeMcpClient` that aggregates `PandaMcpClient`
with one or more `ExternalMcpClient` instances.

#### `PandaMcpClient`

**File:** `bamboo/mcp/panda_mcp_client.py`

Built-in client that exposes PanDA data tools.  No external connection needed.

| Tool | Trigger condition | Routes to | Returns |
|---|---|---|---|
| `fetch_error_dialog_logs` | Symptom nodes too vague; log evidence absent | `task_logs` | `dict[url → content]` |
| `get_parent_task` | `retryID` present but root cause unclear | `external_data` | Parent task dict |
| `get_retry_chain` | Failure spans multiple retry attempts | `external_data` | List of ancestor task summaries |
| `get_task_jobs_summary` | Job-level failure distribution missing | `external_data` | Status counts + top error diags |
| `get_failed_job_details` | Scout failures, site-specific errors needing job-level detail | `external_data` | List of compact job dicts |
| `get_task_jedi_details` | Unclear failure cause despite clean `errorDialog`; scheduling/resource bottleneck suspected | `task_logs` | Enriched JEDI task dict (scheduling params, split rules, resource allocation) |
| `get_task_input_datasets` | Symptoms suggest input data issues (`STAGEIN_FAILED`, dataset not found) | `task_logs` | List of input dataset dicts with file counts |
| `search_panda_server_source` | Task pending due to overestimated resources from scouts; vague errorDialog message with no clear cause | `task_logs` | List of `{file, line, context}` code snippets from panda-server source |
| `search_panda_docs` | Node name or error pattern requires domain-level explanation (e.g. what a task status means, when it is entered, what causes it) | `doc_hints` | Plain-text snippets from the official PanDA WMS documentation, passed as domain background to the reviewer and planner (not stored as graph nodes) |

All tools are safe to call concurrently.  Tools routed to `task_logs` (formatted JSON text)
are processed by the LLM extractor alongside error dialog logs.  Tools routed to
`external_data` are consumed by the structured extractor path.  Results from
`search_panda_docs` go into `doc_hints`, a dedicated dict that flows to the reviewer
and exploration planner as authoritative domain context.

#### `ExternalMcpClient`

**File:** `bamboo/mcp/external_mcp_client.py`

Connects to one external MCP server using the **StreamableHTTP** transport from the official
`mcp` Python SDK.  Tools are discovered at connect time via `session.list_tools()` and are
presented to the LLM alongside the built-in PanDA tools.

- The `mcp` package is a main dependency — no extra install needed.
- If the `mcp` package is missing or the server is unreachable, `connect()` logs the error
  and this client contributes zero tools — the pipeline continues with PanDA tools only.

#### `StdioMcpClient`

**File:** `bamboo/mcp/external_mcp_client.py`

Connects to one external MCP server using the **stdio** transport: bamboo spawns the server
as a subprocess and communicates over its stdin/stdout.  No separately-running server process
is needed — bamboo manages the subprocess lifetime automatically.

- Same `mcp` package requirement and fail-open behaviour as `ExternalMcpClient`.
- The subprocess inherits the current environment plus any extra variables declared in the
  `env` field of the server config entry.

#### `CompositeMcpClient`

**File:** `bamboo/mcp/external_mcp_client.py`

Aggregates any number of `McpClient` instances (PanDA, HTTP, stdio) into one.
`PandaMcpClient` is always first, so its tool names win on any name clash with external servers.

#### Configuring external MCP servers

External servers are declared in a JSON file referenced by `MCP_SERVERS_CONFIG`.  Each entry
must specify **exactly one** transport — `url` (HTTP) or `command` (stdio).

**HTTP transport:**

```json
{
  "servers": [
    {
      "name": "my_atlas",
      "url": "http://localhost:8080/mcp",
      "headers": {"Authorization": "Bearer ${ATLAS_TOKEN}"},
      "enabled": true
    }
  ]
}
```

**stdio transport:**

```json
{
  "servers": [
    {
      "name": "my_server",
      "command": "python3",
      "args": ["-m", "mypackage.server"],
      "env": {"PYTHONPATH": "/path/to/mypackage"},
      "enabled": true
    }
  ]
}
```

Header and `env` values support `${ENV_VAR}` expansion.  Point to the file via `.env`:

```
MCP_SERVERS_CONFIG=/path/to/mcp_servers.json
```

See `bamboo/mcp/server_config.py` for the full schema.

#### Integrating bamboo-mcp

[bamboo-mcp](https://github.com/BNLNPPS/bamboo-mcp) is a companion MCP server that provides
additional tools useful during exploration, notably:

| Tool | Description |
|---|---|
| `panda_log_analysis` | Log analysis + failure classification for a job |
| `panda_task_status` | Detailed task metadata including per-job breakdown |
| `panda_job_status` | Individual job metadata, pilot errors, and file summary |
| `panda_queue_info` | Look up queue and site configuration |
| `panda_harvester_workers` | Fetch Harvester worker (pilot) counts by site |

Results from these tools are stored in `external_data` under `"tool:<name>"` by the explorer's
generic fallback and forwarded to the LLM extractor as additional context.

Install bamboo-mcp in a separate virtual environment, then add this entry to your `mcp_servers.json`:

```json
{
  "servers": [
    {
      "name": "bamboo_mcp",
      "command": "/separate_venv/bin/python3",
      "args": ["-m", "bamboo.server"],
      "env": {"PYTHONPATH": "/path/to/bamboo-mcp/core",
              "PANDA_BASE_URL": "https://bigpanda.monitor.url"},
      "exclude_tools": ["bamboo_.*"],
      "enabled": true
    }
  ]
}
```

---

## Reasoning Navigation Pipeline

### `ReasoningNavigator`

**File:** `bamboo/agents/reasoning_navigator.py`

Diagnoses a problematic task by querying the knowledge databases built by the accumulation
pipeline, then drafts a resolution email for the task submitter.

**Steps:**

1. **Extract clues** — run `KnowledgeGraphExtractor` on the task's structured fields to
   identify symptoms, task features, job features, environment factors, and components.
2. **Graph DB query** — find candidate causes ranked by how many clue types point to them
   (symptoms, task features, job features, environment, components).
3. **Vector DB retrieval** — two-step:
   a. Search node descriptions for similar past incidents (returns `graph_id` values).
   b. Fetch narrative summaries for those graphs.
4. **LLM diagnosis** — synthesise graph evidence + semantic evidence into a root-cause
   statement with confidence score.
5. **Email draft** — generate a professional resolution email for the task submitter.

**Output:** `AnalysisResult` — `root_cause`, `confidence`, `resolution`, `explanation`,
`supporting_evidence`, `email_content`.

---

## Configuration Reference

| Environment variable | Default | Affects |
|---|---|---|
| `EXTRACTION_STRATEGY` | `panda` | `KnowledgeGraphExtractor` strategy selection |
| `LLM_PROVIDER` | `openai` | All LLM calls across all agents |
| `LLM_MODEL` | `gpt-4-turbo-preview` | All LLM calls across all agents |
| `MCP_SERVERS_CONFIG` | _(empty)_ | Path to JSON file listing external MCP servers |

The `search_panda_server_source` tool requires panda-server installed in the same environment:

```
pip install "bamboo[panda]"
```

> **Note:** `panda-server` currently installs with full server-side dependencies.
> This will be updated to a lightweight source-only package once one is available.

The `--max-retries N` flag on `bamboo extract` overrides the reviewer retry limit for a single
run without changing the default.

---

## Failure Handling

All agents follow a **fail-open** policy: if an optional sub-component errors, the pipeline
continues with what it has rather than aborting.

| Component | On failure |
|---|---|
| `KnowledgeReviewer` LLM call | Returns `approved=True`, logs exception |
| `ExplorationPlanner` either LLM call | Returns `None` → explorer falls back to `_select_tools` |
| `ExtraSourceExplorer` LLM call (fallback) | Returns empty `ExplorationResult`, logs exception |
| Individual MCP tool call | Logs warning, skips that tool's result |
| `ExternalMcpClient` connect (server down) | Logs warning, contributes zero tools; PanDA tools still available |
| `ExternalMcpClient` connect (`mcp` not installed) | Logs install hint, contributes zero tools |
| `StdioMcpClient` connect (subprocess fails) | Logs warning, contributes zero tools; PanDA tools still available |
| `StdioMcpClient` connect (`mcp` not installed) | Logs install hint, contributes zero tools |
| `search_panda_server_source` (`pandaserver` not installed) | Logs install hint, returns empty list |
| `search_panda_docs` (GitHub API or network error) | Logs warning, returns empty list; `doc_hints` remains empty |
| `KnowledgeReviewer` repeated rejection | Stores best result after `max_review_retries`, logs warning |
