# PanDA Integration

Bamboo can fetch task data directly from a live **PanDA** (Production and Distributed Analysis)
server via the `panda-client-light` package, eliminating the need to prepare a local JSON file
before running the pipeline.

---

## Module

```
bamboo/utils/panda_client.py
```

All PanDA-facing helpers live here so they can be reused by extractors, agents, scripts, and
the CLI without creating circular imports.  New functions (e.g. job-list fetching, task-status
polling) should be added to this module.

### `fetch_task_data(task_id, verbose=False)`

```python
from bamboo.utils.panda_client import fetch_task_data

task_data = await fetch_task_data(12345)

# with full communication logging
task_data = await fetch_task_data(12345, verbose=True)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | `int` or `str` | PanDA `jediTaskID` |
| `verbose` | `bool` | If `True`, prints every curl command and raw server response to stdout. Useful for debugging auth or network issues. Default: `False`. |

**Returns** – a `dict` of task fields exactly as returned by the PanDA server, ready to pass
as `task_data` to `KnowledgeAccumulator` or any extraction strategy.

**Raises**

| Exception | When |
|-----------|------|
| `ValueError` | `task_id` cannot be converted to `int` |
| `ImportError` | `panda-client-light` is not installed |
| `RuntimeError` | Server returns a non-zero status code, `None`, or a non-dict body |

The call to `pandaclient.Client.get_task_details_json` is wrapped in
`asyncio.run_in_executor` so it never blocks the event loop.

---

## Server Configuration and Obtaining Access Tokens

`panda-client-light` reads the server URL from environment variables.  Set them in your
`.env` file or shell before running Bamboo:

| Variable | Default | Description |
|----------|---------|-------------|
| `PANDA_API_URL` | `http://pandaserver.cern.ch:25080/api/v1` | Plain HTTP API base URL |
| `PANDA_API_URL_SSL` | `https://pandaserver.cern.ch:25443/api/v1` | HTTPS API base URL (used for most calls) |

To point at a development or local PanDA instance:

```bash
export PANDA_API_URL=http://mypanda.example.org:25080/api/v1
export PANDA_API_URL_SSL=https://mypanda.example.org:25443/api/v1
```

Other configuration parameters (e.g. `PANDA_AUTH`) can be set as needed — refer to the
[PanDA client setup guide](https://panda-wms.readthedocs.io/en/latest/client/panda-client.html#setup)
for the full list.

Then you need to obtain an access token for authentication. This typically involves:

```bash
python -c "from pandaclient import Client; print(Client.get_access_token())"
```

> **macOS note:** if you see an error about invalid credentials, you may need to install
> Python's SSL certificates first:
> ```bash
> /Applications/Python\ 3.x/Install\ Certificates.command
> ```

---

## Usage

### Via CLI commands

**Inspect raw task data** without running the full pipeline:

```bash
# Print JSON to stdout
bamboo fetch-task 12345

# Save to a file
bamboo fetch-task 12345 --output task_12345.json

# Show all curl commands and raw server responses (debug auth/network issues)
bamboo fetch-task 12345 --verbose
```

**Populate knowledge base** from a live task (instead of a local JSON file):

```bash
bamboo populate --task-id 12345
bamboo populate --task-id 12345 --email-thread incident.txt
```

**Analyze a task** fetched directly from PanDA:

```bash
bamboo analyze --task-id 12345
bamboo analyze --task-id 12345 --output result.json
```

`--task-id` and `--task-data` are mutually exclusive on both commands.


### Programmatically

```python
import asyncio
from bamboo.utils.panda_client import fetch_task_data
from bamboo.agents.knowledge_accumulator import KnowledgeAccumulator
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient

async def main():
    task_data = await fetch_task_data(12345)

    graph_db = GraphDatabaseClient()
    vector_db = VectorDatabaseClient()
    await graph_db.connect()
    await vector_db.connect()

    from bamboo.agents.knowledge_reviewer import KnowledgeReviewer
    from bamboo.agents.context_enricher import ContextEnricher
    from bamboo.agents.exploration_planner import ExplorationPlanner
    from bamboo.mcp.factory import build_mcp_client
    from bamboo.config import get_settings

    _mcp = build_mcp_client(get_settings())
    reviewer = KnowledgeReviewer()
    explorer = ContextEnricher(_mcp, planner=ExplorationPlanner(_mcp))
    agent = KnowledgeAccumulator(graph_db, vector_db, reviewer=reviewer, explorer=explorer)
    result = await agent.process_knowledge(task_data=task_data)
    print(result.summary)

    await graph_db.close()
    await vector_db.close()

asyncio.run(main())
```

---

## Interactive Mode

When running `bamboo interactive`, the **Populate** and **Analyze** menus both offer
"Fetch from PanDA by task ID" as an alternative to loading a local file:

```
Do you have task data? [y/N]: y
Fetch task data directly from PanDA by task ID? [y/N]: y
Enter PanDA jediTaskID: 12345
✓ Fetched 47 fields for task 12345
```

---

## Error Handling

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ImportError: panda-client-light is required` | Package not installed | `pip install panda-client-light` or `pip install "bamboo[panda]"` |
| `RuntimeError: status=255` | Network failure or wrong server URL | Check `PANDA_API_URL_SSL`; verify the server is reachable |
| `RuntimeError: status=0 … data=None` | Task ID not found on the server | Confirm the `jediTaskID` is correct |
| `ValueError` | Non-numeric string passed as `task_id` | Pass an integer or numeric string |

---

## Testing

Unit tests for `fetch_task_data` are in `tests/test_panda_client.py`.  All tests mock
`pandaclient.Client` via `unittest.mock.patch.dict` on `sys.modules` — no live PanDA
server is required.

```bash
pytest tests/test_panda_client.py -v
```

Covered cases:

- Successful fetch returning a `dict`
- String task ID coerced to `int`
- Non-zero status code → `RuntimeError`
- `None` response body → `RuntimeError`
- Non-dict response body → `RuntimeError`
- Non-numeric task ID → `ValueError`
- Missing `panda-client-light` → `ImportError` with install hint
- Error messages include task ID and `PANDA_API_URL_SSL` hint

---

## PanDA Agent Components

### `PandaKnowledgeExtractor`

**File:** `bamboo/agents/extractors/panda_knowledge_extractor.py`

The default extraction strategy (`EXTRACTION_STRATEGY=panda`).  Combines structured field
parsing with several LLM sub-calls to produce a complete `KnowledgeGraph`.

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

### `PandaMcpClient`

**File:** `bamboo/mcp/panda_mcp_client.py`

The built-in MCP client for PanDA.  No external connection needed.

| Tool | Trigger condition | Routes to | Returns |
|---|---|---|---|
| `fetch_linked_log_files` | Symptom nodes too vague; log evidence absent | `task_logs` | `dict[url → content]` |
| `get_parent_task` | `retryID` present but root cause unclear | `external_data` | Parent task dict |
| `get_retry_chain` | Failure spans multiple retry attempts | `external_data` | List of ancestor task summaries |
| `get_task_jobs_summary` | Job-level failure distribution missing | `external_data` | Status counts + top error diags |
| `get_failed_job_details` | Scout failures, site-specific errors needing job-level detail | `external_data` | List of compact job dicts |
| `get_task_jedi_details` | Unclear failure cause despite clean `errorDialog`; scheduling/resource bottleneck suspected | `task_logs` | Enriched JEDI task dict (scheduling params, split rules, resource allocation) |
| `get_task_input_datasets` | Symptoms suggest input data issues (`STAGEIN_FAILED`, dataset not found) | `task_logs` | List of input dataset dicts with file counts |
| `search_panda_server_source` | Task pending due to overestimated resources from scouts; vague errorDialog message with no clear cause | `task_logs` | List of `{file, line, context}` code snippets from panda-server source |
| `search_panda_docs` | Node name or error pattern requires domain-level explanation | `doc_hints` | Plain-text snippets from PanDA WMS documentation, passed as domain background to the reviewer and planner |

All tools are safe to call concurrently.  Tools routed to `task_logs` are processed by the LLM
extractor alongside error dialog logs.  Tools routed to `external_data` are consumed by the
structured extractor path.  Results from `search_panda_docs` go into `doc_hints`, a dedicated
dict that flows to the reviewer and exploration planner as authoritative domain context.

The `search_panda_server_source` tool requires panda-server installed in the same environment:

```
pip install "bamboo[panda]"
```

> **Note:** `panda-server` currently installs with full server-side dependencies.
> This will be updated to a lightweight source-only package once one is available.

---

### `PandaSourceNavigator`

**File:** `bamboo/agents/panda_source_navigator.py`

Called via `PandaKnowledgeExtractor.prefetch_hints()` → `prefetch_panda_context()` →
`prefetch_panda_source()`.  Iteratively navigates the pandaserver / pandajedi Python source to
answer a code-level question derived from a task's `errorDialog`, and delivers its answer as a
`doc_hints` entry so the downstream reasoner and reviewer have source-code context.

**Navigation steps:**

1. **Term extraction** — an LLM call (`SOURCE_GREP_TERMS_PROMPT`) distils the `errorDialog`
   into 2–5 grep strings: identifiers are copied verbatim; instance-specific values (numbers,
   paths, dataset names) are stripped.  Falls back to splitting on words ≥ 6 characters if the
   LLM returns invalid JSON.

2. **Sliding-window grep** — for each fragment a sliding window tries sub-phrases of decreasing
   length (largest first, minimum 2 words) against every `.py` file in `pandaserver` /
   `pandajedi`.  Files are pre-filtered with a fast `in` check before AST parsing.  Matching
   methods and functions are ranked by how many distinct grep terms appear in their body.

3. **Multi-fragment overlap ranking** — candidates that appear across multiple grep-term
   fragments score higher.  If any candidate scores ≥ 2, only those high-overlap candidates
   are kept (up to 30).

4. **Iterative navigation** (`MAX_ROUNDS = 3`) — in each round every new candidate method is
   read in full, then the LLM decides (via `PANDA_SOURCE_NAV_PROMPT`) whether additional
   follow-up symbols are needed.  The LLM can only request symbols visible in already-read
   source.  Stops when the LLM responds `action: "done"` or no follow-up candidates are found.

5. **Synthesis** — `PANDA_SOURCE_SYNTHESIS_PROMPT` produces a plain-English explanation stored
   as `doc_hints["source code analysis"]`.

**Instrumentation attributes** (set on each instance after `navigate()` returns):

| Attribute | Type | Description |
|---|---|---|
| `last_grep_terms` | `list[str]` | Terms extracted by the LLM in step 1 |
| `last_term_extraction_succeeded` | `bool` | `False` if LLM returned invalid JSON and naive fallback was used |
| `last_candidates_count` | `int` | Number of candidate methods after overlap ranking |
| `last_max_overlap` | `int` | Highest per-qualname overlap score across all grep fragments |
| `last_rounds_used` | `int` | Number of navigation rounds completed (0 if no LLM round ran) |

**Failure signals** (returned strings that callers check for):

| Return value prefix | Meaning |
|---|---|
| `"No methods found …"` | Grep produced zero candidates |
| `"Neither pandaserver …"` | Neither package is installed |
| `"Found candidate methods but …"` | Candidates found but none could be read |

**Fail-open:** `prefetch_panda_source()` catches all exceptions and returns an empty dict.

**Evaluation script:** `bamboo/scripts/eval_source_navigator.py` — batch-evaluates the
navigator over a collection of errorDialog strings (JSON / CSV files or PanDA task IDs),
deduplicates inputs by normalised pattern, and classifies results as `relevant`, `irrelevant`,
`no_candidates`, `too_many_candidates`, `empty_error_dialog`, or `error`.

---

### Integrating bamboo-mcp

[bamboo-mcp](https://github.com/BNLNPPS/bamboo-mcp) is a companion MCP server that provides
additional PanDA-specific tools useful during exploration:

| Tool | Description |
|---|---|
| `panda_log_analysis` | Log analysis + failure classification for a job |
| `panda_task_status` | Detailed task metadata including per-job breakdown |
| `panda_job_status` | Individual job metadata, pilot errors, and file summary |
| `panda_queue_info` | Look up queue and site configuration |
| `panda_harvester_workers` | Fetch Harvester worker (pilot) counts by site |

Results are stored in `external_data` under `"tool:<name>"` and forwarded to the LLM extractor
as additional context.

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

