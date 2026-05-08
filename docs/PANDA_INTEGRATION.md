# PanDA Integration

Bamboo can fetch task data directly from a live **PanDA** (Production and Distributed Analysis)
server via `pandaserver.api.v1.http_client.HttpClient`, eliminating the need to prepare a local
JSON file before running the pipeline.

---

## Server Configuration

`HttpClient` reads the server URL and authentication settings from environment variables.
Set them in your `.env` file or shell before running Bamboo:

| Variable | Default | Description |
|----------|---------|-------------|
| `PANDA_API_URL_SSL` | `https://pandaserver.cern.ch:25443/api/v1` | HTTPS API base URL |
| `PANDA_AUTH` | *(unset)* | Set to `oidc` to use OIDC token auth instead of X.509 |
| `PANDA_AUTH_VO` | *(unset)* | VO name when using OIDC |
| `PANDA_AUTH_ID_TOKEN` | *(unset)* | OIDC token value, or `file:<path>` |
| `X509_USER_PROXY` | *(unset)* | Path to X.509 proxy certificate |

To point at a development or local PanDA instance:

```bash
export PANDA_API_URL_SSL=https://mypanda.example.org:25443/api/v1
```

Refer to the [PanDA client setup guide](https://panda-wms.readthedocs.io/en/latest/client/panda-client.html#setup)
for authentication setup (obtaining an OIDC token or X.509 proxy).

---

## Usage

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

## Testing

Unit tests for `fetch_task_data` are in `tests/test_panda_client.py`.  All tests inject a
fake `pandaserver.api.v1.http_client` module via `patch.dict(sys.modules, ...)` — no live
PanDA server is required.

```bash
pytest tests/test_panda_client.py -v
```

Covered cases:

- Successful fetch returning a `dict`
- String task ID coerced to `int`
- Non-zero status code → `RuntimeError`
- API error (`success=False`) → `RuntimeError` with server message
- Non-numeric task ID → `ValueError`
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

**Evaluation script:** `bamboo/scripts/panda/eval_source_navigator.py` — batch-evaluates the
navigator over a collection of errorDialog strings (JSON / CSV files or PanDA task IDs),
deduplicates inputs by normalised pattern, and classifies results as `relevant`, `irrelevant`,
`no_candidates`, `too_many_candidates`, `empty_error_dialog`, or `error`.

```bash
# Single string smoke-test:
python -m bamboo.scripts.panda.eval_source_navigator "scout_ramCount threshold exceeded"

# Batch from a JSON file of task IDs; write report:
python -m bamboo.scripts.panda.eval_source_navigator --from-file task_ids.json --output report.json

# Terms-only mode (LLM extraction + grep, no navigation rounds):
python -m bamboo.scripts.panda.eval_source_navigator --from-file errors.json --terms-only

# With LLM judge on suspicious results:
python -m bamboo.scripts.panda.eval_source_navigator --from-file errors.json --judge
```

---

### `PandaDocNavigator`

**File:** `bamboo/agents/panda_doc_navigator.py`

Builds a heading-hierarchy graph
from the ReadTheDocs HTML pages, LLM-summarises every node, and stores embeddings in a
dedicated Qdrant collection (`panda_docs`). Index is rebuilt lazily on first use and when
the upstream GitHub tree SHA changes.

**Search uses two parallel strategies:**
- *Flat semantic search* — embeds the query and retrieves top-k Qdrant matches; walks the
  parent chain to attach breadcrumb and parent summary.
- *LLM-guided traversal* — one LLM call per hierarchy level selects relevant pages /
  sections, using the parent node's summary as context at each drill-down step.

Results are merged and deduped by node URL. Each `DocResult` carries: `title`, `url`,
`content` (full section text, no truncation), `parent_summary`, `breadcrumb`, `source`
(`"semantic"` | `"llm_traversal"` | `"both"`).

**Staleness detection:** stores `{"sha": "...", "built_at": "..."}` in
`bamboo/data/panda_docs_meta.json`; compares against the live GitHub tree SHA on every
startup.

**Evaluation script:** `bamboo/scripts/panda/eval_doc_navigator.py` — batch-evaluates
the navigator over a collection of query strings (JSON file or single argument),
reports result counts, source-strategy distribution (semantic / LLM traversal / both),
and optional LLM-as-judge verdicts.

```bash
# Single query smoke-test:
python -m bamboo.scripts.panda.eval_doc_navigator "how to set memory limit for jobs"

# Batch from a JSON file of queries; with LLM judge; write report:
python -m bamboo.scripts.panda.eval_doc_navigator --from-file queries.json --judge --output report.json

# Verbose — print top result titles and URLs per query:
python -m bamboo.scripts.panda.eval_doc_navigator --verbose "exhausted task retry limit"
```

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

