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

### `fetch_task_data(task_id)`

```python
from bamboo.utils.panda_client import fetch_task_data

task_data = await fetch_task_data(12345)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | `int` or `str` | PanDA `jediTaskID` |

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

## Server Configuration

`panda-client-light` reads the server URL from environment variables.  Set them in your
`.env` file or shell before running Bamboo:

| Variable | Default | Description |
|----------|---------|-------------|
| `PANDA_URL` | `http://pandaserver.cern.ch:25080` | Plain HTTP base URL |
| `PANDA_URL_SSL` | `https://pandaserver.cern.ch` | HTTPS base URL (used for most calls) |

To point at a development or local PanDA instance:

```bash
export PANDA_URL=http://mypanda.example.org:25080
export PANDA_URL_SSL=https://mypanda.example.org
```

---

## Usage

### Via CLI scripts

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

### Via the `bamboo` CLI

Inspect raw task data without running the full pipeline:

```bash
# Print JSON to stdout
bamboo fetch-task 12345

# Save to a file
bamboo fetch-task 12345 --output task_12345.json
```

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

    agent = KnowledgeAccumulator(graph_db, vector_db)
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
| `RuntimeError: status=255` | Network failure or wrong server URL | Check `PANDA_URL_SSL`; verify the server is reachable |
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
- Error messages include task ID and `PANDA_URL` hint

