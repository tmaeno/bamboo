# Extraction Strategy Plugin System

## Overview

The Bamboo extraction system uses a **pluggable architecture** for extracting knowledge graphs
from different task management systems.  A strategy encapsulates all system-specific logic:
how to parse fields, how to pre-fetch context, which MCP tools to expose, and which source
navigator to use during exploration.

## Supported Extraction Strategies

| Name | Class | Best for |
|---|---|---|
| `panda` *(default)* | `PandaKnowledgeExtractor` | PanDA WMS task data — see [PanDA Integration](PANDA_INTEGRATION.md) |
| `llm` | `LLMBasedKnowledgeExtractor` | Unstructured data, email threads, natural language text |
| `rule_based` / `jira` | `RuleBasedExtractionStrategy` | Structured key-value data without a dedicated strategy |

## Configuration

```env
# .env
EXTRACTION_STRATEGY=panda   # Default. Options: panda, llm, rule_based, [custom]
```

## Usage

### Basic (auto-configured)

```python
from bamboo.agents.extractors.knowledge_graph_extractor import KnowledgeGraphExtractor

extractor = KnowledgeGraphExtractor()   # uses EXTRACTION_STRATEGY from config

graph = await extractor.extract_from_sources(
    email_text=email,
    task_data=task_dict,
    external_data=external_data,
    task_logs={"scheduler": "...", "worker": "..."},
)
```

### Explicit strategy

```python
extractor = KnowledgeGraphExtractor(strategy="llm")
graph = await extractor.extract_from_sources(task_data=task_data)
```

### List registered strategies

```python
from bamboo.agents.extractors import list_extraction_strategies

for s in list_extraction_strategies():
    print(f"{s['name']}: {s['description']}")
```

## Adding a Custom Strategy

### Step 1: Create the strategy class

```python
# bamboo/agents/extractors/my_system_strategy.py

from bamboo.agents.extractors.base import ExtractionStrategy
from bamboo.models.knowledge_entity import KnowledgeGraph


class MySystemStrategy(ExtractionStrategy):
    """Extract from MySystem task format."""

    # ------------------------------------------------------------------
    # Required — must implement all four
    # ------------------------------------------------------------------

    async def extract(
        self,
        email_text: str = "",
        task_data: dict = None,
        external_data: dict = None,
        task_logs: dict[str, str] = None,
        doc_hints: dict[str, str] = None,
    ) -> KnowledgeGraph:
        # Build and return a KnowledgeGraph from the provided sources.
        ...

    def supports_system(self, system_type: str) -> bool:
        return system_type.lower() == "mysystem"

    @property
    def name(self) -> str:
        return "mysystem"

    @property
    def description(self) -> str:
        return "Extract from MySystem task management"

    # ------------------------------------------------------------------
    # Optional hooks — override only when your system needs them
    # ------------------------------------------------------------------

    async def prefetch_hints(self, task_data=None, email_text="") -> dict[str, str]:
        # Return domain-specific context fetched before extraction.
        # Results are passed to extract() as doc_hints.
        # Omit to use the default no-op (returns {}).
        return await fetch_my_system_docs(task_data or {})

    def source_navigator(self):
        # Return a source navigator for ContextEnricher, or None.
        # Used for source-code-level investigation during exploration.
        return MySystemSourceNavigator()

    def builtin_mcp_clients(self) -> list:
        # Return built-in MCP client instances for this system.
        # build_mcp_client() prepends these before any external servers.
        from bamboo.mcp.my_system_mcp_client import MySystemMcpClient
        return [MySystemMcpClient()]
```

### Step 2: Register the strategy

```python
from bamboo.agents.extractors import register_extraction_strategy
from bamboo.agents.extractors.my_system_strategy import MySystemStrategy

register_extraction_strategy("mysystem", MySystemStrategy)
```

### Step 3: Activate via config

```env
EXTRACTION_STRATEGY=mysystem
```

## Strategy Interface: `ExtractionStrategy`

**File:** `bamboo/agents/extractors/base.py`

### Required (abstract)

| Method | Description |
|---|---|
| `async extract(email_text, task_data, external_data, task_logs, doc_hints)` | Extract and return a `KnowledgeGraph` |
| `supports_system(system_type)` | Return `True` if this strategy handles the given system identifier |
| `name` *(property)* | Short unique identifier (e.g. `"mysystem"`) |
| `description` *(property)* | Human-readable description shown in strategy listings |

### Optional hooks (default no-ops)

| Method | Called by | Default | Override when |
|---|---|---|---|
| `async prefetch_hints(task_data, email_text)` | `KnowledgeGraphExtractor.prefetch_hints()` → accumulator & navigator | `{}` | Your system has domain docs or source analysis to fetch before extraction |
| `source_navigator()` | `build_mcp_client()` callers → `ContextEnricher(source_navigator=...)` | `None` | Your system has navigable source code for root-cause investigation |
| `builtin_mcp_clients()` | `build_mcp_client()` | `[]` | Your system has native MCP tools (task fetching, log retrieval, etc.) |

## Strategy Selection Logic

`KnowledgeGraphExtractor` (and `build_mcp_client`) resolve the strategy once at construction:

1. **Explicit name:** `KnowledgeGraphExtractor(strategy="mysystem")`
2. **Config value:** `EXTRACTION_STRATEGY=mysystem` in `.env`
3. **System-type matching:** `supports_system()` is checked if the name is unknown

## Architecture

```
KnowledgeGraphExtractor
    │
    └─ ExtractionStrategy  (resolved once at construction)
            ├─ prefetch_hints()      → doc_hints → passed to extract()
            ├─ extract()             → KnowledgeGraph
            ├─ source_navigator()    → ContextEnricher(source_navigator=...)
            └─ builtin_mcp_clients() → build_mcp_client() prepends these
```

## Factory Functions

| Function | Description |
|---|---|
| `get_extraction_strategy(name=None)` | Instantiate the active (or named) strategy |
| `register_extraction_strategy(name, cls)` | Register a new strategy class |
| `list_extraction_strategies()` | List all registered strategies with name and description |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ValueError: No extraction strategy found for: mysystem` | Call `register_extraction_strategy` before using it |
| Wrong strategy selected | Pass an explicit `strategy=` argument to `KnowledgeGraphExtractor` |
| `prefetch_hints` not called | Ensure the accumulator/navigator is using `KnowledgeGraphExtractor.prefetch_hints()`, not a direct import |

---

For the PanDA strategy implementation details see [PanDA Integration](PANDA_INTEGRATION.md).  
For the extractor source see `bamboo/agents/extractors/`.
