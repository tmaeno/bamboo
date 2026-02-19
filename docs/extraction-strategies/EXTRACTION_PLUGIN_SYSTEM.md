# Extraction Strategy Plugin System

## Overview

The Bamboo extraction system now uses a **pluggable architecture** for extracting knowledge graphs from different task management systems. This allows you to optimize extraction based on your specific system type and data format.

## Supported Extraction Strategies

### 1. LLM-Based Extraction
- **Name:** `llm`
- **Best For:** Unstructured data, email threads, natural language text
- **Method:** Uses LLM to understand and extract information
- **Advantages:**
  - Handles unstructured/semi-structured data
  - Natural language understanding
  - Flexible interpretation
- **Disadvantages:**
  - Requires LLM API calls
  - Slower than rule-based
  - Costs per API call

### 2. System-Specific Extraction
- **PanDA:** `panda` - Optimized for PanDA, leverages rule-based extraction for PanDA-specific structured fields and formats and LLM-based extraction for unstructured descriptions and comments.

- Can be extended for others

## Configuration

### Setting the Extraction Strategy

In `.env` or configuration:
```env
EXTRACTION_STRATEGY=llm   # Options: llm, panda, [custom strategies]
```

### Environment Variables
```env
# Choose extraction method
EXTRACTION_STRATEGY=llm   # Default: llm
                          # Options: llm, panda, [custom strategies]
```

## Usage

### Basic Usage (Auto-configured)
```python
from bamboo.agents.knowledge_graph_extractor import KnowledgeGraphExtractor

# Uses configured strategy from settings
extractor = KnowledgeGraphExtractor()

graph = await extractor.extract_from_sources(
    email_text=email,
    task_data=task_dict,
    external_data=external_data
)
```

### Specify Strategy
```python
from bamboo.agents.knowledge_graph_extractor import KnowledgeGraphExtractor

# Use specific strategy
extractor = KnowledgeGraphExtractor(strategy="panda")

graph = await extractor.extract_from_sources(
    task_data=task_data,
)
```

### Get Available Strategies

```python
from bamboo.extractors import list_extraction_strategies

strategies = list_extraction_strategies()
for strategy in strategies:
    print(f"{strategy['name']}: {strategy['description']}")
```

## Adding Custom Extraction Strategies

### Step 1: Create Strategy Class

```python
# bamboo/extractors/my_system_strategy.py

from bamboo.extractors.base import ExtractionStrategy
from bamboo.models.knowledge_entity import KnowledgeGraph


class MySystemExtractionStrategy(ExtractionStrategy):
    """Extract from MySystem task format."""

    async def extract(self, email_text="", task_data=None, external_data=None):
        # Extract using MySystem-specific logic
        # Return KnowledgeGraph
        pass

    def supports_system(self, system_type: str) -> bool:
        return system_type.lower() == "mysystem"

    @property
    def name(self) -> str:
        return "mysystem"

    @property
    def description(self) -> str:
        return "Extract from MySystem task management"
```

### Step 2: Register Strategy

```python
# In your initialization code
from bamboo.extractors import register_extraction_strategy
from bamboo.extractors.my_system_strategy import MySystemExtractionStrategy

register_extraction_strategy("mysystem", MySystemExtractionStrategy)
```

### Step 3: Use Strategy
```env
EXTRACTION_STRATEGY=mysystem
```

## Strategy Selection Logic

When you create a `GraphExtractor`:

1. **Explicit strategy:** Uses the specified strategy
   ```python
   GraphExtractor(strategy="jira")  # Uses Jira strategy
   ```

2. **Configuration setting:** Uses `EXTRACTION_STRATEGY`
   ```env
   EXTRACTION_STRATEGY=github
   ```

3. **System support matching:** Finds strategy that supports the value
   - Rule-based supports: jira, github, generic, structured
   - LLM supports: all systems

## Use Cases

### Jira Integration
```python
from bamboo.agents.graph_extractor import GraphExtractor

extractor = GraphExtractor(strategy="jira")

# Jira-optimized extraction
graph = await extractor.extract_from_sources(
    task_data={
        "key": "PROJ-123",
        "summary": "Database connection timeout",
        "description": "...",
        "labels": ["database", "production"],
        "environment": {"os": "Linux", "java": "11"},
        "status": "In Progress"
    }
)
```

### GitHub Integration
```python
extractor = GraphExtractor(strategy="github")

graph = await extractor.extract_from_sources(
    task_data={
        "title": "API timeout on large requests",
        "body": "...",
        "labels": ["bug", "performance"],
        "environment": {"node": "18"}
    }
)
```

### Generic Unstructured Data
```python
extractor = GraphExtractor(strategy="llm")

graph = await extractor.extract_from_sources(
    email_text="Email discussion about the issue...",
    task_data={"description": "Task details..."},
    external_data={"logs": [...], "metrics": [...]}
)
```

## Implementation Details

### Base Class: `ExtractionStrategy`
```python
class ExtractionStrategy(ABC):
    async def extract(self, email_text, task_data, external_data) -> KnowledgeGraph:
        """Extract knowledge graph."""
    
    def supports_system(self, system_type: str) -> bool:
        """Check if supports this system type."""
    
    @property
    def name(self) -> str:
        """Strategy name."""
    
    @property
    def description(self) -> str:
        """Human-readable description."""
```

### Factory Functions

**get_extraction_strategy(strategy)**
- Returns appropriate strategy instance
- Raises ValueError if no strategy found

**list_extraction_strategies()**
- Returns list of registered strategies with details
- Useful for UI/debugging

**register_extraction_strategy(name, strategy_class)**
- Register new strategy
- Called automatically for built-ins

## Architecture Diagram

```
GraphExtractor
    ↓
get_extraction_strategy()
    ↓
Strategy Registry
    ├─ llm
    ├─ rule_based
    ├─ jira
    ├─ github
    └─ [custom strategies]
    ↓
Selected Strategy
    ├─ LLMExtractionStrategy
    ├─ RuleBasedExtractionStrategy
    └─ [CustomStrategy]
    ↓
extract()
    ↓
KnowledgeGraph
```

## Benefits

✅ **Flexibility** - Choose extraction method based on data source  
✅ **Performance** - Use fast rule-based for structured data  
✅ **Accuracy** - Use LLM for complex unstructured data  
✅ **Extensibility** - Easy to add system-specific strategies  
✅ **Maintainability** - Clear separation of concerns  
✅ **Reusability** - Share strategies across projects  

## Troubleshooting

### Strategy Not Found
```
ValueError: No extraction strategy found for: mysystem
```
Solution: Register the strategy before using it

### Wrong Strategy Selected
Use explicit `strategy` parameter:
```python
GraphExtractor(strategy="llm")
```

### Performance Issues
Switch to rule-based for structured data:
```env
EXTRACTION_STRATEGY=rule_based
```

## Future Enhancements

Potential strategies to implement:
- **Azure DevOps** - Azure-specific extraction
- **Linear** - Linear.app optimization
- **Slack** - Slack thread extraction
- **GitLab** - GitLab issue optimization
- **Hybrid** - Combine LLM + rule-based
- **ML-based** - Trained models for specific systems

---

For more information, see the extraction strategies in `bamboo/extractors/`.

