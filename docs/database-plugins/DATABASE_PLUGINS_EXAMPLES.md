# Database Plugin Examples

This document provides practical examples of using and extending the database plugin system.

## Example 1: Using Different Backends

### Development with In-Memory Backend

Create `.env.dev`:
```env
GRAPH_DATABASE_BACKEND=in_memory
VECTOR_DATABASE_BACKEND=in_memory  # Future extension
```

Run your app:
```bash
source .env.dev && python -m bamboo.cli
```

### Production with Neo4j and Qdrant

Create `.env.prod`:
```env
GRAPH_DATABASE_BACKEND=neo4j
VECTOR_DATABASE_BACKEND=qdrant

NEO4J_URI=bolt://production.example.com:7687
NEO4J_USERNAME=prod_user
NEO4J_PASSWORD=secure_password
NEO4J_DATABASE=production_graph

QDRANT_URL=http://qdrant.example.com:6333
QDRANT_API_KEY=prod_api_key
```

## Example 2: Simple Test Using In-Memory Backend

```python
# tests/test_knowledge_extraction.py
import os
import pytest

# Use in-memory backend for testing
os.environ['GRAPH_DATABASE_BACKEND'] = 'in_memory'

from bamboo.database import GraphDatabaseClient
from bamboo.models.graph_element import ErrorNode, CauseNode, GraphRelationship


@pytest.mark.asyncio
async def test_create_and_query_nodes():
    """Test knowledge extraction with in-memory backend."""
    
    graph_db = GraphDatabaseClient()
    await graph_db.connect()
    
    try:
        # Create nodes
        error_node = ErrorNode(
            name="Database Connection Timeout",
            description="Connection to database timed out after 30 seconds",
            error_code="DB_TIMEOUT"
        )
        error_id = await graph_db.create_node(error_node)
        assert error_id is not None
        
        cause_node = CauseNode(
            name="Database Server Overload",
            description="Database server was processing too many queries",
            confidence=0.85,
            frequency=5
        )
        cause_id = await graph_db.create_node(cause_node)
        assert cause_id is not None
        
        # Create relationship
        relationship = GraphRelationship(
            source_id=error_id,
            target_id=cause_id,
            relation_type="indicate",
            confidence=0.85
        )
        result = await graph_db.create_relationship(relationship)
        assert result is True
        
        # Query
        causes = await graph_db.find_causes_by_error("timeout")
        assert len(causes) > 0
        assert causes[0]['cause_name'] == "Database Server Overload"
        
    finally:
        await graph_db.close()


@pytest.mark.asyncio
async def test_canonical_node_deduplication():
    """Test that canonical nodes prevent duplicates."""
    
    graph_db = GraphDatabaseClient()
    await graph_db.connect()
    
    try:
        error_node = ErrorNode(
            name="Initial Name",
            description="Error description"
        )
        
        # First call creates the node
        id1 = await graph_db.get_or_create_canonical_node(
            error_node, 
            "Database Connection Timeout"
        )
        
        # Second call with same canonical name returns same node
        id2 = await graph_db.get_or_create_canonical_node(
            error_node,
            "Database Connection Timeout"
        )
        
        assert id1 == id2, "Same canonical name should return same node"
        
    finally:
        await graph_db.close()
```

## Example 3: Creating a Custom Backend

### PostgreSQL with pgvector Backend

```python
# bamboo/database/backends/postgres_vector_backend.py
"""PostgreSQL with pgvector vector database backend."""

import logging
from typing import Any, Optional

try:
    import asyncpg
    import numpy as np
except ImportError as e:
    raise ImportError(
        "PostgreSQL backend requires 'asyncpg' package. "
        "Install it with: pip install asyncpg"
    ) from e

from bamboo.config import get_settings
from bamboo.database.base import VectorDatabaseBackend

logger = logging.getLogger(__name__)


class PostgresVectorBackend(VectorDatabaseBackend):
    """PostgreSQL with pgvector implementation."""

    def __init__(self):
        """Initialize PostgreSQL backend."""
        self.settings = get_settings()
        self.pool = None
        self.table_name = "vectors"

    async def connect(self):
        """Establish connection to PostgreSQL."""
        try:
            self.pool = await asyncpg.create_pool(
                self.settings.postgres_url,
                min_size=5,
                max_size=20,
            )
            
            # Create table if not exists
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    CREATE EXTENSION IF NOT EXISTS vector;
                    CREATE TABLE IF NOT EXISTS vectors (
                        id BIGSERIAL PRIMARY KEY,
                        vector_id VARCHAR(255) UNIQUE NOT NULL,
                        embedding vector(1536) NOT NULL,
                        content TEXT NOT NULL,
                        section VARCHAR(50),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS 
                        ON vectors USING ivfflat (embedding vector_cosine_ops);
                """)
            
            logger.info("Connected to PostgreSQL with pgvector")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def close(self):
        """Close connection."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection closed")

    async def upsert_section_vector(
        self,
        vector_id: str,
        embedding: list[float],
        content: str,
        section: str,
        metadata: dict[str, Any],
    ) -> str:
        """Insert or update a document."""
        async with self.pool.acquire() as conn:
            # Convert list to PostgreSQL vector format
            vector_str = f"[{','.join(map(str, embedding))}]"
            
            query = """
                INSERT INTO vectors 
                (vector_id, embedding, content, section, metadata)
                VALUES ($1, $2::vector, $3, $4, $5)
                ON CONFLICT (vector_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    content = EXCLUDED.content,
                    section = EXCLUDED.section,
                    metadata = EXCLUDED.metadata
            """
            
            await conn.execute(
                query,
                vector_id,
                vector_str,
                content,
                section,
                metadata
            )
        
        return vector_id

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents."""
        vector_str = f"[{','.join(map(str, query_embedding))}]"
        
        query = f"""
            SELECT 
                vector_id,
                content,
                section,
                metadata,
                1 - (embedding <=> $1::vector) as score
            FROM vectors
            WHERE 1 - (embedding <=> $1::vector) > $2
        """
        
        params = [vector_str, score_threshold]
        
        # Add filter conditions if provided
        if filter_conditions:
            for i, (key, value) in enumerate(filter_conditions.items(), 3):
                query += f" AND metadata->>'{key}' = ${i}"
                params.append(str(value))
        
        query += f" ORDER BY score DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        return [
            {
                "id": row['vector_id'],
                "score": float(row['score']),
                "content": row['content'],
                "entry": row['content'][:100],
                "metadata": row['metadata'] or {}
            }
            for row in rows
        ]

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM vectors WHERE vector_id = $1",
                doc_id
            )
        
        return result != "0"

    async def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a document by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM vectors WHERE vector_id = $1",
                doc_id
            )
        
        if row:
            return {
                "id": row['vector_id'],
                "content": row['content'],
                "entry": row['content'][:100],
                "metadata": row['metadata'] or {}
            }
        
        return None
```

### Register the Backend

In `bamboo/database/factory.py`, add to `_register_builtin_backends()`:

```python
try:
    from bamboo.database.backends.postgres_vector_backend import PostgresVectorBackend
    register_vector_backend("postgres", PostgresVectorBackend)
except ImportError as e:
    logger.debug(f"PostgreSQL backend not available: {e}")
```

### Configure to Use

In `.env`:
```env
VECTOR_DATABASE_BACKEND=postgres
POSTGRES_URL=postgresql://user:password@localhost:5432/bamboo
```

## Example 4: Multi-Backend Configuration

Support different backends per environment:

```python
# config_loader.py
import os
from pathlib import Path

def load_config_for_env(env: str = None):
    """Load environment-specific configuration."""
    
    env = env or os.getenv('ENV', 'development')
    env_file = Path(f".env.{env}")
    
    if env_file.exists():
        # Load environment-specific settings
        from dotenv import load_dotenv
        load_dotenv(env_file)
    else:
        # Fall back to default .env
        from dotenv import load_dotenv
        load_dotenv(".env")
    
    return env
```

Usage:
```bash
# Development with in-memory
ENV=development python app.py

# Staging with Neo4j and Qdrant
ENV=staging python app.py

# Production
ENV=production python app.py
```

## Example 5: Dynamic Backend Switching

```python
# Advanced: Switch backends at runtime
from bamboo.database.factory import register_graph_backend
from bamboo.database.backends.examples.in_memory_backend import InMemoryGraphBackend

# Register additional backend
register_graph_backend("test", InMemoryGraphBackend)

# Can now use: GRAPH_DATABASE_BACKEND=test
```

## Example 6: Testing Multiple Backends

```python
# tests/conftest.py
import pytest

BACKENDS_TO_TEST = ["neo4j", "in_memory"]

@pytest.fixture(params=BACKENDS_TO_TEST)
def graph_db_backend(request, monkeypatch):
    """Test with multiple backends."""
    monkeypatch.setenv("GRAPH_DATABASE_BACKEND", request.param)
    from bamboo.database import GraphDatabaseClient
    return GraphDatabaseClient(), request.param


@pytest.mark.asyncio
async def test_create_node_all_backends(graph_db_backend):
    """Test that node creation works on all backends."""
    graph_db, backend_name = graph_db_backend
    
    await graph_db.connect()
    try:
        from bamboo.models.graph_element import ErrorNode
        
        node = ErrorNode(
            name="Test Error",
            description="Test Description"
        )
        
        node_id = await graph_db.create_node(node)
        assert node_id is not None
        print(f"✓ {backend_name} backend works")
        
    finally:
        await graph_db.close()
```

Run with:
```bash
pytest tests/conftest.py::test_create_node_all_backends -v
```

Output:
```
test_create_node_all_backends[neo4j] PASSED
test_create_node_all_backends[in_memory] PASSED
```

## Example 7: Backend Capability Checking

```python
# check_backends.py
from bamboo.database import (
    list_graph_backends,
    list_vector_backends,
    get_graph_backend,
    get_vector_backend
)

def check_backend_availability():
    """Check which backends are available."""
    
    print("=== Database Backend Status ===")
    
    graph_backends = list_graph_backends()
    vector_backends = list_vector_backends()
    
    print(f"\nGraph Backends Available: {graph_backends}")
    print(f"Vector Backends Available: {vector_backends}")
    
    try:
        graph_backend = get_graph_backend()
        print(f"✓ Graph backend ready: {type(graph_backend).__name__}")
    except ValueError as e:
        print(f"✗ Graph backend error: {e}")
    
    try:
        vector_backend = get_vector_backend()
        print(f"✓ Vector backend ready: {type(vector_backend).__name__}")
    except ValueError as e:
        print(f"✗ Vector backend error: {e}")

if __name__ == "__main__":
    check_backend_availability()
```

Run with:
```bash
python check_backends.py
```

Output:
```
=== Database Backend Status ===

Graph Backends Available: ['neo4j', 'in_memory']
Vector Backends Available: ['qdrant']
✓ Graph backend ready: Neo4jBackend
✓ Vector backend ready: QdrantBackend
```

## Summary

The plugin system enables:
- ✅ Testing without external databases (in-memory)
- ✅ Easy addition of new backends
- ✅ Environment-specific configurations
- ✅ Multi-backend testing
- ✅ Graceful fallbacks
- ✅ Clear error messages

For more information, see:
- `DATABASE_PLUGINS.md` - Complete technical reference
- `DATABASE_PLUGINS_QUICK_REFERENCE.md` - Quick start guide

