"""Tests for knowledge extractors."""

import pytest
from bamboo.agents.graph_extractor import GraphExtractor


@pytest.mark.asyncio
async def test_graph_extractor():
    """Test graph extraction."""
    extractor = GraphExtractor()

    email_text = """
    We encountered an error with task processing. The database connection
    timed out after 30 seconds. This seems to happen when the system is
    under heavy load. We resolved it by increasing the connection pool size
    and adding retry logic.
    """

    task_data = {
        "task_id": "TEST-123",
        "status": "exhausted",
        "last_error": "Database connection timeout",
    }

    # Note: This requires actual LLM API keys to work
    # In a real test, you'd mock the LLM responses
    # graph = await extractor.extract_from_sources(
    #     email_text=email_text,
    #     task_data=task_data,
    # )
    #
    # assert len(graph.nodes) > 0
    # assert len(graph.relationships) > 0

