"""Tests for knowledge extractors."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bamboo.extractors.panda_knowledge_extractor import (
    UNSTRUCTURED_TASK_KEYS,
    ErrorCategoryClassifier,
    ErrorCategoryStore,
    PandaKnowledgeExtractor,
    _category_vector_id,
    _generate_category_label,
)
from bamboo.models.graph_element import NodeType, RelationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_store(
    category: str = "Timeout",
    score: float = 0.9,
    is_new: bool = False,
) -> ErrorCategoryStore:
    """Return an ErrorCategoryStore whose find_or_create() is mocked."""
    store = MagicMock(spec=ErrorCategoryStore)
    store.find_or_create = AsyncMock(return_value=(category, score, is_new))
    store.add_category = AsyncMock(return_value="fake-id")
    store.list_categories = AsyncMock(return_value=[])
    return store


def _make_mock_classifier(
    category: str = "Timeout",
    confidence: float = 0.9,
) -> ErrorCategoryClassifier:
    """Return an ErrorCategoryClassifier whose classify() is mocked."""
    store = _make_mock_store(category, confidence)
    clf = ErrorCategoryClassifier(store=store)
    return clf


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_category_vector_id_stable(self):
        assert _category_vector_id("Timeout") == _category_vector_id("Timeout")
        assert _category_vector_id("Timeout") != _category_vector_id("NetworkError")

    @pytest.mark.asyncio
    async def test_generate_category_label_uses_llm(self):
        """LLM response is cleaned and returned as the label."""
        mock_response = MagicMock()
        mock_response.content = "  TooManyFilesInDataset  "
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            label = await _generate_category_label("failed to insert files ... >200000")
        assert label == "TooManyFilesInDataset"

    @pytest.mark.asyncio
    async def test_generate_category_label_same_for_similar_messages(self):
        """Two messages that differ only in incident-specific tokens must map
        to the same label when the LLM strips those tokens correctly."""
        mock_response = MagicMock()
        mock_response.content = "TooManyFilesInDataset"
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            label1 = await _generate_category_label(
                "failed to insert files for mc20_13TeV:mc20_13TeV.900149."
                "PG_single_nu_Pt50.digit.RDO.e8307_s3482_s3136_d1715/. "
                "Input dataset contains too many files >200000."
            )
            label2 = await _generate_category_label(
                "failed to insert files for user.skondo.Znnjets_mc20_700337_mc20e"
                ".eventpick.AOD.DTRun3_v1.2.15.log. "
                "Input dataset contains too many files >200000."
            )
        assert label1 == label2 == "TooManyFilesInDataset"

    @pytest.mark.asyncio
    async def test_generate_category_label_raises_on_llm_failure(self):
        """LLM errors must propagate so nothing is written to the database."""
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm") as mock_get_llm:
            mock_get_llm.side_effect = RuntimeError("no API key")
            with pytest.raises(RuntimeError, match="no API key"):
                await _generate_category_label("connection refused on port 5432")

    @pytest.mark.asyncio
    async def test_generate_category_label_raises_on_empty_response(self):
        """An empty/non-alphabetic LLM response must also raise."""
        mock_response = MagicMock()
        mock_response.content = "123 !@#"  # no alphabetic characters
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            with pytest.raises(ValueError, match="empty or non-alphabetic"):
                await _generate_category_label("some error message")


# ---------------------------------------------------------------------------
# ErrorCategoryStore
# ---------------------------------------------------------------------------


class TestErrorCategoryStore:
    """Unit-tests using a fake async vector client."""

    def _make_store(
        self, existing_results: list
    ) -> tuple[ErrorCategoryStore, MagicMock]:
        fake_client = MagicMock()
        fake_client.search_similar = AsyncMock(return_value=existing_results)
        fake_client.upsert_section_vector = AsyncMock(return_value="fake-id")

        fake_embeddings = MagicMock()
        fake_embeddings.embed_query = MagicMock(return_value=[0.1] * 8)
        fake_embeddings.embed_documents = MagicMock(return_value=[[0.1] * 8])

        store = ErrorCategoryStore(
            vector_client=fake_client,
            embeddings_client=fake_embeddings,
            new_category_threshold=0.70,
        )
        return store, fake_client

    @pytest.mark.asyncio
    async def test_find_existing_category(self):
        results = [{"id": "x", "score": 0.91, "content": "Timeout",
                    "metadata": {"label": "Timeout"}}]
        store, client = self._make_store(results)
        with patch(
            "bamboo.extractors.panda_knowledge_extractor._generate_category_label",
            new=AsyncMock(return_value="Timeout"),
        ):
            label, score, is_new = await store.find_or_create("request timed out")
        assert label == "Timeout"
        assert score == 0.91
        assert is_new is False
        # The label was found — nothing should be written to the store.
        client.upsert_section_vector.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_new_category_when_no_match(self):
        store, client = self._make_store(existing_results=[])
        with patch(
            "bamboo.extractors.panda_knowledge_extractor._generate_category_label",
            new=AsyncMock(return_value="BizarreCustomFailure"),
        ):
            label, score, is_new = await store.find_or_create("bizarre custom failure xyz")
        assert is_new is True
        assert score == 0.0
        assert label == "BizarreCustomFailure"
        client.upsert_section_vector.assert_called_once()
        call_kwargs = client.upsert_section_vector.call_args.kwargs
        # The stored content must be the clean label, not the raw message.
        assert call_kwargs["content"] == "BizarreCustomFailure"
        assert call_kwargs["metadata"]["label"] == "BizarreCustomFailure"
        assert call_kwargs["section"] == "error_category"

    @pytest.mark.asyncio
    async def test_add_category(self):
        store, client = self._make_store([])
        await store.add_category("CustomDB", "Some exotic database failure")
        client.upsert_section_vector.assert_called_once()
        call_kwargs = client.upsert_section_vector.call_args.kwargs
        assert call_kwargs["metadata"]["label"] == "CustomDB"
        assert call_kwargs["metadata"]["auto_generated"] is False

    @pytest.mark.asyncio
    async def test_list_categories(self):
        results = [
            {
                "id": "a",
                "score": 0.9,
                "content": "desc A",
                "metadata": {"label": "CatA", "auto_generated": False},
            },
            {
                "id": "b",
                "score": 0.8,
                "content": "desc B",
                "metadata": {"label": "CatB", "auto_generated": True},
            },
        ]
        store, _ = self._make_store(results)
        cats = await store.list_categories()
        labels = [c["label"] for c in cats]
        assert "CatA" in labels
        assert "CatB" in labels


# ---------------------------------------------------------------------------
# ErrorCategoryClassifier
# ---------------------------------------------------------------------------


class TestErrorCategoryClassifier:
    @pytest.mark.asyncio
    async def test_classify_returns_store_result(self):
        clf = _make_mock_classifier("DatabaseError", 0.88)
        label, score = await clf.classify("SQL deadlock detected")
        assert label == "DatabaseError"
        assert score == 0.88

    @pytest.mark.asyncio
    async def test_classify_empty_returns_unknown(self):
        clf = _make_mock_classifier()
        label, score = await clf.classify("")
        assert label == "Unknown"
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_classify_whitespace_returns_unknown(self):
        clf = _make_mock_classifier()
        label, score = await clf.classify("   ")
        assert label == "Unknown"
        assert score == 0.0

    def test_store_property(self):
        store = _make_mock_store()
        clf = ErrorCategoryClassifier(store=store)
        assert clf.store is store


# ---------------------------------------------------------------------------
# PandaKnowledgeExtractor
# ---------------------------------------------------------------------------


class TestPandaKnowledgeExtractor:
    def _extractor(self, category="Timeout", confidence=0.9) -> PandaKnowledgeExtractor:
        return PandaKnowledgeExtractor(
            error_classifier=_make_mock_classifier(category, confidence)
        )

    # --- external_data -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_external_data_becomes_feature_nodes(self):
        ext = self._extractor()
        graph = await ext.extract(
            external_data={"env": "production", "region": "us-east-1"}
        )
        names = {n.name for n in graph.nodes}
        assert "env=production" in names
        assert "region=us-east-1" in names
        for node in graph.nodes:
            assert node.node_type == NodeType.TASK_FEATURE

    # --- task_data discrete fields ------------------------------------------

    @pytest.mark.asyncio
    async def test_discrete_task_field_becomes_feature_node(self):
        ext = self._extractor()
        graph = await ext.extract(task_data={"RAM": "4GB", "OS": "Ubuntu 22.04"})
        names = {n.name for n in graph.nodes}
        assert "RAM=4GB" in names
        assert "OS=Ubuntu 22.04" in names

    # --- task_data unstructured fields ---------------------------------------

    @pytest.mark.asyncio
    async def test_unstructured_task_field_becomes_context_node(self):
        ext = self._extractor()
        graph = await ext.extract(
            task_data={"description": "The system crashes intermittently under load."}
        )
        assert len(graph.nodes) == 1
        node = graph.nodes[0]
        assert node.node_type == NodeType.TASK_CONTEXT
        assert node.name == "description"
        assert "crashes" in node.description

    # --- ErrorMessage special handling --------------------------------------

    @pytest.mark.asyncio
    async def test_error_message_produces_context_and_feature_nodes(self):
        ext = self._extractor(category="NetworkError", confidence=0.87)
        graph = await ext.extract(
            task_data={"ErrorMessage": "Connection refused: host unreachable"}
        )
        node_types = {n.node_type for n in graph.nodes}
        assert NodeType.TASK_CONTEXT in node_types
        assert NodeType.TASK_FEATURE in node_types

        feat = next(n for n in graph.nodes if n.node_type == NodeType.TASK_FEATURE)
        assert feat.name == "ErrorCategory=NetworkError"
        assert feat.attribute == "ErrorCategory"
        assert feat.value == "NetworkError"

        ctx = next(n for n in graph.nodes if n.node_type == NodeType.TASK_CONTEXT)
        assert ctx.name == "ErrorMessage"

    @pytest.mark.asyncio
    async def test_error_message_creates_relationship(self):
        ext = self._extractor(category="Timeout", confidence=0.95)
        graph = await ext.extract(
            task_data={"ErrorMessage": "Request timed out after 30s"}
        )
        assert len(graph.relationships) == 1
        rel = graph.relationships[0]
        assert rel.relation_type == RelationType.ASSOCIATED_WITH
        assert rel.source_id == "ErrorMessage"
        assert rel.target_id == "ErrorCategory=Timeout"

    @pytest.mark.asyncio
    async def test_empty_error_message_no_feature_node(self):
        ext = self._extractor()
        graph = await ext.extract(task_data={"ErrorMessage": ""})
        assert all(n.node_type == NodeType.TASK_CONTEXT for n in graph.nodes)
        assert len(graph.relationships) == 0

    # --- mixed input ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_mixed_input(self):
        ext = self._extractor(category="DatabaseError", confidence=0.92)
        # Patch email extraction so it contributes nothing in this test,
        # keeping the assertion count predictable.
        with patch.object(ext, "_extract_from_email", new=AsyncMock(return_value=([], []))):
            graph = await ext.extract(
                email_text="some email text",
                task_data={
                    "priority": "high",
                    "ErrorMessage": "SQL deadlock detected",
                    "description": "Intermittent DB lock contention",
                },
                external_data={"component": "auth-service"},
            )
        names = {n.name for n in graph.nodes}
        assert "component=auth-service" in names
        assert "priority=high" in names
        assert "ErrorCategory=DatabaseError" in names
        assert "ErrorMessage" in names
        assert "description" in names
        assert len(graph.relationships) == 1

    # --- interface -----------------------------------------------------------

    def test_name(self):
        assert PandaKnowledgeExtractor().name == "panda"

    def test_supports_system(self):
        ext = PandaKnowledgeExtractor()
        assert ext.supports_system("panda")
        assert ext.supports_system("PANDA")
        assert ext.supports_system("bamboo_panda")
        assert not ext.supports_system("jira")

    def test_custom_unstructured_keys(self):
        ext = PandaKnowledgeExtractor(unstructured_keys=frozenset({"my_text_field"}))
        assert "my_text_field" in ext._unstructured_keys
        assert "description" not in ext._unstructured_keys


# ---------------------------------------------------------------------------
# Email extraction
# ---------------------------------------------------------------------------

_LLM_EMAIL_RESPONSE = """{
  "nodes": [
    {
      "node_type": "Cause",
      "name": "input dataset exceeds file limit",
      "description": "The dataset had more than 200000 files, exceeding the allowed limit.",
      "metadata": {},
      "steps": []
    },
    {
      "node_type": "Resolution",
      "name": "split dataset into smaller chunks",
      "description": "Divide the input dataset so each subset has fewer than 200000 files.",
      "metadata": {},
      "steps": ["Identify dataset boundaries", "Split into subsets < 200k files", "Re-submit jobs"]
    },
    {
      "node_type": "Task_Context",
      "name": "timeline",
      "description": "Issue first observed during the nightly digitization run on 2025-11-03.",
      "metadata": {},
      "steps": []
    }
  ],
  "relationships": [
    {
      "source_name": "input dataset exceeds file limit",
      "target_name": "split dataset into smaller chunks",
      "relation_type": "solved_by",
      "confidence": 0.95
    },
    {
      "source_name": "timeline",
      "target_name": "input dataset exceeds file limit",
      "relation_type": "contribute_to",
      "confidence": 0.6
    }
  ]
}"""


class TestPandaEmailExtraction:

    def _extractor(self) -> PandaKnowledgeExtractor:
        return PandaKnowledgeExtractor(
            error_classifier=_make_mock_classifier()
        )

    def _mock_llm(self, response_text: str = _LLM_EMAIL_RESPONSE):
        mock_response = MagicMock()
        mock_response.content = response_text
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        return mock_llm

    @pytest.mark.asyncio
    async def test_email_produces_cause_resolution_context_nodes(self):
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._mock_llm()):
            graph = await ext.extract(email_text="... incident email ...")

        node_types = {n.node_type for n in graph.nodes}
        assert NodeType.CAUSE in node_types
        assert NodeType.RESOLUTION in node_types
        assert NodeType.TASK_CONTEXT in node_types

    @pytest.mark.asyncio
    async def test_email_cause_and_resolution_names(self):
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._mock_llm()):
            graph = await ext.extract(email_text="... incident email ...")

        names = {n.name for n in graph.nodes}
        assert "input dataset exceeds file limit" in names
        assert "split dataset into smaller chunks" in names
        assert "timeline" in names

    @pytest.mark.asyncio
    async def test_email_resolution_steps_parsed(self):
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._mock_llm()):
            graph = await ext.extract(email_text="... incident email ...")

        resolution = next(n for n in graph.nodes if n.node_type == NodeType.RESOLUTION)
        assert len(resolution.steps) == 3

    @pytest.mark.asyncio
    async def test_email_relationships_created(self):
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._mock_llm()):
            graph = await ext.extract(email_text="... incident email ...")

        rel_types = {r.relation_type for r in graph.relationships}
        assert RelationType.SOLVED_BY in rel_types
        assert RelationType.CONTRIBUTE_TO in rel_types

    @pytest.mark.asyncio
    async def test_email_empty_text_skips_llm(self):
        """Empty / whitespace-only email_text must not call the LLM."""
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm") as mock_get_llm:
            graph = await ext.extract(email_text="   ")
        mock_get_llm.assert_not_called()
        assert all(
            n.node_type not in (NodeType.CAUSE, NodeType.RESOLUTION)
            for n in graph.nodes
        )

    @pytest.mark.asyncio
    async def test_email_disallowed_node_types_skipped(self):
        """Node types other than Cause/Resolution/Task_Context are silently dropped."""
        bad_response = """{
          "nodes": [
            {"node_type": "Symptom", "name": "bad symptom", "description": "x", "metadata": {}},
            {"node_type": "Cause",   "name": "real cause",  "description": "y", "metadata": {}}
          ],
          "relationships": []
        }"""
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._mock_llm(bad_response)):
            graph = await ext.extract(email_text="some email")

        assert all(n.node_type != NodeType.SYMPTOM for n in graph.nodes)
        assert any(n.name == "real cause" for n in graph.nodes)

    @pytest.mark.asyncio
    async def test_email_malformed_json_returns_empty(self):
        """A non-JSON LLM response must not raise — it yields zero nodes."""
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._mock_llm("not json at all")):
            graph = await ext.extract(email_text="some email")
        assert graph.nodes == []
        assert graph.relationships == []

    @pytest.mark.asyncio
    async def test_email_and_task_data_merged(self):
        """Nodes from email extraction and task_data must coexist in the graph."""
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._mock_llm()):
            graph = await ext.extract(
                email_text="... incident email ...",
                task_data={"priority": "high"},
            )
        node_types = {n.node_type for n in graph.nodes}
        assert NodeType.TASK_FEATURE in node_types   # from task_data
        assert NodeType.CAUSE in node_types           # from email
        assert NodeType.RESOLUTION in node_types      # from email

