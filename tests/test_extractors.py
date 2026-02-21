"""Tests for knowledge extractors."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bamboo.extractors.panda_knowledge_extractor import (
    CanonicalNodeStore,
    ErrorCategoryClassifier,
    ErrorCategoryStore,
    PandaKnowledgeExtractor,
    _canonical_vector_id,
    _generate_category_label,
    _make_cause_resolution_label_fn,
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
    store = MagicMock(spec=ErrorCategoryStore)
    store.find_or_create = AsyncMock(return_value=(category, score, is_new))
    store.add_category = AsyncMock(return_value="fake-id")
    store.list_categories = AsyncMock(return_value=[])
    return store


def _make_mock_canonical_store(
    label: str = "some canonical label",
    score: float = 0.9,
    is_new: bool = False,
) -> CanonicalNodeStore:
    store = MagicMock(spec=CanonicalNodeStore)
    store.find_or_create = AsyncMock(return_value=(label, score, is_new))
    store.add = AsyncMock(return_value="fake-id")
    store.list_all = AsyncMock(return_value=[])
    return store


def _make_mock_classifier(
    category: str = "Timeout",
    confidence: float = 0.9,
) -> ErrorCategoryClassifier:
    return ErrorCategoryClassifier(store=_make_mock_store(category, confidence))


def _make_fake_vector_client(search_results: list) -> MagicMock:
    client = MagicMock()
    client.search_similar = AsyncMock(return_value=search_results)
    client.upsert_section_vector = AsyncMock(return_value="fake-id")
    return client


def _make_fake_embeddings() -> MagicMock:
    emb = MagicMock()
    emb.embed_query = MagicMock(return_value=[0.1] * 8)
    return emb


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_canonical_vector_id_stable(self):
        assert _canonical_vector_id("Cause", "db timeout") == _canonical_vector_id("Cause", "db timeout")
        assert _canonical_vector_id("Cause", "x") != _canonical_vector_id("Resolution", "x")

    def test_canonical_vector_id_different_types(self):
        assert _canonical_vector_id("Cause", "x") != _canonical_vector_id("Resolution", "x")
        assert _canonical_vector_id("ErrorCategory", "x") != _canonical_vector_id("Cause", "x")

    @pytest.mark.asyncio
    async def test_generate_category_label_uses_llm(self):
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
        mock_response = MagicMock()
        mock_response.content = "TooManyFilesInDataset"
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            label1 = await _generate_category_label(
                "failed to insert files for mc20_13TeV. Input dataset contains too many files >200000."
            )
            label2 = await _generate_category_label(
                "failed to insert files for user.skondo.eventpick.AOD. Input dataset contains too many files >200000."
            )
        assert label1 == label2 == "TooManyFilesInDataset"

    @pytest.mark.asyncio
    async def test_generate_category_label_raises_on_llm_failure(self):
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm") as mock_get_llm:
            mock_get_llm.side_effect = RuntimeError("no API key")
            with pytest.raises(RuntimeError, match="no API key"):
                await _generate_category_label("connection refused on port 5432")

    @pytest.mark.asyncio
    async def test_generate_category_label_raises_on_empty_response(self):
        mock_response = MagicMock()
        mock_response.content = "123 !@#"
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            with pytest.raises(ValueError, match="empty or non-alphabetic"):
                await _generate_category_label("some error message")

    @pytest.mark.asyncio
    async def test_make_cause_resolution_label_fn_returns_canonical(self):
        mock_response = MagicMock()
        mock_response.content = "input dataset exceeds file limit"
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            fn = _make_cause_resolution_label_fn("Cause")
            result = await fn("dataset mc20_13TeV contains too many files")
        assert result == "input dataset exceeds file limit"

    @pytest.mark.asyncio
    async def test_make_cause_resolution_label_fn_raises_on_empty(self):
        mock_response = MagicMock()
        mock_response.content = ""
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm
            fn = _make_cause_resolution_label_fn("Cause")
            with pytest.raises(ValueError, match="empty canonical name"):
                await fn("some cause")


# ---------------------------------------------------------------------------
# CanonicalNodeStore
# ---------------------------------------------------------------------------

class TestCanonicalNodeStore:
    """Tests for the generic CanonicalNodeStore used by all canonicalisable types."""

    def _make_store(
        self, search_results: list, label_fn_return: str = "canonical label"
    ) -> CanonicalNodeStore:
        async def _fake_label_fn(raw: str) -> str:
            return label_fn_return

        return CanonicalNodeStore(
            node_type="Cause",
            label_fn=_fake_label_fn,
            vector_client=_make_fake_vector_client(search_results),
            embeddings_client=_make_fake_embeddings(),
            match_threshold=0.82,
        )

    @pytest.mark.asyncio
    async def test_find_existing_returns_stored_label(self):
        results = [{"id": "x", "score": 0.91, "content": "existing cause",
                    "metadata": {"label": "existing cause"}}]
        store = self._make_store(results)
        label, score, is_new = await store.find_or_create("some wordy cause description")
        assert label == "existing cause"
        assert score == 0.91
        assert is_new is False
        store._vector_client.upsert_section_vector.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_match_stores_and_returns_candidate(self):
        store = self._make_store(search_results=[], label_fn_return="input dataset exceeds file limit")
        label, score, is_new = await store.find_or_create("dataset mc20 too many files")
        assert label == "input dataset exceeds file limit"
        assert score == 0.0
        assert is_new is True
        store._vector_client.upsert_section_vector.assert_called_once()
        call_kwargs = store._vector_client.upsert_section_vector.call_args.kwargs
        assert call_kwargs["content"] == "input dataset exceeds file limit"
        assert call_kwargs["section"] == "canonical_node::Cause"
        assert call_kwargs["metadata"]["label"] == "input dataset exceeds file limit"

    @pytest.mark.asyncio
    async def test_section_is_namespaced_by_node_type(self):
        """Different node types must not share the same VectorDB section."""
        store = self._make_store([], "some label")
        await store.find_or_create("raw text")
        call_kwargs = store._vector_client.upsert_section_vector.call_args.kwargs
        assert call_kwargs["section"] == "canonical_node::Cause"

    @pytest.mark.asyncio
    async def test_add_stores_with_auto_generated_false(self):
        store = self._make_store([])
        await store.add("my manual label")
        call_kwargs = store._vector_client.upsert_section_vector.call_args.kwargs
        assert call_kwargs["metadata"]["auto_generated"] is False
        assert call_kwargs["metadata"]["label"] == "my manual label"

    @pytest.mark.asyncio
    async def test_list_all_returns_entries(self):
        results = [
            {"id": "a", "score": 0.9, "content": "cause A",
             "metadata": {"label": "cause A", "auto_generated": False}},
            {"id": "b", "score": 0.8, "content": "cause B",
             "metadata": {"label": "cause B", "auto_generated": True}},
        ]
        store = self._make_store(results)
        entries = await store.list_all()
        labels = [e["label"] for e in entries]
        assert "cause A" in labels
        assert "cause B" in labels


# ---------------------------------------------------------------------------
# ErrorCategoryStore  (specialisation of CanonicalNodeStore)
# ---------------------------------------------------------------------------

class TestErrorCategoryStore:

    def _make_store(self, search_results: list) -> ErrorCategoryStore:
        store = ErrorCategoryStore(
            vector_client=_make_fake_vector_client(search_results),
            embeddings_client=_make_fake_embeddings(),
            new_category_threshold=0.70,
        )
        return store

    @pytest.mark.asyncio
    async def test_find_existing_category(self):
        results = [{"id": "x", "score": 0.91, "content": "Timeout",
                    "metadata": {"label": "Timeout"}}]
        store = self._make_store(results)
        store._label_fn = AsyncMock(return_value="Timeout")
        label, score, is_new = await store.find_or_create("request timed out")
        assert label == "Timeout"
        assert score == 0.91
        assert is_new is False

    @pytest.mark.asyncio
    async def test_creates_new_category_when_no_match(self):
        store = self._make_store([])
        store._label_fn = AsyncMock(return_value="BizarreCustomFailure")
        label, score, is_new = await store.find_or_create("bizarre custom failure xyz")
        assert label == "BizarreCustomFailure"
        assert score == 0.0
        assert is_new is True
        call_kwargs = store._vector_client.upsert_section_vector.call_args.kwargs
        assert call_kwargs["section"] == "canonical_node::ErrorCategory"
        assert call_kwargs["content"] == "BizarreCustomFailure"

    @pytest.mark.asyncio
    async def test_add_category(self):
        store = self._make_store([])
        await store.add_category("CustomDB", "Some exotic database failure")
        store._vector_client.upsert_section_vector.assert_called_once()
        call_kwargs = store._vector_client.upsert_section_vector.call_args.kwargs
        assert call_kwargs["metadata"]["label"] == "CustomDB"
        assert call_kwargs["metadata"]["auto_generated"] is False

    @pytest.mark.asyncio
    async def test_list_categories_adds_description_key(self):
        results = [
            {"id": "a", "score": 0.9, "content": "CatA",
             "metadata": {"label": "CatA", "auto_generated": False}},
        ]
        store = self._make_store(results)
        cats = await store.list_categories()
        assert cats[0]["description"] == cats[0]["label"]

    def test_is_subclass_of_canonical_node_store(self):
        assert issubclass(ErrorCategoryStore, CanonicalNodeStore)

    def test_new_category_threshold_property(self):
        store = ErrorCategoryStore(new_category_threshold=0.65)
        assert store.new_category_threshold == 0.65
        store.new_category_threshold = 0.80
        assert store.match_threshold == 0.80


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
# PandaKnowledgeExtractor — structured fields
# ---------------------------------------------------------------------------

class TestPandaKnowledgeExtractor:

    def _extractor(self, category="Timeout", confidence=0.9) -> PandaKnowledgeExtractor:
        return PandaKnowledgeExtractor(
            error_classifier=_make_mock_classifier(category, confidence),
            cause_store=_make_mock_canonical_store("some cause"),
            resolution_store=_make_mock_canonical_store("some resolution"),
        )

    @pytest.mark.asyncio
    async def test_external_data_becomes_feature_nodes(self):
        ext = self._extractor()
        graph = await ext.extract(external_data={"env": "production", "region": "us-east-1"})
        names = {n.name for n in graph.nodes}
        assert "env=production" in names
        assert "region=us-east-1" in names
        for node in graph.nodes:
            assert node.node_type == NodeType.TASK_FEATURE

    @pytest.mark.asyncio
    async def test_discrete_task_field_becomes_feature_node(self):
        ext = self._extractor()
        graph = await ext.extract(task_data={"coreCount": "8", "cmtConfig": "x86_64"})
        names = {n.name for n in graph.nodes}
        assert "coreCount=8" in names
        assert "cmtConfig=x86_64" in names

    @pytest.mark.asyncio
    async def test_unstructured_task_field_becomes_context_node(self):
        ext = self._extractor()
        graph = await ext.extract(
            task_data={"taskName": "The system crashes intermittently under load."}
        )
        assert len(graph.nodes) == 1
        node = graph.nodes[0]
        assert node.node_type == NodeType.TASK_CONTEXT
        assert node.name == "taskName"

    @pytest.mark.asyncio
    async def test_error_message_produces_symptom_node(self):
        ext = self._extractor(category="NetworkError", confidence=0.87)
        graph = await ext.extract(
            task_data={"errorDialog": "Connection refused: host unreachable"}
        )
        assert len(graph.nodes) == 1
        node = graph.nodes[0]
        assert node.node_type == NodeType.SYMPTOM
        assert node.name == "NetworkError"
        # Raw message preserved as description for traceability + vector search
        assert node.description == "Connection refused: host unreachable"

    @pytest.mark.asyncio
    async def test_error_message_no_relationship_no_context_node(self):
        """errorDialog must not produce a TaskContextNode or any relationship."""
        ext = self._extractor(category="Timeout", confidence=0.95)
        graph = await ext.extract(task_data={"errorDialog": "Request timed out after 30s"})
        assert len(graph.relationships) == 0
        assert all(n.node_type != NodeType.TASK_CONTEXT for n in graph.nodes)

    @pytest.mark.asyncio
    async def test_empty_error_message_produces_no_node(self):
        ext = self._extractor()
        graph = await ext.extract(task_data={"errorDialog": ""})
        assert len(graph.nodes) == 0
        assert len(graph.relationships) == 0

    @pytest.mark.asyncio
    async def test_status_produces_symptom_node(self):
        """task_data['status'] must produce a SymptomNode, not a TaskFeatureNode."""
        ext = self._extractor(category="TaskFailed", confidence=0.90)
        graph = await ext.extract(task_data={"status": "failed"})
        assert len(graph.nodes) == 1
        node = graph.nodes[0]
        assert node.node_type == NodeType.SYMPTOM
        assert node.name == "TaskFailed"
        assert node.description == "failed"
        assert node.metadata["source"] == "task_status"

    @pytest.mark.asyncio
    async def test_status_not_in_discrete_keys(self):
        """'status' must not appear in DISCRETE_TASK_KEYS."""
        from bamboo.extractors.panda_knowledge_extractor import DISCRETE_TASK_KEYS
        assert "status" not in DISCRETE_TASK_KEYS

    @pytest.mark.asyncio
    async def test_split_rule_produces_multiple_feature_nodes(self):
        """splitRule pipe-separated string must produce one TaskFeatureNode per sub-rule."""
        ext = self._extractor()
        graph = await ext.extract(
            task_data={"splitRule": "nGBPerJob=10|nFilesPerJob=5|nMaxFilesPerJob=100"}
        )
        names = {n.name for n in graph.nodes}
        assert "nGBPerJob=10" in names
        assert "nFilesPerJob=5" in names
        assert "nMaxFilesPerJob=100" in names
        assert len(graph.nodes) == 3
        for node in graph.nodes:
            assert node.node_type == NodeType.TASK_FEATURE

    @pytest.mark.asyncio
    async def test_split_rule_invalid_sub_rule_skipped(self):
        """splitRule sub-rules without '=' are skipped with a warning."""
        ext = self._extractor()
        graph = await ext.extract(task_data={"splitRule": "nGBPerJob=10|badentry|nFilesPerJob=5"})
        assert len(graph.nodes) == 2
        names = {n.name for n in graph.nodes}
        assert "nGBPerJob=10" in names
        assert "nFilesPerJob=5" in names

    @pytest.mark.asyncio
    async def test_split_rule_not_in_discrete_keys(self):
        """'splitRule' must not appear in DISCRETE_TASK_KEYS."""
        from bamboo.extractors.panda_knowledge_extractor import DISCRETE_TASK_KEYS
        assert "splitRule" not in DISCRETE_TASK_KEYS

    @pytest.mark.asyncio
    async def test_unknown_task_key_is_skipped(self):
        """Keys not in DISCRETE_TASK_KEYS or UNSTRUCTURED_TASK_KEYS must be skipped."""
        ext = self._extractor()
        graph = await ext.extract(task_data={"unknownBlobField": "some value"})
        assert len(graph.nodes) == 0

    @pytest.mark.asyncio
    async def test_mixed_input(self):
        ext = self._extractor(category="DatabaseError", confidence=0.92)
        with patch.object(ext, "_extract_from_email", new=AsyncMock(return_value=([], []))):
            graph = await ext.extract(
                email_text="some email text",
                task_data={"taskPriority": "900", "errorDialog": "SQL deadlock", "taskName": "DB contention"},
                external_data={"component": "auth-service"},
            )
        names = {n.name for n in graph.nodes}
        node_types = {n.node_type for n in graph.nodes}
        assert "component=auth-service" in names
        assert "taskPriority=900" in names
        assert "DatabaseError" in names          # SymptomNode from errorDialog
        assert NodeType.SYMPTOM in node_types
        assert "taskName" in names               # TaskContextNode
        assert len(graph.relationships) == 0     # no relationships from structured data

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

    def test_cause_and_resolution_stores_injected(self):
        cause = _make_mock_canonical_store("cause label")
        res = _make_mock_canonical_store("res label")
        ext = PandaKnowledgeExtractor(cause_store=cause, resolution_store=res)
        assert ext._cause_store is cause
        assert ext._resolution_store is res


# ---------------------------------------------------------------------------
# PandaKnowledgeExtractor — email extraction
# ---------------------------------------------------------------------------

_LLM_EMAIL_RESPONSE = """{
  "nodes": [
    {
      "node_type": "Cause",
      "name": "input dataset exceeds file limit",
      "description": "The dataset had more than 200000 files.",
      "metadata": {}, "steps": []
    },
    {
      "node_type": "Resolution",
      "name": "split dataset into smaller chunks",
      "description": "Divide the input dataset into subsets below 200000 files.",
      "metadata": {},
      "steps": ["Identify boundaries", "Split into subsets", "Re-submit"]
    },
    {
      "node_type": "Task_Context",
      "name": "timeline",
      "description": "Issue first observed during the nightly run on 2025-11-03.",
      "metadata": {}, "steps": []
    }
  ],
  "relationships": [
    {
      "source_name": "input dataset exceeds file limit",
      "target_name": "split dataset into smaller chunks",
      "relation_type": "solved_by", "confidence": 0.95
    },
    {
      "source_name": "timeline",
      "target_name": "input dataset exceeds file limit",
      "relation_type": "contribute_to", "confidence": 0.6
    }
  ]
}"""


class TestPandaEmailExtraction:

    def _make_mock_llm(self, response_text: str = _LLM_EMAIL_RESPONSE):
        mock_response = MagicMock()
        mock_response.content = response_text
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        return mock_llm

    def _extractor(
        self,
        cause_label: str = "input dataset exceeds file limit",
        resolution_label: str = "split dataset into smaller subsets",
    ) -> PandaKnowledgeExtractor:
        return PandaKnowledgeExtractor(
            error_classifier=_make_mock_classifier(),
            cause_store=_make_mock_canonical_store(cause_label),
            resolution_store=_make_mock_canonical_store(resolution_label),
        )

    @pytest.mark.asyncio
    async def test_email_produces_cause_resolution_context_nodes(self):
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._make_mock_llm()):
            graph = await ext.extract(email_text="... incident email ...")
        node_types = {n.node_type for n in graph.nodes}
        assert NodeType.CAUSE in node_types
        assert NodeType.RESOLUTION in node_types
        assert NodeType.TASK_CONTEXT in node_types

    @pytest.mark.asyncio
    async def test_canonical_name_from_store_used_not_raw(self):
        """The name in the graph node must be from the store, not from the LLM extraction."""
        ext = self._extractor(cause_label="input dataset exceeds file limit",
                               resolution_label="split dataset into smaller subsets")
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._make_mock_llm()):
            graph = await ext.extract(email_text="... incident email ...")
        names = {n.name for n in graph.nodes}
        assert "input dataset exceeds file limit" in names
        assert "split dataset into smaller subsets" in names
        # raw name from LLM response must NOT appear as a node name
        assert "split dataset into smaller chunks" not in names

    @pytest.mark.asyncio
    async def test_canonical_name_used_in_relationship(self):
        ext = self._extractor(resolution_label="split dataset into smaller subsets")
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._make_mock_llm()):
            graph = await ext.extract(email_text="... incident email ...")
        res_rel = next(r for r in graph.relationships if r.relation_type == RelationType.SOLVED_BY)
        assert res_rel.target_id == "split dataset into smaller subsets"

    @pytest.mark.asyncio
    async def test_cause_store_called_for_cause_nodes(self):
        cause_store = _make_mock_canonical_store("input dataset exceeds file limit")
        res_store = _make_mock_canonical_store("split dataset into smaller subsets")
        ext = PandaKnowledgeExtractor(
            error_classifier=_make_mock_classifier(),
            cause_store=cause_store,
            resolution_store=res_store,
        )
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._make_mock_llm()):
            await ext.extract(email_text="... incident email ...")
        cause_store.find_or_create.assert_called_once_with("input dataset exceeds file limit")
        res_store.find_or_create.assert_called_once_with("split dataset into smaller chunks")

    @pytest.mark.asyncio
    async def test_email_resolution_steps_parsed(self):
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._make_mock_llm()):
            graph = await ext.extract(email_text="... incident email ...")
        resolution = next(n for n in graph.nodes if n.node_type == NodeType.RESOLUTION)
        assert len(resolution.steps) == 3

    @pytest.mark.asyncio
    async def test_email_relationships_created(self):
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._make_mock_llm()):
            graph = await ext.extract(email_text="... incident email ...")
        rel_types = {r.relation_type for r in graph.relationships}
        assert RelationType.SOLVED_BY in rel_types
        assert RelationType.CONTRIBUTE_TO in rel_types

    @pytest.mark.asyncio
    async def test_email_empty_text_skips_llm(self):
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm") as mock_get_llm:
            graph = await ext.extract(email_text="   ")
        mock_get_llm.assert_not_called()
        assert all(n.node_type not in (NodeType.CAUSE, NodeType.RESOLUTION) for n in graph.nodes)

    @pytest.mark.asyncio
    async def test_email_disallowed_node_types_skipped(self):
        bad_response = """{
          "nodes": [
            {"node_type": "Symptom", "name": "bad", "description": "x", "metadata": {}},
            {"node_type": "Cause",   "name": "real cause", "description": "y", "metadata": {}}
          ],
          "relationships": []
        }"""
        ext = self._extractor(cause_label="real cause")
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._make_mock_llm(bad_response)):
            graph = await ext.extract(email_text="some email")
        assert all(n.node_type != NodeType.SYMPTOM for n in graph.nodes)
        assert any(n.name == "real cause" for n in graph.nodes)

    @pytest.mark.asyncio
    async def test_email_malformed_json_returns_empty(self):
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._make_mock_llm("not json at all")):
            graph = await ext.extract(email_text="some email")
        assert graph.nodes == []
        assert graph.relationships == []

    @pytest.mark.asyncio
    async def test_email_and_task_data_merged(self):
        ext = self._extractor()
        with patch("bamboo.extractors.panda_knowledge_extractor.get_llm",
                   return_value=self._make_mock_llm()):
            graph = await ext.extract(
                email_text="... incident email ...",
                task_data={"priority": "high"},
            )
        node_types = {n.node_type for n in graph.nodes}
        assert NodeType.TASK_FEATURE in node_types
        assert NodeType.CAUSE in node_types
        assert NodeType.RESOLUTION in node_types
