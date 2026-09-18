from app.services.citation_service import CitationService
from app.services.context_builder import ContextBuilder
from app.services.prompt_builder import PromptBuilder
from app.services.rag_service import RAGService


def test_retrieval_result_is_relevant():
    """A relevant retrieved chunk should be accepted."""

    results = [
        {
            "chunk": "RAG combines information retrieval with generation.",
            "distance": 0.42,
            "metadata": {
                "source": "research.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        }
    ]

    assert results[0]["distance"] < 1.5
    assert "RAG" in results[0]["chunk"]


def test_irrelevant_retrieval_result_is_rejected():
    """A result beyond the configured relevance threshold is rejected."""

    results = [
        {
            "chunk": "Unrelated information.",
            "distance": 2.1,
            "metadata": {
                "source": "research.pdf",
                "page": 7,
                "chunk_id": 4,
            },
        }
    ]

    relevant_results = [
        result
        for result in results
        if result["distance"] <= 1.5
    ]

    assert relevant_results == []


def test_citation_contains_correct_source_and_page():
    """Citations should preserve source and page information."""

    service = CitationService()

    results = [
        {
            "chunk": "RAG retrieves relevant passages.",
            "distance": 0.4,
            "metadata": {
                "source": "paper.pdf",
                "page": 3,
                "chunk_id": 0,
            },
        }
    ]

    citations = service.build_citations(results)

    assert citations == [
        {
            "id": 1,
            "source": "paper.pdf",
            "page": 3,
        }
    ]


def test_duplicate_citations_are_removed():
    """Multiple chunks from the same page should produce one citation."""

    service = CitationService()

    results = [
        {
            "chunk": "First relevant passage.",
            "distance": 0.3,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        },
        {
            "chunk": "Second relevant passage.",
            "distance": 0.4,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
                "chunk_id": 1,
            },
        },
    ]

    citations = service.build_citations(results)

    assert len(citations) == 1
    assert citations[0]["source"] == "paper.pdf"
    assert citations[0]["page"] == 2


def test_context_contains_source_and_page():
    """Retrieved context should preserve citation metadata."""

    builder = ContextBuilder()

    results = [
        {
            "chunk": "RAG combines retrieval and generation.",
            "metadata": {
                "source": "paper.pdf",
                "page": 4,
                "chunk_id": 2,
            },
        }
    ]

    context = builder.build_context(results)

    assert "paper.pdf" in context
    assert "Page: 4" in context
    assert "Chunk: 2" in context
    assert "RAG combines retrieval and generation." in context


def test_prompt_requires_grounded_answer():
    """The prompt should explicitly prevent unsupported answers."""

    builder = PromptBuilder()

    prompt = builder.build(
        question="What is RAG?",
        context=(
            "[Source: paper.pdf | Page: 1 | Chunk: 0]\n"
            "RAG combines retrieval with language generation."
        ),
    )

    assert "only the provided document context" in prompt
    assert "Do not invent facts" in prompt
    assert "information is not available" in prompt
    assert "paper.pdf" in prompt
    assert "Page: 1" in prompt


def test_rag_returns_fallback_when_nothing_is_retrieved():
    """The RAG service should refuse unsupported questions."""

    class FakeRetrievalService:
        def retrieve(self, query, top_k=5):
            return []

    class FakeContextBuilder:
        def build_context(self, results):
            raise AssertionError(
                "Context should not be built without results."
            )

    class FakePromptBuilder:
        def build(self, question, context):
            raise AssertionError(
                "Prompt should not be built without results."
            )

    class FakeLLMService:
        def generate(self, prompt):
            raise AssertionError(
                "LLM should not be called without results."
            )

    class FakeCitationService:
        def build_citations(self, results):
            raise AssertionError(
                "Citations should not be built without results."
            )

    service = RAGService(
        retrieval_service=FakeRetrievalService(),
        context_builder=FakeContextBuilder(),
        prompt_builder=FakePromptBuilder(),
        llm_service=FakeLLMService(),
        citation_service=FakeCitationService(),
    )

    result = service.answer(
        "What is the capital of France?"
    )

    assert result["answer"] == (
        "The information is not available in "
        "the provided documents."
    )

    assert result["results"] == []
    assert result["context"] == ""
    assert result["prompt"] == ""
    assert result["citations"] == []