import pytest

from app.services.rag_service import RAGService


class FakeRetrievalService:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
            }
        )

        return self.results


class FakeContextBuilder:
    def __init__(self):
        self.calls = []

    def build_context(self, results):
        self.calls.append(results)

        return "formatted document context"


class FakePromptBuilder:
    def __init__(self):
        self.calls = []

    def build(self, question, context):
        self.calls.append(
            {
                "question": question,
                "context": context,
            }
        )

        return "formatted prompt"


class FakeLLMService:
    def __init__(self):
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)

        return "generated answer"


class FakeCitationService:
    def __init__(self):
        self.calls = []

    def build_citations(self, results):
        self.calls.append(results)

        return [
            {
                "id": 1,
                "source": "paper.pdf",
                "page": 2,
            }
        ]


def create_service(results):
    retrieval_service = FakeRetrievalService(results)
    context_builder = FakeContextBuilder()
    prompt_builder = FakePromptBuilder()
    llm_service = FakeLLMService()
    citation_service = FakeCitationService()

    service = RAGService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
        citation_service=citation_service,
    )

    return (
        service,
        retrieval_service,
        context_builder,
        prompt_builder,
        llm_service,
        citation_service,
    )


def test_answer_returns_generated_answer():
    results = [
        {
            "chunk": "AI is a field of computer science.",
            "distance": 0.2,
            "metadata": {
                "source": "paper.pdf",
                "page": 1,
                "chunk_id": 0,
            },
        }
    ]

    service, _, _, _, _, _ = create_service(results)

    result = service.answer("What is AI?")

    assert result["answer"] == "generated answer"


def test_answer_retrieves_documents_with_top_k():
    results = [
        {
            "chunk": "Relevant information.",
            "distance": 0.1,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        }
    ]

    service, retrieval_service, _, _, _, _ = create_service(results)

    service.answer(
        question="What is AI?",
        top_k=3,
    )

    assert retrieval_service.calls == [
        {
            "query": "What is AI?",
            "top_k": 3,
        }
    ]


def test_answer_builds_context_from_retrieved_results():
    results = [
        {
            "chunk": "Relevant information.",
            "distance": 0.1,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        }
    ]

    service, _, context_builder, _, _, _ = create_service(results)

    service.answer("What is AI?")

    assert context_builder.calls == [results]


def test_answer_builds_prompt_from_question_and_context():
    results = [
        {
            "chunk": "Relevant information.",
            "distance": 0.1,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        }
    ]

    service, _, _, prompt_builder, _, _ = create_service(results)

    service.answer("What is AI?")

    assert prompt_builder.calls == [
        {
            "question": "What is AI?",
            "context": "formatted document context",
        }
    ]


def test_answer_sends_built_prompt_to_llm():
    results = [
        {
            "chunk": "Relevant information.",
            "distance": 0.1,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        }
    ]

    service, _, _, _, llm_service, _ = create_service(results)

    service.answer("What is AI?")

    assert llm_service.calls == [
        "formatted prompt"
    ]


def test_answer_builds_citations_from_results():
    results = [
        {
            "chunk": "Relevant information.",
            "distance": 0.1,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        }
    ]

    service, _, _, _, _, citation_service = create_service(results)

    service.answer("What is AI?")

    assert citation_service.calls == [results]


def test_answer_returns_citations():
    results = [
        {
            "chunk": "Relevant information.",
            "distance": 0.1,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        }
    ]

    service, _, _, _, _, _ = create_service(results)

    result = service.answer("What is AI?")

    assert result["citations"] == [
        {
            "id": 1,
            "source": "paper.pdf",
            "page": 2,
        }
    ]


def test_answer_returns_results_context_and_prompt():
    results = [
        {
            "chunk": "Relevant information.",
            "distance": 0.1,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        }
    ]

    service, _, _, _, _, _ = create_service(results)

    result = service.answer("What is AI?")

    assert result["results"] == results
    assert result["context"] == "formatted document context"
    assert result["prompt"] == "formatted prompt"


def test_answer_handles_no_retrieved_results():
    service, _, _, _, _, citation_service = create_service([])

    result = service.answer("What is AI?")

    assert result["answer"] == (
        "The information is not available in "
        "the provided documents."
    )
    assert result["results"] == []
    assert result["context"] == ""
    assert result["prompt"] == ""
    assert result["citations"] == []
    assert citation_service.calls == []


def test_answer_rejects_empty_question():
    service, _, _, _, _, _ = create_service([])

    with pytest.raises(ValueError, match="question"):
        service.answer("")


def test_answer_rejects_whitespace_question():
    service, _, _, _, _, _ = create_service([])

    with pytest.raises(ValueError, match="question"):
        service.answer("   ")