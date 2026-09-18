import pytest

from app.services.retrieval_service import RetrievalService


class FakeEmbeddingService:

    def embed_text(self, text):
        return [1.0, 2.0, 3.0]


class FakeVectorStore:

    def __init__(self, results=None):
        self.results = results or []
        self.received_embedding = None
        self.received_top_k = None

    def search(self, query_embedding, top_k=5):
        self.received_embedding = query_embedding
        self.received_top_k = top_k

        return self.results


def test_retrieve_returns_relevant_results():
    results = [
        {
            "chunk": "Relevant research information.",
            "distance": 0.8,
            "metadata": {
                "source": "paper.pdf",
                "page": 1,
            },
        },
        {
            "chunk": "Another relevant passage.",
            "distance": 1.2,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
            },
        },
    ]

    vector_store = FakeVectorStore(results)

    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )

    retrieved = service.retrieve(
        "What is the research about?",
        top_k=5,
    )

    assert len(retrieved) == 2
    assert retrieved[0]["chunk"] == (
        "Relevant research information."
    )
    assert vector_store.received_embedding == [
        1.0,
        2.0,
        3.0,
    ]
    assert vector_store.received_top_k == 5


def test_retrieve_filters_irrelevant_results():
    results = [
        {
            "chunk": "Highly relevant information.",
            "distance": 0.7,
            "metadata": {
                "source": "paper.pdf",
                "page": 1,
            },
        },
        {
            "chunk": "Weakly related information.",
            "distance": 1.4,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
            },
        },
        {
            "chunk": "Unrelated information.",
            "distance": 2.1,
            "metadata": {
                "source": "paper.pdf",
                "page": 3,
            },
        },
    ]

    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(results),
        max_distance=1.5,
    )

    retrieved = service.retrieve(
        "research question",
        top_k=5,
    )

    assert len(retrieved) == 2
    assert retrieved[0]["distance"] == 0.7
    assert retrieved[1]["distance"] == 1.4


def test_retrieve_returns_empty_for_no_results():
    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore([]),
    )

    assert service.retrieve("research question") == []


def test_retrieve_returns_empty_for_empty_query():
    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore([]),
    )

    assert service.retrieve("") == []
    assert service.retrieve("   ") == []


def test_retrieve_rejects_invalid_top_k():
    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore([]),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        service.retrieve(
            "research question",
            top_k=0,
        )


def test_retrieve_can_disable_distance_filter():
    results = [
        {
            "chunk": "Relevant.",
            "distance": 0.8,
            "metadata": {},
        },
        {
            "chunk": "Distant.",
            "distance": 3.0,
            "metadata": {},
        },
    ]

    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(results),
        max_distance=None,
    )

    retrieved = service.retrieve(
        "research question",
        top_k=5,
    )

    assert len(retrieved) == 2