import pytest

from app.services.vector_store import VectorStore


def test_vector_store_initializes():
    store = VectorStore(dimension=3)

    assert store.dimension == 3
    assert store.index.ntotal == 0
    assert store.chunks == []
    assert store.metadata == []


def test_vector_store_adds_embeddings_and_metadata():
    store = VectorStore(dimension=3)

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    chunks = [
        "Artificial intelligence",
        "Machine learning",
    ]

    metadata = [
        {
            "source": "paper.pdf",
            "page": 1,
            "chunk_id": 0,
        },
        {
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": 0,
        },
    ]

    store.add(
        embeddings,
        chunks,
        metadata,
    )

    assert store.index.ntotal == 2
    assert store.chunks == chunks
    assert store.metadata == metadata


def test_vector_store_search_returns_similar_chunks():
    store = VectorStore(dimension=3)

    store.add(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        [
            "AI research",
            "Machine learning",
            "Computer vision",
        ],
        [
            {
                "source": "paper.pdf",
                "page": 1,
                "chunk_id": 0,
            },
            {
                "source": "paper.pdf",
                "page": 2,
                "chunk_id": 0,
            },
            {
                "source": "paper.pdf",
                "page": 3,
                "chunk_id": 0,
            },
        ],
    )

    results = store.search(
        [1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2

    assert results[0]["chunk"] == "AI research"

    assert results[0]["distance"] == pytest.approx(0.0)

    assert results[0]["metadata"] == {
        "source": "paper.pdf",
        "page": 1,
        "chunk_id": 0,
    }


def test_search_empty_store_returns_empty():
    store = VectorStore(dimension=3)

    results = store.search(
        [1.0, 0.0, 0.0]
    )

    assert results == []


def test_add_rejects_mismatched_embedding_chunk_lengths():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add(
            [[1.0, 0.0, 0.0]],
            [
                "chunk 1",
                "chunk 2",
            ],
            [
                {
                    "source": "paper.pdf",
                    "page": 1,
                    "chunk_id": 0,
                },
                {
                    "source": "paper.pdf",
                    "page": 2,
                    "chunk_id": 0,
                },
            ],
        )


def test_add_rejects_mismatched_chunk_metadata_lengths():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add(
            [[1.0, 0.0, 0.0]],
            ["chunk"],
            [],
        )


def test_add_rejects_wrong_dimension():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add(
            [[1.0, 0.0]],
            ["chunk"],
            [
                {
                    "source": "paper.pdf",
                    "page": 1,
                    "chunk_id": 0,
                }
            ],
        )


def test_search_rejects_wrong_dimension():
    store = VectorStore(dimension=3)

    store.add(
        [[1.0, 0.0, 0.0]],
        ["chunk"],
        [
            {
                "source": "paper.pdf",
                "page": 1,
                "chunk_id": 0,
            }
        ],
    )

    with pytest.raises(ValueError):
        store.search(
            [1.0, 0.0]
        )


def test_invalid_dimension_rejected():
    with pytest.raises(ValueError):
        VectorStore(dimension=0)


def test_invalid_top_k_rejected():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.search(
            [1.0, 0.0, 0.0],
            top_k=0,
        )