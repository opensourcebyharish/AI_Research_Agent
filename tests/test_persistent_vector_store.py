from pathlib import Path

import pytest

from app.services.persistent_vector_store import (
    PersistentVectorStore,
)


def test_persistent_vector_store_saves_index(tmp_path):
    store = PersistentVectorStore(
        dimension=3,
        storage_dir=tmp_path,
    )

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    chunks = [
        "Artificial intelligence.",
        "Machine learning.",
    ]

    metadata = [
        {
            "source": "research.pdf",
            "page": 1,
            "chunk_id": 0,
        },
        {
            "source": "research.pdf",
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

    assert (
        Path(tmp_path / "index.faiss").exists()
    )

    assert (
        Path(tmp_path / "metadata.pkl").exists()
    )


def test_persistent_vector_store_loads_after_restart(
    tmp_path,
):
    first_store = PersistentVectorStore(
        dimension=3,
        storage_dir=tmp_path,
    )

    first_store.add(
        [
            [1.0, 0.0, 0.0],
        ],
        [
            "Artificial intelligence.",
        ],
        [
            {
                "source": "research.pdf",
                "page": 1,
                "chunk_id": 0,
            }
        ],
    )

    # Simulate application restart by creating
    # a completely new vector store instance.
    second_store = PersistentVectorStore(
        dimension=3,
        storage_dir=tmp_path,
    )

    assert second_store.index.ntotal == 1

    assert second_store.chunks == [
        "Artificial intelligence."
    ]

    assert second_store.metadata == [
        {
            "source": "research.pdf",
            "page": 1,
            "chunk_id": 0,
        }
    ]


def test_persistent_vector_store_retrieves_after_reload(
    tmp_path,
):
    first_store = PersistentVectorStore(
        dimension=3,
        storage_dir=tmp_path,
    )

    first_store.add(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        [
            "Artificial intelligence.",
            "Machine learning.",
        ],
        [
            {
                "source": "ai.pdf",
                "page": 1,
                "chunk_id": 0,
            },
            {
                "source": "ml.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        ],
    )

    second_store = PersistentVectorStore(
        dimension=3,
        storage_dir=tmp_path,
    )

    results = second_store.search(
        [1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1

    assert results[0]["chunk"] == (
        "Artificial intelligence."
    )

    assert results[0]["metadata"] == {
        "source": "ai.pdf",
        "page": 1,
        "chunk_id": 0,
    }


def test_persistent_vector_store_rejects_corrupt_counts(
    tmp_path,
):
    store = PersistentVectorStore(
        dimension=3,
        storage_dir=tmp_path,
    )

    store.add(
        [[1.0, 0.0, 0.0]],
        ["Test chunk."],
        [
            {
                "source": "test.pdf",
                "page": 1,
                "chunk_id": 0,
            }
        ],
    )

    # Corrupt the metadata by removing one chunk.
    import pickle

    metadata_path = (
        Path(tmp_path) / "metadata.pkl"
    )

    with metadata_path.open("rb") as file:
        data = pickle.load(file)

    data["chunks"] = []

    with metadata_path.open("wb") as file:
        pickle.dump(data, file)

    with pytest.raises(
        ValueError,
        match="index and chunk count",
    ):
        PersistentVectorStore(
            dimension=3,
            storage_dir=tmp_path,
        )