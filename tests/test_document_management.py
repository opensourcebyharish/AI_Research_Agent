from app.services.persistent_vector_store import PersistentVectorStore


def test_remove_source_removes_matching_vectors(tmp_path):
    store = PersistentVectorStore(
        dimension=2,
        storage_dir=tmp_path / "index",
    )

    store.add(
        embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        chunks=[
            "paper one chunk",
            "paper two chunk",
            "paper one second chunk",
        ],
        metadata=[
            {
                "source": "paper1.pdf",
                "page": 1,
                "chunk_id": 0,
            },
            {
                "source": "paper2.pdf",
                "page": 1,
                "chunk_id": 0,
            },
            {
                "source": "paper1.pdf",
                "page": 2,
                "chunk_id": 1,
            },
        ],
    )

    removed = store.remove_source("paper1.pdf")

    assert removed == 2
    assert store.index.ntotal == 1
    assert store.chunks == ["paper two chunk"]
    assert store.metadata == [
        {
            "source": "paper2.pdf",
            "page": 1,
            "chunk_id": 0,
        }
    ]


def test_remove_source_returns_zero_for_unknown_source(tmp_path):
    store = PersistentVectorStore(
        dimension=2,
        storage_dir=tmp_path / "index",
    )

    store.add(
        embeddings=[[1.0, 0.0]],
        chunks=["paper chunk"],
        metadata=[
            {
                "source": "paper.pdf",
                "page": 1,
                "chunk_id": 0,
            }
        ],
    )

    removed = store.remove_source("unknown.pdf")

    assert removed == 0
    assert store.index.ntotal == 1


def test_removed_vectors_stay_removed_after_reload(tmp_path):
    storage_dir = tmp_path / "index"

    store = PersistentVectorStore(
        dimension=2,
        storage_dir=storage_dir,
    )

    store.add(
        embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        chunks=[
            "keep this",
            "remove this",
        ],
        metadata=[
            {
                "source": "keep.pdf",
                "page": 1,
                "chunk_id": 0,
            },
            {
                "source": "remove.pdf",
                "page": 1,
                "chunk_id": 0,
            },
        ],
    )

    store.remove_source("remove.pdf")

    reloaded_store = PersistentVectorStore(
        dimension=2,
        storage_dir=storage_dir,
    )

    assert reloaded_store.index.ntotal == 1
    assert reloaded_store.chunks == ["keep this"]
    assert reloaded_store.metadata == [
        {
            "source": "keep.pdf",
            "page": 1,
            "chunk_id": 0,
        }
    ]