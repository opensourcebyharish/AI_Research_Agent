import json

import pytest

from app.services.document_registry import DocumentRegistry


def test_registry_starts_empty(tmp_path):
    registry = DocumentRegistry(
        tmp_path / "documents.json"
    )

    assert registry.list_documents() == []
    assert registry.count() == 0


def test_add_document(tmp_path):
    registry = DocumentRegistry(
        tmp_path / "documents.json"
    )

    document = registry.add(
        source="paper.pdf",
        pages=10,
        chunks=25,
    )

    assert document == {
        "source": "paper.pdf",
        "pages": 10,
        "chunks": 25,
    }

    assert registry.count() == 1
    assert registry.get("paper.pdf") == document


def test_registry_persists_documents(tmp_path):
    storage_path = tmp_path / "documents.json"

    registry = DocumentRegistry(storage_path)

    registry.add(
        source="research.pdf",
        pages=5,
        chunks=12,
    )

    new_registry = DocumentRegistry(storage_path)

    assert new_registry.count() == 1
    assert new_registry.get("research.pdf") == {
        "source": "research.pdf",
        "pages": 5,
        "chunks": 12,
    }


def test_exists(tmp_path):
    registry = DocumentRegistry(
        tmp_path / "documents.json"
    )

    assert registry.exists("paper.pdf") is False

    registry.add(
        source="paper.pdf",
        pages=3,
        chunks=8,
    )

    assert registry.exists("paper.pdf") is True


def test_duplicate_document_is_rejected(tmp_path):
    registry = DocumentRegistry(
        tmp_path / "documents.json"
    )

    registry.add(
        source="paper.pdf",
        pages=3,
        chunks=8,
    )

    with pytest.raises(
        ValueError,
        match="Document already registered",
    ):
        registry.add(
            source="paper.pdf",
            pages=3,
            chunks=8,
        )


def test_get_missing_document_returns_none(tmp_path):
    registry = DocumentRegistry(
        tmp_path / "documents.json"
    )

    assert registry.get("missing.pdf") is None


def test_remove_document(tmp_path):
    registry = DocumentRegistry(
        tmp_path / "documents.json"
    )

    registry.add(
        source="paper.pdf",
        pages=3,
        chunks=8,
    )

    assert registry.remove("paper.pdf") is True
    assert registry.count() == 0
    assert registry.get("paper.pdf") is None


def test_remove_missing_document(tmp_path):
    registry = DocumentRegistry(
        tmp_path / "documents.json"
    )

    assert registry.remove("missing.pdf") is False


def test_empty_source_is_rejected(tmp_path):
    registry = DocumentRegistry(
        tmp_path / "documents.json"
    )

    with pytest.raises(
        ValueError,
        match="source must not be empty",
    ):
        registry.add(
            source="",
            pages=1,
            chunks=1,
        )


def test_negative_pages_are_rejected(tmp_path):
    registry = DocumentRegistry(
        tmp_path / "documents.json"
    )

    with pytest.raises(
        ValueError,
        match="pages must not be negative",
    ):
        registry.add(
            source="paper.pdf",
            pages=-1,
            chunks=1,
        )


def test_negative_chunks_are_rejected(tmp_path):
    registry = DocumentRegistry(
        tmp_path / "documents.json"
    )

    with pytest.raises(
        ValueError,
        match="chunks must not be negative",
    ):
        registry.add(
            source="paper.pdf",
            pages=1,
            chunks=-1,
        )


def test_registry_file_contains_json(tmp_path):
    storage_path = tmp_path / "documents.json"

    registry = DocumentRegistry(storage_path)

    registry.add(
        source="paper.pdf",
        pages=2,
        chunks=5,
    )

    with storage_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert data == [
        {
            "source": "paper.pdf",
            "pages": 2,
            "chunks": 5,
        }
    ]