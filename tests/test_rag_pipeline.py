from unittest.mock import MagicMock

import pytest
from reportlab.pdfgen import canvas

from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStore


@pytest.fixture
def rag_pipeline():
    document_loader = MagicMock()
    text_processor = MagicMock()
    embedding_service = MagicMock()

    document_loader.load.return_value = [
        {
            "page_number": 1,
            "text": (
                "Artificial intelligence enables machines "
                "to perform tasks that normally require "
                "human intelligence. "
                "Machine learning is a subset of artificial "
                "intelligence that learns patterns from data."
            ),
        }
    ]

    text_processor.clean_text.side_effect = (
        lambda text: text
    )

    text_processor.chunk_text.return_value = [
        (
            "Artificial intelligence enables machines "
            "to perform tasks that normally require "
            "human intelligence. "
            "Machine learning is a subset of artificial "
            "intelligence that learns patterns from data."
        )
    ]

    embedding_service.embed_chunks.return_value = [
        [1.0, 0.0, 0.0]
    ]

    embedding_service.embed_text.return_value = [
        1.0,
        0.0,
        0.0,
    ]

    vector_store = VectorStore(
        dimension=3
    )

    ingestion_service = IngestionService(
        document_loader=document_loader,
        text_processor=text_processor,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    return (
        ingestion_service,
        retrieval_service,
    )


def test_end_to_end_rag_pipeline(
    rag_pipeline,
    tmp_path,
):
    (
        ingestion_service,
        retrieval_service,
    ) = rag_pipeline

    pdf_path = tmp_path / "research.pdf"

    pdf = canvas.Canvas(
        str(pdf_path)
    )

    pdf.drawString(
        100,
        750,
        "Artificial intelligence enables machines "
        "to perform tasks that normally require "
        "human intelligence.",
    )

    pdf.drawString(
        100,
        730,
        "Machine learning is a subset of artificial "
        "intelligence that learns patterns from data.",
    )

    pdf.save()

    chunk_count = ingestion_service.ingest(
        str(pdf_path)
    )

    assert chunk_count > 0

    results = retrieval_service.retrieve(
        "What is machine learning?",
        top_k=2,
    )

    assert len(results) > 0

    retrieved_text = " ".join(
        result["chunk"]
        for result in results
    )

    assert "Machine learning" in retrieved_text

    first_result = results[0]

    assert "metadata" in first_result

    assert "source" in first_result["metadata"]

    assert "page" in first_result["metadata"]

    assert "chunk_id" in first_result["metadata"]

    # The ingestion service now stores a clean,
    # human-readable filename instead of the temporary
    # filesystem path.
    assert first_result["metadata"]["source"] == (
        "research.pdf"
    )

    assert first_result["metadata"]["page"] == 1

    assert first_result["metadata"]["chunk_id"] == 0