from unittest.mock import MagicMock

from app.services.ingestion_service import IngestionService


def test_ingest_uses_original_source_name():
    document_loader = MagicMock()
    text_processor = MagicMock()
    embedding_service = MagicMock()
    vector_store = MagicMock()

    document_loader.load.return_value = [
        {
            "page_number": 1,
            "text": "Retrieval-Augmented Generation combines retrieval and generation.",
        }
    ]

    text_processor.clean_text.return_value = (
        "Retrieval-Augmented Generation combines retrieval and generation."
    )

    text_processor.chunk_text.return_value = [
        "Retrieval-Augmented Generation combines retrieval and generation."
    ]

    embedding_service.embed_chunks.return_value = [
        [0.1, 0.2, 0.3]
    ]

    service = IngestionService(
        document_loader=document_loader,
        text_processor=text_processor,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    result = service.ingest(
        r"C:\Users\hp\AppData\Local\Temp\tmp123.pdf",
        source_name="ai_research_agent_test.pdf",
    )

    assert result == 1

    vector_store.add.assert_called_once()

    _, _, metadata = vector_store.add.call_args.args

    assert metadata == [
        {
            "source": "ai_research_agent_test.pdf",
            "page": 1,
            "chunk_id": 0,
        }
    ]


def test_ingest_defaults_to_filename():
    document_loader = MagicMock()
    text_processor = MagicMock()
    embedding_service = MagicMock()
    vector_store = MagicMock()

    document_loader.load.return_value = [
        {
            "page_number": 2,
            "text": "Test document content.",
        }
    ]

    text_processor.clean_text.return_value = (
        "Test document content."
    )

    text_processor.chunk_text.return_value = [
        "Test document content."
    ]

    embedding_service.embed_chunks.return_value = [
        [0.1, 0.2, 0.3]
    ]

    service = IngestionService(
        document_loader=document_loader,
        text_processor=text_processor,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    result = service.ingest(
        r"C:\documents\research.pdf"
    )

    assert result == 1

    _, _, metadata = vector_store.add.call_args.args

    assert metadata[0]["source"] == "research.pdf"