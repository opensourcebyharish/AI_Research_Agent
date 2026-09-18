from app.services.ingestion_service import IngestionService


class FakeDocumentLoader:

    def __init__(self, pages):
        self.pages = pages

    def load(self, file_path):
        return self.pages


class FakeTextProcessor:

    def clean_text(self, text):
        return text.strip()

    def chunk_text(self, text):
        return [
            f"{text} - chunk 1",
            f"{text} - chunk 2",
        ]


class FakeEmbeddingService:

    def embed_chunks(self, chunks):
        return [
            [1.0, 2.0, 3.0]
            for _ in chunks
        ]


class FakeVectorStore:

    def __init__(self):
        self.embeddings = []
        self.chunks = []
        self.metadata = []

    def add(self, embeddings, chunks, metadata):
        self.embeddings.extend(embeddings)
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)


def create_service(pages):
    return IngestionService(
        document_loader=FakeDocumentLoader(pages),
        text_processor=FakeTextProcessor(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
    )


def test_ingest_indexes_document():
    service = create_service(
        [
            {
                "page_number": 1,
                "text": "Page one",
            },
            {
                "page_number": 2,
                "text": "Page two",
            },
        ]
    )

    result = service.ingest(
        "research.pdf"
    )

    assert result == 4
    assert len(service.vector_store.chunks) == 4
    assert len(service.vector_store.metadata) == 4


def test_ingest_preserves_source_name():
    service = create_service(
        [
            {
                "page_number": 1,
                "text": "Research content",
            }
        ]
    )

    result = service.ingest(
        "temporary_upload.pdf",
        source_name="paper_2026.pdf",
    )

    assert result == 2
    assert service.vector_store.metadata[0]["source"] == (
        "paper_2026.pdf"
    )


def test_ingest_defaults_to_filename():
    service = create_service(
        [
            {
                "page_number": 1,
                "text": "Research content",
            }
        ]
    )

    result = service.ingest(
        "C:/documents/research_paper.pdf"
    )

    assert result == 2
    assert service.vector_store.metadata[0]["source"] == (
        "research_paper.pdf"
    )


def test_ingest_preserves_page_and_chunk_metadata():
    service = create_service(
        [
            {
                "page_number": 3,
                "text": "Important content",
            }
        ]
    )

    service.ingest(
        "research.pdf"
    )

    assert service.vector_store.metadata == [
        {
            "source": "research.pdf",
            "page": 3,
            "chunk_id": 0,
        },
        {
            "source": "research.pdf",
            "page": 3,
            "chunk_id": 1,
        },
    ]


def test_ingest_empty_document():
    service = create_service(
        [
            {
                "page_number": 1,
                "text": "",
            },
            {
                "page_number": 2,
                "text": "   ",
            },
        ]
    )

    result = service.ingest(
        "empty.pdf"
    )

    assert result == 0
    assert service.vector_store.chunks == []


def test_ingest_multiple_documents_into_same_store():
    service = create_service(
        [
            {
                "page_number": 1,
                "text": "First document",
            }
        ]
    )

    first_result = service.ingest(
        "paper_one.pdf"
    )

    service.document_loader.pages = [
        {
            "page_number": 1,
            "text": "Second document",
        }
    ]

    second_result = service.ingest(
        "paper_two.pdf"
    )

    assert first_result == 2
    assert second_result == 2

    assert len(service.vector_store.chunks) == 4

    sources = [
        item["source"]
        for item in service.vector_store.metadata
    ]

    assert sources == [
        "paper_one.pdf",
        "paper_one.pdf",
        "paper_two.pdf",
        "paper_two.pdf",
    ]