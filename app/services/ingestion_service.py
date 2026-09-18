from pathlib import Path


class IngestionService:
    """Load, process, embed, and index documents."""

    def __init__(
        self,
        document_loader,
        text_processor,
        embedding_service,
        vector_store,
    ):
        self.document_loader = document_loader
        self.text_processor = text_processor
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ingest(
        self,
        file_path: str,
        source_name: str | None = None,
    ) -> int:
        """
        Ingest a document into the shared vector store.

        Multiple documents can be added to the same vector store.
        Source and page metadata are preserved for every chunk.

        Args:
            file_path: Actual path of the document to load.
            source_name: Human-readable source name used in metadata.
                If omitted, the filename from file_path is used.

        Returns:
            Number of chunks indexed.
        """

        pages = self.document_loader.load(file_path)

        source = source_name or Path(file_path).name

        chunks = []
        metadata = []

        for page in pages:
            page_number = page["page_number"]
            text = page["text"]

            cleaned_text = self.text_processor.clean_text(
                text
            )

            if not cleaned_text:
                continue

            page_chunks = self.text_processor.chunk_text(
                cleaned_text
            )

            for chunk_index, chunk in enumerate(page_chunks):
                chunks.append(chunk)

                metadata.append(
                    {
                        "source": source,
                        "page": page_number,
                        "chunk_id": chunk_index,
                    }
                )

        if not chunks:
            return 0

        embeddings = self.embedding_service.embed_chunks(
            chunks
        )

        self.vector_store.add(
            embeddings,
            chunks,
            metadata,
        )

        return len(chunks)