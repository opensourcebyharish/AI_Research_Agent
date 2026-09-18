from app.services.citation_service import CitationService
from app.services.context_builder import ContextBuilder
from app.services.document_loader import PDFLoader
from app.services.document_registry import DocumentRegistry
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService
from app.services.persistent_vector_store import (
    PersistentVectorStore,
)
from app.services.prompt_builder import PromptBuilder
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.services.text_processor import TextProcessor


class ApplicationServices:
    """Container for shared application services."""

    def __init__(self):
        self.document_loader = PDFLoader()

        self.text_processor = TextProcessor()

        self.embedding_service = EmbeddingService()

        self.vector_store = PersistentVectorStore(
            dimension=384,
            storage_dir="data/index",
        )

        self.document_registry = DocumentRegistry(
            storage_path="data/documents.json",
        )

        self.ingestion_service = IngestionService(
            document_loader=self.document_loader,
            text_processor=self.text_processor,
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
        )

        self.retrieval_service = RetrievalService(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
        )

        self.context_builder = ContextBuilder()

        self.prompt_builder = PromptBuilder()

        self.citation_service = CitationService()

        self._llm_service = None

        self._rag_service = None

    @property
    def llm_service(self) -> LLMService:
        """Create the LLM service only when it is needed."""

        if self._llm_service is None:
            self._llm_service = LLMService()

        return self._llm_service

    @property
    def rag_service(self) -> RAGService:
        """Create the RAG service only when it is needed."""

        if self._rag_service is None:
            self._rag_service = RAGService(
                retrieval_service=self.retrieval_service,
                context_builder=self.context_builder,
                prompt_builder=self.prompt_builder,
                llm_service=self.llm_service,
                citation_service=self.citation_service,
            )

        return self._rag_service


_services: ApplicationServices | None = None


def get_services() -> ApplicationServices:
    """Return the shared application service container."""

    global _services

    if _services is None:
        _services = ApplicationServices()

    return _services


def get_rag_service() -> RAGService:
    """Return the shared RAG service."""

    return get_services().rag_service


def get_ingestion_service() -> IngestionService:
    """Return the shared ingestion service."""

    return get_services().ingestion_service


def get_document_registry() -> DocumentRegistry:
    """Return the shared document registry."""

    return get_services().document_registry