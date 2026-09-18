import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from app.api.dependencies import (
    get_document_registry,
    get_ingestion_service,
    get_rag_service,
    get_services,
)
from app.core.config import settings
from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService
from app.services.upload_validator import (
    UploadValidationError,
    UploadValidator,
)


router = APIRouter()

logger = logging.getLogger(
    "ai_research_agent.api"
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


class QuestionResponse(BaseModel):
    answer: str
    results: list
    context: str
    prompt: str
    citations: list


_rag_service: RAGService | None = None


upload_validator = UploadValidator(
    max_size_bytes=settings.max_upload_size_bytes,
    allowed_extensions=settings.allowed_file_extensions,
)


def _get_rag_service() -> RAGService:
    """Return the shared RAG service."""

    global _rag_service

    if _rag_service is None:
        _rag_service = get_rag_service()

    return _rag_service


def _save_uploaded_pdf(
    file: UploadFile,
) -> str:
    """Validate and save an uploaded PDF to a temporary file."""

    filename = file.filename or ""

    try:
        safe_filename = upload_validator.validate_filename(
            filename
        )

    except UploadValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    content = file.file.read()

    try:
        upload_validator.validate_bytes(content)

    except UploadValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    with NamedTemporaryFile(
        delete=False,
        suffix=Path(safe_filename).suffix.lower(),
    ) as temporary_file:
        temporary_file.write(content)

        return temporary_file.name


@router.get("/health")
def health_check():
    """Check whether the API process is running."""

    return {
        "status": "ok",
        "service": "AI Research Agent",
    }


@router.get("/ready")
def readiness_check():
    """
    Check whether the application is ready to serve requests.

    Verifies that the persistent vector store and document registry
    are accessible and internally consistent.
    """

    try:
        services = get_services()

        vector_store = services.vector_store
        document_registry = services.document_registry

        vector_count = vector_store.index.ntotal
        chunk_count = len(vector_store.chunks)
        metadata_count = len(vector_store.metadata)
        document_count = document_registry.count()

        if vector_count != chunk_count:
            raise RuntimeError(
                "Vector store and chunk counts do not match."
            )

        if vector_count != metadata_count:
            raise RuntimeError(
                "Vector store and metadata counts do not match."
            )

        return {
            "status": "ready",
            "service": settings.app_name,
            "documents": document_count,
            "chunks": chunk_count,
        }

    except Exception:
        logger.exception(
            "Readiness check failed."
        )

        raise HTTPException(
            status_code=503,
            detail="Service is not ready.",
        )


@router.get("/documents")
def list_documents():
    """Return all documents registered in the knowledge base."""

    registry: DocumentRegistry = get_document_registry()

    return {
        "documents": registry.list_documents(),
        "count": registry.count(),
    }


@router.post("/documents/upload")
def upload_document(
    file: UploadFile | None = None,
):
    """Validate, upload, index, and register a PDF document."""

    if file is None or not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A file is required.",
        )

    try:
        original_filename = upload_validator.validate_filename(
            file.filename
        )

    except UploadValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    registry: DocumentRegistry = get_document_registry()

    if registry.exists(original_filename):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Document already exists: "
                f"{original_filename}"
            ),
        )

    temporary_path = _save_uploaded_pdf(file)

    try:
        ingestion_service: IngestionService = (
            get_ingestion_service()
        )

        chunks_indexed = ingestion_service.ingest(
            temporary_path,
            source_name=original_filename,
        )

        pages = ingestion_service.document_loader.load(
            temporary_path
        )

        pages_processed = sum(
            1
            for page in pages
            if page.get("text", "").strip()
        )

        registry.add(
            source=original_filename,
            pages=pages_processed,
            chunks=chunks_indexed,
        )

        logger.info(
            "Document indexed: %s | pages=%d | chunks=%d",
            original_filename,
            pages_processed,
            chunks_indexed,
        )

        return {
            "message": "Document indexed successfully.",
            "filename": original_filename,
            "pages": pages_processed,
            "chunks_indexed": chunks_indexed,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Document indexing failed: %s",
            original_filename,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to index document.",
        ) from exc

    finally:
        Path(temporary_path).unlink(
            missing_ok=True
        )


@router.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask_question(
    request: QuestionRequest,
):
    """Ask a question against indexed research documents."""

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="question must not be empty",
        )

    if request.top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0",
        )

    try:
        rag_service = _get_rag_service()

        result = rag_service.answer(
            question=request.question,
            top_k=request.top_k,
        )

        logger.info(
            "Research question answered | top_k=%d",
            request.top_k,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to generate research answer."
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate answer.",
        ) from exc