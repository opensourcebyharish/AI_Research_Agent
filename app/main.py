import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging


configure_logging(
    level=settings.log_level,
    log_file=settings.log_file,
)

logger = logging.getLogger(
    "ai_research_agent"
)


app = FastAPI(
    title=settings.app_name,
    description=(
        "A document-grounded AI research assistant "
        "using RAG."
    ),
    version=settings.app_version,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    """Handle unexpected application errors safely."""

    logger.exception(
        "Unhandled exception: %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error."
        },
    )


app.include_router(router)


@app.get("/")
def root():
    """Return basic information about the API."""

    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }