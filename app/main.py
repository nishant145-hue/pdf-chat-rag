
"""
Main entry point of the PDF Chat RAG API.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import manager, router
from app.logger import logger
from app.services.startup_loader import StartupLoader


loader = StartupLoader(manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Startup:
        Restore previously cached documents.

    Shutdown:
        Log application termination.
    """
    logger.info("=" * 60)
    logger.info("PDF Chat RAG API Started")

    try:
        loader.load_documents()

        logger.info("Swagger Docs: http://127.0.0.1:8000/docs")
        logger.info("=" * 60)

        yield

    finally:
        logger.info("=" * 60)
        logger.info("PDF Chat RAG API Stopped")
        logger.info("=" * 60)


app = FastAPI(
    title="PDF Chat RAG API",
    description="""
A Production Ready Retrieval-Augmented Generation (RAG) API.

Features
--------
- Upload multiple PDF documents
- Ask questions about a specific document
- Semantic search using FAISS
- SentenceTransformer embeddings
- Groq LLM integration
- Swagger UI
""",
    version="2.0.0",
    lifespan=lifespan,
)


# -------------------------------------------------------
# CORS
# -------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------
# Register API Routes
# -------------------------------------------------------

app.include_router(router)

