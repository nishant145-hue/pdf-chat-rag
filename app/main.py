"""
Main entry point of the PDF Chat RAG API.
"""
from fastapi import FastAPI
from app.logger import logger
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.document_manager import DocumentManager
from app.services.startup_loader import StartupLoader

manager = DocumentManager()
loader = StartupLoader(manager)

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


# -------------------------------------------------------
# Startup Event
# -------------------------------------------------------

@app.on_event("startup")
async def startup_event():

    logger.info("=" * 60)
    logger.info("🚀 PDF Chat RAG API Started")

    loader.load_documents()

    logger.info("📄 Swagger Docs : http://127.0.0.1:8000/docs")
    logger.info("=" * 60)

# -------------------------------------------------------
# Shutdown Event
# -------------------------------------------------------

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=" * 60)
    logger.info("🛑 PDF Chat RAG API Stopped")
    logger.info("=" * 60)
    
