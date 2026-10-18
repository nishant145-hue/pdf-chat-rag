from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the PDF Chat RAG application.
    Values can be overridden using environment variables
    defined in the .env file.
    """

    # ---------------------------------------------------------
    # API
    # ---------------------------------------------------------

    API_TITLE: str = "PDF Chat RAG API"
    API_VERSION: str = "3.0.0"

    # ---------------------------------------------------------
    # CORS
    # ---------------------------------------------------------

    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    CORS_ALLOW_CREDENTIALS: bool = False
    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------

    BASE_DIR: Path = Path(".")

    UPLOAD_DIR: Path = Path("uploads")

    MAX_UPLOAD_SIZE_BYTES: int = Field(
        default=10 * 1024 * 1024,
        gt=0,
    )

    MAX_PDF_PAGES: int = Field(
        default=500,
        gt=0,
    )

    VECTOR_STORE_DIR: Path = Path("vector_store")

    INDEX_DIR: Path = VECTOR_STORE_DIR / "indexes"

    CHUNK_DIR: Path = VECTOR_STORE_DIR / "chunks"

    METADATA_DIR: Path = VECTOR_STORE_DIR / "metadata"

    # ---------------------------------------------------------
    # Embedding Model
    # ---------------------------------------------------------

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ---------------------------------------------------------
    # Chunking
    # ---------------------------------------------------------

    CHUNK_SIZE: int = 500

    CHUNK_OVERLAP: int = 100

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    TOP_K: int = 3

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------

    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    TEMPERATURE: float = 0.2

    MAX_TOKENS: int = 512

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------

    LOG_LEVEL: str = "INFO"

    LOG_DIR: Path = Path("logs")

    LOG_FILE: str = "rag.log"

    # ---------------------------------------------------------
    # Cache
    # ---------------------------------------------------------

    ENABLE_CACHE: bool = True

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------

    GROQ_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    RERANKER_MODEL: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    FAISS_TOP_K: int = 20

    FINAL_TOP_K: int = 5

    ENABLE_RERANKING: bool = True


settings = Settings()
