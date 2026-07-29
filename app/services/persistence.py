from pathlib import Path
import json
from datetime import datetime

import faiss


class PersistenceService:
    """
    Handles saving and loading
    - FAISS indexes
    - Text chunks
    - Metadata
    """

    def __init__(self):

        self.base_dir = Path("vector_store")

        self.index_dir = self.base_dir / "indexes"
        self.chunk_dir = self.base_dir / "chunks"
        self.metadata_dir = self.base_dir / "metadata"

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Internal Helper
    # ---------------------------------------------------------

    def _safe_name(self, filename: str) -> str:
        """
        Convert filename into a safe file stem.

        Example:
        python.pdf -> python
        My Notes.pdf -> My_Notes
        """

        return (
            Path(filename)
            .stem
            .replace(" ", "_")
        )

    # ---------------------------------------------------------
    # FAISS
    # ---------------------------------------------------------

    def save_index(self, filename: str, index):

        name = self._safe_name(filename)

        path = self.index_dir / f"{name}.faiss"

        faiss.write_index(index, str(path))

    def load_index(self, filename: str):

        name = self._safe_name(filename)

        path = self.index_dir / f"{name}.faiss"

        if not path.exists():
            return None

        return faiss.read_index(str(path))

    # ---------------------------------------------------------
    # Chunks
    # ---------------------------------------------------------

    def save_chunks(
        self,
        filename: str,
        chunks: list[str],
    ):

        name = self._safe_name(filename)

        path = self.chunk_dir / f"{name}_chunks.json"

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                chunks,
                file,
                indent=4,
                ensure_ascii=False,
            )

    def load_chunks(
        self,
        filename: str,
    ):

        name = self._safe_name(filename)

        path = self.chunk_dir / f"{name}_chunks.json"

        if not path.exists():
            return None

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    def save_metadata(
        self,
        filename: str,
        chunk_count: int,
        embedding_model: str,
    ):

        name = self._safe_name(filename)

        path = self.metadata_dir / f"{name}_metadata.json"

        metadata = {

            "filename": filename,

            "chunk_count": chunk_count,

            "embedding_model": embedding_model,

            "created": datetime.now().isoformat()
        }

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

    def load_metadata(
        self,
        filename: str,
    ):

        name = self._safe_name(filename)

        path = self.metadata_dir / f"{name}_metadata.json"

        if not path.exists():
            return None

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def exists(
        self,
        filename: str,
    ) -> bool:
        """
        Returns True only if all persisted
        files exist.
        """

        name = self._safe_name(filename)

        index = self.index_dir / f"{name}.faiss"
        chunks = self.chunk_dir / f"{name}_chunks.json"
        metadata = self.metadata_dir / f"{name}_metadata.json"

        return (
            index.exists()
            and chunks.exists()
            and metadata.exists()
        )

    def delete(
        self,
        filename: str,
    ):

        name = self._safe_name(filename)

        files = [

            self.index_dir / f"{name}.faiss",

            self.chunk_dir / f"{name}_chunks.json",

            self.metadata_dir / f"{name}_metadata.json",
        ]

        for file in files:

            if file.exists():
                file.unlink()