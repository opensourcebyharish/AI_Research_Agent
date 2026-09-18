from pathlib import Path
import json


class DocumentRegistry:
    """Track documents indexed by the research agent."""

    def __init__(
        self,
        storage_path: str | Path = "data/documents.json",
    ):
        self.storage_path = Path(storage_path)

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.documents = self._load()

    def _load(self) -> list[dict]:
        """Load the document registry from disk."""

        if not self.storage_path.exists():
            return []

        try:
            with self.storage_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if not isinstance(data, list):
                return []

            return data

        except (json.JSONDecodeError, OSError):
            return []

    def _save(self) -> None:
        """Persist the document registry to disk."""

        with self.storage_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.documents,
                file,
                indent=2,
            )

    def exists(self, source: str) -> bool:
        """Check whether a document source is already registered."""

        return any(
            document.get("source") == source
            for document in self.documents
        )

    def add(
        self,
        source: str,
        pages: int,
        chunks: int,
    ) -> dict:
        """
        Register a newly indexed document.

        Raises:
            ValueError: If the source is already registered.
        """

        if not source or not source.strip():
            raise ValueError(
                "source must not be empty"
            )

        if pages < 0:
            raise ValueError(
                "pages must not be negative"
            )

        if chunks < 0:
            raise ValueError(
                "chunks must not be negative"
            )

        if self.exists(source):
            raise ValueError(
                f"Document already registered: {source}"
            )

        document = {
            "source": source,
            "pages": pages,
            "chunks": chunks,
        }

        self.documents.append(document)
        self._save()

        return document

    def list_documents(self) -> list[dict]:
        """Return all registered documents."""

        return list(self.documents)

    def get(
        self,
        source: str,
    ) -> dict | None:
        """Return a document by source name."""

        for document in self.documents:
            if document.get("source") == source:
                return document

        return None

    def remove(
        self,
        source: str,
    ) -> bool:
        """
        Remove a document from the registry.

        Note:
            This removes the registry entry only. It does not remove
            vectors from the FAISS index.
        """

        original_count = len(self.documents)

        self.documents = [
            document
            for document in self.documents
            if document.get("source") != source
        ]

        removed = len(self.documents) < original_count

        if removed:
            self._save()

        return removed

    def count(self) -> int:
        """Return the number of registered documents."""

        return len(self.documents)