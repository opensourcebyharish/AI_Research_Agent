from pathlib import Path
import pickle

import faiss
import numpy as np

from app.services.vector_store import VectorStore


class PersistentVectorStore(VectorStore):
    """
    FAISS vector store with disk persistence.

    Stores the FAISS index, chunks, and metadata on disk so the
    indexed documents survive application restarts.
    """

    def __init__(
        self,
        dimension: int,
        storage_dir: str | Path = "data/index",
    ):
        self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_path = (
            self.storage_dir / "index.faiss"
        )

        self.metadata_path = (
            self.storage_dir / "metadata.pkl"
        )

        if (
            self.index_path.exists()
            and self.metadata_path.exists()
        ):
            self._load()
        else:
            super().__init__(dimension)

    def add(
        self,
        embeddings,
        chunks,
        metadata,
    ):
        """Add vectors and persist the updated store."""

        super().add(
            embeddings,
            chunks,
            metadata,
        )

        self.save()

    def remove_source(
        self,
        source: str,
    ) -> int:
        """
        Remove all vectors belonging to a document source.

        Args:
            source: Registered document source name.

        Returns:
            Number of chunks removed.
        """

        if not source or not source.strip():
            raise ValueError(
                "source must not be empty"
            )

        keep_indices = [
            index
            for index, metadata in enumerate(
                self.metadata
            )
            if metadata.get("source") != source
        ]

        removed_count = (
            len(self.metadata)
            - len(keep_indices)
        )

        if removed_count == 0:
            return 0

        if keep_indices:
            vectors = []

            for index in keep_indices:
                vector = self.index.reconstruct(index)
                vectors.append(vector)

            new_index = faiss.IndexFlatL2(
                self.dimension
            )

            new_index.add(
                np.asarray(
                    vectors,
                    dtype="float32",
                )
            )
        else:
            new_index = faiss.IndexFlatL2(
                self.dimension
            )

        self.index = new_index

        self.chunks = [
            self.chunks[index]
            for index in keep_indices
        ]

        self.metadata = [
            self.metadata[index]
            for index in keep_indices
        ]

        self.save()

        return removed_count

    def save(self):
        """Save the FAISS index, chunks, and metadata to disk."""

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        with self.metadata_path.open(
            "wb"
        ) as file:
            pickle.dump(
                {
                    "dimension": self.dimension,
                    "chunks": self.chunks,
                    "metadata": self.metadata,
                },
                file,
            )

    def _load(self):
        """Load the persisted FAISS index and metadata."""

        with self.metadata_path.open(
            "rb"
        ) as file:
            data = pickle.load(file)

        dimension = data["dimension"]

        self.dimension = dimension

        self.index = faiss.read_index(
            str(self.index_path)
        )

        self.chunks = data["chunks"]
        self.metadata = data["metadata"]

        if self.index.d != self.dimension:
            raise ValueError(
                "Persisted FAISS index dimension does not match metadata."
            )

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "Persisted index and chunk count do not match."
            )

        if len(self.chunks) != len(self.metadata):
            raise ValueError(
                "Persisted chunks and metadata count do not match."
            )