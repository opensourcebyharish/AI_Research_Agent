import re


class TextProcessor:
    """Clean document text and create retrieval-friendly chunks."""

    def clean_text(self, text: str) -> str:
        """
        Normalize whitespace while preserving paragraph boundaries.
        """

        if not text:
            return ""

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r" *\n *", "\n", text)

        return text.strip()

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into reasonably meaningful sentences."""

        text = text.strip()

        if not text:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def _split_long_sentence(
        self,
        sentence: str,
        chunk_size: int,
    ) -> list[list[str]]:
        """Split an unusually long sentence into word groups."""

        words = sentence.split()

        if len(words) <= chunk_size:
            return [words]

        return [
            words[i:i + chunk_size]
            for i in range(
                0,
                len(words),
                chunk_size,
            )
        ]

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> list[str]:
        """
        Create retrieval-friendly overlapping text chunks.

        The chunker prefers sentence boundaries. If a sentence is
        longer than the chunk size, it falls back to sliding
        word-based chunks.

        Args:
            text: Text to chunk.
            chunk_size: Maximum number of words in a chunk.
            chunk_overlap: Number of words shared between chunks.

        Returns:
            A list of overlapping text chunks.
        """

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0"
            )

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be >= 0 and < chunk_size"
            )

        if not text or not text.strip():
            return []

        cleaned_text = self.clean_text(text)

        paragraphs = [
            paragraph.strip()
            for paragraph in cleaned_text.split("\n\n")
            if paragraph.strip()
        ]

        sentences = []

        for paragraph in paragraphs:
            sentences.extend(
                self._split_into_sentences(paragraph)
            )

        if not sentences:
            return []

        chunks = []
        current_words = []

        for sentence in sentences:
            words = sentence.split()

            # Long sentence: use sliding word windows.
            if len(words) > chunk_size:
                if current_words:
                    chunks.append(
                        " ".join(current_words)
                    )
                    current_words = []

                step = chunk_size - chunk_overlap

                for start in range(
                    0,
                    len(words),
                    step,
                ):
                    window = words[
                        start:start + chunk_size
                    ]

                    if not window:
                        break

                    chunks.append(
                        " ".join(window)
                    )

                    if (
                        start + chunk_size
                        >= len(words)
                    ):
                        break

                continue

            # If adding the next sentence would exceed the
            # chunk size, finalize the current chunk.
            if (
                current_words
                and len(current_words) + len(words)
                > chunk_size
            ):
                chunks.append(
                    " ".join(current_words)
                )

                overlap_words = current_words[
                    -chunk_overlap:
                ]

                current_words = overlap_words.copy()

            current_words.extend(words)

        if current_words:
            final_chunk = " ".join(current_words)

            if (
                not chunks
                or final_chunk != chunks[-1]
            ):
                chunks.append(final_chunk)

        return chunks