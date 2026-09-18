class ContextBuilder:
    """Build LLM-ready context from retrieved document chunks."""

    def build_context(self, results: list[dict]) -> str:
        """
        Convert retrieval results into formatted context for an LLM.

        Args:
            results: Retrieved chunks containing text and metadata.

        Returns:
            A formatted context string.
        """
        if not results:
            return ""

        context_parts = []

        for result in results:
            chunk = result.get("chunk", "")
            metadata = result.get("metadata", {})

            if not chunk or not chunk.strip():
                continue

            source = metadata.get("source", "unknown")
            page = metadata.get("page", "unknown")
            chunk_id = metadata.get("chunk_id", "unknown")

            context_parts.append(
                f"[Source: {source} | Page: {page} | Chunk: {chunk_id}]\n"
                f"{chunk.strip()}"
            )

        return "\n\n".join(context_parts)