class CitationService:
    """Create structured citations from retrieved document metadata."""

    def build_citations(self, results: list[dict]) -> list[dict]:
        """
        Build unique citations from retrieval results.

        Args:
            results: Retrieved chunks containing metadata.

        Returns:
            A list of unique citation dictionaries.
        """
        if not results:
            return []

        citations = []
        seen = set()

        for result in results:
            metadata = result.get("metadata", {})

            source = metadata.get("source")
            page = metadata.get("page")

            if source is None or page is None:
                continue

            citation_key = (source, page)

            if citation_key in seen:
                continue

            seen.add(citation_key)

            citations.append(
                {
                    "id": len(citations) + 1,
                    "source": source,
                    "page": page,
                }
            )

        return citations