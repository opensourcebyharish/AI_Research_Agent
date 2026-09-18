class RAGService:
    """Orchestrate retrieval, context building, prompting, and generation."""

    def __init__(
        self,
        retrieval_service,
        context_builder,
        prompt_builder,
        llm_service,
        citation_service,
    ):
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service
        self.citation_service = citation_service

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict:
        """
        Generate a grounded answer with source citations.

        Args:
            question: User's question.
            top_k: Number of document chunks to retrieve.

        Returns:
            A dictionary containing the answer, retrieved results,
            context, prompt, and citations.

        Raises:
            ValueError: If the question is empty.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        results = self.retrieval_service.retrieve(
            question,
            top_k=top_k,
        )

        if not results:
            return {
                "answer": (
                    "The information is not available in "
                    "the provided documents."
                ),
                "results": [],
                "context": "",
                "prompt": "",
                "citations": [],
            }

        context = self.context_builder.build_context(results)

        prompt = self.prompt_builder.build(
            question=question,
            context=context,
        )

        answer = self.llm_service.generate(prompt)

        citations = self.citation_service.build_citations(
            results
        )

        return {
            "answer": answer,
            "results": results,
            "context": context,
            "prompt": prompt,
            "citations": citations,
        }