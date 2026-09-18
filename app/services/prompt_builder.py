class PromptBuilder:
    """Build prompts for grounded question answering."""

    SYSTEM_INSTRUCTION = (
        "You are an AI research assistant. "
        "Answer questions using only the provided document context. "
        "Do not invent facts, sources, page numbers, or citations. "
        "If the context does not contain enough information to answer "
        "the question, clearly state that the information is not available "
        "in the provided documents. "
        "When using information from the context, cite it using the "
        "source and page information provided in the context."
    )

    def build(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Build a grounded question-answering prompt.

        Args:
            question: User's question.
            context: Retrieved document context.

        Returns:
            A formatted prompt.

        Raises:
            ValueError: If question or context is empty.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        if not context or not context.strip():
            raise ValueError("context must not be empty")

        return (
            f"{self.SYSTEM_INSTRUCTION}\n\n"
            "DOCUMENT CONTEXT:\n"
            f"{context.strip()}\n\n"
            "USER QUESTION:\n"
            f"{question.strip()}\n\n"
            "ANSWER:"
        )