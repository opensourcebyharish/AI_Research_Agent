class EvaluationService:
    """Evaluate core quality signals of the RAG pipeline."""

    def __init__(self, relevance_threshold: float = 1.5):
        if relevance_threshold <= 0:
            raise ValueError(
                "relevance_threshold must be greater than 0"
            )

        self.relevance_threshold = relevance_threshold

    def evaluate_retrieval(
        self,
        results: list[dict],
    ) -> dict:
        """
        Evaluate whether retrieved passages meet the relevance threshold.

        Returns:
            Dictionary containing retrieval metrics.
        """

        if not results:
            return {
                "total_results": 0,
                "relevant_results": 0,
                "relevance_rate": 0.0,
                "passed": False,
            }

        relevant_results = [
            result
            for result in results
            if result.get(
                "distance",
                float("inf"),
            ) <= self.relevance_threshold
        ]

        total_results = len(results)
        relevant_count = len(relevant_results)

        relevance_rate = (
            relevant_count / total_results
        )

        return {
            "total_results": total_results,
            "relevant_results": relevant_count,
            "relevance_rate": relevance_rate,
            "passed": relevant_count > 0,
        }

    def evaluate_citations(
        self,
        results: list[dict],
        citations: list[dict],
    ) -> dict:
        """
        Evaluate whether citations correspond to retrieved sources/pages.
        """

        if not citations:
            return {
                "total_citations": 0,
                "valid_citations": 0,
                "citation_accuracy": 0.0,
                "passed": False,
            }

        valid_pairs = {
            (
                result.get("metadata", {}).get("source"),
                result.get("metadata", {}).get("page"),
            )
            for result in results
        }

        valid_citations = [
            citation
            for citation in citations
            if (
                citation.get("source"),
                citation.get("page"),
            ) in valid_pairs
        ]

        total_citations = len(citations)
        valid_count = len(valid_citations)

        accuracy = (
            valid_count / total_citations
        )

        return {
            "total_citations": total_citations,
            "valid_citations": valid_count,
            "citation_accuracy": accuracy,
            "passed": valid_count == total_citations,
        }

    def evaluate_groundedness(
        self,
        answer: str,
        context: str,
    ) -> dict:
        """
        Perform a lightweight groundedness check.

        This metric checks whether the answer and context are non-empty.
        It is intentionally deterministic and does not call an LLM.
        """

        answer_present = bool(
            answer and answer.strip()
        )

        context_present = bool(
            context and context.strip()
        )

        passed = (
            answer_present
            and context_present
        )

        return {
            "answer_present": answer_present,
            "context_present": context_present,
            "groundedness_passed": passed,
            "passed": passed,
        }

    def evaluate_unsupported_question(
        self,
        answer: str,
        results: list[dict],
    ) -> dict:
        """
        Evaluate handling of a question for which no evidence was retrieved.
        """

        expected_fallback = (
            "The information is not available in "
            "the provided documents."
        )

        no_results = not results

        correct_refusal = (
            answer.strip() == expected_fallback
            if answer
            else False
        )

        passed = (
            no_results
            and correct_refusal
        )

        return {
            "no_results": no_results,
            "correct_refusal": correct_refusal,
            "passed": passed,
        }

    def evaluate(
        self,
        results: list[dict],
        citations: list[dict],
        answer: str,
        context: str,
    ) -> dict:
        """
        Run the complete deterministic RAG evaluation.

        Returns:
            Complete evaluation scorecard.
        """

        retrieval = self.evaluate_retrieval(
            results
        )

        citation = self.evaluate_citations(
            results,
            citations,
        )

        groundedness = self.evaluate_groundedness(
            answer,
            context,
        )

        checks = [
            retrieval["passed"],
            citation["passed"],
            groundedness["passed"],
        ]

        passed_checks = sum(
            1
            for check in checks
            if check
        )

        total_checks = len(checks)

        overall_score = (
            passed_checks / total_checks
            if total_checks
            else 0.0
        )

        return {
            "retrieval": retrieval,
            "citation": citation,
            "groundedness": groundedness,
            "overall_score": overall_score,
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "passed": (
                passed_checks == total_checks
            ),
        }