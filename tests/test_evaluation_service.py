from app.services.evaluation_service import EvaluationService


def test_retrieval_evaluation_passes_relevant_results():
    service = EvaluationService(
        relevance_threshold=1.5
    )

    results = [
        {
            "distance": 0.4,
            "metadata": {
                "source": "paper.pdf",
                "page": 1,
            },
        },
        {
            "distance": 1.2,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
            },
        },
    ]

    evaluation = service.evaluate_retrieval(
        results
    )

    assert evaluation["total_results"] == 2
    assert evaluation["relevant_results"] == 2
    assert evaluation["relevance_rate"] == 1.0
    assert evaluation["passed"] is True


def test_retrieval_evaluation_rejects_irrelevant_results():
    service = EvaluationService(
        relevance_threshold=1.5
    )

    results = [
        {
            "distance": 2.0,
            "metadata": {
                "source": "paper.pdf",
                "page": 1,
            },
        }
    ]

    evaluation = service.evaluate_retrieval(
        results
    )

    assert evaluation["relevant_results"] == 0
    assert evaluation["relevance_rate"] == 0.0
    assert evaluation["passed"] is False


def test_citation_evaluation_passes_correct_citations():
    service = EvaluationService()

    results = [
        {
            "distance": 0.5,
            "metadata": {
                "source": "paper.pdf",
                "page": 3,
            },
        }
    ]

    citations = [
        {
            "id": 1,
            "source": "paper.pdf",
            "page": 3,
        }
    ]

    evaluation = service.evaluate_citations(
        results,
        citations,
    )

    assert evaluation["total_citations"] == 1
    assert evaluation["valid_citations"] == 1
    assert evaluation["citation_accuracy"] == 1.0
    assert evaluation["passed"] is True


def test_citation_evaluation_rejects_wrong_source():
    service = EvaluationService()

    results = [
        {
            "distance": 0.5,
            "metadata": {
                "source": "paper.pdf",
                "page": 3,
            },
        }
    ]

    citations = [
        {
            "id": 1,
            "source": "wrong.pdf",
            "page": 3,
        }
    ]

    evaluation = service.evaluate_citations(
        results,
        citations,
    )

    assert evaluation["valid_citations"] == 0
    assert evaluation["citation_accuracy"] == 0.0
    assert evaluation["passed"] is False


def test_groundedness_passes_with_answer_and_context():
    service = EvaluationService()

    evaluation = service.evaluate_groundedness(
        answer="RAG combines retrieval and generation.",
        context="RAG combines retrieval and generation.",
    )

    assert evaluation["answer_present"] is True
    assert evaluation["context_present"] is True
    assert evaluation["passed"] is True


def test_groundedness_fails_without_context():
    service = EvaluationService()

    evaluation = service.evaluate_groundedness(
        answer="Some answer.",
        context="",
    )

    assert evaluation["answer_present"] is True
    assert evaluation["context_present"] is False
    assert evaluation["passed"] is False


def test_unsupported_question_evaluation_passes():
    service = EvaluationService()

    answer = (
        "The information is not available in "
        "the provided documents."
    )

    evaluation = service.evaluate_unsupported_question(
        answer=answer,
        results=[],
    )

    assert evaluation["no_results"] is True
    assert evaluation["correct_refusal"] is True
    assert evaluation["passed"] is True


def test_unsupported_question_evaluation_fails_when_answer_is_generated():
    service = EvaluationService()

    evaluation = service.evaluate_unsupported_question(
        answer="Paris is the capital of France.",
        results=[],
    )

    assert evaluation["no_results"] is True
    assert evaluation["correct_refusal"] is False
    assert evaluation["passed"] is False


def test_complete_evaluation_returns_scorecard():
    service = EvaluationService()

    results = [
        {
            "distance": 0.4,
            "metadata": {
                "source": "paper.pdf",
                "page": 1,
            },
        }
    ]

    citations = [
        {
            "id": 1,
            "source": "paper.pdf",
            "page": 1,
        }
    ]

    evaluation = service.evaluate(
        results=results,
        citations=citations,
        answer="RAG retrieves relevant information.",
        context="RAG retrieves relevant information.",
    )

    assert evaluation["passed"] is True
    assert evaluation["passed_checks"] == 3
    assert evaluation["total_checks"] == 3
    assert evaluation["overall_score"] == 1.0