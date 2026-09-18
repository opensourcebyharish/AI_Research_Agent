from app.services.evaluation_service import EvaluationService


def test_ui_scorecard_data_is_available():
    service = EvaluationService()

    results = [
        {
            "distance": 0.4,
            "metadata": {
                "source": "research.pdf",
                "page": 1,
            },
        }
    ]

    citations = [
        {
            "id": 1,
            "source": "research.pdf",
            "page": 1,
        }
    ]

    evaluation = service.evaluate(
        results=results,
        citations=citations,
        answer="RAG combines retrieval and generation.",
        context="RAG combines retrieval and generation.",
    )

    assert evaluation["retrieval"]["relevance_rate"] == 1.0
    assert evaluation["citation"]["citation_accuracy"] == 1.0
    assert evaluation["groundedness"]["passed"] is True
    assert evaluation["overall_score"] == 1.0