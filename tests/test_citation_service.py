from app.services.citation_service import CitationService


def test_build_citations_returns_empty_list_for_empty_results():
    service = CitationService()

    result = service.build_citations([])

    assert result == []


def test_build_citations_creates_citation():
    service = CitationService()

    results = [
        {
            "chunk": "Artificial intelligence is a field of computer science.",
            "distance": 0.2,
            "metadata": {
                "source": "research.pdf",
                "page": 3,
                "chunk_id": 0,
            },
        }
    ]

    result = service.build_citations(results)

    assert result == [
        {
            "id": 1,
            "source": "research.pdf",
            "page": 3,
        }
    ]


def test_build_citations_creates_multiple_citations():
    service = CitationService()

    results = [
        {
            "chunk": "First passage.",
            "distance": 0.2,
            "metadata": {
                "source": "research.pdf",
                "page": 3,
                "chunk_id": 0,
            },
        },
        {
            "chunk": "Second passage.",
            "distance": 0.3,
            "metadata": {
                "source": "research.pdf",
                "page": 7,
                "chunk_id": 0,
            },
        },
    ]

    result = service.build_citations(results)

    assert result == [
        {
            "id": 1,
            "source": "research.pdf",
            "page": 3,
        },
        {
            "id": 2,
            "source": "research.pdf",
            "page": 7,
        },
    ]


def test_build_citations_removes_duplicate_source_and_page():
    service = CitationService()

    results = [
        {
            "chunk": "First chunk from page three.",
            "distance": 0.1,
            "metadata": {
                "source": "research.pdf",
                "page": 3,
                "chunk_id": 0,
            },
        },
        {
            "chunk": "Second chunk from page three.",
            "distance": 0.2,
            "metadata": {
                "source": "research.pdf",
                "page": 3,
                "chunk_id": 1,
            },
        },
    ]

    result = service.build_citations(results)

    assert result == [
        {
            "id": 1,
            "source": "research.pdf",
            "page": 3,
        }
    ]


def test_build_citations_allows_same_page_from_different_sources():
    service = CitationService()

    results = [
        {
            "chunk": "Passage from first document.",
            "distance": 0.1,
            "metadata": {
                "source": "paper_a.pdf",
                "page": 3,
                "chunk_id": 0,
            },
        },
        {
            "chunk": "Passage from second document.",
            "distance": 0.2,
            "metadata": {
                "source": "paper_b.pdf",
                "page": 3,
                "chunk_id": 0,
            },
        },
    ]

    result = service.build_citations(results)

    assert result == [
        {
            "id": 1,
            "source": "paper_a.pdf",
            "page": 3,
        },
        {
            "id": 2,
            "source": "paper_b.pdf",
            "page": 3,
        },
    ]


def test_build_citations_skips_results_without_metadata():
    service = CitationService()

    results = [
        {
            "chunk": "Passage without metadata.",
            "distance": 0.1,
            "metadata": {},
        },
        {
            "chunk": "Valid passage.",
            "distance": 0.2,
            "metadata": {
                "source": "research.pdf",
                "page": 5,
                "chunk_id": 0,
            },
        },
    ]

    result = service.build_citations(results)

    assert result == [
        {
            "id": 1,
            "source": "research.pdf",
            "page": 5,
        }
    ]