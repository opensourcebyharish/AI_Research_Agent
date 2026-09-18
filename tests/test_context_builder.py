from app.services.context_builder import ContextBuilder


def test_build_context_returns_empty_string_for_empty_results():
    builder = ContextBuilder()

    result = builder.build_context([])

    assert result == ""


def test_build_context_formats_single_result():
    builder = ContextBuilder()

    results = [
        {
            "chunk": "Artificial intelligence is a field of computer science.",
            "distance": 0.25,
            "metadata": {
                "source": "research.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        }
    ]

    result = builder.build_context(results)

    assert "[Source: research.pdf | Page: 2 | Chunk: 0]" in result
    assert "Artificial intelligence is a field of computer science." in result


def test_build_context_formats_multiple_results():
    builder = ContextBuilder()

    results = [
        {
            "chunk": "First relevant passage.",
            "distance": 0.2,
            "metadata": {
                "source": "paper.pdf",
                "page": 1,
                "chunk_id": 0,
            },
        },
        {
            "chunk": "Second relevant passage.",
            "distance": 0.4,
            "metadata": {
                "source": "paper.pdf",
                "page": 3,
                "chunk_id": 1,
            },
        },
    ]

    result = builder.build_context(results)

    assert "[Source: paper.pdf | Page: 1 | Chunk: 0]" in result
    assert "[Source: paper.pdf | Page: 3 | Chunk: 1]" in result
    assert "First relevant passage." in result
    assert "Second relevant passage." in result


def test_build_context_separates_multiple_results():
    builder = ContextBuilder()

    results = [
        {
            "chunk": "First passage.",
            "distance": 0.2,
            "metadata": {
                "source": "paper.pdf",
                "page": 1,
                "chunk_id": 0,
            },
        },
        {
            "chunk": "Second passage.",
            "distance": 0.3,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        },
    ]

    result = builder.build_context(results)

    assert "First passage.\n\n[Source:" in result


def test_build_context_skips_empty_chunks():
    builder = ContextBuilder()

    results = [
        {
            "chunk": "",
            "distance": 0.1,
            "metadata": {
                "source": "paper.pdf",
                "page": 1,
                "chunk_id": 0,
            },
        },
        {
            "chunk": "Valid passage.",
            "distance": 0.2,
            "metadata": {
                "source": "paper.pdf",
                "page": 2,
                "chunk_id": 1,
            },
        },
    ]

    result = builder.build_context(results)

    assert "Valid passage." in result
    assert "Page: 1" not in result