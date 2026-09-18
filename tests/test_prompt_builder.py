import pytest

from app.services.prompt_builder import PromptBuilder


def test_prompt_builder_creates_prompt():
    builder = PromptBuilder()

    result = builder.build(
        question="What is artificial intelligence?",
        context="Artificial intelligence is a field of computer science.",
    )

    assert "What is artificial intelligence?" in result
    assert "Artificial intelligence is a field of computer science." in result


def test_prompt_contains_grounding_instruction():
    builder = PromptBuilder()

    result = builder.build(
        question="What is AI?",
        context="AI is a field of computer science.",
    )

    assert "using only the provided document context" in result
    assert "Do not invent facts" in result


def test_prompt_contains_citation_instruction():
    builder = PromptBuilder()

    result = builder.build(
        question="What is AI?",
        context="[Source: paper.pdf | Page: 5 | Chunk: 0]\nAI is a field of computer science.",
    )

    assert "cite it using the source and page information" in result
    assert "Page: 5" in result


def test_prompt_strips_question_and_context_whitespace():
    builder = PromptBuilder()

    result = builder.build(
        question="   What is AI?   ",
        context="   AI is a field of computer science.   ",
    )

    assert "USER QUESTION:\nWhat is AI?" in result
    assert "DOCUMENT CONTEXT:\nAI is a field of computer science." in result


def test_prompt_rejects_empty_question():
    builder = PromptBuilder()

    with pytest.raises(ValueError, match="question"):
        builder.build(
            question="",
            context="Some document context.",
        )


def test_prompt_rejects_empty_context():
    builder = PromptBuilder()

    with pytest.raises(ValueError, match="context"):
        builder.build(
            question="What is AI?",
            context="",
        )