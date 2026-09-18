import pytest

from app.services.text_processor import TextProcessor


def test_clean_text_normalizes_spaces():
    processor = TextProcessor()

    text = "Hello    world\tthis is   a test"

    result = processor.clean_text(text)

    assert result == "Hello world this is a test"


def test_clean_text_removes_excessive_blank_lines():
    processor = TextProcessor()

    text = "Hello\n\n\n\nWorld"

    result = processor.clean_text(text)

    assert result == "Hello\n\nWorld"


def test_clean_text_strips_outer_whitespace():
    processor = TextProcessor()

    text = "   Hello world   "

    result = processor.clean_text(text)

    assert result == "Hello world"


def test_clean_text_handles_empty_text():
    processor = TextProcessor()

    assert processor.clean_text("") == ""


def test_clean_text_preserves_content():
    processor = TextProcessor()

    text = "This is important research content."

    assert processor.clean_text(text) == text


def test_chunk_text_creates_multiple_chunks():
    processor = TextProcessor()

    text = " ".join(f"word{i}" for i in range(1, 11))

    chunks = processor.chunk_text(
        text,
        chunk_size=4,
        chunk_overlap=1,
    )

    assert len(chunks) == 3
    assert chunks[0] == "word1 word2 word3 word4"
    assert chunks[1] == "word4 word5 word6 word7"
    assert chunks[2] == "word7 word8 word9 word10"


def test_chunk_text_preserves_overlap():
    processor = TextProcessor()

    text = "one two three four five six"

    chunks = processor.chunk_text(
        text,
        chunk_size=4,
        chunk_overlap=2,
    )

    assert chunks[0] == "one two three four"
    assert chunks[1] == "three four five six"


def test_chunk_text_handles_short_text():
    processor = TextProcessor()

    text = "one two three"

    chunks = processor.chunk_text(
        text,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert chunks == ["one two three"]


def test_chunk_text_handles_empty_text():
    processor = TextProcessor()

    assert processor.chunk_text("") == []


def test_chunk_text_rejects_invalid_parameters():
    processor = TextProcessor()

    with pytest.raises(ValueError):
        processor.chunk_text("some text", chunk_size=0)

    with pytest.raises(ValueError):
        processor.chunk_text(
            "some text",
            chunk_size=4,
            chunk_overlap=4,
        )