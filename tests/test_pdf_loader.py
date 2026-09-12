from pathlib import Path

import pytest

from app.services.document_loader import PDFLoader


PDF_PATH = Path("data/documents/sample.pdf")


def test_pdf_loader_returns_pages():
    loader = PDFLoader()

    pages = loader.load(PDF_PATH)

    assert len(pages) == 48
    assert isinstance(pages, list)


def test_pdf_loader_page_structure():
    loader = PDFLoader()

    pages = loader.load(PDF_PATH)

    first_page = pages[0]

    assert "page_number" in first_page
    assert "text" in first_page
    assert first_page["page_number"] == 1
    assert isinstance(first_page["text"], str)


def test_pdf_loader_extracts_text():
    loader = PDFLoader()

    pages = loader.load(PDF_PATH)

    all_text = " ".join(page["text"] for page in pages)

    assert len(all_text) > 0
    assert "FACE MASK DETECTION" in all_text.upper()


def test_pdf_loader_rejects_missing_file():
    loader = PDFLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("data/documents/does_not_exist.pdf")


def test_pdf_loader_rejects_non_pdf_file(tmp_path):
    loader = PDFLoader()

    text_file = tmp_path / "sample.txt"
    text_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError):
        loader.load(text_file)