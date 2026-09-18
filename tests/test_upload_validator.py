import pytest

from app.services.upload_validator import (
    UploadValidationError,
    UploadValidator,
)


def test_valid_pdf_is_accepted():
    validator = UploadValidator(
        max_size_bytes=1024,
    )

    filename = validator.validate(
        "research.pdf",
        b"%PDF-1.7\nresearch content",
    )

    assert filename == "research.pdf"


def test_non_pdf_extension_is_rejected():
    validator = UploadValidator()

    with pytest.raises(
        UploadValidationError,
        match="Only PDF files are supported",
    ):
        validator.validate(
            "research.txt",
            b"%PDF-1.7\ncontent",
        )


def test_missing_filename_is_rejected():
    validator = UploadValidator()

    with pytest.raises(
        UploadValidationError,
        match="A filename is required",
    ):
        validator.validate(
            "",
            b"%PDF-1.7\ncontent",
        )


def test_path_traversal_filename_is_rejected():
    validator = UploadValidator()

    with pytest.raises(
        UploadValidationError,
        match="Invalid filename",
    ):
        validator.validate(
            "../research.pdf",
            b"%PDF-1.7\ncontent",
        )


def test_empty_file_is_rejected():
    validator = UploadValidator()

    with pytest.raises(
        UploadValidationError,
        match="uploaded file is empty",
    ):
        validator.validate(
            "research.pdf",
            b"",
        )


def test_invalid_pdf_signature_is_rejected():
    validator = UploadValidator()

    with pytest.raises(
        UploadValidationError,
        match="not a valid PDF",
    ):
        validator.validate(
            "research.pdf",
            b"This is not a PDF file.",
        )


def test_oversized_file_is_rejected():
    validator = UploadValidator(
        max_size_bytes=10,
    )

    content = b"%PDF-1.7\n1234567890"

    with pytest.raises(
        UploadValidationError,
        match="maximum allowed size",
    ):
        validator.validate(
            "research.pdf",
            content,
        )


def test_uppercase_pdf_extension_is_accepted():
    validator = UploadValidator()

    filename = validator.validate(
        "research.PDF",
        b"%PDF-1.7\ncontent",
    )

    assert filename == "research.PDF"


def test_extension_without_dot_is_normalized():
    validator = UploadValidator(
        allowed_extensions=("pdf",),
    )

    filename = validator.validate(
        "research.pdf",
        b"%PDF-1.7\ncontent",
    )

    assert filename == "research.pdf"


def test_invalid_max_size_is_rejected():
    with pytest.raises(
        ValueError,
        match="max_size_bytes must be greater than 0",
    ):
        UploadValidator(
            max_size_bytes=0,
        )


def test_empty_extensions_are_rejected():
    with pytest.raises(
        ValueError,
        match="allowed_extensions must not be empty",
    ):
        UploadValidator(
            allowed_extensions=(),
        )