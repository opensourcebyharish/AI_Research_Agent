from pathlib import Path


class UploadValidationError(ValueError):
    """Raised when an uploaded document fails validation."""


class UploadValidator:
    """Validate uploaded documents before they enter the pipeline."""

    PDF_SIGNATURE = b"%PDF"

    def __init__(
        self,
        max_size_bytes: int = 25 * 1024 * 1024,
        allowed_extensions: tuple[str, ...] = (".pdf",),
    ):
        if max_size_bytes <= 0:
            raise ValueError(
                "max_size_bytes must be greater than 0"
            )

        if not allowed_extensions:
            raise ValueError(
                "allowed_extensions must not be empty"
            )

        self.max_size_bytes = max_size_bytes

        self.allowed_extensions = tuple(
            extension.lower()
            if extension.startswith(".")
            else f".{extension.lower()}"
            for extension in allowed_extensions
        )

    def validate_filename(
        self,
        filename: str,
    ) -> str:
        """
        Validate and normalize a user-supplied filename.

        Returns:
            Safe basename.

        Raises:
            UploadValidationError: If the filename is invalid.
        """

        if not filename or not filename.strip():
            raise UploadValidationError(
                "A filename is required."
            )

        safe_name = Path(filename).name

        if safe_name != filename:
            raise UploadValidationError(
                "Invalid filename."
            )

        extension = Path(safe_name).suffix.lower()

        if extension not in self.allowed_extensions:
            raise UploadValidationError(
                "Only PDF files are supported."
            )

        return safe_name

    def validate_bytes(
        self,
        content: bytes,
    ) -> None:
        """
        Validate uploaded file bytes.

        Raises:
            UploadValidationError: If the content is invalid.
        """

        if not content:
            raise UploadValidationError(
                "The uploaded file is empty."
            )

        if len(content) > self.max_size_bytes:
            max_size_mb = (
                self.max_size_bytes
                / (1024 * 1024)
            )

            raise UploadValidationError(
                f"File exceeds the maximum allowed size "
                f"of {max_size_mb:g} MB."
            )

        if not content.startswith(
            self.PDF_SIGNATURE
        ):
            raise UploadValidationError(
                "The uploaded file is not a valid PDF."
            )

    def validate(
        self,
        filename: str,
        content: bytes,
    ) -> str:
        """
        Validate both filename and file content.

        Returns:
            Safe normalized filename.
        """

        safe_name = self.validate_filename(
            filename
        )

        self.validate_bytes(
            content
        )

        return safe_name