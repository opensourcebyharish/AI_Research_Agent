from pathlib import Path

import pymupdf


class PDFLoader:
    """Load a PDF file and extract its text page by page."""

    def load(self, file_path: str | Path) -> list[dict[str, object]]:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A list containing one dictionary per page.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError: If the file is not a PDF.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        pages = []

        with pymupdf.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                    }
                )

        return pages