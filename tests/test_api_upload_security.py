from io import BytesIO

from fastapi import UploadFile
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_upload_rejects_empty_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "empty.pdf",
                BytesIO(b""),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_rejects_fake_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "fake.pdf",
                BytesIO(b"this is not a pdf"),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "valid pdf" in (
        response.json()["detail"].lower()
    )


def test_upload_rejects_path_traversal():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "../malicious.pdf",
                BytesIO(
                    b"%PDF-1.7\n"
                    b"fake content"
                ),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "invalid filename" in (
        response.json()["detail"].lower()
    )


def test_upload_rejects_non_pdf_content():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "research.pdf",
                BytesIO(
                    b"PK\x03\x04"
                    b"not really a pdf"
                ),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "valid pdf" in (
        response.json()["detail"].lower()
    )