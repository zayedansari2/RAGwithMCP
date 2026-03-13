"""Document processing: read files and split into chunks."""

import io
import os
from pathlib import Path


def extract_text(file_path: str) -> str:
    """Extract plain text from a file (PDF, DOCX, TXT, MD)."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file_path)
    elif suffix == ".docx":
        return _extract_docx(file_path)
    elif suffix in (".txt", ".md", ".rst", ".csv"):
        return _extract_text_file(file_path)
    else:
        # Attempt to read as plain text
        return _extract_text_file(file_path)


def _extract_pdf(file_path: str) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError("pypdf is required to read PDF files. Run: pip install pypdf")


def _extract_docx(file_path: str) -> str:
    try:
        import docx

        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        raise ImportError(
            "python-docx is required to read DOCX files. Run: pip install python-docx"
        )


def _extract_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start += chunk_size - chunk_overlap

    return chunks


def process_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Extract text from a file and split it into chunks."""
    text = extract_text(file_path)
    return split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
