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
    else:
        return _extract_text_file(file_path)


def _extract_pdf(file_path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is required to read PDF files. Run: pip install pypdf"
        )
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx(file_path) -> str:
    try:
        import docx
    except ImportError:
        raise ImportError(
            "python-docx is required to read DOCX files. Run: pip install python-docx"
        )
    doc = docx.Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs]
    return "\n\n".join(paragraphs)


def _extract_text_file(file_path) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split text into overlapping chunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )
    text = text.strip()
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def process_file(
    file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list:
    """Extract text from a file and split it into chunks."""
    text = extract_text(file_path)
    return split_text(text, chunk_size, chunk_overlap)
