"""
ingest.py - Extract and clean text from uploaded PDF files using PyPDF2
"""

import os
import PyPDF2
from utils import clean_text


def extract_text_from_pdf(filepath: str) -> str:
    """
    Extract raw text from every page of a PDF, then clean it.

    Args:
        filepath: Absolute or relative path to the PDF file.

    Returns:
        A single cleaned string containing the full document text.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the PDF is empty or could not be read.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")

    raw_pages: list[str] = []

    with open(filepath, "rb") as pdf_file:
        try:
            reader = PyPDF2.PdfReader(pdf_file)
        except Exception as exc:
            raise ValueError(f"Could not parse PDF: {exc}") from exc

        if len(reader.pages) == 0:
            raise ValueError("PDF contains no pages.")

        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                raw_pages.append(page_text)
            except Exception:
                # Skip unreadable pages silently
                raw_pages.append("")

    full_raw_text = "\n\n".join(raw_pages)

    if not full_raw_text.strip():
        raise ValueError(
            "No text could be extracted from the PDF. "
            "The document may be scanned or image-based."
        )

    cleaned = clean_text(full_raw_text)
    return cleaned


def get_pdf_metadata(filepath: str) -> dict:
    """
    Extract basic metadata from a PDF file.

    Returns a dict with keys: title, author, num_pages.
    """
    metadata = {"title": None, "author": None, "num_pages": 0}
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            metadata["num_pages"] = len(reader.pages)
            if reader.metadata:
                metadata["title"] = reader.metadata.get("/Title")
                metadata["author"] = reader.metadata.get("/Author")
    except Exception:
        pass
    return metadata