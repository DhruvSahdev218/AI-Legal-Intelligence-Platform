"""
chunker.py - Split cleaned document text into overlapping sentence-based chunks
using NLTK sentence tokenisation.
"""

import nltk
from utils import is_valid_sentence

# Download the Punkt tokeniser data on first use (silent if already cached)
def _ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def split_into_chunks(
    text: str,
    chunk_size: int = 5,
    overlap: int = 1,
) -> list[str]:
    """
    Tokenise *text* into sentences, then group them into overlapping chunks.

    Args:
        text:       Cleaned document text.
        chunk_size: Number of sentences per chunk.
        overlap:    Number of sentences shared between adjacent chunks
                    (sliding-window style).

    Returns:
        A list of chunk strings.  Each chunk is a paragraph of several
        sentences that form a coherent semantic unit.
    """
    _ensure_nltk_data()

    if not text.strip():
        return []

    # Sentence tokenisation
    sentences: list[str] = nltk.sent_tokenize(text)

    # Filter out noise (very short fragments, page headers, lone numbers …)
    sentences = [s.strip() for s in sentences if is_valid_sentence(s)]

    if not sentences:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)          # how far to advance each time

    for start in range(0, len(sentences), step):
        window = sentences[start : start + chunk_size]
        chunk_text = " ".join(window).strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Stop once we've included the last sentence
        if start + chunk_size >= len(sentences):
            break

    return chunks


def chunk_statistics(chunks: list[str]) -> dict:
    """Return basic statistics about a list of chunks (for logging / debug)."""
    if not chunks:
        return {"count": 0, "avg_words": 0, "min_words": 0, "max_words": 0}

    word_counts = [len(c.split()) for c in chunks]
    return {
        "count": len(chunks),
        "avg_words": round(sum(word_counts) / len(word_counts), 1),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
    }