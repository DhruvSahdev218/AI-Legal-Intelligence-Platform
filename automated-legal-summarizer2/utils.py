"""
utils.py - Text cleaning and preprocessing utilities
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean raw extracted text from PDFs.
    Removes noise, normalizes whitespace, fixes common PDF artifacts.
    """
    if not text:
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # Remove null bytes and other control characters (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Fix hyphenated line breaks (common in PDFs): "condi-\ntion" → "condition"
    text = re.sub(r"-\s*\n\s*", "", text)

    # Replace multiple newlines with a single paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace single newlines (likely soft wraps) with a space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse multiple spaces/tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove page numbers (standalone digits on a line)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove repeated dashes or underscores (table borders, dividers)
    text = re.sub(r"[-_=]{3,}", "", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    # Final strip
    text = text.strip()

    return text


def truncate_text(text: str, max_chars: int = 4000) -> str:
    """Truncate text to a maximum number of characters, ending at a sentence."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to end at the last sentence boundary
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.7:
        return truncated[: last_period + 1]
    return truncated


def word_count(text: str) -> int:
    """Return approximate word count of text."""
    return len(text.split())


def is_valid_sentence(sentence: str, min_words: int = 5) -> bool:
    """Check if a string is a meaningful sentence worth indexing."""
    words = sentence.split()
    if len(words) < min_words:
        return False
    # Must contain at least one alphabetic character
    if not any(c.isalpha() for c in sentence):
        return False
    return True