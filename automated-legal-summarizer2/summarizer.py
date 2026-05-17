"""
summarizer.py - Generate abstractive summaries of legal documents using
a fine-tuned T5 model (google/flan-t5-base — instruction-aware and
stronger than plain t5-base for summarisation tasks).
"""

from transformers import pipeline, Pipeline
from utils import truncate_text

# Singleton pipeline — loaded once
_SUMMARIZER: Pipeline | None = None
_MODEL_NAME = "google/flan-t5-base"

# T5 can handle ~512 tokens; we guard with a char-level cutoff
_MAX_INPUT_CHARS = 3500
_MAX_NEW_TOKENS = 300
_MIN_NEW_TOKENS = 60


def _get_summarizer() -> Pipeline:
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = pipeline(
            "summarization",
            model=_MODEL_NAME,
            tokenizer=_MODEL_NAME,
            device=-1,          # CPU; change to 0 for GPU
        )
    return _SUMMARIZER


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def summarize_text(text: str) -> str:
    """
    Generate a concise summary of the supplied text.

    For long documents the text is first truncated to fit within the model's
    context window.  Chunked documents can be summarised by concatenating a
    representative sample of their chunks before calling this function.

    Args:
        text: Cleaned document text (or concatenated chunks).

    Returns:
        A one-to-three-paragraph summary string.
    """
    if not text.strip():
        return "No content available to summarise."

    # Prepend an instruction prefix for Flan-T5
    prompt = "Summarize the following legal document:\n\n" + text
    prompt = truncate_text(prompt, max_chars=_MAX_INPUT_CHARS)

    summarizer = _get_summarizer()

    try:
        result = summarizer(
            prompt,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=_MIN_NEW_TOKENS,
            do_sample=False,
            truncation=True,
        )
        summary: str = result[0]["summary_text"].strip()
    except Exception as exc:
        return f"Summarisation failed: {exc}"

    return summary


def summarize_chunks(chunks: list[str], sample_size: int = 20) -> str:
    """
    Summarise a document represented as a list of chunks.

    Takes an evenly-spaced sample of *sample_size* chunks from across the
    document to give the model a representative cross-section, then summarises
    the concatenated sample.

    Args:
        chunks:      All chunks for the document.
        sample_size: Maximum number of chunks to include in the input.

    Returns:
        Summary string.
    """
    if not chunks:
        return "No content available to summarise."

    # Even-step sampling to cover the whole document
    if len(chunks) <= sample_size:
        sampled = chunks
    else:
        step = len(chunks) / sample_size
        sampled = [chunks[int(i * step)] for i in range(sample_size)]

    combined = " ".join(sampled)
    return summarize_text(combined)