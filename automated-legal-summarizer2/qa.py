"""
qa.py - Answer user questions about a legal document using a two-stage pipeline:

  Stage 1 – Retrieval:  Use FAISS (cosine similarity over SentenceTransformer
                         embeddings) to find the top-k most relevant chunks.

  Stage 2 – Reading:    Feed the retrieved context + question to a
                         RoBERTa-based extractive QA model
                         (deepset/roberta-base-squad2) to extract the answer
                         span and a confidence score.
"""

from transformers import pipeline, Pipeline
from embedder import embed_query, search_index
import faiss

# Singleton QA pipeline
_QA_PIPELINE: Pipeline | None = None
_QA_MODEL = "deepset/roberta-base-squad2"

_TOP_K_RETRIEVAL = 5          # chunks to retrieve before QA
_MAX_CONTEXT_CHARS = 3000     # guard against oversized context


def _get_qa_pipeline() -> Pipeline:
    global _QA_PIPELINE
    if _QA_PIPELINE is None:
        _QA_PIPELINE = pipeline(
            "question-answering",
            model=_QA_MODEL,
            tokenizer=_QA_MODEL,
            device=-1,
        )
    return _QA_PIPELINE


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _build_context(retrieved: list[dict], max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """Concatenate retrieved chunks into a single context string."""
    parts: list[str] = []
    total = 0
    for item in retrieved:
        chunk = item["chunk"]
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return " ".join(parts)


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def answer_question(
    question: str,
    index: faiss.Index,
    chunks: list[str],
    top_k: int = _TOP_K_RETRIEVAL,
) -> dict:
    """
    Answer a question about a document using retrieval-augmented QA.

    Args:
        question: The user's natural-language question.
        index:    Loaded FAISS index for the document.
        chunks:   Corresponding list of text chunks.
        top_k:    Number of chunks to retrieve.

    Returns:
        Dict with keys:
            answer      – extracted answer span (str)
            confidence  – model confidence in [0, 1] (float)
            context     – the retrieved context used (str)
            sources     – list of retrieved chunk dicts with score
    """
    if not question.strip():
        return {
            "answer": "Please provide a question.",
            "confidence": 0.0,
            "context": "",
            "sources": [],
        }

    # ── Stage 1: Retrieval ──────────────────────────────────────────────────
    query_vec = embed_query(question)
    retrieved = search_index(index, query_vec, chunks, top_k=top_k)

    if not retrieved:
        return {
            "answer": "No relevant passages found in the document.",
            "confidence": 0.0,
            "context": "",
            "sources": [],
        }

    context = _build_context(retrieved)

    # ── Stage 2: Extractive QA ─────────────────────────────────────────────
    qa = _get_qa_pipeline()

    try:
        result = qa(
            question=question,
            context=context,
            max_answer_len=150,
            handle_impossible_answer=True,
        )
        answer: str = result.get("answer", "").strip()
        score: float = float(result.get("score", 0.0))
    except Exception as exc:
        return {
            "answer": f"QA model error: {exc}",
            "confidence": 0.0,
            "context": context,
            "sources": retrieved,
        }

    # If the model returned an empty answer treat it as not found
    if not answer:
        answer = "The answer could not be found in the retrieved passages."
        score = 0.0

    return {
        "answer": answer,
        "confidence": round(score, 4),
        "context": context,
        "sources": retrieved,
    }