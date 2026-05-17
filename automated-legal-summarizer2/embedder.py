"""
embedder.py - Convert text chunks into dense vector embeddings using
SentenceTransformers and build a FAISS index for fast similarity search.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Singleton model — loaded once and reused across calls
_MODEL: SentenceTransformer | None = None
_MODEL_NAME = "all-MiniLM-L6-v2"          # 22 M params, 384-dim, fast & accurate


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def embed_chunks(chunks: list[str]) -> np.ndarray:
    """
    Encode a list of text chunks into L2-normalised embedding vectors.

    Args:
        chunks: List of text strings to embed.

    Returns:
        Float32 numpy array of shape (N, embedding_dim).
    """
    if not chunks:
        raise ValueError("Cannot embed an empty list of chunks.")

    model = _get_model()
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → cosine sim == dot product
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an inner-product (cosine) FAISS index from a matrix of embeddings.

    Because the vectors are L2-normalised, inner-product == cosine similarity.

    Args:
        embeddings: Float32 array of shape (N, dim).

    Returns:
        A populated faiss.IndexFlatIP ready for similarity search.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # exact inner-product search
    index.add(embeddings)
    return index


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        Float32 array of shape (1, dim).
    """
    model = _get_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.astype(np.float32)


def search_index(
    index: faiss.IndexFlatIP,
    query_vec: np.ndarray,
    chunks: list[str],
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query vector.

    Args:
        index:     Populated FAISS index.
        query_vec: Shape (1, dim) query embedding.
        chunks:    Original list of text chunks (parallel to index rows).
        top_k:     Number of results to return.

    Returns:
        List of dicts: [{"chunk": str, "score": float, "index": int}, ...]
        sorted by descending similarity score.
    """
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:          # FAISS placeholder for missing results
            continue
        results.append({
            "chunk": chunks[idx],
            "score": float(score),
            "index": int(idx),
        })

    return results