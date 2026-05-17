"""
store.py - Persist and load the FAISS index and chunk list to the indices/ folder.

Each processed document gets its own sub-folder identified by a doc_id
(typically the original filename without extension).

Layout on disk:
    indices/
    └── <doc_id>/
        ├── faiss.index   ← FAISS binary index
        └── chunks.json   ← JSON array of chunk strings
"""

import os
import json
import faiss
import numpy as np

INDICES_DIR = os.path.join(os.path.dirname(__file__), "indices")


def _doc_dir(doc_id: str) -> str:
    """Return (and create if needed) the storage directory for a document."""
    path = os.path.join(INDICES_DIR, doc_id)
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
#  Save                                                                        #
# --------------------------------------------------------------------------- #

def save_index(doc_id: str, index: faiss.Index, chunks: list[str]) -> str:
    """
    Persist a FAISS index and its corresponding chunks to disk.

    Args:
        doc_id: Unique identifier for the document (e.g. filename stem).
        index:  Populated FAISS index.
        chunks: List of text chunks parallel to the index rows.

    Returns:
        Path to the directory where data was saved.
    """
    directory = _doc_dir(doc_id)

    # Save FAISS index
    faiss_path = os.path.join(directory, "faiss.index")
    faiss.write_index(index, faiss_path)

    # Save chunks as JSON
    chunks_path = os.path.join(directory, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return directory


# --------------------------------------------------------------------------- #
#  Load                                                                        #
# --------------------------------------------------------------------------- #

def load_index(doc_id: str) -> tuple[faiss.Index, list[str]]:
    """
    Load a previously saved FAISS index and chunk list from disk.

    Args:
        doc_id: Unique identifier used when saving.

    Returns:
        Tuple of (faiss_index, chunks_list).

    Raises:
        FileNotFoundError: If the index files do not exist for this doc_id.
    """
    directory = _doc_dir(doc_id)
    faiss_path = os.path.join(directory, "faiss.index")
    chunks_path = os.path.join(directory, "chunks.json")

    if not os.path.exists(faiss_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"No stored index found for document '{doc_id}'. "
            "Please upload and process the document first."
        )

    index = faiss.read_index(faiss_path)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


def index_exists(doc_id: str) -> bool:
    """Return True if a persisted index exists for the given doc_id."""
    directory = os.path.join(INDICES_DIR, doc_id)
    return (
        os.path.exists(os.path.join(directory, "faiss.index"))
        and os.path.exists(os.path.join(directory, "chunks.json"))
    )


def list_indexed_documents() -> list[str]:
    """Return a list of all doc_ids that have been indexed."""
    if not os.path.isdir(INDICES_DIR):
        return []
    return [
        name
        for name in os.listdir(INDICES_DIR)
        if os.path.isdir(os.path.join(INDICES_DIR, name))
    ]