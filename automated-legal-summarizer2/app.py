
import os
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from ingest import extract_text_from_pdf, get_pdf_metadata
from chunker import split_into_chunks, chunk_statistics
from embedder import embed_chunks, build_faiss_index
from store import save_index, load_index, index_exists, list_indexed_documents
from summarizer import summarize_chunks
from qa import answer_question

# ─── App setup ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
(BASE_DIR / "indices").mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload


# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_doc_id(filename: str) -> str:
    """Create a safe, unique document identifier from an original filename."""
    stem = Path(secure_filename(filename)).stem
    short_id = uuid.uuid4().hex[:8]
    return f"{stem}_{short_id}"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/documents", methods=["GET"])
def documents():
    docs = list_indexed_documents()
    return jsonify({"documents": docs})


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accepts a multipart/form-data request with a single PDF file.
    Extracts text → chunks → embeddings → FAISS index.
    Returns the doc_id and processing statistics.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    # Save raw upload
    doc_id = make_doc_id(file.filename)
    save_path = UPLOAD_DIR / f"{doc_id}.pdf"
    file.save(str(save_path))

    # Pipeline
    try:
        # 1. Extract
        text = extract_text_from_pdf(str(save_path))
        metadata = get_pdf_metadata(str(save_path))

        # 2. Chunk
        chunks = split_into_chunks(text, chunk_size=5, overlap=1)
        if not chunks:
            return jsonify({"error": "Could not extract meaningful text from PDF."}), 422
        stats = chunk_statistics(chunks)

        # 3. Embed
        embeddings = embed_chunks(chunks)

        # 4. Index
        faiss_index = build_faiss_index(embeddings)

        # 5. Persist
        save_index(doc_id, faiss_index, chunks)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

    return jsonify({
        "doc_id": doc_id,
        "filename": file.filename,
        "metadata": metadata,
        "chunk_stats": stats,
        "message": "Document successfully processed and indexed.",
    })


@app.route("/summarize", methods=["POST"])
def summarize():
    """
    Body (JSON): { "doc_id": "<id>" }
    Returns a generated summary of the document.
    """
    data = request.get_json(force=True, silent=True) or {}
    doc_id = data.get("doc_id", "").strip()

    if not doc_id:
        return jsonify({"error": "doc_id is required."}), 400
    if not index_exists(doc_id):
        return jsonify({"error": f"Document '{doc_id}' not found. Upload it first."}), 404

    try:
        _, chunks = load_index(doc_id)
        summary = summarize_chunks(chunks)
    except Exception as e:
        return jsonify({"error": f"Summarisation failed: {e}"}), 500

    return jsonify({"doc_id": doc_id, "summary": summary})


@app.route("/ask", methods=["POST"])
def ask():
    """
    Body (JSON): { "doc_id": "<id>", "question": "<question>" }
    Returns the extracted answer, confidence score, and source chunks.
    """
    data = request.get_json(force=True, silent=True) or {}
    doc_id = data.get("doc_id", "").strip()
    question = data.get("question", "").strip()

    if not doc_id:
        return jsonify({"error": "doc_id is required."}), 400
    if not question:
        return jsonify({"error": "question is required."}), 400
    if not index_exists(doc_id):
        return jsonify({"error": f"Document '{doc_id}' not found. Upload it first."}), 404

    try:
        faiss_index, chunks = load_index(doc_id)
        result = answer_question(question, faiss_index, chunks)
    except Exception as e:
        return jsonify({"error": f"QA failed: {e}"}), 500

    return jsonify({
        "doc_id": doc_id,
        "question": question,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources": result["sources"][:3],   # top 3 source passages
    })


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)