import os
import re
import shutil
import uuid
from pathlib import Path
from threading import Lock
from typing import Generator, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_DIR = Path(__file__).resolve().parent.parent
_data_dir = Path(os.getenv("RAG_DATA_DIR", str(PROJECT_DIR))).expanduser()
DATA_DIR = (_data_dir if _data_dir.is_absolute() else PROJECT_DIR / _data_dir).resolve()
PDF_DIR = DATA_DIR / "data" / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma_db"
SESSION_DIR = DATA_DIR / "sessions"
COLLECTION_NAME = "project_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_LOCK = Lock()

# Global cached embedding model instance to avoid reloading heavy models
_CACHED_EMBEDDINGS = None
_EMBEDDINGS_LOCK = Lock()


def normalize_session_id(session_id: str | None) -> str:
    value = (session_id or "").strip()
    if not value:
        return "_shared"
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value)
    normalized = normalized.strip("-_")
    return normalized or "_shared"


def get_session_root(session_id: str | None = None) -> Path:
    session_name = normalize_session_id(session_id)
    return SESSION_DIR / session_name


def get_session_pdf_dir(session_id: str | None = None) -> Path:
    if not session_id:
        return PDF_DIR
    return get_session_root(session_id)


def get_session_chroma_dir(session_id: str | None = None) -> Path:
    if not session_id:
        return CHROMA_DIR
    return get_session_root(session_id) / "chroma_db"


def get_active_chroma_marker(session_id: str | None = None) -> Path:
    if session_id:
        return get_session_root(session_id) / "active_chroma_dir.txt"
    return DATA_DIR / "active_chroma_dir.txt"


def get_active_chroma_dir(session_id: str | None = None) -> Path:
    """Return the newest valid Chroma directory for the active index."""
    default_dir = get_session_chroma_dir(session_id)
    marker = get_active_chroma_marker(session_id)
    if marker.exists():
        try:
            marker_value = marker.read_text(encoding="utf-8").strip()
            if marker_value:
                candidate = Path(marker_value)
                if candidate.is_dir():
                    return candidate
        except Exception:
            pass

    candidates = [default_dir]
    candidates.extend(
        sorted(
            default_dir.parent.glob(f"{default_dir.name}_*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return default_dir


def write_active_chroma_dir(chroma_dir: Path, session_id: str | None = None) -> Path:
    marker = get_active_chroma_marker(session_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(chroma_dir), encoding="utf-8")
    return chroma_dir


def get_embeddings():
    """Create or return cached HuggingFace embeddings model for fast reuse."""
    global _CACHED_EMBEDDINGS
    if _CACHED_EMBEDDINGS is not None:
        return _CACHED_EMBEDDINGS

    with _EMBEDDINGS_LOCK:
        if _CACHED_EMBEDDINGS is None:
            _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"batch_size": 64},  # Optimized batch encoding
            )
    return _CACHED_EMBEDDINGS


def _load_pdf_pages_stream(pdf_path: Path) -> Generator[Document, None, None]:
    """
    Stream PDF pages lazily with high performance.
    Uses PyMuPDF (fitz) if available for 10x-50x speedup and low memory on 100MB+ files;
    falls back to PyPDFLoader if pymupdf is not available.
    """
    try:
        import pymupdf  # Fast C++ engine

        doc = pymupdf.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text and text.strip():
                yield Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": page_num + 1},
                )
        doc.close()
        return
    except ImportError:
        pass

    # Fallback to pypdf
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(pdf_path))
    for page in loader.lazy_load():
        text = page.page_content or ""
        if text.strip():
            metadata = dict(page.metadata or {})
            metadata["source"] = pdf_path.name
            metadata["page"] = int(metadata.get("page", 0)) + 1
            page.metadata = metadata
            yield page


def create_vector_store(session_id: str | None = None, batch_size: int = 128):
    """
    Load PDFs and rebuild the persistent Chroma vector store safely.
    Optimized for large documents up to 100 MB via generator streaming and batched chunk indexing.
    """
    pdf_dir = get_session_pdf_dir(session_id)
    chroma_dir = get_session_chroma_dir(session_id)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}. Upload or add a PDF first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )

    embeddings = get_embeddings()
    active_dir = chroma_dir.parent / f"{chroma_dir.name}_{uuid.uuid4().hex}"

    vector_store = None
    total_chunks = 0
    buffer_chunks: List[Document] = []

    try:
        for pdf_path in pdf_files:
            try:
                for doc in _load_pdf_pages_stream(pdf_path):
                    page_chunks = splitter.split_documents([doc])
                    buffer_chunks.extend(page_chunks)

                    # Ingest in batches to prevent huge RAM spikes on 100MB files
                    if len(buffer_chunks) >= batch_size:
                        if vector_store is None:
                            vector_store = Chroma.from_documents(
                                documents=buffer_chunks,
                                embedding=embeddings,
                                persist_directory=str(active_dir),
                                collection_name=COLLECTION_NAME,
                            )
                        else:
                            vector_store.add_documents(buffer_chunks)
                        total_chunks += len(buffer_chunks)
                        buffer_chunks = []
            except Exception as exc:
                print(f"Skipping error in PDF {pdf_path.name}: {exc}")

        # Ingest remaining chunks
        if buffer_chunks:
            if vector_store is None:
                vector_store = Chroma.from_documents(
                    documents=buffer_chunks,
                    embedding=embeddings,
                    persist_directory=str(active_dir),
                    collection_name=COLLECTION_NAME,
                )
            else:
                vector_store.add_documents(buffer_chunks)
            total_chunks += len(buffer_chunks)
            buffer_chunks = []

        if total_chunks == 0 or vector_store is None:
            raise ValueError("No readable text chunks could be extracted from the PDFs.")

        write_active_chroma_dir(active_dir, session_id)

        if chroma_dir.exists() and chroma_dir.is_dir():
            try:
                shutil.rmtree(chroma_dir)
            except PermissionError:
                print(
                    f"Keeping existing active directory {chroma_dir} in place because it is still in use. The new index is active at {active_dir}."
                )

        print(f"Vector store ({total_chunks} chunks) saved to: {active_dir}")
        return vector_store
    except Exception:
        if active_dir.exists():
            shutil.rmtree(active_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    create_vector_store()
