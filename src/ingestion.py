import os
import re
import shutil
import uuid
from pathlib import Path
from threading import Lock

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
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
    """Create the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def _load_pdf_documents(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    documents = []

    for page in pages:
        metadata = dict(page.metadata or {})
        metadata["source"] = pdf_path.name
        metadata["page"] = int(metadata.get("page", 0)) + 1
        page.metadata = metadata
        if page.page_content and page.page_content.strip():
            documents.append(page)

    return documents


def create_vector_store(session_id: str | None = None):
    """Load PDFs and rebuild the persistent Chroma vector store safely."""
    pdf_dir = get_session_pdf_dir(session_id)
    chroma_dir = get_session_chroma_dir(session_id)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}. Upload or add a PDF first.")

    documents = []
    for pdf_path in pdf_files:
        try:
            documents.extend(_load_pdf_documents(pdf_path))
        except Exception as exc:
            print("Skipping an unreadable PDF.")

    if not documents:
        raise ValueError("No readable text was found in the PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks could be created from the PDFs.")

    embeddings = get_embeddings()
    active_dir = chroma_dir.parent / f"{chroma_dir.name}_{uuid.uuid4().hex}"

    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(active_dir),
            collection_name=COLLECTION_NAME,
        )
        write_active_chroma_dir(active_dir, session_id)

        if chroma_dir.exists() and chroma_dir.is_dir():
            try:
                shutil.rmtree(chroma_dir)
            except PermissionError:
                print(f"Keeping existing active directory {chroma_dir} in place because it is still in use. The new index is active at {active_dir}.")

        print(f"Vector store saved to: {active_dir}")
        return vector_store
    except Exception:
        if active_dir.exists():
            shutil.rmtree(active_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    create_vector_store()