from langchain_chroma import Chroma

from src.ingestion import COLLECTION_NAME, get_active_chroma_dir, get_embeddings


def get_vector_store(session_id: str | None = None):
    """Open the persistent Chroma vector store."""
    chroma_dir = get_active_chroma_dir(session_id)
    if not chroma_dir.exists():
        return None

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(chroma_dir),
        embedding_function=get_embeddings(),
    )


def retrieve_documents(question: str, k: int = 5, session_id: str | None = None):
    """Retrieve the most relevant document chunks."""
    if not question or not question.strip():
        return []

    vector_store = get_vector_store(session_id) if session_id else get_vector_store()
    if vector_store is None:
        return []

    try:
        return vector_store.similarity_search(question.strip(), k=k)
    except Exception:
        print("Document retrieval failed.")
        return []