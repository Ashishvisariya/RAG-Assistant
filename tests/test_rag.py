
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from app import app
from src import retrieval


client = TestClient(app)


def test_root_and_health_endpoints():
    root = client.get("/")
    assert root.status_code == 200
    body = root.json()
    assert "Advanced RAG System" in body["message"]
    assert body["health"] == "/health"

    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "healthy"


def test_invalid_question_is_rejected():
    result = client.post("/ask", json={"question": "   "})
    assert result.status_code in {400, 422}


def test_upload_rejects_non_pdf():
    result = client.post(
        "/upload",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )
    assert result.status_code == 400
    assert "PDF" in result.json()["detail"]


def test_retrieve_documents_returns_results(monkeypatch):
    fake_doc = MagicMock()
    fake_doc.page_content = "RAG combines retrieval with generation."
    fake_doc.metadata = {"source": "guide.pdf", "page": 1}

    fake_store = MagicMock()
    fake_store.similarity_search.return_value = [fake_doc]

    monkeypatch.setattr(retrieval, "get_vector_store", lambda: fake_store)

    results = retrieval.retrieve_documents("What is RAG?", k=3)

    assert len(results) == 1
    assert results[0].page_content == "RAG combines retrieval with generation."
    fake_store.similarity_search.assert_called_once_with("What is RAG?", k=3)


def test_retrieve_documents_returns_empty_list(monkeypatch):
    fake_store = MagicMock()
    fake_store.similarity_search.return_value = []

    monkeypatch.setattr(retrieval, "get_vector_store", lambda: fake_store)

    assert retrieval.retrieve_documents("Unknown question") == []


def test_generate_questions_handles_empty_database(monkeypatch):
    monkeypatch.setattr("src.graph.retrieve_documents", lambda *args, **kwargs: [])
    questions = __import__("src.graph", fromlist=["generate_questions"]).generate_questions(5)
    assert questions == []


def test_graph_answer_payload_contains_metadata(monkeypatch):
    fake_doc = SimpleNamespace(
        page_content="The model uses dense retrieval and generation.",
        metadata={"source": "guide.pdf", "page": 2},
    )

    monkeypatch.setattr("src.graph.retrieve_documents", lambda *args, **kwargs: [fake_doc])

    monkeypatch.setattr(
        "src.graph.get_llm",
        lambda: RunnableLambda(lambda payload: "This answer is supported by the document."),
    )

    result = __import__("src.graph", fromlist=["ask_question"]).ask_question("What is the model doing?")

    assert result["answer"]
    assert result["answer_type"] in {"document_grounded", "general"}
    assert "sources" in result


def test_active_chroma_dir_prefers_latest_snapshot(tmp_path, monkeypatch):
    import src.ingestion as ingestion

    monkeypatch.setattr(ingestion, "PROJECT_DIR", tmp_path)
    marker = tmp_path / "active_chroma_dir.txt"
    old_dir = tmp_path / "chroma_db"
    new_dir = tmp_path / "chroma_db_202406011200"
    old_dir.mkdir()
    new_dir.mkdir()
    marker.write_text(str(new_dir))

    assert ingestion.get_active_chroma_dir() == new_dir


def test_session_scopes_use_isolated_paths():
    import src.ingestion as ingestion

    safe_session = "user-42"
    pdf_dir = ingestion.get_session_pdf_dir(safe_session)
    chroma_dir = ingestion.get_session_chroma_dir(safe_session)

    assert pdf_dir.name == "user-42"
    assert "sessions" in str(pdf_dir)
    assert chroma_dir.parent == pdf_dir
    assert str(chroma_dir).replace("\\", "/").endswith("user-42/chroma_db")
