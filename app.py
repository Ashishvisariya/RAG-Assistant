import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph import ask_question, generate_questions
from src.ingestion import create_vector_store, get_session_pdf_dir, normalize_session_id

PROJECT_DIR = Path(__file__).resolve().parent
MAX_FILE_SIZE = 10 * 1024 * 1024
INDEX_LOCK = Lock()


def get_allowed_origins() -> list[str]:
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5501,http://localhost:5501,http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:8001,http://localhost:8001",
    )
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="Advanced RAG System",
    description="AI-powered document assistant using Groq + LangGraph + Chroma.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=128)


class SuggestedQuestionsRequest(BaseModel):
    count: int = Field(default=5, ge=1, le=10)
    session_id: str | None = Field(default=None, max_length=128)


@app.get("/")
def root():
    return {
        "message": "Advanced RAG System API is running.",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }


@app.get("/health")
def health():
    from src.retrieval import get_vector_store

    return {
        "status": "healthy",
        "database_ready": get_vector_store() is not None,
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
    }


@app.get("/ready")
def ready():
    try:
        from src.graph import get_llm

        get_llm()
        groq_ready = True
    except Exception:
        groq_ready = False

    from src.retrieval import get_vector_store

    return {
        "status": "ok" if groq_ready else "degraded",
        "groq_ready": groq_ready,
        "database_ready": get_vector_store() is not None,
    }


@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    session_id = None
    if request.session_id:
        session_id = normalize_session_id(request.session_id)

    try:
        payload = ask_question(question, session_id=session_id)
        return {
            "question": question,
            "answer": payload.get("answer", ""),
            "answer_type": payload.get("answer_type", "general"),
            "sources": payload.get("sources", []),
            "request_id": payload.get("request_id"),
            "session_id": session_id,
        }
    except Exception as exc:
        print("Question answering failed.")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while answering your question.",
        ) from exc


@app.post("/questions")
def questions(request: SuggestedQuestionsRequest):
    session_id = None
    if request.session_id:
        session_id = normalize_session_id(request.session_id)

    try:
        result = generate_questions(request.count, session_id=session_id)
        if not result:
            return {"questions": [], "message": "No indexed documents are available.", "session_id": session_id}
        return {"questions": result, "session_id": session_id}
    except Exception as exc:
        print("Question generation failed.")
        raise HTTPException(
            status_code=500,
            detail="Could not generate suggested questions.",
        ) from exc


@app.get("/documents")
def documents(session_id: str = Query(min_length=1, max_length=128)):
    cleaned_session_id = normalize_session_id(session_id)
    pdf_dir = get_session_pdf_dir(cleaned_session_id)
    files = sorted(path.name for path in pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    return {"documents": files, "session_id": cleaned_session_id}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), session_id: str = Form(default="")):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")

    filename = Path(file.filename.replace("\\", "/")).name
    if not filename or filename in {".", ".."} or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")

    if len(session_id) > 128:
        raise HTTPException(status_code=422, detail="Session ID must be 128 characters or fewer.")

    cleaned_session_id = None
    if session_id.strip():
        cleaned_session_id = normalize_session_id(session_id)

    try:
        content = await file.read(MAX_FILE_SIZE + 1)
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="PDF must be 10 MB or smaller.")
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="The uploaded file does not have a valid PDF header.")

        pdf_dir = get_session_pdf_dir(cleaned_session_id)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        destination = pdf_dir / filename

        with INDEX_LOCK:
            if destination.exists():
                raise HTTPException(
                    status_code=409,
                    detail="A PDF with this filename already exists. Rename the file and try again.",
                )

            destination.write_bytes(content)
            try:
                create_vector_store(cleaned_session_id)
            except Exception as exc:
                destination.unlink(missing_ok=True)
                print("PDF indexing failed.")
                raise HTTPException(
                    status_code=500,
                    detail="PDF indexing failed. Check that the PDF contains readable text and inspect the backend logs.",
                ) from exc

        return {
            "message": "PDF uploaded and indexed successfully.",
            "filename": filename,
            "status": "indexed",
            "session_id": cleaned_session_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        print("PDF upload failed.")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during PDF upload.",
        ) from exc
    finally:
        await file.close()

