# Advanced RAG System

A document Q&A application built with FastAPI, LangChain, LangGraph, ChromaDB, Groq, and a lightweight frontend. It lets users upload PDF files, index them into a vector database, and ask grounded questions about the document content.

## Features

- Upload PDF documents
- Extract and chunk text from PDFs
- Build a persistent vector store with ChromaDB
- Retrieve relevant document chunks using semantic search
- Generate grounded answers with Groq + LangGraph
- Provide general fallback answers when no document match is found
- Simple web UI for chat and PDF management
- Session-scoped document indexing

## Tech Stack

- Python 3.12+
- FastAPI
- Uvicorn
- LangChain / LangGraph
- LangChain Groq
- ChromaDB
- Hugging Face Embeddings
- PyPDF
- HTML / CSS / Vanilla JavaScript
- Docker + Docker Compose + Nginx

## Project Structure

```text
Advanced RAG System/
├── app.py
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── nginx.conf
├── render.yaml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .dockerignore
├── README.md
├── active_chroma_dir.txt
├── data/
│   └── pdfs/
├── frontend/
│   ├── config.js
│   ├── index.html
│   ├── script.js
│   └── style.css
├── sessions/
├── src/
│   ├── __init__.py
│   ├── graph.py
│   ├── ingestion.py
│   ├── prompts.py
│   └── retrieval.py
├── tests/
│   └── test_rag.py
└── .gitignore
```

## Environment Setup

1. Create a `.env` file in the project root.
2. Copy the example values from `.env.example`.
3. Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
ALLOWED_ORIGINS=http://localhost:80,http://127.0.0.1:80,http://localhost:5501,http://127.0.0.1:5501
```

## Local Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Serve the frontend from the `frontend` folder:

```bash
cd frontend
python -m http.server 5501
```

Then open:

- Frontend: `http://127.0.0.1:5501`
- API docs: `http://127.0.0.1:8001/docs`

## Docker Deployment

Build and run the full stack:

```bash
docker compose up --build -d
```

Then access:

- Frontend: `http://localhost`
- Backend: `http://localhost:8001`
- API docs: `http://localhost:8001/docs`

## Render Deployment

Deploy the repository as a Render Blueprint using `render.yaml`. It creates:

- `advanced-rag-api`: Docker Web Service using `Dockerfile.backend`, with `/health` as its health check and a 10 GB persistent disk mounted at `/var/data`.
- `advanced-rag-ui`: Static Site publishing `frontend/`. Its build command writes the backend service host into `frontend/config.js`.

The Blueprint sets `RAG_DATA_DIR=/var/data`, so session PDFs, Chroma indexes, index markers, and the Hugging Face model cache live on the persistent disk. The backend uses the Render-injected `$PORT`. The Web Service's `starter` plan is required for a persistent disk; ephemeral storage on a free instance is not a durable substitute.

Before deploying, provide `GROQ_API_KEY` as a secret in Render. `GROQ_MODEL` defaults to `openai/gpt-oss-20b`. The Blueprint allows the default frontend origin `https://advanced-rag-ui.onrender.com`; update `ALLOWED_ORIGINS` if Render assigns a different hostname or you add a custom domain. The frontend backend URL is linked from the Web Service host automatically.

If deploying the services separately instead of using the Blueprint, configure the backend with:

- Build: Dockerfile at `./Dockerfile.backend`, context `.`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Persistent disk: mount at `/var/data`; set `RAG_DATA_DIR=/var/data`
- Environment: `GROQ_API_KEY` (secret), `GROQ_MODEL`, `RAG_DATA_DIR`, `HF_HOME=/var/data/huggingface`, and `ALLOWED_ORIGINS=https://<your-static-site-host>`

Configure the Static Site with root directory `.`, publish directory `frontend`, and a build step that writes `window.RAG_API_BASE_URL` in `frontend/config.js` to `https://<your-backend-host>` before publishing. Never put the Groq key in frontend settings or files.

The API currently has no user authentication or authorization. Session IDs separate browser data but are not an access-control boundary; do not treat the public deployment as suitable for confidential documents without adding authentication and per-user authorization.

Install test dependencies and run tests with:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Notes

- The app stores uploaded PDFs and vector indexes in the `sessions/` directory per session.
- The current implementation uses the browser session to associate a user with their uploaded PDFs.
- If you move across different origins or browser tabs, the stored session may differ and you may need to re-upload or refresh the page.

## Troubleshooting

### Missing Groq API key

Make sure your `.env` file contains a valid `GROQ_API_KEY`.

### No answers after upload

Check that:

- the PDF was uploaded successfully,
- the browser and backend are using the same session,
- the uploaded file contains readable text,
- the backend service is running.

### CORS errors

Ensure the frontend origin is included in `ALLOWED_ORIGINS` in your `.env` file.

## License

This project is for educational and personal use.
