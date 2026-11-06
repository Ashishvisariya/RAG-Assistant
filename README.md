# Advanced CRAG System (Corrective RAG + Tavily Web Search)

An AI-powered document intelligence assistant featuring:
- **Corrective RAG (CRAG)** via LangGraph: Self-evaluating retrieval that automatically grades document relevance and triggers web search when needed.
- **Tavily Web Search**: Real-time web retrieval to answer queries when local documents lack information or have partial context.
- **100 MB Large Document Indexing**: Powered by fast C++ PyMuPDF streaming and batched vector store ingestion.
- **Modern HTML/CSS/JS Frontend**: Clean, responsive web UI with CRAG badge indicators, document management, and markdown/source rendering.
- **FastAPI Backend**: Robust API with session isolation, CORS, and health checks.

---

## 🚀 Quick Start (HTML / CSS / JS UI)

### 1. Configure Environment Variables
Make sure your `.env` contains your Groq and Tavily API keys:

```env
GROQ_API_KEY="your-groq-api-key"
GROQ_MODEL="openai/gpt-oss-20b"
TAVILY_API_KEY="your-tavily-api-key"
```

### 2. Start the Backend API
Run the FastAPI backend on port `8000`:

```powershell
python -m uvicorn app:app --reload --port 8000
```

### 3. Open the Frontend
Open `frontend/index.html` in your browser or run a simple local web server:

```powershell
cd frontend
python -m http.server 5500
```
Then visit `http://127.0.0.1:5500` in your web browser.

---

## 🌟 Key Features

1. **Corrective RAG (CRAG) Routing**:
   - 🎯 **Document Grounded**: When local indexed PDFs contain high-confidence answers.
   - ⚖️ **Hybrid (Doc + Tavily Web)**: When document matches are ambiguous or partial.
   - 🌐 **Tavily Web Search**: When local PDFs do not contain relevant information, the query is rewritten into an optimal search query and answered with live Tavily web results.
   - 💡 **General Knowledge**: Fallback answering with validation against hallucinations.

2. **100 MB Document Uploads**:
   - Uses **PyMuPDF (`pymupdf`)** for fast, low-memory C++ document streaming (10x-50x faster than standard PDF loaders).
   - Ingests text into ChromaDB in batches of 128 chunks to prevent memory spikes on large PDF documents.
   - Cached singleton HuggingFace embeddings for fast vectorization.

3. **Frontend UI**:
   - Visual badges for CRAG decisions (🎯 Document Grounded, 🌐 Tavily Web, ⚖️ Hybrid).
   - Clickable source pills for both PDF page numbers and web links.
   - Document upload status and indexed document listing.
   - Question suggestion generator.
