import os
import re
import uuid
from pathlib import Path
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from src.prompts import ANSWER_PROMPT, GRADE_PROMPT, QUESTIONS_PROMPT, REWRITE_PROMPT, VALIDATE_PROMPT, WEB_SEARCH_PROMPT
from src.retrieval import retrieve_documents

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured in the environment.")
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.1,
        timeout=60,
        max_retries=2,
    )


def tavily_search(query: str, max_results: int = 4) -> List[Document]:
    """Execute search using Tavily API if configured."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("TAVILY_API_KEY is not configured.")
        return []

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results, search_depth="basic")
        results = response.get("results", [])

        web_docs = []
        for res in results:
            content = res.get("content", "")
            title = res.get("title", "")
            url = res.get("url", "web")
            if content:
                formatted_content = f"Title: {title}\nURL: {url}\nContent: {content}"
                web_docs.append(
                    Document(
                        page_content=formatted_content,
                        metadata={"source": url, "page": "web", "title": title},
                    )
                )
        return web_docs
    except Exception as exc:
        print(f"Tavily search error: {exc}")
        return []


class CRAGState(TypedDict):
    question: str
    documents: list
    relevant_documents: list
    web_documents: list
    combined_documents: list
    answer: str
    answer_type: str
    sources: list
    request_id: str
    retries: int
    validation_attempts: int
    session_id: str | None
    crag_decision: str  # "correct", "ambiguous", "incorrect"


def retrieve(state: CRAGState):
    """Retrieve initial documents from vector store."""
    docs = retrieve_documents(state["question"], k=5, session_id=state.get("session_id"))
    return {"documents": docs}


def grade_documents(state: CRAGState):
    """
    Evaluate retrieved documents.
    In Corrective RAG (CRAG):
    - High confidence / sufficient relevant docs -> Correct (use internal knowledge)
    - Low confidence / few relevant docs -> Ambiguous (combine internal + web search)
    - Zero relevant docs -> Incorrect (fallback to web search)
    """
    docs = state.get("documents", [])
    if not docs:
        return {"relevant_documents": [], "crag_decision": "incorrect"}

    grader = GRADE_PROMPT | get_llm() | StrOutputParser()
    relevant_docs = []
    for doc in docs:
        try:
            result = grader.invoke({"question": state["question"], "document": doc.page_content})
            if result.strip().upper().startswith("YES"):
                relevant_docs.append(doc)
        except Exception as exc:
            print(f"Grading error: {exc}")

    num_relevant = len(relevant_docs)
    total_docs = len(docs)

    if num_relevant >= 2 or (total_docs == 1 and num_relevant == 1):
        decision = "correct"
    elif num_relevant > 0:
        decision = "ambiguous"
    else:
        decision = "incorrect"

    return {"relevant_documents": relevant_docs, "crag_decision": decision}


def route_crag_decision(state: CRAGState) -> Literal["generate_document", "transform_query_web", "combine_web"]:
    decision = state.get("crag_decision", "incorrect")
    if decision == "correct":
        return "generate_document"
    elif decision == "ambiguous":
        return "combine_web"
    else:
        return "transform_query_web"


def transform_query_web(state: CRAGState):
    """Transform user question into an optimal Tavily web search query and retrieve web documents."""
    question = state["question"]
    try:
        query_chain = WEB_SEARCH_PROMPT | get_llm() | StrOutputParser()
        search_query = query_chain.invoke({"question": question}).strip()
    except Exception:
        search_query = question

    search_query = search_query.strip('"').strip("'")
    if not search_query:
        search_query = question

    web_docs = tavily_search(search_query, max_results=4)
    relevant = list(state.get("relevant_documents", []))
    combined = relevant + web_docs

    return {
        "web_documents": web_docs,
        "combined_documents": combined,
    }


def generate_document_answer(state: CRAGState):
    """Generate answer strictly from relevant local documents."""
    docs = state.get("relevant_documents", [])
    if not docs:
        return {
            "answer": "I could not find relevant information in the uploaded documents to answer this question.",
            "answer_type": "general",
            "sources": [],
        }

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, page {doc.metadata.get('page', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    )

    answer = (ANSWER_PROMPT | get_llm() | StrOutputParser()).invoke({
        "context": context,
        "question": state["question"],
    })

    sources = [
        {"source": doc.metadata.get("source", "Unknown"), "page": doc.metadata.get("page", "Unknown")}
        for doc in docs
    ]
    return {"answer": answer.strip(), "answer_type": "document_grounded", "sources": sources}


def generate_web_answer(state: CRAGState):
    """Generate answer from Tavily web search documents (when document retrieval was incorrect)."""
    docs = state.get("web_documents", [])
    if not docs:
        # Fallback to general LLM response if web search produced no results or no API key
        return generate_general_answer(state)

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Web')}\n{doc.page_content}"
        for doc in docs
    )

    answer = (ANSWER_PROMPT | get_llm() | StrOutputParser()).invoke({
        "context": context,
        "question": state["question"],
    })

    sources = [
        {"source": doc.metadata.get("source", "Web"), "page": "web", "title": doc.metadata.get("title", "")}
        for doc in docs
    ]
    return {"answer": answer.strip(), "answer_type": "web_grounded", "sources": sources}


def generate_combined_answer(state: CRAGState):
    """Generate answer combining both local documents and web search results (ambiguous case)."""
    docs = state.get("combined_documents", [])
    if not docs:
        return generate_general_answer(state)

    context_parts = []
    sources = []
    for doc in docs:
        source_label = doc.metadata.get("source", "Unknown")
        page_label = doc.metadata.get("page", "Unknown")
        context_parts.append(f"Source: {source_label}, page/type: {page_label}\n{doc.page_content}")
        sources.append({
            "source": source_label,
            "page": page_label,
            "title": doc.metadata.get("title", ""),
        })

    context = "\n\n".join(context_parts)
    answer = (ANSWER_PROMPT | get_llm() | StrOutputParser()).invoke({
        "context": context,
        "question": state["question"],
    })

    return {"answer": answer.strip(), "answer_type": "hybrid_crag", "sources": sources}


def generate_general_answer(state: CRAGState):
    answer = (
        ChatPromptTemplate.from_template(
            """
            Answer the user's question using general knowledge, but clearly state that the answer is not verified against uploaded PDFs or current web search.
            If you are unsure, say so clearly.

            Question: {question}
            Answer:
            """
        )
        | get_llm()
        | StrOutputParser()
    ).invoke({"question": state["question"]})
    return {"answer": answer.strip(), "answer_type": "general", "sources": []}


def validate_answer(state: CRAGState):
    """Verify that the generated answer is grounded in the provided documents."""
    if state.get("answer_type") not in ("document_grounded", "hybrid_crag", "web_grounded"):
        return {"validation_attempts": state.get("validation_attempts", 0) + 1}

    docs = (
        state.get("combined_documents")
        or state.get("relevant_documents")
        or state.get("web_documents")
        or []
    )
    if not docs:
        return {"validation_attempts": state.get("validation_attempts", 0) + 1}

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, page: {doc.metadata.get('page', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    )

    validation = (VALIDATE_PROMPT | get_llm() | StrOutputParser()).invoke({
        "context": context,
        "question": state["question"],
        "answer": state["answer"],
    })

    if validation.strip().upper() != "SUPPORTED":
        return {
            "answer": "I could not verify that the answer is fully supported by the retrieved sources.",
            "answer_type": "general",
            "sources": [],
            "validation_attempts": state.get("validation_attempts", 0) + 1,
        }

    return {"validation_attempts": state.get("validation_attempts", 0) + 1}


def rewrite_question(state: CRAGState):
    rewritten = (REWRITE_PROMPT | get_llm() | StrOutputParser()).invoke({"question": state["question"]})
    return {"question": rewritten.strip(), "retries": state.get("retries", 0) + 1}


def generate_questions(num_questions: int = 5, session_id: str | None = None):
    docs = retrieve_documents(
        "key concepts main findings definitions important facts and details",
        k=min(5, max(1, num_questions)),
        session_id=session_id,
    )
    if not docs:
        return []

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, page {doc.metadata.get('page', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    )

    response = (QUESTIONS_PROMPT | get_llm() | StrOutputParser()).invoke({
        "count": max(1, min(10, int(num_questions))),
        "context": context,
    })

    items = []
    for line in response.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        candidate = re.sub(r"^\d+[\.)]\s*", "", candidate)
        candidate = candidate.strip()
        if candidate and candidate not in items:
            items.append(candidate)
    return items[: max(1, min(10, int(num_questions)))]


# Build CRAG StateGraph
builder = StateGraph(CRAGState)

builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade_documents)
builder.add_node("generate_document", generate_document_answer)
builder.add_node("transform_query_web", transform_query_web)
builder.add_node("generate_web", generate_web_answer)
builder.add_node("combine_web", transform_query_web)
builder.add_node("generate_combined", generate_combined_answer)
builder.add_node("validate", validate_answer)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")

# CRAG conditional branching based on grading confidence
builder.add_conditional_edges(
    "grade",
    route_crag_decision,
    {
        "generate_document": "generate_document",
        "combine_web": "combine_web",
        "transform_query_web": "transform_query_web",
    },
)

builder.add_edge("generate_document", "validate")
builder.add_edge("transform_query_web", "generate_web")
builder.add_edge("generate_web", "validate")
builder.add_edge("combine_web", "generate_combined")
builder.add_edge("generate_combined", "validate")
builder.add_edge("validate", END)

rag_graph = builder.compile()


def ask_question(question: str, session_id: str | None = None):
    normalized = (question or "").strip()
    if not normalized:
        return {
            "answer": "Please provide a question.",
            "answer_type": "general",
            "sources": [],
            "request_id": str(uuid.uuid4()),
            "crag_decision": "none",
        }

    result = rag_graph.invoke({
        "question": normalized,
        "documents": [],
        "relevant_documents": [],
        "web_documents": [],
        "combined_documents": [],
        "answer": "",
        "answer_type": "general",
        "sources": [],
        "request_id": str(uuid.uuid4()),
        "retries": 0,
        "validation_attempts": 0,
        "session_id": session_id,
        "crag_decision": "none",
    })

    return {
        "answer": result.get("answer", ""),
        "answer_type": result.get("answer_type", "general"),
        "sources": result.get("sources", []),
        "request_id": result.get("request_id"),
        "crag_decision": result.get("crag_decision", "none"),
    }
