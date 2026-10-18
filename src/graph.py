import os
import re
import uuid
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from src.prompts import ANSWER_PROMPT, GRADE_PROMPT, QUESTIONS_PROMPT, REWRITE_PROMPT, VALIDATE_PROMPT
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


class RAGState(TypedDict):
    question: str
    documents: list
    relevant_documents: list
    answer: str
    answer_type: str
    sources: list
    request_id: str
    retries: int
    validation_attempts: int
    session_id: str | None


def retrieve(state: RAGState):
    docs = retrieve_documents(state["question"], k=5, session_id=state.get("session_id"))
    return {"documents": docs}


def grade_documents(state: RAGState):
    docs = state.get("documents", [])
    if not docs:
        return {"relevant_documents": []}

    grader = GRADE_PROMPT | get_llm() | StrOutputParser()
    relevant_docs = []
    for doc in docs:
        result = grader.invoke({"question": state["question"], "document": doc.page_content})
        if result.strip().upper().startswith("YES"):
            relevant_docs.append(doc)
    return {"relevant_documents": relevant_docs}


def route_after_grading(state: RAGState):
    return "document" if state.get("relevant_documents") else "general"


def generate_document_answer(state: RAGState):
    docs = state.get("relevant_documents", [])
    if not docs:
        return {"answer": "I could not find relevant information in the indexed documents to answer this question.", "answer_type": "general", "sources": []}

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


def generate_general_answer(state: RAGState):
    answer = (
        ChatPromptTemplate.from_template(
            """
            Answer the user's question using general knowledge, but clearly state that the answer is not verified against uploaded PDFs.
            If you are unsure, say so clearly.

            Question: {question}
            Answer:
            """
        )
        | get_llm()
        | StrOutputParser()
    ).invoke({"question": state["question"]})
    return {"answer": answer.strip(), "answer_type": "general", "sources": []}


def validate_answer(state: RAGState):
    if state.get("answer_type") != "document_grounded":
        return {"validation_attempts": state.get("validation_attempts", 0) + 1}

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, page {doc.metadata.get('page', 'Unknown')}\n{doc.page_content}"
        for doc in state.get("relevant_documents", [])
    )

    validation = (VALIDATE_PROMPT | get_llm() | StrOutputParser()).invoke({
        "context": context,
        "question": state["question"],
        "answer": state["answer"],
    })

    if validation.strip().upper() != "SUPPORTED":
        return {
            "answer": "I could not verify that the answer is fully supported by the retrieved documents.",
            "answer_type": "general",
            "sources": [],
            "validation_attempts": state.get("validation_attempts", 0) + 1,
        }

    return {"validation_attempts": state.get("validation_attempts", 0) + 1}


def rewrite_question(state: RAGState):
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


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade_documents)
builder.add_node("document", generate_document_answer)
builder.add_node("general", generate_general_answer)
builder.add_node("validate", validate_answer)
builder.add_node("rewrite", rewrite_question)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges(
    "grade",
    route_after_grading,
    {
        "document": "document",
        "general": "general",
    },
)
builder.add_edge("document", "validate")
builder.add_edge("general", END)
builder.add_edge("validate", END)

rag_graph = builder.compile()


def ask_question(question: str, session_id: str | None = None):
    normalized = (question or "").strip()
    if not normalized:
        return {"answer": "Please provide a question.", "answer_type": "general", "sources": [], "request_id": str(uuid.uuid4())}

    result = rag_graph.invoke({
        "question": normalized,
        "documents": [],
        "relevant_documents": [],
        "answer": "",
        "answer_type": "general",
        "sources": [],
        "request_id": str(uuid.uuid4()),
        "retries": 0,
        "validation_attempts": 0,
        "session_id": session_id,
    })

    answer = result.get("answer", "").strip() or "No answer generated."
    return {
        "answer": answer,
        "answer_type": result.get("answer_type", "general"),
        "sources": result.get("sources", []),
        "request_id": result.get("request_id", str(uuid.uuid4())),
    }


if __name__ == "__main__":
    request = input("What do you want to know? ").strip()
    if request:
        print(ask_question(request))
    else:
        print("Please enter a question.")
