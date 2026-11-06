from langchain_core.prompts import ChatPromptTemplate


GRADE_PROMPT = ChatPromptTemplate.from_template(
    """
You are evaluating whether a document passage is relevant to a question.

Question:
{question}

Document passage:
{document}

Reply with exactly one word:
YES - if the passage contains information relevant to the question.
NO - if the passage is not relevant.
"""
)


WEB_SEARCH_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert at formulating web search queries for Tavily Search.
Given a user question, rewrite it into a concise, focused web search query.
Do not add boolean operators like AND/OR, do not wrap in quotes unless exact match is needed, and keep it under 10 words.

User question: {question}

Web search query:
"""
)


REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """
Rewrite the question to improve semantic search over a collection of PDF
documents. Preserve the user's original intent. Do not answer the question.

Original question:
{question}

Improved search query:
"""
)


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the supplied context.

Rules:
1. Do not invent facts that are not supported by the context.
2. If the context does not contain the answer, say so clearly.
3. Cite supporting passages using their source labels, for example [report.pdf, page 3] or [web: domain.com].
4. Give a clear, direct answer with useful details.
5. You may organize longer answers with headings or bullet points.

Context:
{context}

Question:
{question}

Answer:
"""
)


VALIDATE_PROMPT = ChatPromptTemplate.from_template(
    """
You are checking whether an answer is supported by the supplied documents.

Question:
{question}

Documents:
{context}

Proposed answer:
{answer}

Reply with exactly one word:
SUPPORTED - if the substantive claims in the answer are supported by the documents.
UNSUPPORTED - if the answer contains unsupported substantive claims.

If the documents do not contain enough information to answer the question,
an answer that clearly acknowledges this limitation is SUPPORTED.
"""
)


QUESTIONS_PROMPT = ChatPromptTemplate.from_template(
    """
Create {count} useful questions that can be answered from the supplied
document excerpts.

Rules:
- Use only information present in the excerpts.
- Make the questions distinct.
- Do not provide answers.
- Return a numbered list and no introductory paragraph.

Document excerpts:
{context}
"""
)
