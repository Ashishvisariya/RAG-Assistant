const API_BASE = (() => {
    const origin = window.location.origin;
    if (origin && origin !== "null" && !origin.startsWith("file://")) {
        return origin;
    }
    return "http://127.0.0.1:8001";
})();
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const suggestions = document.getElementById("suggestions");
const newChatBtn = document.getElementById("new-chat");
const suggestBtn = document.getElementById("suggest-btn");
const apiStatus = document.getElementById("api-status");
const statusDot = document.getElementById("status-dot");
const pdfInput = document.getElementById("pdf-file");
const uploadStatus = document.getElementById("upload-status");
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const documentList = document.getElementById("document-list");

const SESSION_STORAGE_KEY = "rag_session_id";
let busy = false;

function getSessionId() {
    const existing = localStorage.getItem(SESSION_STORAGE_KEY);
    if (existing) {
        return existing;
    }
    const sessionId = `user-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    return sessionId;
}

function escapeHtml(value = "") {
    return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatMarkdown(value = "") {
    let html = escapeHtml(value);
    html = html.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${escapeHtml(code.trim())}</code></pre>`);
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    html = html.replace(/\[(.+?)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    html = html.replace(/\n\n/g, "<br><br>");
    html = html.replace(/\n/g, "<br>");
    return html;
}

function setBusy(nextValue) {
    busy = nextValue;
    sendBtn.disabled = nextValue;
    suggestBtn.disabled = nextValue;
    sendBtn.textContent = nextValue ? "…" : "➤";
}

async function checkApi() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) throw new Error("API unavailable");
        const payload = await response.json();
        apiStatus.textContent = payload.status === "healthy" ? "API connected" : "API ready";
        statusDot.classList.remove("offline");
    } catch {
        apiStatus.textContent = "API offline";
        statusDot.classList.add("offline");
    }
}

function addMessage(role, text, options = {}) {
    const message = document.createElement("div");
    message.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "Y" : "AI";

    const content = document.createElement("div");
    content.className = "message-content";

    const label = document.createElement("div");
    label.className = "message-role";
    label.textContent = role === "user" ? "You" : "Assistant";

    const textNode = document.createElement("div");
    textNode.className = `message-text${options.isError ? " error" : ""}`;
    textNode.textContent = text;

    content.appendChild(label);
    content.appendChild(textNode);

    if (options.answerType || options.sources?.length) {
        const meta = document.createElement("div");
        meta.className = "message-meta";

        if (options.answerType) {
            const badge = document.createElement("span");
            badge.className = `answer-badge ${options.answerType === "document_grounded" ? "document" : "general"}`;
            badge.textContent = options.answerType === "document_grounded" ? "Document-grounded" : "General answer";
            meta.appendChild(badge);
        }

        if (options.sources?.length) {
            const sourceList = document.createElement("div");
            sourceList.className = "source-list";
            options.sources.forEach((source) => {
                const chip = document.createElement("span");
                chip.className = "source-pill";
                chip.textContent = `${source.source || "Document"}, p. ${source.page || "n/a"}`;
                sourceList.appendChild(chip);
            });
            meta.appendChild(sourceList);
        }

        content.appendChild(meta);
    }

    message.appendChild(avatar);
    message.appendChild(content);
    messages.appendChild(message);
    message.scrollIntoView({ behavior: "smooth", block: "end" });
    return textNode;
}

function resetWelcomeState() {
    welcome.hidden = true;
}

async function askQuestion(question) {
    const trimmed = (question || "").trim();
    if (busy || !trimmed) return;

    addMessage("user", trimmed);
    questionInput.value = "";
    resizeTextarea();
    setBusy(true);
    resetWelcomeState();

    const answerNode = addMessage("assistant", "Thinking…");

    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: trimmed, session_id: getSessionId() }),
        });

        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Could not get an answer.");

        const answerText = payload.answer || "No answer was returned.";
        const answerType = payload.answer_type || "general";
        const sources = Array.isArray(payload.sources) ? payload.sources : [];

        answerNode.innerHTML = formatMarkdown(answerText);
        const meta = document.createElement("div");
        meta.className = "message-meta";
        const badge = document.createElement("span");
        badge.className = `answer-badge ${answerType === "document_grounded" ? "document" : "general"}`;
        badge.textContent = answerType === "document_grounded" ? "Document-grounded" : "General answer";
        meta.appendChild(badge);

        if (sources.length) {
            const sourceList = document.createElement("div");
            sourceList.className = "source-list";
            sources.forEach((source) => {
                const chip = document.createElement("span");
                chip.className = "source-pill";
                chip.textContent = `${source.source || "Document"}, p. ${source.page || "n/a"}`;
                sourceList.appendChild(chip);
            });
            meta.appendChild(sourceList);
        }

        const copy = document.createElement("button");
        copy.type = "button";
        copy.className = "copy-btn";
        copy.textContent = "Copy";
        copy.addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(answerText);
                copy.textContent = "Copied";
                setTimeout(() => { copy.textContent = "Copy"; }, 1200);
            } catch {
                copy.textContent = "Copy failed";
            }
        });
        meta.appendChild(copy);

        answerNode.parentElement.appendChild(meta);
    } catch (error) {
        answerNode.classList.add("error");
        answerNode.textContent = `Error: ${error.message}`;
    } finally {
        setBusy(false);
        questionInput.focus();
    }
}

async function generateSuggestedQuestions() {
    if (busy) return;
    setBusy(true);
    suggestBtn.textContent = "Generating…";

    try {
        const response = await fetch(`${API_BASE}/questions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ count: 5, session_id: getSessionId() }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Could not generate questions.");

        const questions = Array.isArray(payload.questions)
            ? payload.questions
            : String(payload.questions || "").split(/\r?\n/).map((line) => line.replace(/^\d+[.)]\s*/, "").trim()).filter(Boolean);

        suggestions.innerHTML = "";
        if (!questions.length) {
            const empty = document.createElement("button");
            empty.type = "button";
            empty.className = "suggestion";
            empty.textContent = "No indexed documents found yet.";
            empty.disabled = true;
            suggestions.appendChild(empty);
            return;
        }

        questions.slice(0, 5).forEach((question) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "suggestion";
            button.textContent = question;
            button.addEventListener("click", () => askQuestion(question));
            suggestions.appendChild(button);
        });
    } catch (error) {
        addMessage("assistant", `Could not generate questions: ${error.message}`, { isError: true });
    } finally {
        setBusy(false);
        suggestBtn.textContent = "Generate questions";
    }
}

function resizeTextarea() {
    questionInput.style.height = "auto";
    questionInput.style.height = `${Math.min(questionInput.scrollHeight, 180)}px`;
}

function handleUpload(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
        uploadStatus.textContent = "Please select a PDF file.";
        pdfInput.value = "";
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        uploadStatus.textContent = "Maximum file size is 10 MB.";
        pdfInput.value = "";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", getSessionId());
    uploadStatus.textContent = "Uploading and indexing…";
    pdfInput.disabled = true;

    fetch(`${API_BASE}/upload`, { method: "POST", body: formData })
        .then(async (response) => {
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(payload.detail || "Upload failed.");
            uploadStatus.textContent = `Indexed successfully: ${payload.filename}`;
            documentList.innerHTML = `<div class="document-pill"><span>${payload.filename}</span><span>✓</span></div>`;
        })
        .catch((error) => {
            uploadStatus.textContent = `Error: ${error.message}`;
        })
        .finally(() => {
            pdfInput.disabled = false;
            pdfInput.value = "";
        });
}

chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    askQuestion(questionInput.value);
});

questionInput.addEventListener("input", resizeTextarea);
questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});

newChatBtn.addEventListener("click", () => {
    messages.innerHTML = "";
    welcome.hidden = false;
    questionInput.value = "";
    resizeTextarea();
    questionInput.focus();
});

suggestBtn.addEventListener("click", generateSuggestedQuestions);

pdfInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    handleUpload(file);
});

sidebarToggle.addEventListener("click", () => {
    sidebar.classList.toggle("open");
    sidebarBackdrop.hidden = !sidebar.classList.contains("open");
});

sidebarBackdrop.addEventListener("click", () => {
    sidebar.classList.remove("open");
    sidebarBackdrop.hidden = true;
});

resizeTextarea();
checkApi();