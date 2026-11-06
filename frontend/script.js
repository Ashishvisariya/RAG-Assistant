const API_BASE = (() => {
    const configured = window.RAG_API_BASE_URL?.trim();
    if (configured) return configured.replace(/\/+$/, "");
    if (["localhost", "127.0.0.1"].includes(window.location.hostname)) {
        return "http://127.0.0.1:8000";
    }
    return window.location.origin;
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
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // 100 MB
let busy = false;

async function requestJson(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, options);
    const text = await response.text();
    let payload = null;

    if (text.trim()) {
        try {
            payload = JSON.parse(text);
        } catch {
            throw new Error(`The server returned an invalid response (HTTP ${response.status}).`);
        }
    }

    if (!response.ok) {
        throw new Error(payload?.detail || `Request failed (HTTP ${response.status}).`);
    }
    if (payload === null) {
        throw new Error("The server returned an empty response.");
    }
    return payload;
}

function getSessionId() {
    const existing = localStorage.getItem(SESSION_STORAGE_KEY);
    if (existing) {
        return existing;
    }
    const sessionId = `user-${crypto.randomUUID()}`;
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    return sessionId;
}

async function loadDocuments() {
    try {
        const query = new URLSearchParams({ session_id: getSessionId() });
        const payload = await requestJson(`/documents?${query}`);
        documentList.replaceChildren();
        payload.documents.forEach((filename) => {
            const document = document.createElement("div");
            document.className = "document-pill";
            const name = document.createElement("span");
            name.textContent = filename;
            const status = document.createElement("span");
            status.textContent = "Indexed";
            document.append(name, status);
            documentList.appendChild(document);
        });
    } catch {
        documentList.replaceChildren();
    }
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
        const payload = await requestJson("/health");
        let statusText = "API connected";
        if (payload.tavily_configured) {
            statusText += " (CRAG + Tavily ready)";
        }
        apiStatus.textContent = statusText;
        statusDot.classList.remove("offline");
    } catch {
        apiStatus.textContent = "API offline (start backend on :8000)";
        statusDot.classList.add("offline");
    }
}

function getBadgeConfig(answerType, decision) {
    if (decision === "correct" || answerType === "document_grounded") {
        return { label: "🎯 Document Grounded", css: "document" };
    }
    if (decision === "ambiguous" || answerType === "hybrid_crag") {
        return { label: "⚖️ Hybrid (Doc + Tavily Web)", css: "hybrid" };
    }
    if (answerType === "web_grounded") {
        return { label: "🌐 Tavily Web Search", css: "web" };
    }
    return { label: "💡 General Knowledge", css: "general" };
}

function addMessage(role, text, options = {}) {
    const message = document.createElement("div");
    message.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "Y" : "CR";

    const content = document.createElement("div");
    content.className = "message-content";

    const label = document.createElement("div");
    label.className = "message-role";
    label.textContent = role === "user" ? "You" : "CRAG Assistant";

    const textNode = document.createElement("div");
    textNode.className = `message-text${options.isError ? " error" : ""}`;
    textNode.textContent = text;

    content.appendChild(label);
    content.appendChild(textNode);

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

    const answerNode = addMessage("assistant", "Evaluating with CRAG & retrieving…");

    try {
        const payload = await requestJson("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: trimmed, session_id: getSessionId() }),
        });

        const answerText = payload.answer || "No answer was returned.";
        const answerType = payload.answer_type || "general";
        const decision = payload.crag_decision || "none";
        const sources = Array.isArray(payload.sources) ? payload.sources : [];

        answerNode.innerHTML = formatMarkdown(answerText);

        const meta = document.createElement("div");
        meta.className = "message-meta";

        // CRAG badge
        const badgeConfig = getBadgeConfig(answerType, decision);
        const badge = document.createElement("span");
        badge.className = `answer-badge ${badgeConfig.css}`;
        badge.textContent = badgeConfig.label;
        meta.appendChild(badge);

        if (sources.length) {
            const sourceList = document.createElement("div");
            sourceList.className = "source-list";
            sources.forEach((source) => {
                const isWeb = String(source.source || "").startsWith("http");
                if (isWeb) {
                    const link = document.createElement("a");
                    link.className = "source-pill source-web";
                    link.href = source.source;
                    link.target = "_blank";
                    link.rel = "noopener noreferrer";
                    link.textContent = `🌐 ${source.title || source.source}`;
                    sourceList.appendChild(link);
                } else {
                    const chip = document.createElement("span");
                    chip.className = "source-pill";
                    chip.textContent = `📄 ${source.source || "Document"}, p. ${source.page || "n/a"}`;
                    sourceList.appendChild(chip);
                }
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
        const payload = await requestJson("/questions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ count: 5, session_id: getSessionId() }),
        });

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
        uploadStatus.textContent = "Please select a valid PDF file.";
        pdfInput.value = "";
        return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
        uploadStatus.textContent = `File exceeds the 100 MB limit (${(file.size / (1024 * 1024)).toFixed(1)} MB).`;
        pdfInput.value = "";
        return;
    }

    const fileSizeMb = (file.size / (1024 * 1024)).toFixed(1);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", getSessionId());
    uploadStatus.textContent = `Indexing ${file.name} (${fileSizeMb} MB)…`;
    pdfInput.disabled = true;

    requestJson("/upload", { method: "POST", body: formData })
        .then((payload) => {
            uploadStatus.textContent = `Indexed successfully: ${payload.filename}`;
            const document = document.createElement("div");
            document.className = "document-pill";
            const name = document.createElement("span");
            name.textContent = payload.filename;
            const status = document.createElement("span");
            status.textContent = "Indexed";
            document.append(name, status);
            documentList.appendChild(document);
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
loadDocuments();
