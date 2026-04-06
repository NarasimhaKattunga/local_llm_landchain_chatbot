import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from local_rag_chatbot_engine import (
    build_chain,
    format_docs,
    format_sources,
    get_history,
    summarize_if_needed,
)

# ─── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NK · RAG Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:    #0a0b0f;
    --bg-secondary:  #111318;
    --bg-card:       #16181f;
    --bg-hover:      #1e2028;
    --accent:        #00e5a0;
    --accent-dim:    #00e5a020;
    --accent-mid:    #00e5a060;
    --text-primary:  #eef0f7;
    --text-secondary:#8b90a4;
    --text-muted:    #4a4f63;
    --border:        #1f2230;
    --border-accent: #00e5a030;
    --user-bubble:   #0f2a1e;
    --ai-bubble:     #13151c;
    --danger:        #ff4f6b;
    --warn:          #ffb547;
    --radius:        14px;
    --radius-sm:     8px;
}

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Syne', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Main layout ── */
[data-testid="stAppViewContainer"] > .main {
    background: var(--bg-primary) !important;
    padding: 0 !important;
}
.block-container {
    padding: 2rem 2.5rem 6rem !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* ── Header ── */
.rag-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
.rag-header-icon {
    width: 52px; height: 52px;
    background: var(--accent-dim);
    border: 1.5px solid var(--accent-mid);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
}
.rag-header-text h1 {
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    color: var(--text-primary) !important;
    margin: 0 !important; padding: 0 !important;
    line-height: 1.1 !important;
}
.rag-header-text p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--accent) !important;
    margin: 4px 0 0 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 var(--accent-mid); }
    50% { opacity: 0.7; box-shadow: 0 0 0 5px transparent; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.2rem !important; }

.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.sidebar-brand-icon {
    font-size: 22px;
    width: 40px; height: 40px;
    background: var(--accent-dim);
    border: 1px solid var(--border-accent);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
}
.sidebar-brand-label {
    font-size: 0.85rem; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.01em;
}
.sidebar-brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: var(--text-muted);
    letter-spacing: 0.06em; text-transform: uppercase;
}

.sidebar-section-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
    margin-top: 8px !important;
}

/* Sidebar inputs */
[data-testid="stSidebar"] input[type="text"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 10px 12px !important;
}
[data-testid="stSidebar"] input[type="text"]:focus {
    border-color: var(--accent-mid) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
    outline: none !important;
}

/* Sidebar button */
[data-testid="stSidebar"] .stButton button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 9px 14px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    border-color: var(--danger) !important;
    color: var(--danger) !important;
    background: #ff4f6b0f !important;
}

/* Sidebar stats */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    margin-bottom: 8px;
}
.stat-card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 4px;
}
.stat-card-value {
    font-size: 1.1rem; font-weight: 700;
    color: var(--accent);
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 0 20px !important;
    gap: 0 !important;
}

/* Avatar override */
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    display: none !important;
}

/* User message */
[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
div[data-testid="stChatMessage"]:has([aria-label="user avatar"]) .stMarkdown {
    background: var(--user-bubble) !important;
    border: 1px solid #00e5a020 !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 14px 18px !important;
    margin-left: auto !important;
    max-width: 82% !important;
}

/* Custom message bubbles via markdown injection */
.msg-user {
    display: flex; justify-content: flex-end; margin-bottom: 20px;
}
.msg-user-inner {
    background: var(--user-bubble);
    border: 1px solid var(--border-accent);
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    max-width: 82%;
    font-size: 0.95rem; line-height: 1.6;
    color: var(--text-primary);
}
.msg-user-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: var(--accent);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.msg-ai {
    display: flex; justify-content: flex-start; margin-bottom: 20px;
}
.msg-ai-inner {
    background: var(--ai-bubble);
    border: 1px solid var(--border);
    border-radius: 4px 18px 18px 18px;
    padding: 16px 18px;
    max-width: 90%;
    font-size: 0.95rem; line-height: 1.7;
    color: var(--text-primary);
    position: relative;
}
.msg-ai-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; color: var(--accent);
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 10px;
    background: var(--accent-dim);
    border: 1px solid var(--border-accent);
    border-radius: 100px; padding: 3px 10px;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 4px 8px !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-mid) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.92rem !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: italic !important;
}
[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border-radius: 10px !important;
    color: #000 !important;
    transition: opacity 0.2s ease !important;
}
[data-testid="stChatInput"] button:hover { opacity: 0.85 !important; }

/* ── Expander (sources) ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    margin-top: 10px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.04em !important;
    padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover { color: var(--accent) !important; }
[data-testid="stExpander"] pre, [data-testid="stExpander"] code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    line-height: 1.7 !important;
}

/* ── Alert / error boxes ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    border-width: 1px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-muted);
}
.empty-state-icon { font-size: 3rem; margin-bottom: 16px; opacity: 0.5; }
.empty-state-title {
    font-size: 1rem; font-weight: 700;
    color: var(--text-secondary);
    margin-bottom: 8px;
}
.empty-state-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: var(--text-muted);
    letter-spacing: 0.04em;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 24px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = "nk_default_session"
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

# ─── Load chain (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_chain():
    def on_error(msg):
        st.error(msg)
        st.stop()
    return build_chain(on_error=on_error)

_, retriever, _, summary_chain, chain_with_memory = load_chain()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🧠</div>
        <div>
            <div class="sidebar-brand-label">NK RAG Engine</div>
            <div class="sidebar-brand-sub">Local · Offline · Private</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">⚙️ &nbsp;Session Control</div>', unsafe_allow_html=True)
    session_id = st.text_input(
        label="session_id",
        value=st.session_state.session_id,
        label_visibility="collapsed",
        placeholder="Session ID…",
    )
    if session_id != st.session_state.session_id:
        st.session_state.session_id = session_id
        st.session_state.display_messages = []
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("🗑️  Clear conversation history"):
        summarize_if_needed(st.session_state.session_id, summary_chain)
        st.session_state.display_messages = []
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">📊 &nbsp;Session Stats</div>', unsafe_allow_html=True)

    total_msgs   = len(st.session_state.display_messages)
    user_msgs    = sum(1 for m in st.session_state.display_messages if m["role"] == "user")
    sources_used = sum(1 for m in st.session_state.display_messages if m.get("sources"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-label">Messages</div>
            <div class="stat-card-value">{total_msgs}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-label">Queries</div>
            <div class="stat-card-value">{user_msgs}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-card-label">Sources Retrieved</div>
        <div class="stat-card-value">{sources_used}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">🔧 &nbsp;Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4a4f63;line-height:2;">
        🤖 &nbsp;Ollama · Llama3<br>
        🔢 &nbsp;nomic-embed-text<br>
        🗂️ &nbsp;FAISS · Local Index<br>
        🦜 &nbsp;LangChain · RAG<br>
        ⚡ &nbsp;Streamlit · UI
    </div>
    """, unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <div class="rag-header-icon">🧠</div>
    <div class="rag-header-text">
        <h1>NK Local RAG : Document Intelligence : Chatbot</h1>
        <p><span class="status-dot"></span>Connected · Ollama · FAISS · Local</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Render chat history ──────────────────────────────────────────────────────
if not st.session_state.display_messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">💬</div>
        <div class="empty-state-title">No conversation yet</div>
        <div class="empty-state-sub">Ask anything about your indexed documents below</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.display_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"""
                <div class="msg-user">
                    <div class="msg-user-inner">
                        <div class="msg-user-meta">👤 &nbsp;You</div>
                        {msg["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"""
                <div class="msg-ai">
                    <div class="msg-ai-inner">
                        <div class="msg-ai-badge">🧠 &nbsp;RAG · Assistant</div>
                        {msg["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander("📎 &nbsp;View Sources"):
                        st.code(msg["sources"], language=None)

# ─── Chat input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("✦  Ask a question about your documents…"):
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-user-inner">
                <div class="msg-user-meta">👤 &nbsp;You</div>
                {user_input}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Summarize history
    try:
        summarize_if_needed(st.session_state.session_id, summary_chain)
    except Exception as e:
        st.warning(f"⚠️ Could not summarize conversation history.\n\n{e}")

    # Retrieve docs
    with st.spinner("🔍 &nbsp;Searching knowledge base…"):
        try:
            st.session_state.last_docs = retriever.invoke(user_input)
        except Exception as e:
            st.error(f"❌ Failed to retrieve documents. Check Ollama is running.\n\n{e}")
            st.stop()

    context = format_docs(st.session_state.last_docs)

    # Generate response
    with st.chat_message("assistant"):
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
             color:#00e5a0;letter-spacing:0.1em;text-transform:uppercase;
             margin-bottom:10px;opacity:0.8;">
            🧠 &nbsp;RAG · Assistant
        </div>
        """, unsafe_allow_html=True)
        try:
            full_answer = st.write_stream(
                chain_with_memory.stream(
                    {"input": user_input, "context": context},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )
            )
        except Exception as e:
            st.error(f"❌ Failed to generate a response.\n\n{e}")
            st.stop()

        sources_text = format_sources(st.session_state.last_docs) if st.session_state.last_docs else ""
        if sources_text:
            with st.expander("📎 &nbsp;View Sources"):
                st.code(sources_text, language=None)

    st.session_state.display_messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources_text,
    })