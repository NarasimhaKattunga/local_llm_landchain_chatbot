import os
import sys
from typing import List

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DB_URL = "sqlite:///llm_local_chat_history.db"
SUMMARY_PREFIX = "SUMMARY:"
MAX_HISTORY = 8
VECTOR_DB_PATH = "local_faiss_db"

# ─── HISTORY MANAGEMENT ───────────────────────────────────────────────────────
def get_history(session_id: str) -> SQLChatMessageHistory:
    """Fetch chat history for a session."""
    return SQLChatMessageHistory(session_id=session_id, connection=DB_URL)

def get_filtered_messages(session_id: str) -> List:
    """Return messages excluding stored summaries."""
    return [
        msg for msg in get_history(session_id).messages
        if not (
            isinstance(msg, SystemMessage)
            and isinstance(msg.content, str)
            and msg.content.startswith(SUMMARY_PREFIX)
        )
    ]

def extract_summary(session_id: str) -> str:
    """Retrieve stored summary if available."""
    for msg in get_history(session_id).messages:
        if (
            isinstance(msg, SystemMessage)
            and isinstance(msg.content, str)
            and msg.content.startswith(SUMMARY_PREFIX)
        ):
            return msg.content.replace(SUMMARY_PREFIX, "").strip()
    return ""

def summarize_if_needed(session_id: str, summary_chain):
    """Summarize long chat histories and replace with compact summary."""
    messages = get_filtered_messages(session_id)
    
    if len(messages) < MAX_HISTORY * 2:
        return

    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in messages
    )

    existing_summary = extract_summary(session_id)
    if existing_summary:
        history_text = f"Previous summary: {existing_summary}\n\n{history_text}"

    new_summary = summary_chain.invoke({"history_text": history_text})

    history = get_history(session_id)
    history.clear()
    history.add_message(SystemMessage(content=f"{SUMMARY_PREFIX} {new_summary}"))

# ─── DOCUMENT HELPERS ─────────────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1))

def format_sources(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page")
        page_info = f", p.{page + 1}" if page is not None else ""
        formatted.append(f"[{i}] {source}{page_info}")
    return "\n".join(formatted)

# ─── INITIALIZATION ───────────────────────────────────────────────────────────
def init_llm(on_error):
    try:
        return ChatOllama(model="llama3.2", temperature=0.7, num_predict=500)
    except Exception as e:
        on_error(f"Ollama initialization failed:\n{e}")
        raise

def init_embeddings(on_error):
    try:
        return OllamaEmbeddings(model="nomic-embed-text")
    except Exception as e:
        on_error(f"Embedding model initialization failed:\n{e}")
        raise

def load_vectorstore(embeddings, on_error):
    if not os.path.isdir(VECTOR_DB_PATH):
        on_error("FAISS index missing. Run ingest.py first.")
        raise RuntimeError("FAISS index missing. Run ingest.py first.")

    try:
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        on_error(f"Failed to load FAISS index:\n{e}")
        raise

# ─── PROMPTS ──────────────────────────────────────────────────────────────────
def build_main_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
Answer ONLY using the provided context.

Rules:
- Each sentence MUST end with citations like [1], [2].
- No separate sources list.
- If unknown, say you don't know.

Context:\n{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

def build_summary_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Summarize this conversation in 3 sentences with key details."),
        ("human", "{history_text}")
    ])

# ─── CHAIN BUILDER ────────────────────────────────────────────────────────────
def build_chain(on_error=None):
    def fail(msg):
        if on_error:
            on_error(msg)
        else:
            print(f"Error: {msg}")
            sys.exit(1)
        raise RuntimeError(msg)

    # Initialize core components
    llm = init_llm(fail)
    embeddings = init_embeddings(fail)
    vectorstore = load_vectorstore(embeddings, fail)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Prompts
    main_prompt = build_main_prompt()
    summary_prompt = build_summary_prompt()

    # Chains
    summary_chain = summary_prompt | llm | StrOutputParser()

    rag_chain = (
        {
            "context": lambda x: x["context"],
            "input": lambda x: x["input"],
            "history": lambda x: x.get("history", []),
        }
        | main_prompt
        | llm
        | StrOutputParser()
    )

    chain_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return llm, retriever, main_prompt, summary_chain, chain_with_memory