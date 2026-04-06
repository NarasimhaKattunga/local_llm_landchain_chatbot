# Local RAG Chatbot : document intelligence

Chat with your own documents - PDF's , docs and text files — privately, offline, for free.

No API keys. No cloud. Everything runs on your machine.

---

## What it does

You upload docs. You ask questions. The chatbot finds the most relevant parts of your documents and answers using a local AI model. It also remembers your conversation across sessions.

-----

## How it works

```
┌─────────────────────────────────────────────────────────┐
│                    Your Browser                         │
│                  Streamlit UI :8501                     │
└───────────────────────┬─────────────────────────────────┘
                        │  question
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  LangChain Chain                        │
│                                                         │
│   ┌──────────────┐        ┌───────────────────────┐     │
│   │  Condense    │        │  Conversation Memory  │     │
│   │  question    │        │  SQLite · auto-summary│     │
│   └──────┬───────┘        └───────────────────────┘     │
│          │ standalone query                             │
└──────────┼──────────────────────────────────────────────┘
           │
     ┌─────▼──────────────────────────────┐
     │         nomic-embed-text           │
     │    converts query → 768-dim vector │
     └─────┬──────────────────────────────┘
           │ query vector
     ┌─────▼──────────────────────────────┐
     │         FAISS Vector Index         │
     │    finds top-4 matching chunks     │
     │    from your indexed PDFs          │
     └─────┬──────────────────────────────┘
           │ relevant context chunks
     ┌─────▼──────────────────────────────┐
     │       Ollama · Llama3              │
     │    generates answer from context   │
     │    runs 100% locally               │
     └─────┬──────────────────────────────┘
           │ answer + sources
           ▼
        Your screen
```

---

## Tech stack

| Component | Tool | Purpose |
|-----------|------|---------|
| UI | Streamlit | Web chat interface |
| Orchestration | LangChain | Connects all the pieces |
| LLM | Ollama · Llama3 | Generates answers locally |
| Embeddings | nomic-embed-text | Converts text to vectors |
| Vector search | FAISS | Finds relevant document chunks |
| Memory | SQLite | Stores conversation history |

---

## Ingestion pipeline (one-time setup)

Run this once before chatting. It reads your PDFs and builds the search index.

```
  Your Docs
      │
      ▼
  DirectoryLoader          ← reads all files from ./docs
      │
      ▼
  TextSplitter             ← cuts into 512-token chunks (64 overlap)
      │
      ▼
  nomic-embed-text         ← converts each chunk to a vector
      │
      ▼
  FAISS.save_local()       ← writes index.faiss + index.pkl to disk
```

```bash
python feed_docs.py
```

---

## Installation

### 1 — Clone the repo

```bash
git clone <your-repo-url>
cd local_llm_langchain_chatbot
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Mac / Linux
.venv\Scripts\activate         # Windows
```

### 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4 — Install Ollama and pull models

Download Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama serve
ollama pull llama3
ollama pull nomic-embed-text
```

### 5 — Index your documents

Put your PDFs in the `./docs` folder, then run:

```bash
python feed_docs.py
```

### 6 — Start the chatbot

```bash
streamlit run qa_chatbot_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project structure

```
local_llm_langchain_chatbot/
│
├── feed_docs.py              ← run once to index your PDFs
├── local_raw_chatbot_engine.py   ← RAG + memory logic
├── qa_chatbot_app.py      ← Streamlit UI
│
├── docs/                  ← put your docs here
├── faiss_db/              ← generated vector index (auto-created)
├── llm_local_chat_history.db        ← conversation memory (auto-created)
│
└── requirements.txt
```

---

## Memory and summarization

Every conversation is saved to `llm_local_chat_history.db`. When the history grows long, older messages are automatically summarized and stored as a system note — so the chatbot stays fast without losing context.

```
Turn 1  ──┐
Turn 2  ──┤  → stored in SQLite
Turn 3  ──┘
  ...
Turn 8  ──┐
           ├─→  summarized  →  stored as SUMMARY: ...
Turn 9  ──┘
```

You can tune this in `local_raw_chatbot_engine.py`:

```python
MAX_HISTORY = 8     # how many turns before summarization kicks in
k           = 5     # how many document chunks to retrieve
temperature = 0.7   # 0 = focused answers, 1 = more creative
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Ollama not running | `ollama serve` |
| Model not found | `ollama pull llama3` and `ollama pull nomic-embed-text` |
| No FAISS index | `python feed_docs.py` |
| Import errors | Restart Streamlit after code changes |
| Playwright errors | `playwright install` |

---

## What's coming next

- Async support for faster responses
- Multi-user session handling
- FastAPI + Docker deployment
- Kafka integration for streaming pipelines
- Logging and observability

---

## License

MIT — free to use, modify, and distribute.

---

Built for local-first AI. Your data stays on your machine, always.
