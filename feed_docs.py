import sys
import os
import logging
import time
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# ─── Config ───────────────────────────────────────────────────────────────────
DOC_DIR          = "docs/"
VECTOR_STORE_DIR = "local_faiss_db"
LOCAL_EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 100
BATCH_SIZE       = 50
MAX_RETRIES      = 3

SUPPORTED_EXTENSIONS = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
}

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def progress_bar(current: int, total: int, width: int = 40) -> str:
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = current / total * 100
    return f"[{bar}] {current}/{total} ({pct:.1f}%)"


def embed_with_retry(vectorstore, batch, embeddings, attempt=1):
    try:
        if vectorstore is None:
            return FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
            return vectorstore
    except Exception as e:
        if attempt < MAX_RETRIES:
            wait = 2 ** attempt
            log.warning(f"Embedding failed (attempt {attempt}/{MAX_RETRIES}). Retrying in {wait}s... [{e}]")
            time.sleep(wait)
            return embed_with_retry(vectorstore, batch, embeddings, attempt + 1)
        raise


# ─── Step 1: Validate directory ───────────────────────────────────────────────
def validate_doc_dir(doc_dir: str) -> dict:
    log.info("Validating documents directory...")
    if not os.path.isdir(doc_dir):
        log.error(f"Directory '{doc_dir}' not found. Create it and add documents first.")
        sys.exit(1)

    found = {ext: [] for ext in SUPPORTED_EXTENSIONS}
    for f in os.listdir(doc_dir):
        ext = Path(f).suffix.lower()
        if ext in found:
            found[ext].append(f)

    total = sum(len(v) for v in found.items())
    if total == 0:
        log.error(f"No supported files found in '{doc_dir}'. Add PDF, DOCX, or TXT files.")
        sys.exit(1)

    for ext, files in found.items():
        if files:
            log.info(f"  {ext.upper()}: {len(files)} file(s) → {', '.join(files)}")

    return found


# ─── Step 2: Load all documents ───────────────────────────────────────────────
def load_documents(doc_dir: str) -> list:
    log.info("Loading documents...")
    all_docs = []
    failed = []

    for filepath in Path(doc_dir).iterdir():
        ext = filepath.suffix.lower()
        loader_cls = SUPPORTED_EXTENSIONS.get(ext)
        if not loader_cls:
            continue
        try:
            loader = loader_cls(str(filepath))
            docs = loader.load()
            all_docs.extend(docs)
            log.info(f"  ✓ {filepath.name} — {len(docs)} page(s)/section(s)")
        except Exception as e:
            log.warning(f"  ✗ Failed to load '{filepath.name}': {e}")
            failed.append(filepath.name)

    if failed:
        log.warning(f"Skipped {len(failed)} file(s) due to errors: {', '.join(failed)}")

    if not all_docs:
        log.error("No content could be extracted from any file. Aborting.")
        sys.exit(1)

    log.info(f"Loaded {len(all_docs)} page(s)/section(s) total.")
    return all_docs


# ─── Step 3: Split into chunks ────────────────────────────────────────────────
def split_documents(documents: list) -> list:
    log.info(f"Splitting documents (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
    except Exception as e:
        log.error(f"Failed to split documents: {e}")
        sys.exit(1)

    if not chunks:
        log.error("No chunks were created. Check document content.")
        sys.exit(1)

    log.info(f"Created {len(chunks)} chunk(s).")
    return chunks


# ─── Step 4: Embed and store in FAISS ─────────────────────────────────────────
def embed_and_store(chunks: list) -> None:
    log.info(f"Initialising embeddings with model '{LOCAL_EMBED_MODEL}'...")
    try:
        embeddings = OllamaEmbeddings(model=LOCAL_EMBED_MODEL)
        embeddings.embed_query("ping")
    except Exception as e:
        log.error(
            f"Could not connect to Ollama. Make sure Ollama is running and "
            f"'{LOCAL_EMBED_MODEL}' is pulled.\n  Run: ollama pull {LOCAL_EMBED_MODEL}\n  Error: {e}"
        )
        sys.exit(1)

    log.info(f"Embedding {len(chunks)} chunk(s) in batches of {BATCH_SIZE}...")
    vectorstore = None
    total = len(chunks)

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        end   = min(i + BATCH_SIZE, total)
        print(f"\r\033[K  {progress_bar(end, total)}  batch {i // BATCH_SIZE + 1}", end="", flush=True)
        try:
            vectorstore = embed_with_retry(vectorstore, batch, embeddings)
        except Exception as e:
            print()
            log.error(f"Embedding failed after {MAX_RETRIES} attempts: {e}")
            sys.exit(1)

    print()

    if vectorstore is None:
        log.error("Vectorstore is empty after embedding. Aborting.")
        sys.exit(1)

    Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    try:
        vectorstore.save_local(VECTOR_STORE_DIR)
    except Exception as e:
        log.error(f"Failed to save FAISS index: {e}")
        sys.exit(1)

    log.info(f"FAISS index saved to '{VECTOR_STORE_DIR}/'.")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    start = time.time()
    log.info("=== Document Ingest Pipeline Started ===")

    found_files = validate_doc_dir(DOC_DIR)
    documents   = load_documents(DOC_DIR)
    chunks      = split_documents(documents)
    embed_and_store(chunks)

    total_files = sum(len(v) for v in found_files.values())
    elapsed = time.time() - start
    log.info(f"=== Done! {total_files} file(s) indexed in {elapsed:.1f}s. Ready to query. ===")


if __name__ == "__main__":
    main()