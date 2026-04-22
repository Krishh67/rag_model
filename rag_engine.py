"""
rag_engine.py
=============
Production-ready RAG engine for local PDFs.

Key capabilities
----------------
• Batch-safe processing   — large PDFs are embedded in configurable batches
                            so memory stays bounded even for 300 MB+ files.
• Incremental updates     — tracks processed files in processed_files.json;
                            already-indexed PDFs are skipped on re-run.
• Resume capability       — if a run is interrupted mid-file a partial FAISS
                            index is kept on disk; on restart the engine picks
                            up from the last successfully saved batch.
• Metadata extraction     — parses year and part label out of filenames like
                            "CHI_2022_part1.pdf".
• Gemini 2.0 Flash LLM   — via LangChain's google-genai wrapper; set
                            GEMINI_API_KEY in the environment before calling
                            query().

Public API
----------
load_pdfs_from_folder(folder)          -> list[Path]
chunk_text(documents)                  -> list[Document]
create_embeddings()                    -> HuggingFaceEmbeddings
save_db(chunks, embeddings)            -> FAISS
load_db(embeddings)                    -> FAISS | None
add_new_documents(chunks, embeddings)  -> FAISS
query(question, vectorstore, api_key, k) -> dict
"""

import gc
import json
import os
import re
from pathlib import Path
from typing import Generator

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# ─── Global constants ─────────────────────────────────────────────────────────

DATA_FOLDER        = Path("data")          # where PDFs live
VECTOR_DB_DIR      = Path("vector_db")     # persisted FAISS index
PROCESSED_LOG      = Path("processed_files.json")  # incremental-update ledger

EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL       = 'gemini-2.5-flash-lite'

CHUNK_SIZE         = 500    # characters per text chunk
CHUNK_OVERLAP      = 50     # character overlap between consecutive chunks
EMBED_BATCH_SIZE   = 64     # chunks embedded per GPU/CPU call  ← tune for RAM
TOP_K              = 5      # chunks retrieved per query


# ─── Metadata helpers ─────────────────────────────────────────────────────────

def _extract_metadata(filename: str) -> dict:
    """
    Parse structured metadata from a filename like 'CHI_2022_part1.pdf'.

    Rules
    -----
    • year  — first 4-digit sequence found in the stem (e.g. 2022)
    • part  — text following the last occurrence of 'part' (e.g. 'part1')
    • file  — full stem without extension

    Returns a dict with keys: 'source_file', 'year', 'part'.
    """
    stem = Path(filename).stem          # e.g. "CHI_2022_part1"

    # Year: first 4-digit number (relaxed — no word boundary required)
    year_match = re.search(r'(19|20)\d{2}', stem)
    year = year_match.group(0) if year_match else "unknown"

    # Part: text after last 'part' (case-insensitive)
    part_match = re.search(r'part\w*', stem, re.IGNORECASE)
    part = part_match.group(0).lower() if part_match else "full"

    return {"source_file": stem, "year": year, "part": part}


# ─── Processed-files ledger ───────────────────────────────────────────────────

def _load_processed_log() -> dict:
    """
    Load the JSON ledger that records which files have been fully indexed.
    Returns an empty dict if the file does not exist yet.
    """
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_processed_log(log: dict) -> None:
    """Persist the processed-files ledger to disk (atomic overwrite)."""
    PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROCESSED_LOG.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    tmp.replace(PROCESSED_LOG)


def _mark_file_processed(filename: str) -> None:
    """Add *filename* to the processed ledger and save."""
    log = _load_processed_log()
    log[filename] = True
    _save_processed_log(log)


def is_file_processed(filename: str) -> bool:
    """Return True if *filename* has already been fully indexed."""
    return _load_processed_log().get(filename, False)


# ─── 1. load_pdfs_from_folder ─────────────────────────────────────────────────

def load_pdfs_from_folder(folder: str | Path = DATA_FOLDER) -> list[Path]:
    """
    Return a sorted list of all *.pdf paths inside *folder*.

    Parameters
    ----------
    folder : str | Path
        Directory to scan.  Defaults to DATA_FOLDER ("data/").

    Returns
    -------
    list[Path]
        Sorted list of PDF paths found in the folder.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(
            f"Data folder '{folder}' not found. "
            "Create it and place your PDF files inside."
        )
    pdfs = sorted(folder.glob("*.pdf"))
    return pdfs


# ─── 2. chunk_text ────────────────────────────────────────────────────────────

def chunk_text(documents: list, filename: str = "") -> list:
    """
    Split a list of LangChain Documents into fixed-size overlapping chunks.

    Additionally enriches each chunk's metadata with:
      • source_file  — stem of the original PDF filename
      • year         — 4-digit year parsed from the filename
      • part         — part label parsed from the filename
      • page         — 0-based page index (already set by PyPDFLoader)

    Parameters
    ----------
    documents : list[Document]
        Raw pages loaded by PyPDFLoader.
    filename : str
        Original PDF filename used for metadata extraction.

    Returns
    -------
    list[Document]
        Chunked documents ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,   # stores character offset in metadata
    )
    chunks = splitter.split_documents(documents)

    # Enrich metadata
    extra = _extract_metadata(filename) if filename else {}
    for chunk in chunks:
        chunk.metadata.update(extra)
        # Normalise page to int (PyPDFLoader stores it as int already)
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 0

    return chunks


# ─── 3. create_embeddings ─────────────────────────────────────────────────────

def create_embeddings() -> HuggingFaceEmbeddings:
    """
    Instantiate a HuggingFaceEmbeddings model (MiniLM-L6-v2, 384-dim).

    The model runs entirely locally — no API key required.
    First call downloads the model weights (~90 MB) and caches them.

    Returns
    -------
    HuggingFaceEmbeddings
        Ready-to-use embeddings object.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},          # change to "cuda" for GPU
        encode_kwargs={"normalize_embeddings": True, "batch_size": EMBED_BATCH_SIZE},
    )


# ─── Batch generator ──────────────────────────────────────────────────────────

def _batched(items: list, size: int) -> Generator[list, None, None]:
    """Yield successive *size*-length slices of *items*."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ─── 4. save_db ───────────────────────────────────────────────────────────────

def save_db(
    chunks: list,
    embeddings: HuggingFaceEmbeddings,
    batch_size: int = EMBED_BATCH_SIZE * 4,
    progress_cb=None,
) -> FAISS:
    """
    Build a brand-new FAISS index from *chunks* and save it to disk.

    Documents are embedded in batches of *batch_size* chunks to keep memory
    usage bounded when processing very large PDFs.

    Parameters
    ----------
    chunks : list[Document]
        All chunked documents to index.
    embeddings : HuggingFaceEmbeddings
        Embedding model.
    batch_size : int
        Number of chunks embedded per call (default 256).
    progress_cb : callable | None
        Optional callback(current_idx, total) for progress reporting.

    Returns
    -------
    FAISS
        Populated vector store (also persisted to disk).
    """
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = None
    total = len(chunks)

    for batch_idx, batch in enumerate(_batched(chunks, batch_size)):
        if vectorstore is None:
            # Create the index from the first batch
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            # Merge subsequent batches
            batch_vs = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_vs)
            del batch_vs

        # Checkpoint after every batch ← resume from here if interrupted
        vectorstore.save_local(str(VECTOR_DB_DIR))

        processed = min((batch_idx + 1) * batch_size, total)
        if progress_cb:
            progress_cb(processed, total)

        gc.collect()   # encourage memory release

    return vectorstore


# ─── 5. load_db ───────────────────────────────────────────────────────────────

def load_db(embeddings: HuggingFaceEmbeddings) -> "FAISS | None":
    """
    Reload a previously saved FAISS index from VECTOR_DB_DIR.

    Parameters
    ----------
    embeddings : HuggingFaceEmbeddings
        Must use the same model that was used when save_db() was called.

    Returns
    -------
    FAISS | None
        The loaded vector store, or None if no index exists.
    """
    index_file = VECTOR_DB_DIR / "index.faiss"
    if not index_file.exists():
        return None
    return FAISS.load_local(
        str(VECTOR_DB_DIR),
        embeddings,
        allow_dangerous_deserialization=True,  # required by LangChain ≥ 0.2
    )


# ─── 6. add_new_documents ─────────────────────────────────────────────────────

def add_new_documents(
    new_chunks: list,
    embeddings: HuggingFaceEmbeddings,
    batch_size: int = EMBED_BATCH_SIZE * 4,
    progress_cb=None,
) -> FAISS:
    """
    Merge *new_chunks* into an existing FAISS index (or create one if absent).

    This is the incremental-update entry point.  It loads the current index,
    appends documents in batches, and saves after each batch so progress is
    never lost.

    Parameters
    ----------
    new_chunks : list[Document]
        New document chunks to add.
    embeddings : HuggingFaceEmbeddings
        Same embedding model as used for the existing index.
    batch_size : int
        Chunks embedded per batch call.
    progress_cb : callable | None
        Optional callback(current_idx, total).

    Returns
    -------
    FAISS
        Updated vector store.
    """
    vectorstore = load_db(embeddings)
    total = len(new_chunks)

    for batch_idx, batch in enumerate(_batched(new_chunks, batch_size)):
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            batch_vs = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_vs)
            del batch_vs

        # Save checkpoint after every batch (resume support)
        vectorstore.save_local(str(VECTOR_DB_DIR))

        processed = min((batch_idx + 1) * batch_size, total)
        if progress_cb:
            progress_cb(processed, total)

        gc.collect()

    return vectorstore


# ─── 7. query ─────────────────────────────────────────────────────────────────

# Prompt that strictly constraints the LLM to retrieved context only.
_RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful research assistant with access to indexed academic documents.
Use the context passages below to answer the question as thoroughly as possible.
If the context contains partial information, use it and say what you found.
Only say you cannot find information if the context is completely irrelevant or empty.

Context passages:
{context}

Question: {question}

Answer (based on the context above):"""
)


def _format_docs(docs: list) -> str:
    """Concatenate document page_content into a single context string."""
    return "\n\n".join(d.page_content for d in docs)


def query(
    question: str,
    vectorstore: FAISS,
    api_key: str,
    k: int = TOP_K,
) -> dict:
    """
    Retrieve relevant chunks from `vectorstore` and generate an answer via
    Gemini 2.0 Flash using a modern LCEL chain (no deprecated RetrievalQA).

    Parameters
    ----------
    question : str
        Natural-language question from the user.
    vectorstore : FAISS
        Populated FAISS vector store.
    api_key : str
        Google Gemini API key.
    k : int
        Number of chunks to retrieve (default TOP_K = 5).

    Returns
    -------
    dict
        {
          "answer":  str,            # model's answer
          "sources": list[dict]      # deduplicated source metadata
        }
        Each source dict: {"source_file", "year", "part", "page"}.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.1,
        convert_system_message_to_human=True,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # ── LCEL chain: retrieve → format → prompt → LLM → parse ─────────────────
    # The retriever is invoked separately so we can also return source docs.
    source_docs = retriever.invoke(question)

    rag_chain = (
        {
            "context":  lambda _: _format_docs(source_docs),
            "question": RunnablePassthrough(),
        }
        | _RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)

    # ── Deduplicate and format source metadata ────────────────────────────────
    sources = []
    seen = set()
    for doc in source_docs:
        m   = doc.metadata
        key = (m.get("source_file", "?"), m.get("page", "?"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source_file": m.get("source_file", m.get("source", "unknown")),
                "year":        m.get("year", "unknown"),
                "part":        m.get("part", "full"),
                "page":        m.get("page", "N/A"),
            })

    return {"answer": answer, "sources": sources}
