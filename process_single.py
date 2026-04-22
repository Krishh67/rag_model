"""
process_single.py
=================
Run steps 3-7 of the RAG pipeline for ONE PDF in your terminal.
Live progress printed after every batch.

Usage:
    python process_single.py                       # defaults to 2_2021.pdf
    python process_single.py data/some_other.pdf   # any file
"""

import sys
import io
import time
from pathlib import Path

# Force UTF-8 so prints work on any Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rag_engine import (
    chunk_text,
    create_embeddings,
    add_new_documents,
    _mark_file_processed,
    is_file_processed,
    EMBED_BATCH_SIZE,
)
from langchain_community.document_loaders import PyMuPDFLoader  # robust for large PDFs

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/2_2021.pdf")
BATCH_SIZE  = EMBED_BATCH_SIZE * 4   # 256 chunks per embed call

# ── Helpers ────────────────────────────────────────────────────────────────────
def sep():
    print("-" * 62)

def hdr(title):
    sep()
    print(f"  {title}")
    sep()

def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    overall_start = time.time()
    print()
    hdr("DocMind -- Single-file processor")
    print(f"  Target : {TARGET_FILE}")
    print(f"  Batch  : {BATCH_SIZE} chunks per embed call")
    print()

    # Guard: file must exist
    if not TARGET_FILE.exists():
        print(f"  [ERROR] File not found: {TARGET_FILE}")
        sys.exit(1)

    size_mb = round(TARGET_FILE.stat().st_size / (1024 ** 2), 1)
    print(f"  File size : {size_mb} MB")

    # Guard: already processed?
    if is_file_processed(TARGET_FILE.name):
        print(f"\n  [SKIP] '{TARGET_FILE.name}' already in processed_files.json.")
        print("  Nothing to do. Delete processed_files.json to reprocess.\n")
        sys.exit(0)

    # ------------------------------------------------------------------
    # STEP 3 -- Load PDF
    # ------------------------------------------------------------------
    hdr("STEP 3 -- Load PDF (PyPDFLoader)")
    t0 = time.time()
    print("  Reading pages... (may take several minutes for large files)")
    loader = PyMuPDFLoader(str(TARGET_FILE))  # handles large/compressed PDFs without limit errors
    pages  = loader.load()
    print(f"  [OK] Loaded {len(pages)} pages  [{fmt_time(time.time() - t0)}]")

    # ------------------------------------------------------------------
    # STEP 4 -- Chunk text
    # ------------------------------------------------------------------
    hdr("STEP 4 -- Split into chunks (500 chars / 50 overlap)")
    t0 = time.time()
    chunks = chunk_text(pages, filename=TARGET_FILE.name)
    del pages   # free memory
    total_chunks = len(chunks)
    print(f"  [OK] {total_chunks} chunks created  [{fmt_time(time.time() - t0)}]")
    print(f"  Sample metadata: {chunks[0].metadata}")

    # ------------------------------------------------------------------
    # STEP 5+6 -- Embed in batches + save FAISS checkpoint after each
    # ------------------------------------------------------------------
    hdr("STEP 5+6 -- Embed chunks & save FAISS checkpoints")
    print("  Loading embedding model (MiniLM-L6-v2)...")
    emb = create_embeddings()
    print("  [OK] Model ready\n")

    embed_start = time.time()

    def progress_cb(done: int, total: int):
        batch_num = max(done // BATCH_SIZE, 1)
        elapsed   = time.time() - embed_start
        avg_time  = elapsed / batch_num
        remaining = avg_time * ((total - done) / BATCH_SIZE)
        pct       = done / total * 100
        print(
            f"  Batch {batch_num:>3}  |  {done:>5}/{total} chunks  "
            f"({pct:5.1f}%)  |  elapsed {fmt_time(elapsed)}"
            f"  |  ~{fmt_time(remaining)} left"
        )

    vs = add_new_documents(
        new_chunks=chunks,
        embeddings=emb,
        batch_size=BATCH_SIZE,
        progress_cb=progress_cb,
    )
    print(f"\n  [OK] Embedding done  [{fmt_time(time.time() - embed_start)}]")
    print("  [SAVED] FAISS index -> vector_db/")

    # ------------------------------------------------------------------
    # STEP 7 -- Mark file as processed
    # ------------------------------------------------------------------
    hdr("STEP 7 -- Mark file as processed")
    _mark_file_processed(TARGET_FILE.name)
    print(f"  [OK] '{TARGET_FILE.name}' written to processed_files.json")

    # Summary
    sep()
    print(f"  [DONE] Total time : {fmt_time(time.time() - overall_start)}")
    print(f"         Chunks     : {total_chunks}  from  {size_mb} MB")
    print(f"         Next step  : streamlit run app.py")
    sep()
    print()

if __name__ == "__main__":
    main()
