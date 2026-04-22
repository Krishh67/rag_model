"""
diagnose.py
===========
Checks what is actually stored in the FAISS index.
Run with: python diagnose.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rag_engine import create_embeddings, load_db

print("\n--- Loading embeddings model ---")
emb = create_embeddings()

print("--- Loading FAISS index ---")
vs = load_db(emb)
if vs is None:
    print("ERROR: No FAISS index found in vector_db/")
    sys.exit(1)

print(f"--- Index loaded. Total vectors: {vs.index.ntotal} ---\n")

# Do a test similarity search
test_query = "main topic"
print(f"Test query: '{test_query}'")
docs = vs.similarity_search(test_query, k=3)

print(f"\nRetrieved {len(docs)} chunks:\n")
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(f"Metadata : {doc.metadata}")
    print(f"Content  : {repr(doc.page_content[:300])}")
    print()
