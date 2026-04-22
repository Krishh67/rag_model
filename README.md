# 🧠 DocMind — Local PDF RAG System

End-to-end **Retrieval-Augmented Generation** over local PDFs.  
Built with LangChain · FAISS · HuggingFace Embeddings · Gemini 2.0 Flash · Streamlit.

---

## 📁 Project Structure

```
rag/
├── app.py                  ← Streamlit UI  (run this)
├── rag_engine.py           ← All modular RAG functions
├── requirements.txt
├── .streamlit/
│   └── config.toml         ← Dark theme
├── data/                   ← ⬅ Place your PDFs here
│   ├── CHI_2022_part1.pdf
│   └── CHI_2022_part2.pdf
├── vector_db/              ← Auto-created: FAISS index
│   ├── index.faiss
│   └── index.pkl
└── processed_files.json    ← Auto-created: incremental-update ledger
```

---

## 🚀 Quick Start

### 1 — Install dependencies
```powershell
pip install -r requirements.txt
```

### 2 — Add your PDFs
```
data/
  CHI_2022_part1.pdf
  CHI_2022_part2.pdf
  ...
```

### 3 — Get a free Gemini API key
1. Visit <https://aistudio.google.com/app/apikey>
2. Click **Create API key**
3. Paste it in the sidebar when the app opens.

Alternatively set it as an environment variable:
```powershell
$env:GEMINI_API_KEY = "AIza..."
```

### 4 — Run
```powershell
streamlit run app.py
```

Open **http://localhost:8501**, paste your API key, click **Process Data**, then ask questions.

---

## 🧩 Module Reference (`rag_engine.py`)

| Function | Signature | Description |
|---|---|---|
| `load_pdfs_from_folder` | `(folder) → list[Path]` | Scan `data/` for all `.pdf` files |
| `chunk_text` | `(docs, filename) → list[Document]` | 500-char / 50-overlap chunks + metadata |
| `create_embeddings` | `() → HuggingFaceEmbeddings` | Local MiniLM-L6-v2 (no API needed) |
| `save_db` | `(chunks, embeddings, batch_sz, cb) → FAISS` | Build fresh FAISS index (batched) |
| `load_db` | `(embeddings) → FAISS \| None` | Reload persisted index from `vector_db/` |
| `add_new_documents` | `(chunks, embeddings, batch_sz, cb) → FAISS` | Incremental merge into existing index |
| `query` | `(question, vectorstore, api_key, k) → dict` | RetrievalQA via Gemini 2.0 Flash |

---

## ⚙️ Configuration (edit `rag_engine.py`)

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Character overlap |
| `EMBED_BATCH_SIZE` | 64 | Chunks per embedding call |
| `TOP_K` | 5 | Retrieved chunks per query |
| `GEMINI_MODEL` | `gemini-2.0-flash` | LLM model name |
| `DATA_FOLDER` | `data/` | PDF source directory |
| `VECTOR_DB_DIR` | `vector_db/` | FAISS index save path |

---

## 🔄 Incremental + Resume Logic

```
First run:
  ┌─ Scan data/ ──────────────────────────────┐
  │  CHI_2022_part1.pdf  → not in ledger → process
  │  CHI_2022_part2.pdf  → not in ledger → process
  └───────────────────────────────────────────┘
        ↓ after each file: write to processed_files.json

Second run (or after interrupt):
  ┌─ Scan data/ ──────────────────────────────┐
  │  CHI_2022_part1.pdf  → in ledger → ✅ SKIP
  │  CHI_2022_part2.pdf  → not in ledger → process
  └───────────────────────────────────────────┘

Batch checkpointing:
  Every N chunks → save FAISS index to disk
  → even a crash mid-file loses only the current batch
```

---

## 📝 Metadata Extracted

For a file named `CHI_2022_part1.pdf`:

| Field | Value |
|---|---|
| `source_file` | `CHI_2022_part1` |
| `year` | `2022` |
| `part` | `part1` |
| `page` | page index (int) |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `data/ not found` | Create the folder and add PDFs |
| Slow first run | MiniLM model downloading (~90 MB) — wait once |
| `GOOGLE_API_KEY not set` | Paste key in sidebar before querying |
| Out of memory | Lower **Embed batch size** slider in sidebar |
| Want GPU | Set `"device": "cuda"` in `create_embeddings()` |
| Reset everything | Delete `vector_db/` and `processed_files.json` |
