"""
app.py
======
Streamlit front-end for the local-PDF RAG system.

Usage
-----
    streamlit run app.py

Environment
-----------
Set  GEMINI_API_KEY  before launching, or paste the key in the sidebar.
"""

import os
import gc
import time
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

from rag_engine import (
    DATA_FOLDER,
    VECTOR_DB_DIR,
    EMBED_BATCH_SIZE,
    TOP_K,
    load_pdfs_from_folder,
    chunk_text,
    create_embeddings,
    save_db,
    load_db,
    add_new_documents,
    is_file_processed,
    _mark_file_processed,
    query,
)

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocMind — Local PDF RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1744 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    border: 1px solid rgba(102,126,234,0.35);
    border-radius: 18px;
    padding: 30px 40px;
    text-align: center;
    margin-bottom: 26px;
    backdrop-filter: blur(8px);
}
.hero h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px;
}
.hero p { color: #94a3b8; font-size: 1rem; margin: 0; }

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
    backdrop-filter: blur(6px);
    transition: border-color .25s;
}
.card:hover { border-color: rgba(167,139,250,0.45); }

/* ── Labels ── */
.label {
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 8px;
}

/* ── Answer box ── */
.answer-box {
    background: linear-gradient(135deg,rgba(96,165,250,.1),rgba(167,139,250,.1));
    border: 1px solid rgba(96,165,250,.3);
    border-radius: 12px;
    padding: 20px 24px;
    color: #e2e8f0;
    font-size: .97rem;
    line-height: 1.75;
    white-space: pre-wrap;
}

/* ── Chips ── */
.chip {
    display: inline-block;
    background: rgba(167,139,250,.15);
    border: 1px solid rgba(167,139,250,.4);
    border-radius: 24px;
    padding: 4px 13px;
    font-size: .8rem;
    color: #c4b5fd;
    margin: 3px 3px 3px 0;
    font-weight: 500;
}
.chip-green {
    background: rgba(52,211,153,.12);
    border-color: rgba(52,211,153,.4);
    color: #34d399;
}
.chip-blue {
    background: rgba(96,165,250,.12);
    border-color: rgba(96,165,250,.4);
    color: #60a5fa;
}
.chip-yellow {
    background: rgba(251,191,36,.12);
    border-color: rgba(251,191,36,.4);
    color: #fbbf24;
}

/* ── File rows ── */
.file-row {
    display: flex;
    align-items: center;
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 5px;
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.06);
    font-size: .88rem;
    color: #cbd5e1;
}
.file-row .icon { margin-right: 9px; font-size: 1rem; }

/* ── Log area ── */
.log-box {
    background: rgba(0,0,0,.35);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'Courier New', monospace;
    font-size: .8rem;
    color: #94a3b8;
    max-height: 220px;
    overflow-y: auto;
    line-height: 1.6;
}

/* ── Input overrides ── */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid rgba(167,139,250,.4) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,.18) !important;
}

/* ── Button overrides ── */
.stButton > button {
    background: linear-gradient(135deg,#667eea,#764ba2) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: opacity .2s, transform .15s !important;
}
.stButton > button:hover { opacity: .85 !important; transform: translateY(-1px) !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg,#667eea,#a78bfa) !important; border-radius: 4px !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,.04);
    border-radius: 10px;
    padding: 12px 16px;
    border: 1px solid rgba(255,255,255,.07);
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricValue"] { color: #a78bfa !important; }

hr { border-color: rgba(255,255,255,.07) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,.35); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ─────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "vectorstore":   None,
        "embeddings":    None,
        "db_ready":      False,
        "processing":    False,
        "process_log":   [],   # list of log-line strings
        "query_history": [],   # list of {question, answer, sources}
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

_init()


# ─── Cached embedding model (once per session) ────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return create_embeddings()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.markdown("*Local PDF · RAG · Gemini 2.0 Flash*")
    st.divider()

    # ── API key ──────────────────────────────────────────────────────────────
    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input(
        "key",
        type="password",
        placeholder="AIza…",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get a free key at https://aistudio.google.com/app/apikey",
        label_visibility="collapsed",
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        st.markdown('<span class="chip chip-green">✓ Key set</span>', unsafe_allow_html=True)
    else:
        st.caption("⚠️ Required for Q&A")

    st.divider()

    # ── Retrieval settings ───────────────────────────────────────────────────
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Top-K chunks", 1, 10, TOP_K,
                       help="Chunks retrieved per query")
    batch_sz = st.slider("Embed batch size", 32, 256, EMBED_BATCH_SIZE * 4, step=32,
                          help="Chunks embedded per batch (lower = less RAM)")

    st.divider()

    # ── DB status ────────────────────────────────────────────────────────────
    st.markdown("### 💾 Vector Database")
    db_exists = (VECTOR_DB_DIR / "index.faiss").exists()
    if db_exists:
        st.markdown('<span class="chip chip-green">✓ Index on disk</span>', unsafe_allow_html=True)
        if st.button("📂 Load Existing DB", use_container_width=True):
            with st.spinner("Loading FAISS index…"):
                emb = get_embeddings()
                vs  = load_db(emb)
                if vs:
                    st.session_state.vectorstore = vs
                    st.session_state.embeddings  = emb
                    st.session_state.db_ready    = True
                    st.success("Database loaded!")
                else:
                    st.error("Load failed — index may be corrupt.")
    else:
        st.markdown('<span class="chip chip-yellow">No index yet</span>', unsafe_allow_html=True)
        st.caption("Click **Process Data** to build it.")

    st.divider()

    # ── Session stats ────────────────────────────────────────────────────────
    st.markdown("### 📊 Session")
    c1, c2 = st.columns(2)
    try:
        from rag_engine import _load_processed_log
        n_proc = len(_load_processed_log())
    except Exception:
        n_proc = 0
    c1.metric("Indexed PDFs", n_proc)
    c2.metric("Queries",      len(st.session_state.query_history))


# ─── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>🧠 DocMind</h1>
    <p>Local PDF knowledge base · Powered by Gemini 2.0 Flash</p>
</div>
""", unsafe_allow_html=True)

# ─── Layout ───────────────────────────────────────────────────────────────────

left, right = st.columns([1, 1.4], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — File overview + Process button
# ══════════════════════════════════════════════════════════════════════════════

with left:
    st.markdown('<div class="label">📁 PDFs in data/</div>', unsafe_allow_html=True)

    # Scan data folder
    try:
        pdf_paths = load_pdfs_from_folder(DATA_FOLDER)
    except FileNotFoundError:
        pdf_paths = []
        st.error(f"❌ Folder **{DATA_FOLDER}/** not found. Create it and add PDFs.")

    if not pdf_paths:
        st.markdown("""
        <div class="card" style="text-align:center;padding:34px;">
            <div style="font-size:2.5rem;margin-bottom:10px;">📂</div>
            <div style="color:#94a3b8;">No PDFs found in <code>data/</code></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # File list with per-file status
        for p in pdf_paths:
            processed = is_file_processed(p.name)
            icon  = "✅" if processed else "⏳"
            badge = (
                '<span class="chip chip-green" style="float:right;font-size:.72rem">indexed</span>'
                if processed else
                '<span class="chip chip-yellow" style="float:right;font-size:.72rem">pending</span>'
            )
            size_mb = round(p.stat().st_size / (1024**2), 1)
            st.markdown(f"""
            <div class="file-row">
                <span class="icon">{icon}</span>
                <span style="flex:1">{p.name} <span style="color:#64748b;font-size:.78rem">({size_mb} MB)</span></span>
                {badge}
            </div>
            """, unsafe_allow_html=True)

        pending = [p for p in pdf_paths if not is_file_processed(p.name)]
        processed_count = len(pdf_paths) - len(pending)

        st.markdown(f"""
        <div style="margin:10px 0 18px;color:#64748b;font-size:.85rem;">
            {processed_count} / {len(pdf_paths)} files indexed · {len(pending)} pending
        </div>
        """, unsafe_allow_html=True)

    # ── Process Data button ────────────────────────────────────────────────
    process_btn = st.button(
        "⚡ Process Data",
        use_container_width=True,
        disabled=not pdf_paths or st.session_state.processing,
    )

    # ── Processing log ────────────────────────────────────────────────────
    if st.session_state.process_log:
        st.markdown('<div class="label" style="margin-top:18px;">📋 Processing Log</div>', unsafe_allow_html=True)
        log_html = "<br>".join(st.session_state.process_log[-40:])   # last 40 lines
        st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────────────
    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
1. **Scan** — all PDFs in `data/` are listed automatically.
2. **Skip** — already-indexed files are detected via `processed_files.json`.
3. **Load** — `PyPDFLoader` streams pages one at a time.
4. **Chunk** — 500-char chunks with 50-char overlap, metadata tagged.
5. **Embed** — MiniLM-L6-v2 (local, CPU) in batches → FAISS index.
6. **Save** — checkpoint after every batch; resume on interruption.
7. **Query** — question → top-5 chunks → Gemini 2.0 Flash → answer.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING LOGIC
# ══════════════════════════════════════════════════════════════════════════════

if process_btn and pdf_paths:
    st.session_state.processing  = True
    st.session_state.process_log = []

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        st.session_state.process_log.append(f"<span style='color:#4ade80'>[{ts}]</span> {msg}")

    pending_files = [p for p in pdf_paths if not is_file_processed(p.name)]

    if not pending_files:
        st.info("✅ All files already processed. Load the existing DB from the sidebar.")
        st.session_state.processing = False
    else:
        emb = get_embeddings()
        st.session_state.embeddings = emb

        # Status placeholders in the LEFT column (inside 'left' block context)
        with left:
            status_text   = st.empty()
            progress_bar  = st.progress(0)
            file_progress = st.empty()

        total_files = len(pending_files)

        for file_idx, pdf_path in enumerate(pending_files):
            fname = pdf_path.name
            log(f"📄 Starting: <b>{fname}</b>")

            file_frac_start = file_idx / total_files
            file_frac_end   = (file_idx + 1) / total_files

            # ── Load pages lazily ──────────────────────────────────────────
            status_text.markdown(
                f'<span style="color:#e2e8f0;font-size:.9rem">'
                f'Loading <b>{fname}</b> ({file_idx+1}/{total_files})…</span>',
                unsafe_allow_html=True,
            )

            try:
                loader = PyPDFLoader(str(pdf_path))
                pages  = loader.load()  # list[Document] — one per page
            except Exception as exc:
                log(f"❌ Failed to load {fname}: {exc}")
                continue

            log(f"   Loaded {len(pages)} pages")

            # ── Chunk entire file ──────────────────────────────────────────
            all_chunks = chunk_text(pages, filename=fname)
            total_chunks = len(all_chunks)
            log(f"   {total_chunks} chunks created (batch_size={batch_sz})")
            del pages
            gc.collect()

            # ── Embed + add in batches ─────────────────────────────────────
            def _progress(done, total, fi=file_idx, ft=total_files,
                          fstart=file_frac_start, fend=file_frac_end):
                frac = fstart + (done / max(total, 1)) * (fend - fstart)
                progress_bar.progress(min(frac, 1.0))
                pct = int(done / max(total, 1) * 100)
                file_progress.markdown(
                    f'<span style="color:#94a3b8;font-size:.82rem">'
                    f'{fname} — {done}/{total} chunks ({pct}%)</span>',
                    unsafe_allow_html=True,
                )

            try:
                vs = add_new_documents(
                    new_chunks=all_chunks,
                    embeddings=emb,
                    batch_size=batch_sz,
                    progress_cb=_progress,
                )
                st.session_state.vectorstore = vs
            except Exception as exc:
                log(f"❌ Embedding error ({fname}): {exc}")
                continue
            finally:
                del all_chunks
                gc.collect()

            _mark_file_processed(fname)
            log(f"✅ Done: <b>{fname}</b>")

        # Final state
        progress_bar.progress(1.0)
        status_text.markdown(
            '<span style="color:#4ade80;font-weight:600">✅ All files processed!</span>',
            unsafe_allow_html=True,
        )
        file_progress.empty()
        st.session_state.db_ready   = True
        st.session_state.processing = False
        log("=" * 40)
        log("🎉 Vector database ready. Ask your questions!")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Q&A Interface
# ══════════════════════════════════════════════════════════════════════════════

with right:
    st.markdown('<div class="label">💬 Ask Your Documents</div>', unsafe_allow_html=True)

    db_ready = st.session_state.db_ready and st.session_state.vectorstore is not None

    if not db_ready:
        st.markdown("""
        <div class="card" style="text-align:center;padding:48px 32px;">
            <div style="font-size:3rem;margin-bottom:14px;">🔍</div>
            <div style="color:#94a3b8;font-size:.97rem;">
                Process your PDFs or load an existing<br>database to start asking questions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Query form ────────────────────────────────────────────────────
        with st.form("query_form", clear_on_submit=True):
            question = st.text_area(
                "q",
                placeholder="What are the key findings of CHI 2022? Summarise the methodology…",
                height=108,
                label_visibility="collapsed",
            )
            ask_btn = st.form_submit_button("🔍 Ask DocMind", use_container_width=True)

        if ask_btn and question.strip():
            if not api_key:
                st.error("⚠️ Enter your Gemini API key in the sidebar first.")
            else:
                with st.spinner("Searching knowledge base and generating answer…"):
                    try:
                        result = query(
                            question=question.strip(),
                            vectorstore=st.session_state.vectorstore,
                            api_key=api_key,
                            k=top_k,
                        )
                        st.session_state.query_history.insert(0, {
                            "question": question.strip(),
                            "answer":   result["answer"],
                            "sources":  result["sources"],
                        })
                    except Exception as exc:
                        st.error(f"❌ Query error: {exc}")

        # ── Results (newest first, tabbed) ────────────────────────────────
        hist = st.session_state.query_history
        if hist:
            display = hist[:5]   # show last 5 queries as tabs
            tabs = st.tabs([f"Q{i+1}" for i in range(len(display))])

            for tab_i, tab in enumerate(tabs):
                entry = display[tab_i]
                with tab:
                    # Question
                    st.markdown(
                        f'<div class="label">Question</div>'
                        f'<div style="color:#e2e8f0;font-weight:500;margin-bottom:14px;">'
                        f'{entry["question"]}</div>',
                        unsafe_allow_html=True,
                    )
                    # Answer
                    st.markdown('<div class="label">Answer</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="answer-box">{entry["answer"]}</div>',
                        unsafe_allow_html=True,
                    )
                    # Sources
                    if entry["sources"]:
                        st.markdown(
                            '<div class="label" style="margin-top:14px;">Sources</div>',
                            unsafe_allow_html=True,
                        )
                        chips = ""
                        for src in entry["sources"]:
                            pf   = src["source_file"]
                            yr   = src["year"]
                            pt   = src["part"]
                            pg   = src["page"]
                            pg_s = f"p.{pg+1}" if isinstance(pg, int) else str(pg)
                            chips += (
                                f'<span class="chip">'
                                f'📄 {pf} · {yr} · {pt} · {pg_s}'
                                f'</span>'
                            )
                        st.markdown(chips, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Clear history"):
                st.session_state.query_history = []
                st.rerun()


# ─── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style="text-align:center;color:#334155;font-size:.8rem;padding-bottom:10px;">
    🦜 LangChain · 🤗 HuggingFace Embeddings · 🔍 FAISS · ✨ Gemini 2.0 Flash · 🎈 Streamlit
</div>
""", unsafe_allow_html=True)
