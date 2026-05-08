"""
Streamlit frontend — deployed to Streamlit Cloud.
All heavy processing (vision LLM, vector store, Ragas) runs in the FastAPI
backend on Railway; this file only handles UI and HTTP calls.
"""

import io
import json
import os
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Financial Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.4rem 1.8rem; border-radius: 12px;
    margin-bottom: 1.5rem; color: white;
}
.hero h1 { margin: 0 0 0.3rem 0; font-size: 2rem; }
.hero p  { margin: 0; opacity: 0.9; }
.chat-user {
    background: rgba(102, 126, 234, 0.25);
    border: 1px solid rgba(102, 126, 234, 0.4);
    color: inherit;
    border-radius: 18px 18px 4px 18px;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0 0.5rem auto;
    max-width: 82%; width: fit-content;
    overflow-wrap: break-word; word-break: break-word;
}
.chat-ai {
    background: rgba(255,255,255,0.06);
    border-left: 4px solid #667eea;
    border-radius: 4px 18px 18px 18px;
    padding: 0.7rem 1rem;
    margin: 0.5rem auto 0.5rem 0;
    max-width: 92%; color: inherit;
    overflow-wrap: break-word; word-break: break-word;
    white-space: pre-wrap;
}
</style>""", unsafe_allow_html=True)

# ── backend URL ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# ── API client helpers ────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs) -> dict:
    """Call the backend API, raise on HTTP error, return JSON."""
    try:
        resp = getattr(requests, method)(f"{BACKEND_URL}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach backend at **{BACKEND_URL}**. "
            "Start the API server locally with `uvicorn api:app --reload` "
            "or set the correct BACKEND_URL."
        )
    except requests.exceptions.HTTPError as e:
        detail = str(e)
        try:
            body = e.response.json()
            detail = body.get("detail") or body.get("message") or str(e)
        except Exception:
            pass
        raise RuntimeError(f"Backend error: {detail}")


def api_extract(raw_bytes: bytes, filename: str, provider: str, use_docling: bool) -> dict:
    return _api("post", "/extract",
                files={"file": (filename, raw_bytes, "application/octet-stream")},
                data={"provider": provider, "use_docling": str(use_docling).lower()},
                timeout=120)


def api_ask(raw_bytes: bytes, filename: str, question: str,
            provider: str, use_docling: bool) -> dict:
    return _api("post", "/ask",
                files={"file": (filename, raw_bytes, "application/octet-stream")},
                data={"question": question, "provider": provider,
                      "use_docling": str(use_docling).lower()},
                timeout=120)


def api_evaluate(eval_pairs: list, provider: str) -> list:
    samples = [{"question": p["question"], "answer": p["answer"],
                "contexts": p["contexts"]} for p in eval_pairs]
    return _api("post", "/evaluate",
                json={"samples": samples, "provider": provider},
                timeout=300)


def api_kb_count() -> int:
    try:
        return _api("get", "/kb/count", timeout=10)["count"]
    except Exception:
        return 0


def api_kb_add(doc_id: str, text: str, metadata: dict):
    _api("post", "/kb/add",
         json={"doc_id": doc_id, "text": text, "metadata": metadata},
         timeout=60)


def api_kb_search(query: str) -> str:
    return _api("post", "/kb/search", json={"query": query}, timeout=30)["context"]


def api_health() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "raw_bytes": None,
    "image_bytes": None,   # PNG for display
    "filename": None,
    "extracted": None,
    "chat": [],
    "eval_pairs": [],
    "monitor": None,
    "provider": "groq",
    "use_docling": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.monitor is None:
    from backend.monitoring import CostMonitor
    st.session_state.monitor = CostMonitor()

monitor = st.session_state.monitor

# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(val) -> str:
    try:
        return f"${float(val):,.2f}"
    except Exception:
        return str(val) if val is not None else "N/A"


def _raw_to_display_png(raw: bytes, filename: str) -> bytes | None:
    """Convert raw file bytes to PNG for UI display."""
    if filename.lower().endswith(".pdf"):
        try:
            import fitz
            doc = fitz.open(stream=raw, filetype="pdf")
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
            return pix.tobytes("png")
        except Exception as e:
            st.error(f"PDF preview error: {e}")
            return None
    return raw


# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🤖 AI Financial Assistant</h1>
  <p>Upload financial documents · Extract key fields · Ask questions powered by Vision + RAG</p>
</div>""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Backend status
    st.header("Backend")
    backend_ok = api_health()
    if backend_ok:
        st.success(f"🟢 Connected  \n`{BACKEND_URL}`")
    else:
        st.error(f"🔴 Unreachable  \n`{BACKEND_URL}`")
        st.caption("Set `BACKEND_URL` in `.env` or start the API:\n`uvicorn api:app --reload`")

    st.divider()
    st.header("LLM Provider")
    provider_labels = {
        "groq": "Groq (free) — Llama 4 Scout",
        "anthropic": "Anthropic — Claude Sonnet 4.6",
    }
    selected = st.radio(
        "Choose provider",
        options=["groq", "anthropic"],
        format_func=lambda p: provider_labels[p],
        index=["groq", "anthropic"].index(st.session_state.provider),
    )
    st.session_state.provider = selected

    st.divider()
    st.header("About")
    st.markdown(
        "- **Backend**: FastAPI on Railway\n"
        "- **Frontend**: Streamlit Cloud\n"
        "- **Vector DB**: ChromaDB\n"
        "- **Evaluation**: Ragas\n"
        "- **Tracing**: LangSmith"
    )
    st.divider()
    if st.session_state.raw_bytes:
        st.success(f"📄 {st.session_state.filename}")
        if st.button("🗑️ Clear document", use_container_width=True):
            for k in ("raw_bytes", "image_bytes", "filename", "extracted"):
                st.session_state[k] = None
            st.session_state.chat = []
            st.session_state.eval_pairs = []
            st.session_state.use_docling = False
            st.rerun()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Document Analysis", "💬 Ask Questions",
    "📚 Knowledge Base", "📊 Monitoring", "📈 Evaluation"
])

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 — Document Analysis                                  ║
# ╚══════════════════════════════════════════════════════════════╝
with tab1:
    col_up, col_prev = st.columns([1, 1], gap="large")

    with col_up:
        st.subheader("Upload Financial Document")
        uploaded = st.file_uploader(
            "PNG · JPG · JPEG · WEBP · BMP · PDF",
            type=["png", "jpg", "jpeg", "webp", "bmp", "pdf"],
        )

        if uploaded:
            raw = uploaded.read()
            png = _raw_to_display_png(raw, uploaded.name)
            if png:
                st.session_state.raw_bytes = raw
                st.session_state.image_bytes = png
                st.session_state.filename = uploaded.name
                st.session_state.extracted = None
                st.success(f"Loaded: **{uploaded.name}**")

        if st.session_state.raw_bytes:
            is_pdf = (st.session_state.filename or "").lower().endswith(".pdf")
            if is_pdf:
                st.session_state.use_docling = st.toggle(
                    "Use Docling (text extraction, cheaper for text PDFs)",
                    value=st.session_state.use_docling,
                )

            label = "🔍 Extract with Docling" if st.session_state.use_docling else "🔍 Extract Document Fields"
            if st.button(label, type="primary", use_container_width=True, disabled=not backend_ok):
                with st.spinner("Sending to backend for analysis…"):
                    t0 = time.perf_counter()
                    try:
                        result = api_extract(
                            st.session_state.raw_bytes,
                            st.session_state.filename,
                            st.session_state.provider,
                            st.session_state.use_docling,
                        )
                        elapsed = (time.perf_counter() - t0) * 1000
                        st.session_state.extracted = result
                        if "_usage" in result:
                            monitor.record(
                                f"Extraction ({result.get('_method', 'vision')})",
                                result["_usage"], elapsed,
                            )
                        st.success(f"Done via **{result.get('_method', 'vision')}**!")
                    except Exception as exc:
                        st.error(str(exc))

    with col_prev:
        if st.session_state.image_bytes:
            st.subheader("Preview")
            st.image(Image.open(io.BytesIO(st.session_state.image_bytes)),
                     use_container_width=True)

    if st.session_state.extracted:
        st.divider()
        f = st.session_state.extracted
        st.subheader("📋 Extracted Fields")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Type", (f.get("document_type") or "—").replace("_", " ").title())
        m2.metric("Vendor", f.get("vendor_name") or "—")
        m3.metric("Date", f.get("billing_date") or "—")
        m4.metric("Total", _fmt(f.get("total_amount")))

        c1, c2, c3 = st.columns(3)
        c1.metric("Subtotal", _fmt(f.get("subtotal")))
        c2.metric("Tax",      _fmt(f.get("tax")))
        c3.metric("Fees",     _fmt(f.get("fees")))

        items = f.get("line_items") or []
        if items:
            st.subheader("Line Items")
            df = pd.DataFrame(items)
            if "amount" in df.columns:
                df["amount"] = df["amount"].apply(_fmt)
            st.dataframe(df, use_container_width=True, hide_index=True)

        if f.get("summary"):
            st.info(f"**Summary:** {f['summary']}")

        st.divider()
        if st.button("📚 Save to Knowledge Base", use_container_width=True, disabled=not backend_ok):
            with st.spinner("Saving…"):
                try:
                    api_kb_add(
                        doc_id=f"doc_{int(time.time())}",
                        text=json.dumps(f, indent=2),
                        metadata={"source": st.session_state.filename or "uploaded_doc"},
                    )
                    st.success("Saved! Future Q&A will use this as context.")
                except Exception as exc:
                    st.error(str(exc))


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 — Ask Questions                                      ║
# ╚══════════════════════════════════════════════════════════════╝
with tab2:
    if not st.session_state.raw_bytes:
        st.info("⬅️ Upload a document in **Document Analysis** first.")
    else:
        doc_col, chat_col = st.columns([5, 7], gap="large")

        with doc_col:
            st.subheader("Document")
            st.image(Image.open(io.BytesIO(st.session_state.image_bytes)),
                     use_container_width=True)
            if st.session_state.extracted:
                with st.expander("Quick facts"):
                    ef = st.session_state.extracted
                    st.write(f"**Type:** {ef.get('document_type','—')}")
                    st.write(f"**Vendor:** {ef.get('vendor_name','—')}")
                    st.write(f"**Total:** {_fmt(ef.get('total_amount'))}")
                    st.write(f"**Date:** {ef.get('billing_date','—')}")

        with chat_col:
            st.subheader("Ask About This Document")

            for msg in st.session_state.chat:
                safe = msg["content"].replace("$", "&#36;")
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">🧑 {safe}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-ai">🤖 {safe}</div>',
                                unsafe_allow_html=True)

            st.write("")
            q_cols = st.columns(2)
            quick_qs = ["Why was this charge deducted?", "What are all the line items?",
                        "Is this charge legitimate?", "Summarize this document"]
            pending_q = None
            for i, qtext in enumerate(quick_qs):
                if q_cols[i % 2].button(qtext, use_container_width=True, key=f"qbtn_{i}",
                                        disabled=not backend_ok):
                    pending_q = qtext

            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input("Your question",
                    placeholder="e.g. Why was $320.45 charged for AWS EC2?",
                    label_visibility="collapsed")
                submitted = st.form_submit_button("Send ▶", use_container_width=True,
                                                  disabled=not backend_ok)

            question = pending_q or (user_input.strip() if submitted and user_input.strip() else None)

            if question:
                st.session_state.chat.append({"role": "user", "content": question})
                with st.spinner("Asking backend…"):
                    t0 = time.perf_counter()
                    try:
                        result = api_ask(
                            st.session_state.raw_bytes,
                            st.session_state.filename,
                            question,
                            st.session_state.provider,
                            st.session_state.use_docling,
                        )
                        elapsed = (time.perf_counter() - t0) * 1000
                        st.session_state.chat.append(
                            {"role": "assistant", "content": result["answer"]})
                        st.session_state.eval_pairs.append({
                            "question": question,
                            "answer": result["answer"],
                            "contexts": [c for c in
                                         result.get("retrieved_context", "").split("\n\n")
                                         if c.strip()] or ["No context retrieved"],
                        })
                        if "_usage" in result:
                            monitor.record("Q&A", result["_usage"], elapsed)
                    except Exception as exc:
                        st.session_state.chat.append(
                            {"role": "assistant", "content": f"⚠️ {exc}"})
                st.rerun()

            if st.session_state.chat:
                if st.button("🗑️ Clear conversation", use_container_width=True):
                    st.session_state.chat = []
                    st.rerun()


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 3 — Knowledge Base                                     ║
# ╚══════════════════════════════════════════════════════════════╝
with tab3:
    st.subheader("📚 Billing Knowledge Base")
    st.write("Policies stored on the backend are retrieved automatically during Q&A.")

    kb_left, kb_right = st.columns(2, gap="large")

    with kb_left:
        st.write("**Built-in billing policies (loaded on backend startup):**")
        policies_path = Path("data/billing_policies")
        if policies_path.exists():
            for pf in sorted(policies_path.glob("*.txt")):
                with st.expander(f"📄 {pf.stem.replace('_',' ').title()}"):
                    st.text(pf.read_text(encoding="utf-8")[:600] + "\n…")
        st.metric("Total chunks in vector store", api_kb_count())

    with kb_right:
        st.write("**Add custom policy:**")
        title_in = st.text_input("Title", placeholder="e.g. Company Expense Policy 2025")
        body_in = st.text_area("Content", height=180)

        if st.button("➕ Add to Knowledge Base", use_container_width=True,
                     disabled=not backend_ok):
            if body_in.strip():
                with st.spinner("Sending to backend…"):
                    try:
                        api_kb_add(
                            doc_id=f"custom_{int(time.time())}",
                            text=body_in,
                            metadata={"source": title_in or "custom_policy",
                                      "type": "billing_policy"},
                        )
                        st.success("Added!")
                    except Exception as exc:
                        st.error(str(exc))
            else:
                st.warning("Please enter content first.")

        st.divider()
        st.write("**Test retrieval:**")
        test_q = st.text_input("Search query", placeholder="e.g. AWS EC2 charge")
        if st.button("🔍 Search", use_container_width=True,
                     disabled=not backend_ok) and test_q:
            with st.spinner("Searching…"):
                try:
                    ctx = api_kb_search(test_q)
                    st.text_area("Retrieved context:", ctx or "Nothing found.", height=200)
                except Exception as exc:
                    st.error(str(exc))


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 4 — Monitoring                                         ║
# ╚══════════════════════════════════════════════════════════════╝
with tab4:
    st.subheader("📊 Usage & Cost Monitoring")

    logs = monitor.logs
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Requests", len(logs))
    m2.metric("Total Cost", f"${monitor.total_cost:.6f}")
    m3.metric("Total Tokens", f"{monitor.total_tokens:,}")
    m4.metric("Avg Cost / Request",
              f"${monitor.total_cost / max(len(logs), 1):.6f}")

    if logs:
        st.divider()
        st.dataframe(pd.DataFrame(monitor.as_records()),
                     use_container_width=True, hide_index=True)

        if len(logs) > 1:
            st.divider()
            timestamps = [l.timestamp for l in logs]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Input Tokens", x=timestamps,
                                 y=[l.input_tokens for l in logs]))
            fig.add_trace(go.Bar(name="Output Tokens", x=timestamps,
                                 y=[l.output_tokens for l in logs]))
            fig.update_layout(barmode="stack", title="Token Usage per Request")
            st.plotly_chart(fig, use_container_width=True)

            cumulative, running = [], 0.0
            for l in logs:
                running += l.cost_usd
                cumulative.append(round(running, 8))
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=timestamps, y=cumulative,
                                      mode="lines+markers", name="Cumulative Cost"))
            fig2.update_layout(title="Cumulative Cost Over Session", yaxis_title="USD")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No requests yet. Upload a document and ask questions to see data.")

    with st.expander("💰 Model Pricing Reference"):
        st.table(pd.DataFrame({
            "Model": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001",
                      "llama-4-scout (Groq)", "llama-3.3-70b (Groq)"],
            "Input / 1M tokens": ["$3.00", "$0.25", "$0.11", "$0.59"],
            "Output / 1M tokens": ["$15.00", "$1.25", "$0.34", "$0.79"],
        }))


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 5 — Ragas Evaluation                                   ║
# ╚══════════════════════════════════════════════════════════════╝
with tab5:
    st.subheader("📈 Ragas Evaluation")
    st.write("Evaluate AI response quality after asking questions in the **Ask Questions** tab.")

    col_i1, col_i2 = st.columns(2)
    col_i1.info("**Faithfulness** — Is the answer grounded in the retrieved context? (0–1)")
    col_i2.info("**Response Relevancy** — Is the answer relevant to the question? (0–1)")

    pairs = st.session_state.eval_pairs
    st.metric("Q&A pairs available", len(pairs))

    if not pairs:
        st.warning("No Q&A pairs yet — go to **Ask Questions** and ask at least one question.")
    else:
        for i, p in enumerate(pairs):
            q_short = p["question"][:80] + "…" if len(p["question"]) > 80 else p["question"]
            with st.expander(f"Q{i+1}: {q_short}"):
                st.write(f"**Answer:** {p['answer'][:300]}")
                st.caption(f"{len(p['contexts'])} context chunk(s) retrieved")

        st.divider()
        if st.button("🚀 Run Ragas Evaluation", type="primary",
                     use_container_width=True, disabled=not backend_ok):
            with st.spinner(f"Evaluating {len(pairs)} pair(s)… this may take a minute."):
                try:
                    records = api_evaluate(pairs, st.session_state.provider)
                    st.success("Evaluation complete!")
                    df_eval = pd.DataFrame(records)
                    for col in df_eval.select_dtypes("float").columns:
                        df_eval[col] = df_eval[col].round(3)
                    st.dataframe(df_eval, use_container_width=True, hide_index=True)

                    score_cols = [c for c in df_eval.columns
                                  if c not in ("user_input", "response",
                                               "retrieved_contexts", "reference")]
                    if score_cols:
                        st.divider()
                        st.subheader("Average Scores")
                        avg_cols = st.columns(len(score_cols))
                        for i, sc in enumerate(score_cols):
                            avg = df_eval[sc].mean()
                            icon = "🟢" if avg >= 0.7 else "🟡" if avg >= 0.4 else "🔴"
                            avg_cols[i].metric(f"{icon} {sc}", f"{avg:.3f}")

                    with st.expander("Score guide"):
                        st.markdown(
                            "| Range | Quality |\n|---|---|\n"
                            "| 0.8–1.0 | Excellent |\n| 0.6–0.8 | Good |\n"
                            "| 0.4–0.6 | Moderate |\n| 0.0–0.4 | Poor |"
                        )
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

        if st.button("🗑️ Clear pairs", use_container_width=True):
            st.session_state.eval_pairs = []
            st.rerun()
