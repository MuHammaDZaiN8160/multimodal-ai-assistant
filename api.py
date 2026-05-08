"""
FastAPI backend — deployed to Railway.
Exposes document extraction, Q&A, knowledge-base management, and Ragas evaluation
as REST endpoints consumed by the Streamlit frontend.
"""

import os
import sys
import time

import fitz  # PyMuPDF
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── make backend/ and config/ importable ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

app = FastAPI(title="AI Financial Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _pdf_to_png(raw: bytes) -> bytes:
    doc = fitz.open(stream=raw, filetype="pdf")
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
    return pix.tobytes("png")


def _to_image_bytes(raw: bytes, filename: str, use_docling: bool) -> bytes:
    """Return PNG image bytes (convert PDF if needed and not using Docling)."""
    if filename.lower().endswith(".pdf") and not use_docling:
        return _pdf_to_png(raw)
    return raw


# ── health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── document extraction ────────────────────────────────────────────────────────

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    provider: str = Form("groq"),
    use_docling: bool = Form(False),
):
    raw = await file.read()
    image_bytes = _to_image_bytes(raw, file.filename or "", use_docling)

    from backend.document_processor import extract_document_fields

    try:
        result = extract_document_fields(
            image_bytes=image_bytes,
            provider=provider,
            raw_bytes=raw,
            filename=file.filename or "",
            use_docling=use_docling,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


# ── Q&A ────────────────────────────────────────────────────────────────────────

@app.post("/ask")
async def ask(
    file: UploadFile = File(...),
    question: str = Form(...),
    provider: str = Form("groq"),
    use_docling: bool = Form(False),
):
    try:
        raw = await file.read()
        image_bytes = _to_image_bytes(raw, file.filename or "", use_docling)

        from backend.document_processor import answer_document_question
        from backend.vector_store import get_knowledge_base

        context = get_knowledge_base().retrieve_context(question)
        result = answer_document_question(
            image_bytes=image_bytes,
            question=question,
            context=context,
            provider=provider,
            raw_bytes=raw,
            filename=file.filename or "",
            use_docling=use_docling,
        )
        result["retrieved_context"] = context
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Ragas evaluation ───────────────────────────────────────────────────────────

class EvalPayload(BaseModel):
    samples: list[dict]
    provider: str = "groq"


@app.post("/evaluate")
def evaluate(payload: EvalPayload):
    from backend.evaluation import EvalSample, run_ragas_evaluation

    samples = [EvalSample(**s) for s in payload.samples]
    try:
        return run_ragas_evaluation(samples, provider=payload.provider)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── knowledge base ─────────────────────────────────────────────────────────────

@app.get("/kb/count")
def kb_count():
    from backend.vector_store import get_knowledge_base
    return {"count": get_knowledge_base().collection_count()}


class KBAddPayload(BaseModel):
    doc_id: str
    text: str
    metadata: dict = {}


@app.post("/kb/add")
def kb_add(payload: KBAddPayload):
    from backend.vector_store import get_knowledge_base
    get_knowledge_base().add_document_context(
        doc_id=payload.doc_id,
        text=payload.text,
        metadata=payload.metadata,
    )
    return {"status": "added"}


class KBSearchPayload(BaseModel):
    query: str
    n_results: int = 5


@app.post("/kb/search")
def kb_search(payload: KBSearchPayload):
    from backend.vector_store import get_knowledge_base
    context = get_knowledge_base().retrieve_context(
        payload.query, n_results=payload.n_results
    )
    return {"context": context}


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
