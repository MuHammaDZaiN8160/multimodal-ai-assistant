import base64
import io
import json
import os
import tempfile

from PIL import Image
from langsmith import traceable

from config.settings import (
    ANTHROPIC_API_KEY,
    GROQ_API_KEY,
    ANTHROPIC_VISION_MODEL,
    ANTHROPIC_FAST_MODEL,
    GROQ_VISION_MODEL,
    GROQ_FAST_MODEL,
)

# ── shared helpers ────────────────────────────────────────────────────────────

def _encode_image(image_bytes: bytes) -> tuple[str, str]:
    img = Image.open(io.BytesIO(image_bytes))
    buf = io.BytesIO()
    fmt = img.format or "PNG"
    if fmt not in ("PNG", "JPEG", "WEBP", "GIF"):
        fmt = "PNG"
    img.save(buf, format=fmt)
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    return b64, f"image/{fmt.lower()}"


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        try:
            return json.loads(raw[start:end]) if start != -1 else {}
        except Exception:
            return {"summary": raw, "line_items": []}


_EXTRACTION_PROMPT = """You are a financial document analyzer. Extract all information from this document into the following JSON format. Use null for any field not visible.

{
  "document_type": "credit_card_statement | invoice | receipt | expense_report",
  "vendor_name": "...",
  "billing_date": "YYYY-MM-DD or as shown",
  "account_number": "last 4 digits only if visible",
  "line_items": [
    {"date": "...", "description": "...", "amount": 0.00}
  ],
  "subtotal": 0.00,
  "tax": 0.00,
  "fees": 0.00,
  "total_amount": 0.00,
  "currency": "USD",
  "previous_balance": 0.00,
  "payments_received": 0.00,
  "summary": "One-sentence description of this document and its main charges."
}

Return ONLY the JSON object with no additional text."""

_QA_SYSTEM = (
    "You are an expert financial document assistant. "
    "Analyze the provided document carefully. "
    "Give precise, grounded answers referencing specific line items, amounts, and dates. "
    "If billing policy context is provided, incorporate it to explain charges more fully. "
    "Be concise and factual."
)


# ── Anthropic (vision) ────────────────────────────────────────────────────────

@traceable(run_type="llm", name="anthropic_extract_fields")
def _anthropic_extract(image_bytes: bytes) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    b64, media_type = _encode_image(image_bytes)
    resp = client.messages.create(
        model=ANTHROPIC_VISION_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": _EXTRACTION_PROMPT},
        ]}],
    )
    data = _parse_json(resp.content[0].text)
    data["_usage"] = {"model": ANTHROPIC_VISION_MODEL,
                      "input_tokens": resp.usage.input_tokens,
                      "output_tokens": resp.usage.output_tokens}
    data["_method"] = "anthropic-vision"
    return data


@traceable(run_type="llm", name="anthropic_answer_question")
def _anthropic_qa(image_bytes: bytes, question: str, context: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    b64, media_type = _encode_image(image_bytes)
    user_text = (f"Relevant context:\n\n{context}\n\n---\n\nQuestion: {question}"
                 if context else f"Question: {question}")
    resp = client.messages.create(
        model=ANTHROPIC_VISION_MODEL, max_tokens=1500, system=_QA_SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": user_text},
        ]}],
    )
    return {"answer": resp.content[0].text,
            "_usage": {"model": ANTHROPIC_VISION_MODEL,
                       "input_tokens": resp.usage.input_tokens,
                       "output_tokens": resp.usage.output_tokens}}


# ── Groq (vision) ─────────────────────────────────────────────────────────────

@traceable(run_type="llm", name="groq_extract_fields")
def _groq_extract(image_bytes: bytes) -> dict:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    b64, media_type = _encode_image(image_bytes)
    resp = client.chat.completions.create(
        model=GROQ_VISION_MODEL, max_tokens=2048,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            {"type": "text", "text": _EXTRACTION_PROMPT},
        ]}],
    )
    data = _parse_json(resp.choices[0].message.content)
    data["_usage"] = {"model": GROQ_VISION_MODEL,
                      "input_tokens": resp.usage.prompt_tokens,
                      "output_tokens": resp.usage.completion_tokens}
    data["_method"] = "groq-vision"
    return data


@traceable(run_type="llm", name="groq_answer_question")
def _groq_qa(image_bytes: bytes, question: str, context: str) -> dict:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    b64, media_type = _encode_image(image_bytes)
    user_text = (f"Relevant context:\n\n{context}\n\n---\n\nQuestion: {question}"
                 if context else f"Question: {question}")
    resp = client.chat.completions.create(
        model=GROQ_VISION_MODEL, max_tokens=1500,
        messages=[
            {"role": "system", "content": _QA_SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": user_text},
            ]},
        ],
    )
    return {"answer": resp.choices[0].message.content,
            "_usage": {"model": GROQ_VISION_MODEL,
                       "input_tokens": resp.usage.prompt_tokens,
                       "output_tokens": resp.usage.completion_tokens}}


# ── Docling (PDF text extraction → LLM) ──────────────────────────────────────

def _docling_extract_text(raw_bytes: bytes, filename: str) -> str:
    """Extract structured markdown text from a PDF using Docling."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError("Docling not installed. Run: pip install docling")

    suffix = os.path.splitext(filename)[1].lower() or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw_bytes)
        tmp_path = f.name

    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        return result.document.export_to_markdown()
    finally:
        os.unlink(tmp_path)


@traceable(run_type="chain", name="docling_extract_fields")
def _docling_extract(raw_bytes: bytes, filename: str, provider: str) -> dict:
    """Docling text extraction → LLM field parsing (no vision, cheaper for text PDFs)."""
    doc_text = _docling_extract_text(raw_bytes, filename)
    prompt = f"{_EXTRACTION_PROMPT}\n\nDocument content:\n\n{doc_text}"

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_FAST_MODEL, max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content
        usage = {"model": GROQ_FAST_MODEL,
                 "input_tokens": resp.usage.prompt_tokens,
                 "output_tokens": resp.usage.completion_tokens}
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=ANTHROPIC_FAST_MODEL, max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text
        usage = {"model": ANTHROPIC_FAST_MODEL,
                 "input_tokens": resp.usage.input_tokens,
                 "output_tokens": resp.usage.output_tokens}

    data = _parse_json(raw)
    data["_usage"] = usage
    data["_method"] = f"docling+{provider}-text"
    data["_docling_text"] = doc_text[:500] + "…" if len(doc_text) > 500 else doc_text
    return data


@traceable(run_type="chain", name="docling_answer_question")
def _docling_qa(raw_bytes: bytes, filename: str, question: str,
                context: str, provider: str) -> dict:
    """Docling text extraction → LLM Q&A (no vision)."""
    doc_text = _docling_extract_text(raw_bytes, filename)
    user_text = (
        f"Document content:\n\n{doc_text}\n\n"
        + (f"Relevant billing policy context:\n\n{context}\n\n" if context else "")
        + f"Question: {question}"
    )

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_FAST_MODEL, max_tokens=1500,
            messages=[{"role": "system", "content": _QA_SYSTEM},
                      {"role": "user", "content": user_text}],
        )
        return {"answer": resp.choices[0].message.content,
                "_usage": {"model": GROQ_FAST_MODEL,
                           "input_tokens": resp.usage.prompt_tokens,
                           "output_tokens": resp.usage.completion_tokens}}
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=ANTHROPIC_FAST_MODEL, max_tokens=1500, system=_QA_SYSTEM,
            messages=[{"role": "user", "content": user_text}],
        )
        return {"answer": resp.content[0].text,
                "_usage": {"model": ANTHROPIC_FAST_MODEL,
                           "input_tokens": resp.usage.input_tokens,
                           "output_tokens": resp.usage.output_tokens}}


# ── public API ────────────────────────────────────────────────────────────────

def extract_document_fields(
    image_bytes: bytes,
    provider: str = "anthropic",
    raw_bytes: bytes = None,
    filename: str = "",
    use_docling: bool = False,
) -> dict:
    """Extract structured fields from a financial document image."""
    if use_docling and raw_bytes:
        return _docling_extract(raw_bytes, filename, provider)
    if provider == "groq":
        return _groq_extract(image_bytes)
    return _anthropic_extract(image_bytes)


def answer_document_question(
    image_bytes: bytes,
    question: str,
    context: str = "",
    provider: str = "anthropic",
    raw_bytes: bytes = None,
    filename: str = "",
    use_docling: bool = False,
) -> dict:
    """Answer a user question about a financial document."""
    if use_docling and raw_bytes:
        return _docling_qa(raw_bytes, filename, question, context, provider)
    if provider == "groq":
        return _groq_qa(image_bytes, question, context)
    return _anthropic_qa(image_bytes, question, context)
