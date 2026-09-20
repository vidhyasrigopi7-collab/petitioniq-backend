from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import fitz
import requests
import json
import re
import os
import uuid
import chromadb
from llm_providers import get_llm_provider, set_runtime_provider, get_runtime_settings
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
def _active_embed_label():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1)
        if r.status_code == 200:
            return f"ollama:{EMBED_MODEL}"
    except Exception:
        pass
    return "local-vector-rag"
def _active_model_label():  
    provider = get_runtime_settings()["provider"]
    model = {
        "claude": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        "groq": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "gemini": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    }.get(provider, MODEL)
    return f"{provider}:{model}"

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_store")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
DOCS = {}
LAWBOOK_COLLECTION_NAME = "lawbook_corpus"
lawbook_collection = chroma_client.get_or_create_collection(name=LAWBOOK_COLLECTION_NAME)
LAWBOOKS_META_PATH = os.path.join(CHROMA_PATH, "lawbooks_meta.json")
def _load_lawbooks_meta():
    if os.path.exists(LAWBOOKS_META_PATH):
        try:
            with open(LAWBOOKS_META_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []
def _save_lawbooks_meta(meta):
    os.makedirs(CHROMA_PATH, exist_ok=True)
    with open(LAWBOOKS_META_PATH, "w") as f:
        json.dump(meta, f)
LAWBOOKS_META = _load_lawbooks_meta()
@app.get("/api")
def root():
    return {"api": "ok", "model": MODEL, "embed_model": EMBED_MODEL}
@app.get("/health")
def health():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = [x["name"] for x in r.json().get("models", [])]
        def is_installed(name, model_list):
            return any(m == name or m.startswith(name + ":") for m in model_list)
        return {
            "api": "ok", "ollama": "ok", "model": MODEL,
            "model_installed": is_installed(MODEL, models),
            "embed_model": EMBED_MODEL,
            "embed_model_installed": is_installed(EMBED_MODEL, models),
            "available_models": models
        }
    except Exception as e:
        return {"api": "ok", "ollama": "error", "model": MODEL, "model_installed": False, "error": str(e)}
def extract_pdf(data):
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page_no, page in enumerate(doc, start=1):
        txt = page.get_text("text")
        if txt.strip():
            pages.append(f"\n--- PAGE {page_no} ---\n{txt}")
    doc.close()
    return "\n".join(pages)
def clean(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
def chunk_text(text, chunk_size=2000, overlap=150):
    """Splits text into overlapping chunks so context isn't lost at boundaries."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks
import hashlib
import math
def _local_fallback_embed(text, dim=384):
    words = re.findall(r'\w+', text.lower())
    vec = [0.0] * dim
    if not words:
        return vec
    for w in words:
        h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        vec[idx] += sign
    for i in range(len(words) - 1):
        bg = words[i] + "_" + words[i+1]
        h = int(hashlib.md5(bg.encode('utf-8')).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        vec[idx] += sign * 1.5
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec
def embed_text(text):
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=10
        )
        if r.status_code == 200:
            return r.json().get("embedding")
    except Exception:
        pass
    return _local_fallback_embed(text)
def embed_chunks(chunks):
    return [embed_text(c) for c in chunks]
FIELD_QUERIES = [
    {"query": "Court name, court number, judge, coram, case number, document type", "top_k": 3},
    {"query": "Petitioner, respondent, applicant, appellant, accused, versus, all parties named in the case including multiple petitioners", "top_k": 4},
    {"query": "Advocate or counsel appearing for the petitioner or respondent", "top_k": 3},
    {"query": "Acts and sections cited, offence sections, under section, charged under, IPC, CrPC, BNS, BNSS", "top_k": 4},
    {"query": "Prayer, relief sought, wherefore it is prayed, grant bail, common prayer", "top_k": 3},
    {"query": "Key legal issues, arguments and legal summary discussed in the document", "top_k": 3},
]
def build_collection(full_text):
    """Chunks + embeds the document and stores it in a fresh Chroma collection.
    The collection is kept alive (not deleted) so /chat can query it later."""
    chunks = chunk_text(full_text)
    if not chunks:
        chunks = [full_text[:2000]]
    embeddings = embed_chunks(chunks)
    collection_name = f"doc_{uuid.uuid4().hex}"
    collection = chroma_client.create_collection(name=collection_name)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )
    return collection, collection_name
def retrieve(collection, query, top_k=4):
    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0] if results["documents"] else []
def retrieve_lawbooks(query, top_k=4):
    """Semantic search over the persistent law-book/external-source knowledge base."""
    count = lawbook_collection.count()
    if count == 0:
        return []
    query_embedding = embed_text(query)
    results = lawbook_collection.query(
        query_embeddings=[query_embedding], n_results=min(top_k, count)
    )
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    return [{"text": d, "title": m.get("title", "Unknown source")} for d, m in zip(docs, metas)]
def format_law_evidence(items, char_limit=6000):
    if not items:
        return ""
    blocks = [f"[Source: {it['title']}]\n{it['text']}" for it in items]
    return "\n\n---\n\n".join(blocks)[:char_limit]
def build_evidence(collection, all_docs):
    retrieved = [all_docs[0]]  # anchor: doc start (title/case header)
    for field in FIELD_QUERIES:
        retrieved.extend(retrieve(collection, field["query"], top_k=field["top_k"]))
    retrieved.append(all_docs[-1])  # anchor: doc end (signatures/final prayer)
    unique = []
    for chunk in retrieved:
        if chunk not in unique:
            unique.append(chunk)
    return "\n\n---\n\n".join(unique)[:8000]
def analyze_with_llama(evidence):
    prompt = f"""
You are an expert Indian legal-document analyst.
You are given evidence retrieved from a PDF using semantic search.
The evidence chunks are separated by "---" and may be out of order.
Your job is to extract ONLY information actually present in the evidence.
Do NOT guess.
Do NOT invent.
Do NOT use general legal knowledge to fill missing fields.
IMPORTANT:
The PDF may be a petition, application, judgment, order, or legal article.
If a field is present, return its actual value.
If a field is absent, return:
"Not Found in the Document"
Do NOT confuse:
- article author with advocate
- cited case with petitioner/respondent
- legal discussion with prayer
- section explanation with section number
For Sections, return ONLY section numbers that literally appear in the
evidence next to words like "Section", "u/s", "under Section", or "Sec.".
Copy them exactly as written. If multiple different sections are mentioned
in different places (e.g. offence sections AND a procedural section used
to file this specific petition), include all of them, comma-separated.
Never output a section number that does not literally appear in the
evidence text.
Example:
"389, 438, 439"
NOT:
"389 of the Criminal Procedure Code deals with..."
For Acts, return actual Act names only.
For Prayer, return the actual relief requested.
If the document contains a prayer such as:
"it is therefore prayed that..."
extract the complete meaningful prayer.
For Petitioner and Respondent, look around:
Petitioner / Respondent / Vs / Versus / Applicant / State.
If there are MULTIPLE petitioners (e.g. a common order covering several
connected petitions/appeals, or petitioners listed as A1, A2, A3 etc.),
list ALL of their names, separated by "; ". Do the same for multiple
respondents. Do not return only the first one found.
For Advocate, look around:
Advocate / Counsel / For the Petitioner / For the Respondent.
Return ONLY valid JSON.
FORMAT:
{{
  "document_type": "",
  "case_number": "",
  "court_name": "",
  "court_number": "",
  "petitioner": "",
  "respondent": "",
  "advocate": "",
  "acts": "",
  "sections": "",
  "case_type": "",
  "prayer": "",
  "legal_summary": "",
  "key_legal_issues": ""
}}
Legal Summary:
Write 5-7 sentences based ONLY on the document.
Key Legal Issues:
Give the actual legal issues discussed.
RETRIEVED EVIDENCE FROM PDF:
{evidence}
"""
    return get_llm_provider().generate_json(prompt, num_predict=2000)
@app.post("/analyze")
async def analyze(petition: UploadFile = File(...)):
    if not petition.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    data = await petition.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty PDF.")
    try:
        full_text = clean(extract_pdf(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")
    if not full_text:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")
    return _analyze_petition_from_text(full_text, petition.filename)
def _analyze_petition_from_text(full_text, filename):
    """Core of /analyze, reusable so other endpoints (e.g. the combined
    /documents/handle flow) don't need a second HTTP round-trip."""
    chunks = chunk_text(full_text)
    if not chunks:
        chunks = [full_text[:2000]]
    collection, collection_name = build_collection(full_text)
    evidence = build_evidence(collection, chunks)
    result = analyze_with_llama(evidence)
    fields = [
        "document_type", "case_number", "court_name", "court_number", "petitioner",
        "respondent", "advocate", "acts", "sections", "case_type", "prayer",
        "legal_summary", "key_legal_issues"
    ]
    for field in fields:
        if field not in result or result[field] in (None, ""):
            result[field] = "Not Found in the Document"
    doc_id = uuid.uuid4().hex
    DOCS[doc_id] = {
        "collection_name": collection_name,
        "filename": filename,
        "details": result,
    }
    return {
        "doc_id": doc_id,"filename": filename,
        "summary": result["legal_summary"],
        "details": result,
        "text_preview": full_text[:5000],
        "full_text_length": len(full_text),
        "llm_model": _active_model_label(),
        "embed_model": _active_embed_label()
    }
def _ingest_lawbook(data, filename, title=""):
    """Core of /lawbooks/upload, reusable so the combined /documents/handle
    flow can ingest a reference PDF without a second HTTP round-trip."""
    try:
        full_text = clean(extract_pdf(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")
    if not full_text:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")
    chunks = chunk_text(full_text, chunk_size=1500, overlap=150)
    if not chunks:
        chunks = [full_text[:1500]]
    embeddings = embed_chunks(chunks)
    source_id = uuid.uuid4().hex
    display_title = (title or "").strip() or filename
    ids = [f"{source_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source_id": source_id, "title": display_title, "chunk": i}
        for i in range(len(chunks))
    ]
    lawbook_collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    entry = {
        "source_id": source_id,
        "title": display_title,
        "filename": filename,
        "chunks": len(chunks),
    }
    LAWBOOKS_META.append(entry)
    _save_lawbooks_meta(LAWBOOKS_META)
    return entry, full_text
@app.post("/lawbooks/upload")
async def upload_lawbook(source: UploadFile = File(...), title: str = ""):
    """Ingests a law book / bare act / judgment PDF into the persistent
    reference knowledge base, separate from any single petition's document
    collection. Chat and counter-petition drafting can both cite this."""
    if not source.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    data = await source.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty PDF.")

    entry, _ = _ingest_lawbook(data, source.filename, title)
    return entry
def _heuristic_is_petition(text):
    t = text[:6000].lower()
    signals = [
        "petitioner", "respondent", "versus", " vs ", "in the court of", "cr.m.p",
        "crl.m.p", "writ petition", "criminal appeal", "prayed that", "sworn affidavit",
        "applicant", "learned counsel", "learned advocate",
    ]
    hits = sum(1 for kw in signals if kw in t)
    return hits >= 2
def _classify_document(full_text):
    """Fast, accurate heuristic classification: distinguishes statutes/bare acts from court petitions."""
    sample = full_text[:5000].lower()
    statute_keywords = [
        "bare act", "an act to", "short title and commencement", "be it enacted",
        "chapter i", "chapter ii", "section 1.", "the indian penal code",
        "the code of criminal procedure", "the bharatiya nyaya", "the bharatiya nagarik",
        "commentary on", "all india reporter", "supreme court cases", "act no."
    ]
    if any(k in sample for k in statute_keywords):
        if not ("versus" in sample and ("crl.o.p" in sample or "crl.m.p" in sample or "in the high court" in sample)):
            return "reference"
    petition_keywords = [
        "in the high court", "before the hon'ble", "in the court of",
        "crl.o.p", "crl.m.p", "writ petition", "w.p. no", "criminal appeal",
        "sworn affidavit", "wherefore it is prayed", "most respectfully prayed",
        "petitioner", "respondent"
    ]
    if sum(1 for k in petition_keywords if k in sample) >= 3:
        return "petition"
    return "petition" if _heuristic_is_petition(full_text) else "reference"
def _summarize_reference(full_text, title):
    sample = full_text[:10000]
    prompt = f"""You are a legal research assistant. The user just uploaded a reference
document titled "{title}". Using ONLY the text below:
1. Write a concise summary (4-6 sentences) of what this document covers.
2. End with one short follow-up question inviting the user to say which section, topic, or
   point they'd like explained next.
Do not invent content that isn't present in the text.
DOCUMENT TEXT:
{sample}
"""
    return get_llm_provider().generate_text(prompt, num_predict=600)
def _answer_reference_question(full_text, question, title):
    sample = full_text[:10000]
    prompt = f"""You are a legal research assistant answering a question about a document
titled "{title}", using ONLY the text below. If the answer isn't in the text, say so plainly instead of guessing. Be concise and direct.
DOCUMENT TEXT:
{sample}
Question: {question}
Answer:"""
    return get_llm_provider().generate_text(prompt, num_predict=800)
@app.post("/documents/handle")
async def handle_document(
    file: UploadFile = File(...),
    question: str = Form(""),
    doc_type: str = Form("auto"),
):
    """Processes uploaded PDF according to doc_type ('petition', 'reference', or 'auto')."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty PDF.")
    try:
        full_text = clean(extract_pdf(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")
    if not full_text:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")
    d_type = (doc_type or "auto").strip().lower()
    if d_type in ("petition", "court_petition"):
        kind = "petition"
    elif d_type in ("reference", "lawbook", "law_book"):
        kind = "reference"
    else:
        kind = _classify_document(full_text)
    q = (question or "").strip()
    if kind == "petition":
        analysis = _analyze_petition_from_text(full_text, file.filename)
        counter = _generate_counter_for_doc(analysis["doc_id"], q)
        return {"kind": "petition", "analysis": analysis, "counter": counter}
    # Reference / law-book upload: treated as a standalone document for this
    # session only — it answers questions purely from its own text and is
    # NOT added to any shared/persistent corpus.
    title = os.path.splitext(file.filename)[0]
    reference_doc_id = uuid.uuid4().hex
    DOCS[reference_doc_id] = {
        "kind": "reference",
        "filename": file.filename,
        "title": title,
        "full_text": full_text,
    }
    if q:
        answer = _answer_reference_question(full_text, q, title)
    else:
        answer = _summarize_reference(full_text, title)
    return {
        "kind": "reference",
        "doc_id": reference_doc_id,
        "title": title,
        "answer": answer,
    }
@app.get("/lawbooks/list")
def list_lawbooks():
    return {"sources": LAWBOOKS_META}
@app.delete("/lawbooks/{source_id}")
def delete_lawbook(source_id: str):
    global LAWBOOKS_META
    if not any(m["source_id"] == source_id for m in LAWBOOKS_META):
        raise HTTPException(status_code=404, detail="Source not found.")
    lawbook_collection.delete(where={"source_id": source_id})
    LAWBOOKS_META = [m for m in LAWBOOKS_META if m["source_id"] != source_id]
    _save_lawbooks_meta(LAWBOOKS_META)
    return {"deleted": source_id}
class SettingsRequest(BaseModel):
    provider: str
    api_key: str = ""
@app.get("/settings")
def get_settings():
    return get_runtime_settings()
@app.post("/settings")
def update_settings(req: SettingsRequest):
    provider = req.provider.lower()
    if provider not in ("ollama", "claude", "groq", "gemini"):
        raise HTTPException(status_code=400, detail="Unknown provider.")
    set_runtime_provider(provider, req.api_key or None)
    return get_runtime_settings()
class ChatRequest(BaseModel):
    doc_id: str
    question: str
    history: list[dict] = []
STRATEGY_PATTERNS = [
    r"\bhow (can|do|to|could|would|possible|.*possible).{0,25}\bwin\b",
    r"\bwin.{0,20}\bcase\b",
    r"\bwinning\b",
    r"\bchances? of (winning|success)\b",
    r"\b(strong|weak) point",
    r"\bconvert.{0,20}(weak|strong)",
    r"\blegal strategy\b",
    r"\bhow to (argue|defend|fight)\b",
    r"\bbest (argument|defense|strategy)\b",
    r"\bwill (i|we|petitioner|they).{0,15}\bwin\b",
]
STRATEGY_REFUSAL = (
    "I can only summarize what's actually written in this document — I can't predict how "
    "a case will turn out, rate how strong or weak it is, or suggest legal strategy or "
    "arguments. Questions about strategy or the likely outcome need a licensed advocate. "
    "I'm happy to tell you what the document itself says — for example, what either side "
    "argued, or what the court observed."
)
@app.post("/chat")
async def chat(req: ChatRequest):
    doc = DOCS.get(req.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found. Please analyze the PDF again.")
    if doc.get("kind") == "reference":
        q = req.question.strip()
        if not q:
            raise HTTPException(status_code=400, detail="Please enter a question.")

        history_text = ""
        for turn in req.history[-4:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            history_text += f"{role}: {turn.get('content', '')}\n"
        # Answer purely from this specific uploaded document's own text —
        # no shared/persistent law corpus involved.
        doc_text = doc.get("full_text", "")[:8000]
        title = doc.get("title", doc.get("filename", "the document"))
        prompt = f"""You are a legal research assistant answering questions about a document
titled "{title}", using ONLY the text below. If the answer isn't in the text, say so plainly
instead of guessing. Be direct and concise.

IMPORTANT: Answer only about "{title}" itself. The conversation history below may mention an
unrelated court petition or case from earlier in this session — do NOT connect, compare, or
draw any conclusion linking this document to that (or any other) case unless the user's
CURRENT question explicitly asks you to relate it to that specific case. If the question is
simply asking you to explain a provision, just explain the provision.

DOCUMENT TEXT:
{doc_text}
CONVERSATION (for context only — do not assume it's about the same subject as the current question):
{history_text}
User Question: {q}
Answer:"""
        answer = get_llm_provider().generate_text(prompt, num_predict=500)
        return {"answer": answer, "kind": "reference"}
    q_lower = req.question.lower()
    if any(re.search(p, q_lower) for p in STRATEGY_PATTERNS):
        return {"answer": STRATEGY_REFUSAL}
    collection = chroma_client.get_collection(name=doc["collection_name"])
    relevant_chunks = retrieve(collection, req.question, top_k=5)
    evidence = "\n\n---\n\n".join(relevant_chunks)[:10000]
    law_items = retrieve_lawbooks(req.question, top_k=4)
    law_evidence = format_law_evidence(law_items)
    law_block = (
        f"\n\nEXTERNAL LEGAL REFERENCE MATERIAL (law books / bare acts uploaded separately — "
        f"you may use this to explain what a cited section/act says, and MUST mention the "
        f"[Source: ...] title when you rely on it; never invent section text not shown here):\n"
        f"{law_evidence}\n"
        if law_evidence else ""
    )
    history_text = ""
    for turn in req.history[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        history_text += f"{role}: {turn.get('content', '')}\n"
    prompt = f"""You are a legal assistant answering questions about a specific court document
("{doc['filename']}"). Answer ONLY using the evidence below (the document, and the external
legal reference material if provided). If the answer is not in the evidence, say you cannot
find that information. Be concise (2-5 sentences) and do not invent facts.
You NEVER predict case outcomes, rate how "strong" or "weak" a case is, or advise on legal
strategy. You CAN factually summarize arguments, evidence, or reasoning that the document
itself already contains (e.g. "what did the Government Advocate argue" or "what did the
court say about X"), and you CAN explain what an external act/section says if it's present
in the reference material below — that is reporting, not giving advice.
DOCUMENT EVIDENCE:
{evidence}
{law_block}
CONVERSATION SO FAR:
{history_text}
User's new question: {req.question}
Answer:"""
    answer = get_llm_provider().generate_text(prompt, num_predict=1000)
    return {"answer": answer, "law_sources_used": [it["title"] for it in law_items]}
COUNTER_QUERIES = [
    {"query": "Petitioner's allegations, claims, and facts stated against the respondent", "top_k": 4},
    {"query": "Prayer, relief sought, wherefore it is prayed", "top_k": 3},
    {"query": "Acts and sections cited, under section, charged under", "top_k": 3},
    {"query": "Facts, background, chronology of events narrated in the petition", "top_k": 4},
]
def build_counter_evidence(collection, all_docs):
    retrieved = [all_docs[0]]
    for q in COUNTER_QUERIES:
        retrieved.extend(retrieve(collection, q["query"], top_k=q["top_k"]))
    retrieved.append(all_docs[-1])
    unique = []
    for c in retrieved:
        if c not in unique:
            unique.append(c)
    return "\n\n---\n\n".join(unique)[:8000]
class CounterRequest(BaseModel):
    doc_id: str
    extra_instructions: str = ""
def _generate_counter_for_doc(doc_id, extra_instructions=""):
    doc = DOCS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found. Please analyze the PDF again.")
    collection = chroma_client.get_collection(name=doc["collection_name"])
    stored = collection.get()
    all_chunks = stored["documents"] if stored else []
    if not all_chunks:
        raise HTTPException(status_code=400, detail="No stored content for this document.")
    evidence = build_counter_evidence(collection, all_chunks)
    details = doc["details"]

    prompt = f"""You are an expert Indian legal drafter. Draft a formal COUNTER PETITION /
COUNTER AFFIDAVIT in reply to the petition described below, from the perspective of the
Respondent(s) opposing the Petitioner's claims.

Use ONLY facts, sections, and case details that actually appear in the evidence. Do NOT
invent case numbers, dates, or facts not present.

Placeholders — use them SPARINGLY: only for a genuinely missing fact that cannot be inferred
(e.g. the Respondent advocate's name/address). Never repeat the same generic placeholder or
denial sentence across multiple paragraphs. Each numbered reply must respond to the SPECIFIC
allegation in that paragraph — vary the wording (admit / deny / not aware — put to strict
proof / deny for want of knowledge) based on what that particular allegation actually says,
instead of reusing one boilerplate sentence everywhere.

Structure the draft using standard Indian court format:
1. Cause title (Court name, case number, Petitioner vs Respondent, styled as the
   Respondent's counter)
2. Preliminary objections (only if maintainability/limitation points are evident from the
   evidence)
3. Para-wise reply: for each major allegation in the petition, a corresponding numbered
   reply that specifically addresses that allegation's own facts
4. Statement of facts from the Respondent's side (a single placeholder here is enough if the
   source petition is silent — do not repeat it paragraph by paragraph)
5. Grounds opposing the relief sought
6. Prayer (praying the original relief be dismissed/rejected, plus any counter-relief that
   is actually supported by the evidence)
Known extracted details of the original petition:
{json.dumps(details, indent=2)}
RETRIEVED EVIDENCE FROM THE ORIGINAL PETITION:
{evidence}
ADDITIONAL INSTRUCTIONS FROM USER (may be empty): {extra_instructions}
Write the full counter-petition as plain text (not JSON), with standard legal paragraph
numbering."""
    draft = get_llm_provider().generate_text(prompt, num_predict=1200)
    doc["counter_draft"] = draft
    return {
        "doc_id": doc_id,
        "counter_petition": draft,
        "llm_model": _active_model_label(),
    }
@app.post("/generate-counter")
async def generate_counter(req: CounterRequest):
    return _generate_counter_for_doc(req.doc_id, req.extra_instructions)
@app.get("/generate-counter/{doc_id}")
def get_counter(doc_id: str):
    doc = DOCS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if "counter_draft" not in doc:
        raise HTTPException(status_code=404, detail="No counter-petition generated yet for this document.")
    return {"doc_id": doc_id, "counter_petition": doc["counter_draft"]}
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend-new", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)