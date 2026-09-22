"""FastAPI app.  Run:  uvicorn app.main:app --reload"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from google.genai import errors as genai_errors
from pydantic import BaseModel

from .config import load_settings
from .ingest import make_chunks, read_pages
from .library import Library
from .llm import Answerer

INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


# ---------- request / response shapes ----------

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int


class UploadProblem(BaseModel):
    filename: str
    error: str


class UploadResult(BaseModel):
    added: List[DocumentInfo]
    errors: List[UploadProblem]


class AskRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None  # search only these documents; omit for all


class SourceOut(BaseModel):
    n: int
    filename: str
    page: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceOut]


class HealthResponse(BaseModel):
    status: str
    passages: int
    gemini_configured: bool
    storage: str


# ---------- app ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.library = Library(settings)
    app.state.answerer = Answerer(settings.gemini_api_key, settings.gemini_model)
    if not app.state.answerer.configured:
        print("Note: GEMINI_API_KEY is not set. You can add PDFs, but asking questions will fail.")
    print(f"Ready: http://localhost:8000  (storage: {app.state.library.storage})")
    yield
    app.state.library.close()


app = FastAPI(title="Qdrant PDF Q&A", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_HTML)


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    library, answerer = request.app.state.library, request.app.state.answerer
    return HealthResponse(
        status="ok", passages=library.count(), gemini_configured=answerer.configured, storage=library.storage
    )


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents(request: Request):
    return request.app.state.library.list_documents()


@app.post("/documents", response_model=UploadResult)
def upload_documents(request: Request, files: List[UploadFile] = File(...)):
    settings, library = request.app.state.settings, request.app.state.library
    added, problems = [], []

    for upload in files:
        name = upload.filename or "unnamed.pdf"
        try:
            data = upload.file.read()
            if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
                raise ValueError("Not a PDF file.")
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                raise ValueError(f"Larger than {settings.max_upload_mb} MB.")
            pages = read_pages(data)
            chunks = make_chunks(pages, settings.chunk_words, settings.chunk_overlap)
            if not chunks:
                raise ValueError("No text found. Scanned PDFs need OCR first.")
            doc_id = library.add_document(name, chunks)
            added.append(DocumentInfo(doc_id=doc_id, filename=name, pages=len(pages), chunks=len(chunks)))
        except Exception as exc:
            problems.append(UploadProblem(filename=name, error=str(exc)))

    if not added:
        raise HTTPException(400, "; ".join(f"{p.filename}: {p.error}" for p in problems) or "No files received.")
    return UploadResult(added=added, errors=problems)


@app.delete("/documents/{doc_id}")
def delete_document(request: Request, doc_id: str):
    request.app.state.library.delete_document(doc_id)
    return {"ok": True}


@app.post("/ask", response_model=AskResponse)
def ask(request: Request, body: AskRequest):
    settings, library, answerer = request.app.state.settings, request.app.state.library, request.app.state.answerer
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Please type a question.")

    hits = library.search(question, body.doc_ids, settings.top_k)
    if not hits:
        return AskResponse(answer="There is nothing to search yet. Add a PDF first.", sources=[])

    try:
        answer = answerer.answer(question, hits)
    except genai_errors.APIError as exc:
        status = exc.code if isinstance(exc.code, int) and 400 <= exc.code < 600 else 502
        if status == 429:
            raise HTTPException(429, "Rate limit reached. Wait a moment and try again.") from exc
        if status in (401, 403):
            raise HTTPException(status, "Gemini rejected the API key. Check GEMINI_API_KEY.") from exc
        raise HTTPException(status, exc.message or str(exc)) from exc

    sources = [
        SourceOut(n=i, filename=h.filename, page=h.page, score=h.score, text=h.text)
        for i, h in enumerate(hits, start=1)
    ]
    return AskResponse(answer=answer, sources=sources)
