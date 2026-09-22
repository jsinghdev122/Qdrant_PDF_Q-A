# Qdrant PDF Q&A

Add PDFs to a library, then ask questions across them. Passages are stored in the [Qdrant] vector database, the best matches are found for each question, and Google Gemini writes an answer that cites them by page.

## How it works

1. **Read**: `pypdf` extracts the text of each PDF page.
2. **Chunk**: each page is cut into overlapping windows of about 180 words. Every passage remembers its file and page.
3. **Embed**: a small local model run through `fastembed` turns each passage into a vector. This needs no API key and no GPU.
4. **Store**: vectors go into a Qdrant collection (cosine similarity) with the filename, page and text as payload. The library persists between restarts.
5. **Search**: a question is embedded and Qdrant returns the closest passages. You can limit the search to chosen documents with a payload filter.
6. **Answer**: only those passages (not the whole PDF) are sent to Gemini, which answers and cites them as `[1]`, `[2]`. The page shows the sources with their page numbers.

Adding PDFs works without a Gemini key. Only asking questions needs one.

## Setup

```
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

set GEMINI_API_KEY=your-key    # macOS/Linux: export GEMINI_API_KEY='your-key'
uvicorn app.main:app --reload
```

Open http://localhost:8000. The first start downloads the embedding model (roughly tens of MB).

### Using a Qdrant server instead

By default Qdrant runs inside the app and saves to `./qdrant_data`. To use a real server:

```


## Settings (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | none | Needed to ask questions (`GOOGLE_API_KEY` also works) |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Model that writes answers |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `QDRANT_URL` / `QDRANT_API_KEY` | none | Use a Qdrant server or cloud cluster |
| `QDRANT_PATH` | `./qdrant_data` | Folder for local mode |
| `QDRANT_COLLECTION` | `pdf_passages` | Collection name |
| `TOP_K` | `6` | Passages sent to the model per question |
| `CHUNK_WORDS` / `CHUNK_OVERLAP` | `180` / `40` | Passage size and overlap, in words |
| `MAX_UPLOAD_MB` | `30` | Largest PDF accepted |

If you change `EMBED_MODEL`, use a new `QDRANT_COLLECTION` (or delete `qdrant_data`), because vectors from different models are not comparable.

## API

Interactive docs: http://localhost:8000/docs

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/documents` | List documents in the library |
| POST | `/documents` | Upload one or more PDFs (multipart field `files`) |
| DELETE | `/documents/{doc_id}` | Remove a document and its passages |
| POST | `/ask` | `{"question": "...", "doc_ids": ["..."]}` returns the answer and sources |
| GET | `/health` | Passage count, storage location, whether a Gemini key is set |


The tests cover the PDF reading and chunking. They need only `pypdf`.

## Project layout


app/
  main.py      routes and startup
  library.py   embeddings + Qdrant (add, search, list, delete)
  ingest.py    PDF -> page-aware passages
  llm.py       prompt and Gemini call
  config.py    settings
static/index.html   the web page


## Limits

- Scanned PDFs without a text layer have no text to read and are rejected.
- Answers are only as good as the passages found. Very broad questions ("summarize everything") work better on a single document.
- Questions are independent; there is no chat memory.
- The document list is built by scanning stored passages, which is fine for hundreds of documents but not for huge libraries.
