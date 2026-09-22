"""Settings, read from environment variables."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    gemini_api_key: Optional[str]
    gemini_model: str
    embed_model: str
    qdrant_url: Optional[str]      # set to use a Qdrant server (e.g. Docker)
    qdrant_api_key: Optional[str]
    qdrant_path: str               # used when no server URL is set
    collection: str
    top_k: int                     # passages sent to the model per question
    chunk_words: int               # words per passage
    chunk_overlap: int             # words shared between neighbouring passages
    max_upload_mb: int


def load_settings() -> Settings:
    env = os.environ.get
    key = (env("GEMINI_API_KEY") or env("GOOGLE_API_KEY") or "").strip()
    return Settings(
        gemini_api_key=key or None,
        gemini_model=env("GEMINI_MODEL", "gemini-3.6-flash"),
        embed_model=env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        qdrant_url=env("QDRANT_URL") or None,
        qdrant_api_key=env("QDRANT_API_KEY") or None,
        qdrant_path=env("QDRANT_PATH", "./qdrant_data"),
        collection=env("QDRANT_COLLECTION", "pdf_passages"),
        top_k=int(env("TOP_K", "6")),
        chunk_words=int(env("CHUNK_WORDS", "180")),
        chunk_overlap=int(env("CHUNK_OVERLAP", "40")),
        max_upload_mb=int(env("MAX_UPLOAD_MB", "30")),
    )
