"""PDF bytes -> passages that remember which page they came from."""
from io import BytesIO
from typing import Dict, List

from pypdf import PdfReader


def read_pages(data: bytes) -> List[str]:
    """Text of each page, in order (an empty string for pages without text)."""
    reader = PdfReader(BytesIO(data))
    return [(page.extract_text() or "") for page in reader.pages]


def split_words(text: str, size: int, overlap: int) -> List[str]:
    """Sliding window over words: `size` words per piece, `overlap` shared with the next."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return pieces


def make_chunks(pages: List[str], size: int, overlap: int) -> List[Dict]:
    """Passages as {"page": 1-based page number, "text": ...}. Windows never cross pages."""
    chunks = []
    for number, text in enumerate(pages, start=1):
        for piece in split_words(text, size, overlap):
            chunks.append({"page": number, "text": piece})
    return chunks
