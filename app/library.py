"""The vector database: embeds passages and stores/searches them in Qdrant."""
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from .config import Settings


@dataclass(frozen=True)
class Hit:
    filename: str
    page: int
    text: str
    score: float  # cosine similarity, higher = closer in meaning


class Library:
    def __init__(self, settings: Settings):
        self._collection = settings.collection
        self._lock = threading.RLock()  # requests run in worker threads

        print(f"Loading embedding model {settings.embed_model} (first run downloads it)...")
        self._embedder = TextEmbedding(model_name=settings.embed_model)
        dimension = len(next(iter(self._embedder.embed(["dimension probe"]))))

        if settings.qdrant_url:
            self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            self.storage = settings.qdrant_url
            self._is_server = True
        else:
            self._client = QdrantClient(path=settings.qdrant_path)
            self.storage = f"local folder {settings.qdrant_path}"
            self._is_server = False

        self._prepare_collection(dimension)

    def _prepare_collection(self, dimension: int) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
        else:
            existing = self._client.get_collection(self._collection).config.params.vectors
            size = getattr(existing, "size", None)
            if size is not None and size != dimension:
                raise RuntimeError(
                    f"Collection '{self._collection}' stores {size}-dimension vectors but the "
                    f"embedding model produces {dimension}. Use another QDRANT_COLLECTION or delete the old data."
                )
        if self._is_server:  # payload indexes only matter on a real server
            try:
                self._client.create_payload_index(
                    self._collection, field_name="doc_id", field_schema=models.PayloadSchemaType.KEYWORD
                )
            except Exception:
                pass  # already exists

    def _embed(self, texts: Iterable[str]) -> List[List[float]]:
        return [vector.tolist() for vector in self._embedder.embed(list(texts))]

    def add_document(self, filename: str, chunks: Sequence[Dict]) -> str:
        """Store a document's passages. Returns the new document id."""
        doc_id = uuid.uuid4().hex
        with self._lock:
            vectors = self._embed(chunk["text"] for chunk in chunks)
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"doc_id": doc_id, "filename": filename, "page": chunk["page"], "text": chunk["text"]},
                )
                for chunk, vector in zip(chunks, vectors)
            ]
            for start in range(0, len(points), 128):
                self._client.upsert(collection_name=self._collection, points=points[start:start + 128])
        return doc_id

    def search(self, question: str, doc_ids: Optional[Sequence[str]], limit: int) -> List[Hit]:
        """Closest passages to the question, optionally only inside the given documents."""
        query_filter = None
        if doc_ids:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))]
            )
        with self._lock:
            vector = self._embed([question])[0]
            response = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        return [
            Hit(
                filename=point.payload["filename"],
                page=point.payload["page"],
                text=point.payload["text"],
                score=point.score,
            )
            for point in response.points
        ]

    def list_documents(self) -> List[Dict]:
        """One entry per document. `pages` is the last page that had text."""
        found: Dict[str, Dict] = {}
        with self._lock:
            offset = None
            while True:
                records, offset = self._client.scroll(
                    collection_name=self._collection,
                    limit=256,
                    offset=offset,
                    with_payload=["doc_id", "filename", "page"],
                    with_vectors=False,
                )
                for record in records:
                    info = found.setdefault(
                        record.payload["doc_id"],
                        {"doc_id": record.payload["doc_id"], "filename": record.payload["filename"], "pages": 0, "chunks": 0},
                    )
                    info["chunks"] += 1
                    info["pages"] = max(info["pages"], record.payload["page"])
                if offset is None:
                    break
        return sorted(found.values(), key=lambda d: d["filename"].lower())

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                    )
                ),
            )

    def count(self) -> int:
        with self._lock:
            return self._client.count(collection_name=self._collection, exact=True).count

    def close(self) -> None:
        self._client.close()
