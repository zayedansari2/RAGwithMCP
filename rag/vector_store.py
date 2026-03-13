"""ChromaDB-backed persistent vector store for RAG."""
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import chromadb

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "documents"

_embedding_fn = None


def _make_local_embedding_fn(dim: int):
    """
    Return a fully offline embedding function based on the hashing trick.

    It converts text into a bag-of-words feature vector using two independent hash
    functions to reduce collisions, then L2-normalises the result.  The vectors are
    not as rich as those produced by a neural encoder, but they are completely
    self-contained (no model download required) and give reasonable nearest-neighbour
    retrieval for keyword-heavy queries.
    """

    def embed(texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            words = re.findall(r"\w+", text.lower())
            for word in words:
                h1 = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
                h2 = int(hashlib.sha256(word.encode()).hexdigest(), 16) % dim
                vec[h1] += float(1)
                vec[h2] += float(1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec.tolist())
        return results

    return embed


def _get_embedding_fn():
    """Return (and cache) the best available embedding function."""
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        # warmup
        model.encode(["warmup"])

        def _st_embed(texts):
            return model.encode(texts).tolist()

        _embedding_fn = _st_embed
    except Exception:
        _embedding_fn = _make_local_embedding_fn(384)
    return _embedding_fn


def get_client():
    db_path = Path(CHROMA_DB_PATH)
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(str(db_path))


def get_collection(client):
    """Get or create the documents collection (no built-in EF — we embed manually)."""
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_document(file_name: str, chunks: list) -> int:
    """
    Add document chunks to the vector store.

    Returns the number of chunks added.
    """
    embed = _get_embedding_fn()
    embeddings = embed(chunks)
    client = get_client()
    collection = get_collection(client)
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    metadatas = [{"source": file_name, "chunk_index": i} for i in range(len(chunks))]
    collection.add(
        ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas
    )
    return len(chunks)


def search(
    query: str, n_results: int = 5, source_filter: Optional[str] = None
) -> list[dict]:
    """
    Search the vector store for relevant chunks.

    Returns a list of dicts with keys: text, source, chunk_index, distance.
    """
    client = get_client()
    collection = get_collection(client)
    total = collection.count()
    if total == 0:
        return []
    embed = _get_embedding_fn()
    query_embedding = embed([query])[0]
    n_results = min(n_results, total)
    where = {"source": source_filter} if source_filter else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index"),
                "distance": dist,
            }
        )
    return output


def list_documents() -> list[dict]:
    """
    List all unique documents stored in the vector store.

    Returns a list of dicts with keys: name, chunk_count.
    """
    client = get_client()
    collection = get_collection(client)
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    sources: dict = {}
    for meta in results["metadatas"]:
        source = meta.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    return [
        {"name": name, "chunk_count": count} for name, count in sorted(sources.items())
    ]


def delete_document(file_name: str) -> int:
    """
    Delete all chunks belonging to a document.

    Returns the number of chunks deleted.
    """
    client = get_client()
    collection = get_collection(client)
    results = collection.get(where={"source": file_name}, include=["metadatas"])
    ids = results["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def get_total_chunks() -> int:
    """Return the total number of chunks stored."""
    client = get_client()
    return get_collection(client).count()
