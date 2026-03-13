"""ChromaDB-backed persistent vector store for RAG."""

import os
import re
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import chromadb

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "documents"

# Module-level embedding function cache (initialised once per process)
_embedding_fn = None


def _make_local_embedding_fn(dim: int = 512):
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
                h1 = hash(word) % dim
                h2 = hash(word + "\x00") % dim
                vec[h1] += 1.0
                vec[h2] += 0.5
            norm = float(np.linalg.norm(vec))
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

        def _st_embed(texts: list[str]) -> list[list[float]]:
            return model.encode(texts, normalize_embeddings=True).tolist()

        # Quick sanity check — will fail if model weights aren't cached
        _st_embed(["warmup"])
        _embedding_fn = _st_embed
        return _embedding_fn
    except Exception:
        pass

    _embedding_fn = _make_local_embedding_fn()
    return _embedding_fn


def get_client() -> chromadb.PersistentClient:
    db_path = Path(CHROMA_DB_PATH)
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def get_collection(client: Optional[chromadb.PersistentClient] = None):
    """Get or create the documents collection (no built-in EF — we embed manually)."""
    if client is None:
        client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_document(
    file_name: str,
    chunks: list[str],
    client: Optional[chromadb.PersistentClient] = None,
) -> int:
    """
    Add document chunks to the vector store.

    Returns the number of chunks added.
    """
    if not chunks:
        return 0

    embed = _get_embedding_fn()
    embeddings = embed(chunks)

    collection = get_collection(client)
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    metadatas = [{"source": file_name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return len(chunks)


def search(
    query: str,
    n_results: int = 5,
    source_filter: Optional[str] = None,
    client: Optional[chromadb.PersistentClient] = None,
) -> list[dict]:
    """
    Search the vector store for relevant chunks.

    Returns a list of dicts with keys: text, source, chunk_index, distance.
    """
    collection = get_collection(client)
    if collection.count() == 0:
        return []

    embed = _get_embedding_fn()
    query_embedding = embed([query])

    where = {"source": source_filter} if source_filter else None
    n_results = min(n_results, collection.count())

    results = collection.query(
        query_embeddings=query_embedding,
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
                "chunk_index": meta.get("chunk_index", 0),
                "distance": dist,
            }
        )
    return output


def list_documents(client: Optional[chromadb.PersistentClient] = None) -> list[dict]:
    """
    List all unique documents stored in the vector store.

    Returns a list of dicts with keys: name, chunk_count.
    """
    collection = get_collection(client)
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    sources: dict[str, int] = {}
    for meta in results["metadatas"]:
        source = meta.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    return [{"name": name, "chunk_count": count} for name, count in sorted(sources.items())]


def delete_document(
    file_name: str, client: Optional[chromadb.PersistentClient] = None
) -> int:
    """
    Delete all chunks belonging to a document.

    Returns the number of chunks deleted.
    """
    collection = get_collection(client)
    results = collection.get(where={"source": file_name}, include=["metadatas"])
    ids = results["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def get_total_chunks(client: Optional[chromadb.PersistentClient] = None) -> int:
    """Return the total number of chunks stored."""
    return get_collection(client).count()
