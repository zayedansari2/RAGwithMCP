#!/usr/bin/env python3
"""
MCP server that exposes the RAG knowledge base as tools.

Run with:
    python mcp_server.py

Or via the MCP CLI:
    mcp dev mcp_server.py

The server uses stdio transport so it can be connected to any MCP-compatible
client (Claude Desktop, Cursor, Cline, etc.).
"""

import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from rag import document_processor, vector_store

load_dotenv()

mcp = FastMCP(
    name="RAG Knowledge Base",
    instructions=(
        "This server gives you access to a personal knowledge base built from "
        "uploaded documents. Use search_knowledge_base to find relevant context "
        "before answering questions about the stored documents."
    ),
)


@mcp.tool()
def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """
    Search the knowledge base for chunks relevant to a query.

    Args:
        query: The search query or question.
        n_results: Number of results to return (default 5, max 20).

    Returns:
        Formatted string with the most relevant text chunks and their sources.
    """
    n_results = max(1, min(n_results, 20))
    results = vector_store.search(query, n_results=n_results)

    if not results:
        return "No relevant content found in the knowledge base."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] Source: {r['source']} (chunk {r['chunk_index']})\n{r['text']}"
        )
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def list_documents() -> str:
    """
    List all documents stored in the knowledge base.

    Returns:
        A formatted list of document names and their chunk counts.
    """
    docs = vector_store.list_documents()
    if not docs:
        return "The knowledge base is empty. No documents have been uploaded yet."

    lines = [f"Knowledge base contains {len(docs)} document(s):\n"]
    for doc in docs:
        lines.append(f"  • {doc['name']}  ({doc['chunk_count']} chunks)")
    total = vector_store.get_total_chunks()
    lines.append(f"\nTotal chunks: {total}")
    return "\n".join(lines)


@mcp.tool()
def add_document_text(document_name: str, text: str) -> str:
    """
    Add raw text to the knowledge base.

    Use this when you have text content you want to store (e.g. content fetched
    from the web, or text generated during a conversation).

    Args:
        document_name: A descriptive name for the document (e.g. 'meeting_notes.txt').
        text: The full text content to store.

    Returns:
        Confirmation message with the number of chunks stored.
    """
    if not document_name.strip():
        return "Error: document_name cannot be empty."
    if not text.strip():
        return "Error: text cannot be empty."

    chunks = document_processor.split_text(text)
    if not chunks:
        return f"No content could be extracted from the provided text."

    n = vector_store.add_document(document_name.strip(), chunks)
    return f"✅ '{document_name}' added to the knowledge base ({n} chunks stored)."


@mcp.tool()
def delete_document(document_name: str) -> str:
    """
    Delete a document and all its chunks from the knowledge base.

    Args:
        document_name: The exact name of the document to delete.

    Returns:
        Confirmation message.
    """
    if not document_name.strip():
        return "Error: document_name cannot be empty."

    deleted = vector_store.delete_document(document_name.strip())
    if deleted == 0:
        return f"Document '{document_name}' was not found in the knowledge base."
    return f"🗑️ Deleted '{document_name}' ({deleted} chunks removed)."


if __name__ == "__main__":
    mcp.run()
