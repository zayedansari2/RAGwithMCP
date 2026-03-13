"""MCP server exposing the RAG knowledge base as tools via stdio transport."""
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from rag import document_processor, vector_store

app = Server("rag-knowledge-base")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_knowledge_base",
            description="Search for relevant context chunks by query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_documents",
            description="List all documents stored in the knowledge base.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="add_document_text",
            description="Add raw text to the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content to add.",
                    },
                    "name": {
                        "type": "string",
                        "description": "A name / identifier for this document.",
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Characters per chunk (default 1000).",
                        "default": 1000,
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "description": "Overlap between chunks (default 200).",
                        "default": 200,
                    },
                },
                "required": ["text", "name"],
            },
        ),
        Tool(
            name="delete_document",
            description="Remove a document from the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact document name to delete.",
                    }
                },
                "required": ["name"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_knowledge_base":
        query = arguments["query"]
        n_results = int(arguments.get("n_results", 5))
        results = vector_store.search(query, n_results=n_results)
        if not results:
            return [TextContent(type="text", text="No relevant chunks found.")]
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] Source: {r['source']} (chunk {r['chunk_index']}, distance {r['distance']:.4f})\n{r['text']}"
            )
        return [TextContent(type="text", text="\n\n".join(lines))]

    elif name == "list_documents":
        docs = vector_store.list_documents()
        if not docs:
            return [TextContent(type="text", text="No documents in the knowledge base.")]
        lines = [f"- {d['name']} ({d['chunk_count']} chunks)" for d in docs]
        total = vector_store.get_total_chunks()
        lines.append(f"\nTotal chunks: {total}")
        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "add_document_text":
        text = arguments["text"]
        doc_name = arguments["name"]
        chunk_size = int(arguments.get("chunk_size", 1000))
        chunk_overlap = int(arguments.get("chunk_overlap", 200))
        chunks = document_processor.split_text(text, chunk_size, chunk_overlap)
        n = vector_store.add_document(doc_name, chunks)
        return [TextContent(type="text", text=f"Added '{doc_name}' ({n} chunks).")]

    elif name == "delete_document":
        doc_name = arguments["name"]
        deleted = vector_store.delete_document(doc_name)
        if deleted == 0:
            return [TextContent(type="text", text=f"Document '{doc_name}' not found.")]
        return [TextContent(type="text", text=f"Deleted '{doc_name}' ({deleted} chunks removed).")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


if __name__ == "__main__":
    import asyncio

    asyncio.run(stdio_server(app))
