# RAGwithMCP

A local project that combines **Retrieval-Augmented Generation (RAG)** with the **Model Context Protocol (MCP)** so you can:

1. **Upload documents** (PDF, DOCX, TXT, MD, CSV) via a web GUI.
2. **Chat with your documents** using an LLM powered by [OpenRouter](https://openrouter.ai).
3. **Connect any MCP-compatible LLM** (Claude Desktop, Cursor, Cline, etc.) to your personal knowledge base via the included MCP server.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│               Gradio Web GUI                │
│  Upload ▸ Manage ▸ Chat ▸ MCP Server Info   │
└────────────────────┬────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   RAG Pipeline          │
        │  document_processor.py  │  ← text extraction + chunking
        │  vector_store.py        │  ← ChromaDB (local, persistent)
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐        ┌──────────────────────────┐
        │   llm/openrouter.py     │        │   mcp_server.py          │
        │   OpenRouter API        │        │   MCP tools (stdio)      │
        │   openrouter/auto       │        │   search / list / add /  │
        └─────────────────────────┘        │   delete documents       │
                                           └──────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and set your OPEN_ROUTER_API_KEY
```

### 3. Start the web GUI

```bash
python main.py
# Open http://localhost:7860 in your browser
```

### 4. (Optional) Start the MCP server

In a **separate terminal**:

```bash
python mcp_server.py
```

---

## Using the Web GUI

| Tab | What you can do |
|-----|-----------------|
| **📤 Upload Documents** | Upload PDF / DOCX / TXT / MD / CSV files; configure chunk size & overlap |
| **🗂️ Manage Documents** | View all stored documents and delete individual ones |
| **💬 Chat** | Ask questions answered with RAG context from your uploaded files |
| **🔌 MCP Server** | Instructions for connecting the MCP server to your LLM |

---

## MCP Server

The MCP server exposes your knowledge base as tools via the **stdio** transport.

### Available tools

| Tool | Description |
|------|-------------|
| `search_knowledge_base` | Find relevant chunks by query |
| `list_documents` | List all uploaded documents |
| `add_document_text` | Add raw text to the knowledge base |
| `delete_document` | Remove a document |

### Connect to Claude Desktop

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "python",
      "args": ["/absolute/path/to/RAGwithMCP/mcp_server.py"]
    }
  }
}
```

### Connect with the MCP CLI

```bash
mcp dev mcp_server.py
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPEN_ROUTER_API_KEY` | *(required)* | Your [OpenRouter](https://openrouter.ai) API key |
| `OPENROUTER_MODEL` | `openrouter/auto` | Model to use (e.g. `openrouter/auto`) |
| `CHROMA_DB_PATH` | `./chroma_db` | Path to the ChromaDB persistent store |
| `PORT` | `7860` | Port for the Gradio web interface |

---

## Project Structure

```
RAGwithMCP/
├── main.py                  # Entry point (Gradio GUI)
├── mcp_server.py            # MCP server (stdio transport)
├── requirements.txt
├── .env.example
├── rag/
│   ├── document_processor.py  # File reading + text chunking
│   └── vector_store.py        # ChromaDB wrapper
├── llm/
│   └── openrouter.py          # OpenRouter API client
└── gui/
    └── app.py                 # Gradio interface
```
