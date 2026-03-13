# RAGwithMCP

A local **Retrieval-Augmented Generation (RAG)** application with an [MCP](https://modelcontextprotocol.io) server, built with ChromaDB, Gradio, and OpenRouter.

## Features

- 📤 **Upload documents** — PDF, DOCX, TXT, MD, CSV, RST
- 💬 **Chat with your documents** — RAG pipeline retrieves relevant chunks and passes them as context to an LLM via OpenRouter
- 🗂️ **Manage documents** — list and delete stored documents
- 🔌 **MCP server** — expose your knowledge base as tools for Claude Desktop, Cursor, or any MCP-compatible client

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For PDF and DOCX support, also install:

```bash
pip install pypdf python-docx
```

For better embeddings (optional):

```bash
pip install sentence-transformers
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your OPEN_ROUTER_API_KEY
```

Get a free API key at <https://openrouter.ai/keys>.

### 3. Launch the web interface

```bash
python main.py
```

Then open <http://localhost:7860> in your browser.

### 4. (Optional) Start the MCP server

```bash
python mcp_server.py
```

Connect it to Claude Desktop by adding to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "python",
      "args": ["/path/to/RAGwithMCP/mcp_server.py"]
    }
  }
}
```

## Project Structure

```
RAGwithMCP/
├── main.py               # Entry point — launches the Gradio UI
├── mcp_server.py         # MCP server (stdio transport)
├── gui/
│   └── app.py            # Gradio web interface
├── llm/
│   └── openrouter.py     # OpenRouter API client
├── rag/
│   ├── document_processor.py  # File ingestion & text splitting
│   └── vector_store.py        # ChromaDB vector store wrapper
├── requirements.txt
└── .env.example
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPEN_ROUTER_API_KEY` | — | **Required.** Your OpenRouter API key. |
| `OPENROUTER_MODEL` | `openrouter/auto` | Model to use for chat completions. |
| `CHROMA_DB_PATH` | `./chroma_db` | Path to the ChromaDB data directory. |

