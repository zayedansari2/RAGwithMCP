"""Gradio web interface for RAG document management and chat."""

import os
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
from dotenv import load_dotenv

from rag import document_processor, vector_store
from llm import openrouter

load_dotenv()

DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/auto")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _upload_file(file_obj, chunk_size: int, chunk_overlap: int) -> str:
    """Process an uploaded file and store its chunks in the vector store."""
    if file_obj is None:
        return "No file provided."

    file_path = file_obj.name
    file_name = Path(file_path).name

    try:
        chunks = document_processor.process_file(
            file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if not chunks:
            return f"⚠️ No text could be extracted from '{file_name}'."

        n = vector_store.add_document(file_name, chunks)
        return f"✅ '{file_name}' uploaded successfully ({n} chunks stored)."
    except Exception as exc:
        return f"❌ Error uploading '{file_name}': {exc}"


def _list_documents() -> str:
    """Return a markdown table of stored documents."""
    docs = vector_store.list_documents()
    if not docs:
        return "_No documents stored yet. Upload some files to get started!_"

    lines = ["| Document | Chunks |", "|----------|--------|"]
    for doc in docs:
        lines.append(f"| {doc['name']} | {doc['chunk_count']} |")
    total = vector_store.get_total_chunks()
    lines.append(f"\n**Total chunks in knowledge base:** {total}")
    return "\n".join(lines)


def _delete_document(doc_name: str) -> str:
    """Delete a document from the vector store."""
    doc_name = doc_name.strip()
    if not doc_name:
        return "⚠️ Please enter a document name."

    deleted = vector_store.delete_document(doc_name)
    if deleted == 0:
        return f"⚠️ Document '{doc_name}' not found."
    return f"🗑️ Deleted '{doc_name}' ({deleted} chunks removed)."


def _chat(
    user_message: str,
    history: list,
    n_results: int,
    model: str,
    api_key: str,
) -> tuple[list, str]:
    """Process a chat message using RAG context."""
    if not user_message.strip():
        return history, ""

    effective_api_key = api_key.strip() or None

    try:
        # Retrieve relevant chunks
        results = vector_store.search(user_message, n_results=n_results)
        context_chunks = [r["text"] for r in results]

        # Build history for multi-turn conversation (last 10 turns)
        chat_history = []
        for human, assistant in history[-10:]:
            chat_history.append({"role": "user", "content": human})
            chat_history.append({"role": "assistant", "content": assistant})

        # Get response
        reply = openrouter.rag_chat(
            user_question=user_message,
            context_chunks=context_chunks,
            chat_history=chat_history,
            model=model,
            api_key=effective_api_key,
        )

        # Append source attribution
        if results:
            sources = sorted({r["source"] for r in results})
            reply += f"\n\n---\n*Sources: {', '.join(sources)}*"

    except ValueError as exc:
        reply = f"⚠️ Configuration error: {exc}"
    except Exception as exc:
        reply = f"❌ Error: {exc}"

    history.append((user_message, reply))
    return history, ""


def _clear_chat() -> tuple[list, str]:
    """Reset the chatbot to an empty conversation."""
    return [], ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="RAG with MCP") as app:
        gr.Markdown(
            """
# 📚 RAG with MCP
Upload your documents, chat with them using a powerful LLM, and connect an MCP server 
to your favorite AI assistant.
"""
        )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # Tab 1: Upload Documents
            # ------------------------------------------------------------------
            with gr.Tab("📤 Upload Documents"):
                gr.Markdown("### Upload files to the knowledge base")
                gr.Markdown(
                    "Supported formats: **PDF**, **DOCX**, **TXT**, **MD**, **CSV**, **RST**"
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        upload_file = gr.File(
                            label="Select a file to upload",
                            file_types=[".pdf", ".docx", ".txt", ".md", ".csv", ".rst"],
                        )
                    with gr.Column(scale=1):
                        chunk_size = gr.Slider(
                            minimum=200,
                            maximum=4000,
                            value=1000,
                            step=100,
                            label="Chunk size (characters)",
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0,
                            maximum=500,
                            value=200,
                            step=50,
                            label="Chunk overlap (characters)",
                        )

                upload_btn = gr.Button("Upload & Process", variant="primary")
                upload_status = gr.Markdown()

                upload_btn.click(
                    fn=_upload_file,
                    inputs=[upload_file, chunk_size, chunk_overlap],
                    outputs=upload_status,
                )

            # ------------------------------------------------------------------
            # Tab 2: Manage Documents
            # ------------------------------------------------------------------
            with gr.Tab("🗂️ Manage Documents"):
                gr.Markdown("### Documents in the knowledge base")

                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                doc_list = gr.Markdown()

                refresh_btn.click(fn=_list_documents, inputs=[], outputs=doc_list)
                app.load(fn=_list_documents, inputs=[], outputs=doc_list)

                gr.Markdown("---")
                gr.Markdown("### Delete a document")
                with gr.Row():
                    delete_input = gr.Textbox(
                        label="Document name (exact match)",
                        placeholder="e.g. report.pdf",
                    )
                    delete_btn = gr.Button("🗑️ Delete", variant="stop")
                delete_status = gr.Markdown()

                delete_btn.click(
                    fn=_delete_document,
                    inputs=[delete_input],
                    outputs=delete_status,
                )

            # ------------------------------------------------------------------
            # Tab 3: Chat
            # ------------------------------------------------------------------
            with gr.Tab("💬 Chat"):
                gr.Markdown(
                    "### Chat with your knowledge base\n"
                    "Ask questions and the assistant will use your uploaded documents as context."
                )

                with gr.Accordion("⚙️ Settings", open=False):
                    with gr.Row():
                        model_input = gr.Textbox(
                            label="OpenRouter model",
                            value=DEFAULT_MODEL,
                            placeholder="e.g. openrouter/auto",
                        )
                        n_results_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Number of context chunks",
                        )
                    api_key_input = gr.Textbox(
                        label="OpenRouter API key (leave blank to use .env)",
                        placeholder="sk-or-...",
                        type="password",
                    )

                chatbot = gr.Chatbot(height=450, label="Conversation")
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder="Ask something about your documents…",
                        scale=8,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                clear_btn = gr.Button("🗑️ Clear conversation", variant="secondary")

                send_btn.click(
                    fn=_chat,
                    inputs=[msg_input, chatbot, n_results_slider, model_input, api_key_input],
                    outputs=[chatbot, msg_input],
                )
                msg_input.submit(
                    fn=_chat,
                    inputs=[msg_input, chatbot, n_results_slider, model_input, api_key_input],
                    outputs=[chatbot, msg_input],
                )
                clear_btn.click(fn=_clear_chat, outputs=[chatbot, msg_input])

            # ------------------------------------------------------------------
            # Tab 4: MCP Server Info
            # ------------------------------------------------------------------
            with gr.Tab("🔌 MCP Server"):
                gr.Markdown(
                    """
### Connect an LLM to your knowledge base via MCP

The **MCP (Model Context Protocol) server** exposes your knowledge base as tools 
that any compatible LLM or agent can use.

#### Starting the MCP server

Run the following command in a separate terminal:

```bash
python mcp_server.py
```

The server uses **stdio** transport (default for MCP), so you can connect it 
directly to Claude Desktop, Cursor, or any MCP-compatible client.

#### Available MCP Tools

| Tool | Description |
|------|-------------|
| `search_knowledge_base` | Search for relevant context chunks by query |
| `list_documents` | List all documents in the knowledge base |
| `add_document_text` | Add raw text to the knowledge base |
| `delete_document` | Remove a document from the knowledge base |

#### Example `claude_desktop_config.json`

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

#### Connecting with `mcp` CLI

```bash
mcp dev mcp_server.py
```
"""
                )

    return app


def launch(share: bool = False, port: int = 7860):
    app = build_app()
    app.launch(
        share=share,
        server_port=port,
        theme=gr.themes.Soft(),
    )
