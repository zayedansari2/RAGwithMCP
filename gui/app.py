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


def _upload_file(file_obj, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
    """Process an uploaded file and store its chunks in the vector store."""
    if file_obj is None:
        return "No file provided."
    file_path = file_obj.name
    file_name = Path(file_path).name
    try:
        chunks = document_processor.process_file(file_path, chunk_size, chunk_overlap)
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
    n_results: int = 5,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
) -> tuple:
    """Process a chat message using RAG context."""
    user_message = user_message.strip()
    if not user_message:
        return history, ""
    effective_api_key = api_key.strip() or None
    try:
        results = vector_store.search(user_message, n_results=n_results)
        context_chunks = [r["text"] for r in results]
        chat_history = []
        for human, assistant in history:
            chat_history.append({"role": "user", "content": human})
            chat_history.append({"role": "assistant", "content": assistant})
        reply = openrouter.rag_chat(
            user_message,
            context_chunks,
            chat_history=chat_history,
            model=model,
            api_key=effective_api_key,
        )
        sources = sorted({r["source"] for r in results})
        if sources:
            reply += "\n\n---\n*Sources: " + ", ".join(sources) + "*"
        history = history + [[user_message, reply]]
        return history, ""
    except ValueError as exc:
        return history + [[user_message, f"⚠️ Configuration error: {exc}"]], ""
    except Exception as exc:
        return history + [[user_message, f"❌ Error: {exc}"]], ""


def _clear_chat() -> str:
    """Reset the chatbot to an empty conversation."""
    return []


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RAG with MCP") as app:
        gr.Markdown(
            "\n# 📚 RAG with MCP\n"
            "Upload your documents, chat with them using a powerful LLM, and connect an MCP server \n"
            "to your favourite AI assistant.\n"
        )
        with gr.Tabs():
            with gr.Tab("📤 Upload Documents"):
                gr.Markdown("### Upload files to the knowledge base")
                gr.Markdown(
                    "Supported formats: **PDF**, **DOCX**, **TXT**, **MD**, **CSV**, **RST**"
                )
                with gr.Row():
                    with gr.Column():
                        upload_file = gr.File(label="Select a file to upload")
                        chunk_size = gr.Slider(
                            minimum=100,
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
                    with gr.Column():
                        upload_status = gr.Markdown()
                upload_btn.click(
                    _upload_file,
                    inputs=[upload_file, chunk_size, chunk_overlap],
                    outputs=upload_status,
                )

            with gr.Tab("🗂️ Manage Documents"):
                gr.Markdown("### Documents in the knowledge base")
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                doc_list = gr.Markdown()
                refresh_btn.click(_list_documents, outputs=doc_list)
                app.load(_list_documents, outputs=doc_list)
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
                    _delete_document, inputs=delete_input, outputs=delete_status
                )

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
                chatbot = gr.Chatbot(label="Conversation")
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder="Ask something about your documents…",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ Clear conversation")
                send_btn.click(
                    _chat,
                    inputs=[msg_input, chatbot, n_results_slider, model_input, api_key_input],
                    outputs=[chatbot, msg_input],
                )
                msg_input.submit(
                    _chat,
                    inputs=[msg_input, chatbot, n_results_slider, model_input, api_key_input],
                    outputs=[chatbot, msg_input],
                )
                clear_btn.click(_clear_chat, outputs=chatbot)

            with gr.Tab("🔌 MCP Server"):
                gr.Markdown(
                    "\n### Connect an LLM to your knowledge base via MCP\n\n"
                    "The **MCP (Model Context Protocol) server** exposes your knowledge base as tools \n"
                    "that any compatible LLM or agent can use.\n\n"
                    "#### Starting the MCP server\n\n"
                    "Run the following command in a separate terminal:\n\n"
                    "```bash\npython mcp_server.py\n```\n\n"
                    "The server uses **stdio** transport (default for MCP), so you can connect it \n"
                    "directly to Claude Desktop, Cursor, or any MCP-compatible client.\n\n"
                    "#### Available MCP Tools\n\n"
                    "| Tool | Description |\n"
                    "|------|-------------|\n"
                    "| `search_knowledge_base` | Search for relevant context chunks by query |\n"
                    "| `list_documents` | List all documents in the knowledge base |\n"
                    "| `add_document_text` | Add raw text to the knowledge base |\n"
                    "| `delete_document` | Remove a document from the knowledge base |\n\n"
                    "#### Example `claude_desktop_config.json`\n\n"
                    "```json\n"
                    "{\n"
                    '  "mcpServers": {\n'
                    '    "rag-knowledge-base": {\n'
                    '      "command": "python",\n'
                    '      "args": ["/path/to/RAGwithMCP/mcp_server.py"]\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "```\n\n"
                    "#### Connecting with `mcp` CLI\n\n"
                    "```bash\nmcp dev mcp_server.py\n```\n"
                )
    return app


def launch(share: bool = False, port: int = 7860):
    app = build_app()
    app.launch(share=share, server_port=port, theme=gr.themes.Soft())
