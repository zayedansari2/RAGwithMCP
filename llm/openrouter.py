"""OpenRouter API client for chat completions."""
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/auto")


def get_api_key() -> str:
    key = os.environ.get("OPEN_ROUTER_API_KEY", "")
    if not key:
        raise ValueError(
            "OPEN_ROUTER_API_KEY is not set. Copy .env.example to .env and add your API key."
        )
    return key


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> str:
    """
    Send a chat completion request to OpenRouter.

    Args:
        messages: List of {"role": "...", "content": "..."} dicts.
        model: OpenRouter model identifier.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        api_key: Override the API key from the environment.

    Returns:
        The assistant's reply text.
    """
    key = api_key or get_api_key()
    response = requests.post(
        OPENROUTER_BASE_URL + "/chat/completions",
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/zayedansari2/RAGwithMCP",
            "X-Title": "RAGwithMCP",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected API response format: {data}") from exc


def rag_chat(
    user_question: str,
    context_chunks: list,
    chat_history: Optional[list[dict]] = None,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> str:
    """
    Ask a question with RAG context injected into the system prompt.

    Args:
        user_question: The user's question.
        context_chunks: Relevant document chunks retrieved from the vector store.
        chat_history: Previous messages in the conversation (optional).
        model: OpenRouter model identifier.
        api_key: Override the API key from the environment.

    Returns:
        The assistant's reply text.
    """
    context_block = "\n\n---\n\n".join(context_chunks)
    if context_block:
        system_prompt = (
            "You are a helpful assistant. Answer the user's question based on the provided context. "
            "If the context does not contain enough information, say so clearly and answer from your general knowledge if possible.\n\n"
            "Context:\n" + context_block
        )
    else:
        system_prompt = (
            "You are a helpful assistant. No relevant context was found in the knowledge base for this question. "
            "Answer using your general knowledge."
        )
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_question})
    return chat(messages, model=model, api_key=api_key)
