#!/usr/bin/env python3
"""Entry point: launch the RAG with MCP web interface."""

import os
import argparse
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="RAG with MCP — web interface")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 7860)),
        help="Port to listen on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public Gradio share link",
    )
    args = parser.parse_args()

    from gui.app import launch

    launch(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
