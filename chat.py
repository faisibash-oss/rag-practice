"""
Interactive terminal chat over the documents ingested by ingest.py.

For each question you type:
1. Embed the question locally with sentence-transformers (same model as ingest.py)
2. Retrieve the most relevant chunks from the local Chroma vector store
3. Send those chunks + your question to Claude
4. Stream and print Claude's answer

Run:
    python chat.py
"""

import os
import sys

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-6"
TOP_K = 4  # how many chunks to retrieve per question
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the \
provided context, which was retrieved from the user's own documents.

Rules:
- Base your answer strictly on the context below. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, say so plainly \
instead of guessing.
- Cite which source file(s) you drew from when relevant, using the source names \
given in the context.
- Be concise and direct."""


def load_api_key() -> None:
    """Load ANTHROPIC_API_KEY from the environment or a local .env file, and
    give clear setup instructions if it's still missing."""
    load_dotenv()  # picks up a .env file in the project root, if present

    if os.environ.get("ANTHROPIC_API_KEY"):
        return

    print(
        "ANTHROPIC_API_KEY is not set.\n\n"
        "This script expects it to already be available as an environment "
        "variable, or in a .env file in the project root.\n\n"
        "To set it for this project, create a file named '.env' next to this "
        "script containing:\n\n"
        "    ANTHROPIC_API_KEY=your-api-key-here\n\n"
        "(.env is already listed in .gitignore, so it won't be committed.)\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def format_context(results) -> str:
    """Turn Chroma query results into a numbered, source-labeled context block."""
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    parts = []
    for i, (text, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{text}")

    return "\n\n---\n\n".join(parts)


def main() -> None:
    load_api_key()

    if not os.path.isdir(CHROMA_DIR):
        raise SystemExit(
            f"No vector store found at '{CHROMA_DIR}/'. Run 'python ingest.py' first."
        )

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    anthropic_client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    print(f"Ready. {collection.count()} chunks loaded. Ask a question (Ctrl+C to quit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not question:
            continue

        query_embedding = embedder.encode([question]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)

        if not results["documents"][0]:
            print("Claude: I couldn't find any relevant content in the knowledge base.\n")
            continue

        context = format_context(results)
        user_message = f"Context:\n\n{context}\n\nQuestion: {question}"

        print("Claude: ", end="", flush=True)
        with anthropic_client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
