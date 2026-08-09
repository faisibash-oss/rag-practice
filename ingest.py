"""
Ingest all Markdown files in /docs into a local Chroma vector store.

Steps:
1. Read every .md file in docs/
2. Split each file into overlapping text chunks
3. Embed each chunk locally with sentence-transformers (all-MiniLM-L6-v2)
4. Store the chunks + embeddings in a persistent Chroma collection

Run this whenever you add or change files in docs/:
    python ingest.py
"""

import glob
import os

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 150  # characters shared between consecutive chunks


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, breaking on paragraph boundaries
    where possible so we don't cut sentences in half."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > chunk_size:
            chunks.append(current)
            # start the next chunk with the tail of the previous one, for overlap
            current = current[-overlap:] + "\n\n" + paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph

    if current.strip():
        chunks.append(current.strip())

    return chunks


def load_documents(docs_dir: str) -> list[dict]:
    """Read every .md file and split it into chunks with source metadata."""
    documents = []
    md_files = sorted(glob.glob(os.path.join(docs_dir, "*.md")))

    if not md_files:
        raise SystemExit(f"No .md files found in '{docs_dir}/'.")

    for path in md_files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        filename = os.path.basename(path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "id": f"{filename}::chunk-{i}",
                    "text": chunk,
                    "source": filename,
                }
            )

    return documents


def main() -> None:
    print(f"Loading documents from '{DOCS_DIR}/'...")
    documents = load_documents(DOCS_DIR)
    print(f"Split into {len(documents)} chunks from "
          f"{len(set(d['source'] for d in documents))} files.")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding chunks...")
    texts = [d["text"] for d in documents]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print(f"Writing to Chroma at '{CHROMA_DIR}/'...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop any existing collection so re-running ingest.py doesn't duplicate data
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=[d["id"] for d in documents],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": d["source"]} for d in documents],
    )

    print(f"Done. Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
