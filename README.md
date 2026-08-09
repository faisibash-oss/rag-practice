# rag-practice

A minimal, fully local **Retrieval-Augmented Generation (RAG)** pipeline —
ask questions in the terminal and get answers grounded in your own
documents, powered by [Claude](https://www.anthropic.com/claude) for
generation and free local models for everything else.

No cloud vector database, no paid embedding API, no web UI — just two
Python scripts and a `/docs` folder.

## What is RAG?

Large language models are trained on a fixed snapshot of data — they
don't know your private documents, and can't cite where an answer came
from. **Retrieval-Augmented Generation** fixes this by retrieving the
most relevant passages from *your* documents at question time and handing
them to the model as context, so it answers using your material instead
of (or in addition to) what it memorized during training.

RAG has two phases:

1. **Ingest** (run once, or whenever your docs change): split documents
   into chunks, convert each chunk into a numerical vector (an
   "embedding"), and store those vectors in a searchable index.
2. **Query** (run on every question): embed the question with the same
   model, find the stored chunks whose vectors are most similar, and pass
   those chunks + the question to an LLM to generate a grounded answer.

The `docs/` folder in this repo has more detail on each concept — it's
also the sample content this pipeline is built to search over.

## How this pipeline works, step by step

```
docs/*.md ──► ingest.py ──► chroma_db/ (local vector store)
                                   │
your question ──► chat.py ────────┤
                                   ▼
                         embed question, search chroma_db/
                                   │
                                   ▼
                    top-k matching chunks + your question
                                   │
                                   ▼
                          Claude (claude-sonnet-4-6)
                                   │
                                   ▼
                            answer, streamed to terminal
```

**Components:**

| Piece | What it does | Tech |
|---|---|---|
| Vector store | Stores document chunks + their embeddings, supports similarity search | [Chroma](https://www.trychroma.com/) — runs locally, no API key |
| Embedding model | Turns text into vectors so semantically similar text ends up numerically close | [sentence-transformers](https://www.sbert.net/), model `all-MiniLM-L6-v2` — runs locally, no API key |
| LLM | Reads the retrieved context + your question and writes the answer | [Claude API](https://platform.claude.com), model `claude-sonnet-4-6` |
| `ingest.py` | One-time (or as-needed) script: reads `docs/*.md`, chunks them, embeds the chunks, writes them to Chroma | — |
| `chat.py` | Interactive loop: embeds your question, retrieves matching chunks, calls Claude, streams the answer | — |

**In `ingest.py`:**

1. Every `.md` file in `docs/` is read and split into ~1000-character
   chunks (with a small overlap between consecutive chunks, so an idea
   split across a chunk boundary isn't lost).
2. Each chunk is embedded locally with `all-MiniLM-L6-v2` — a small,
   fast sentence-transformers model that runs entirely on your machine.
3. Chunks + embeddings + source filenames are written to a persistent
   Chroma collection on disk (`chroma_db/`).

**In `chat.py`:**

1. You type a question.
2. The question is embedded with the *same* embedding model used during
   ingestion (this consistency matters — embeddings from different
   models aren't comparable).
3. Chroma returns the top 4 most similar chunks.
4. Those chunks are inserted into a prompt alongside your question and
   sent to Claude, with a system prompt instructing it to answer only
   from the provided context.
5. Claude's answer streams back to the terminal.

Only your question and the retrieved chunks are ever sent to the Claude
API — the embedding and retrieval steps are 100% local.

## Prerequisites

- **Python 3.9+**. Check with `python --version`. If you don't have
  Python installed, get it from [python.org/downloads](https://www.python.org/downloads/)
  (on Windows, make sure to check "Add python.exe to PATH" during install).
- **An Anthropic API key**. Get one at
  [platform.claude.com](https://platform.claude.com) if you don't have
  one already.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

### API key

This project reads `ANTHROPIC_API_KEY` from your environment, or from a
`.env` file in the project root (loaded automatically via `python-dotenv`).

If you already have `ANTHROPIC_API_KEY` set as a system/user environment
variable, you don't need to do anything else — `chat.py` will pick it up.

Otherwise, create a `.env` file in this folder:

```bash
cp .env.example .env
```

Then edit `.env` and replace the placeholder with your real key:

```
ANTHROPIC_API_KEY=sk-ant-your-real-key-here
```

`.env` is already excluded via `.gitignore`, so it won't be committed.

## Usage

**1. Ingest the documents** (run this first, and again any time you add
or edit files in `docs/`):

```bash
python ingest.py
```

The first run will download the embedding model (~90MB) — subsequent
runs use the cached copy and are much faster.

**2. Chat with your documents:**

```bash
python chat.py
```

Type a question and press Enter. Ask something like *"What is RAG?"* or
*"How does chunking work?"* — the answers should be grounded in the
placeholder docs. Press `Ctrl+C` to quit.

## Adding your own content

The `docs/` folder currently contains placeholder content about RAG
concepts, so you can verify the pipeline works end-to-end before adding
real material. To use your own documents:

1. Add or replace `.md` files in `docs/`.
2. Re-run `python ingest.py` to rebuild the vector store.
3. Run `python chat.py` and ask questions about the new content.

Only Markdown (`.md`) files are picked up by `ingest.py` — to support
other formats (PDF, plain text, etc.), you'd extend `load_documents()` in
`ingest.py` to read and convert them to text first.

## Project structure

```
rag-practice/
├── docs/                  # Source documents (Markdown)
├── ingest.py               # Chunk + embed + store docs in Chroma
├── chat.py                 # Ask questions, retrieve context, call Claude
├── requirements.txt
├── .env.example             # Template for your API key
└── chroma_db/               # Generated by ingest.py — not committed
```

## Notes on design choices

- **Chunking** is done with simple paragraph-aware fixed-size splitting
  (no external chunking library) — see `docs/how-chunking-works.md` for
  why chunk size and overlap matter.
- **Chroma runs embedded/local** (`PersistentClient`), not as a separate
  server — good for prototyping and small-to-medium document sets. For
  larger scale, Chroma can also run as a standalone server, or you could
  swap in another vector store (Pinecone, Qdrant, pgvector, etc.) without
  changing the rest of the pipeline.
- **Retrieval is unranked beyond cosine similarity** — no re-ranking
  step. For a small, focused document set like this one, similarity
  search alone is usually good enough; larger or noisier corpora often
  benefit from adding a re-ranking model on top of the initial retrieval.
