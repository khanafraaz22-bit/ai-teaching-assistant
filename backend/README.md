# YT Course Assistant — Backend

AI-powered learning assistant for YouTube course playlists.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI |
| Database | MongoDB (Motor async) |
| Vector DB | ChromaDB (dev) |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Auth | JWT + bcrypt |

---

## Project Structure

```
backend/
├── main.py                         # App entry point
├── requirements.txt
├── .env.example                    # Copy to .env
└── app/
    ├── api/
    │   └── v1/
    │       ├── router.py           # Aggregates all routes
    │       └── endpoints/
    │           ├── auth.py         # Signup, login, password reset
    │           ├── courses.py      # Playlist ingestion, video listing
    │           ├── chat.py         # RAG chatbot
    │           └── learning.py     # Summarize, mindmap, exam, search
    ├── config/
    │   └── settings.py             # All env-driven config (Pydantic)
    ├── core/                       # Auth logic, security helpers
    ├── database/
    │   ├── mongodb.py              # Motor connection manager
    │   └── vectordb.py             # ChromaDB connection manager
    ├── models/
    │   └── schemas.py              # All Pydantic models & DB schemas
    ├── services/
    │   ├── auth/                   # JWT, hashing, email verification
    │   ├── youtube/                # Playlist parser, transcript extractor
    │   ├── rag/                    # Embedding, retrieval, LLM chain
    │   └── learning/               # Summarizer, mindmap, exam generator
    ├── utils/
    │   └── logging.py              # Loguru setup
    └── workers/                    # Background tasks (Celery — Phase 5)
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- MongoDB running locally (`mongod`)
- OpenAI API key

### 2. Setup

```bash
# Clone and enter the backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set SECRET_KEY and OPENAI_API_KEY at minimum
```

### 3. Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Explore the API

Open: http://localhost:8000/docs

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `SECRET_KEY` | ✅ | JWT signing key — generate with `openssl rand -hex 32` |
| `OPENAI_API_KEY` | ✅ | Your OpenAI API key |
| `MONGODB_URL` | — | Default: `mongodb://localhost:27017` |
| `MONGODB_DB_NAME` | — | Default: `yt_course_assistant` |
| `CHROMA_PERSIST_DIR` | — | Default: `./chroma_store` |
| `DEBUG` | — | Default: `false` |

---

## Development Phases

| Phase | Status | Description |
|---|---|---|
| 1 — Architecture | ✅ Done | Folder structure, config, DB connections, schemas |
| 2 — YouTube Processing | 🔜 Next | Playlist parser, transcript extraction, chunking |
| 3 — RAG Chatbot | 🔜 | Embedding, vector search, LLM Q&A |
| 4 — Learning Tools | 🔜 | Summarizer, mind map, exam generator |
| 5 — Auth | 🔜 | JWT, bcrypt, email verification |
