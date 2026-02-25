# ReviewGPT

A RAG-based web application that lets businesses upload their Google Maps reviews and chat with their data through a conversational interface.

## Architecture Overview

```
┌─────────────────┐        ┌──────────────────────────────────────────┐
│                 │        │              Backend (FastAPI)            │
│  Next.js        │  HTTP  │                                          │
│  Frontend       │◄──────►│  /upload  ──► embedding_service         │
│  (Vercel)       │        │                    │                     │
│                 │        │  /chat    ──► retrieval_service          │
└─────────────────┘        │                    │                     │
                           │               llm_service               │
                           └──────────────────┬───────────────────────┘
                                              │
                     ┌────────────────────────┼────────────────┐
                     │                        │                │
              ┌──────▼──────┐        ┌────────▼──────┐  ┌─────▼──────┐
              │  Supabase   │        │  Groq API     │  │ HuggingFace│
              │  pgvector   │        │  (LLM)        │  │ Embeddings │
              │  (vectors + │        │               │  │            │
              │   metadata) │        └───────────────┘  └────────────┘
              └─────────────┘
```

### Data Flow

1. **Upload**: User uploads a CSV of Google Maps reviews → reviews are chunked, embedded via `multilingual-e5-large`, and stored in Supabase pgvector with session/business ID metadata.
2. **Chat**: User sends a query → query is embedded → nearest-neighbor search in Supabase → top-k review chunks are retrieved → LLM (Groq) synthesizes a response.
3. **Multi-tenancy**: All vector records are tagged with a `business_id` (derived from the upload session) so data is fully isolated per user/business.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js (Vercel) |
| Backend | FastAPI, Python 3.11+ (Render.com) |
| Embeddings | `multilingual-e5-large` via sentence-transformers |
| Vector DB | Supabase (PostgreSQL + pgvector) |
| LLM | Groq API (llama3 / mixtral) |
| RAG | Manual implementation (LlamaIndex later) |

## Supported Languages

English, French, Arabic, Spanish, Portuguese

## Project Structure

```
ReviewGPT/
├── backend/
│   ├── main.py                    # FastAPI entrypoint
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── upload.py              # CSV upload & ingestion endpoints
│   │   └── chat.py               # Conversational query endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding_service.py   # Embed text with multilingual-e5-large
│   │   ├── retrieval_service.py   # pgvector similarity search
│   │   └── llm_service.py        # Groq LLM calls & prompt construction
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic request/response models
│   ├── db/
│   │   ├── __init__.py
│   │   ├── supabase_client.py    # Supabase SDK + asyncpg pool singletons
│   │   └── migrations/
│   │       └── 001_create_reviews.sql  # reviews table + pgvector indexes
│   └── core/
│       ├── __init__.py
│       └── config.py             # Env var loading via pydantic-settings
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- A [Supabase](https://supabase.com) project with the `pgvector` extension enabled
- A [Groq](https://console.groq.com) API key

### 1. Clone the repository

```bash
git clone https://github.com/myriam-sys/ReviewGPT.git
cd ReviewGPT
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your actual credentials
```

### 5. Run the backend locally

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

### 6. Run the database migration

Open the [Supabase SQL editor](https://supabase.com/dashboard/project/_/sql) for your project and run the contents of [backend/db/migrations/001_create_reviews.sql](backend/db/migrations/001_create_reviews.sql).

This migration:
- Enables the `vector` extension (pgvector)
- Creates the `reviews` table with all required columns, including a `vector(1024)` column for embeddings (populated in Phase 3)
- Creates indexes on `session_id` and `has_text` for fast lookups
- Enables Row-Level Security so the anon key cannot read rows directly

```sql
-- Quick reference — run the full file, not just this snippet:
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS reviews (
    review_id  UUID PRIMARY KEY,
    session_id UUID NOT NULL,
    ...
);
```

The migration is idempotent (`CREATE … IF NOT EXISTS`) and safe to run more than once.

## Deployment

| Service | Target |
|---|---|
| Backend | [Render.com](https://render.com) (Web Service, Python) |
| Frontend | [Vercel](https://vercel.com) |
| Database | [Supabase](https://supabase.com) |

## License

MIT
