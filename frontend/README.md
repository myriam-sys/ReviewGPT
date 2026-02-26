# ReviewGPT — Frontend

Next.js 14 (App Router) frontend for the ReviewGPT RAG application.

## Setup

```bash
cd frontend
npm install
npm run dev
```

The app starts at **http://localhost:3000**.

The FastAPI backend must be running on **http://localhost:8000**.
To start the backend from the repo root:

```bash
uvicorn backend.main:app --reload
```

## Environment

Copy `.env.local` is already pre-configured for local development:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production (Vercel), set `NEXT_PUBLIC_API_URL` to your Render backend URL in the
Vercel project settings under **Environment Variables**.

## Pages

| Route | Description |
|---|---|
| `/` | Upload a CSV/XLSX file and track embedding progress |
| `/chat/[session_id]` | Chat with an uploaded dataset |

## Stack

- **Next.js 14** (App Router, TypeScript)
- **Tailwind CSS** (dark theme)
- **fetch** — no external HTTP libraries
