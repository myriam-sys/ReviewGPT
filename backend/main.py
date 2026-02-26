"""
ReviewGPT — FastAPI application entrypoint.

Initializes the FastAPI app, registers routers, pre-loads the embedding model
at startup, and exposes a health check endpoint.  All application-level
middleware (CORS, auth, rate limiting) will be configured here as the project
grows.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import chat, upload

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ReviewGPT API",
    description="RAG-based conversational interface for Google Maps reviews.",
    version="0.1.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


# ── Startup ───────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event() -> None:
    """
    Pre-load the embedding model so the first upload request is not penalised
    by the model-load time.

    Model weights are downloaded from HuggingFace on the very first run and
    cached in ``~/.cache/huggingface/`` for all subsequent starts.
    """
    try:
        from backend.services.embedding_service import load_model

        load_model()
    except Exception:
        # Model load failure is non-fatal at startup — embedding will be
        # retried lazily on the first upload.  Log prominently so it is
        # visible in server logs without crashing the app.
        logger.exception(
            "Failed to pre-load embedding model at startup — "
            "embedding will be attempted lazily on first upload."
        )


# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Liveness probe — confirms the API is running."""
    return {"status": "ok"}
