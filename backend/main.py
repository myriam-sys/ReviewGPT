"""
ReviewGPT — FastAPI application entrypoint.

Initializes the FastAPI app, registers routers, and exposes a health check
endpoint. All application-level middleware (CORS, auth, rate limiting) will
be configured here as the project grows.
"""

from fastapi import FastAPI

from backend.routers import upload

app = FastAPI(
    title="ReviewGPT API",
    description="RAG-based conversational interface for Google Maps reviews.",
    version="0.1.0",
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload.router, prefix="/upload", tags=["upload"])

# TODO: Register chat router (Phase 4)
# from backend.routers import chat
# app.include_router(chat.router, prefix="/chat", tags=["chat"])

# TODO: Add CORS middleware before deploying frontend


# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Liveness probe — confirms the API is running."""
    return {"status": "ok"}
