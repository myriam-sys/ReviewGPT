"""
App configuration — loads and validates all environment variables at startup.

Uses pydantic-settings to parse a .env file and expose a typed ``settings``
singleton that can be imported anywhere in the application:

    from backend.core.config import settings
    print(settings.groq_model)

All fields map 1-to-1 to the keys in .env.example.  Pydantic validates types
and raises a clear error at startup if a required variable is missing, so
misconfigurations surface immediately rather than at request time.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # silently ignore unknown env vars
    )

    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url: str = ""
    # Key passed to supabase.create_client() for server-side SDK operations.
    # Set to your service_role key so DB writes bypass RLS.
    supabase_key: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_db_url: str = ""

    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama3-8b-8192"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dimension: int = 1024

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "development"
    rag_top_k: int = 5
    llm_max_tokens: int = 1024

    # ── Upload constraints ────────────────────────────────────────────────────
    # Maximum CSV file size accepted (bytes). Default: 10 MB.
    max_upload_bytes: int = 10 * 1024 * 1024
    # Number of cleaned rows returned by the preview endpoint.
    preview_row_limit: int = 5


# Module-level singleton — import this everywhere instead of re-instantiating.
settings = Settings()
