"""Application configuration.

All settings are read from environment variables (or a local `.env` file).
Never hard-code secrets — copy `.env.example` to `.env` and fill it in.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings loaded from the environment / .env file.

    pydantic-settings maps env vars case-insensitively, so the environment
    variable ``GROQ_API_KEY`` populates the ``groq_api_key`` field below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_env: str = "dev"

    # --- LLM provider (free tiers) ---
    # Get a free key at https://console.groq.com  (primary generator)
    groq_api_key: str = ""
    # Optional fallback — free key at https://aistudio.google.com/apikey
    gemini_api_key: str = ""

    # Groq is OpenAI-compatible. Verified 2026-08-26: use openai/gpt-oss-20b for
    # RAG answer generation. Groq renames models often — discover the current list
    # via GET {groq_base_url}/models. (The old llama-3.3-* ids no longer exist.)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-20b"

    # --- Embeddings (local, free, no API key) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store ---
    chroma_dir: str = ".chroma"


# Import-time singleton used across the app.
settings = Settings()
