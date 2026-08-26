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

    # Exact model id from your provider's console. Groq renames models often,
    # so confirm the current id before deploying (see README).
    llm_model: str = "llama-3.3-70b-versatile"

    # --- Embeddings (local, free, no API key) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store ---
    chroma_dir: str = ".chroma"


# Import-time singleton used across the app.
settings = Settings()
