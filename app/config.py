from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = "./storage/chroma"
    ENABLE_ADVANCED_SEARCH: bool = True
    MAX_CONTEXT_CHUNKS: int = 10
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
