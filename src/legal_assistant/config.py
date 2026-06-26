from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/legal_assistant"
    redis_url: str = "redis://localhost:6379/0"

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_persist_dir: str = ""

    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    weather_provider: str = "open_meteo"
    qweather_api_key: str = ""
    gaode_api_key: str = ""

    max_history_turns: int = 20
    redis_session_ttl_seconds: int = 86400

    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    api_key: str = ""
    skip_auto_ingest: bool = False

    legal_disclaimer: str = "本回答仅供参考，不构成法律意见，具体问题请咨询执业律师。"


settings = Settings()
