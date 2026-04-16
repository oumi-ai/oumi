from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    github_token: str | None = None
    github_repo: str = "oumi-ai/oumi"
    cache_ttl_seconds: int = 300
    cache_dir: str = ".cache/github"
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False


settings = Settings()
