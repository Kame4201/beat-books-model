from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration for beat-books-model service."""

    # Database (READ-ONLY access to shared DB)
    DATABASE_URL: str

    # Model artifacts
    MODEL_ARTIFACTS_PATH: str = "model_artifacts"

    # Feature engineering
    DEFAULT_ROLLING_WINDOWS: str = "3,5,10"  # game windows for rolling averages

    # App
    ENV: str = "local"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"  # nosec B104
    API_PORT: int = 8002

    # Service version (read from pyproject.toml or set via env)
    VERSION: str = "0.1.0"

    model_config = {"env_file": ".env"}


settings = Settings()  # type: ignore[call-arg]  # populated by env/.env at runtime
