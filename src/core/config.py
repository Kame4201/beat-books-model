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
    LOG_LEVEL: str = "INFO"

    model_config = {"env_file": ".env"}


settings = Settings()
