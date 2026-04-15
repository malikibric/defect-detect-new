from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "DefectDetect Web API"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
    
    # Security
    SECRET_KEY: str = "a_very_secret_key_change_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Database (asyncpg)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/defectdetect"
    DATABASE_SYNC_URL: str = "postgresql+psycopg://postgres:password@localhost:5432/defectdetect"
    DB_ECHO: bool = False
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # File storage
    STORAGE_BACKEND: str = "local"
    LOCAL_STORAGE_ROOT: str = "output/uploads"
    MAX_UPLOAD_SIZE_MB: int = 100

    # Optional S3 settings (used when STORAGE_BACKEND=s3)
    S3_BUCKET_NAME: str = ""
    S3_REGION: str = ""

    @field_validator("API_V1_PREFIX")
    @classmethod
    def validate_prefix(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("API_V1_PREFIX must start with '/'")
        return value.rstrip("/") or "/"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
