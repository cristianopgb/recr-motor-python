import os
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "REC Roteirizador - Sistema 2 (Motor Python)"
    app_version: str = "1.0.0"
    ambiente: str = os.getenv("AMBIENTE", "development")

    # Callback para o Sistema 1
    callback_enabled: bool = os.getenv("CALLBACK_ENABLED", "false").strip().lower() in {"1", "true", "sim", "yes"}
    callback_url: str = os.getenv("CALLBACK_URL", "").strip()
    callback_secret: str = os.getenv("CALLBACK_SECRET", "").strip()
    callback_timeout_seconds: int = int(os.getenv("CALLBACK_TIMEOUT_SECONDS", "20"))


settings = Settings()
