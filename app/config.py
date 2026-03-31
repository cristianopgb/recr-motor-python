from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "REC Roteirizador - Sistema 2 (Motor Python)"
    app_version: str = "1.0.0"
    ambiente: str = "development"


settings = Settings()
