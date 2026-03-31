from fastapi import FastAPI

from app.api import health, roteirizacao
from app.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(roteirizacao.router, prefix="/roteirizacao", tags=["roteirizacao"])
