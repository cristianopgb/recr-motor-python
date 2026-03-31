from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.roteirizacao import router as roteirizacao_router

app = FastAPI(
    title="REC Roteirizador - Sistema 2 (Motor Python)",
    description="API do motor de roteirização da REC Transportes",
    version="1.0.0",
)

app.include_router(health_router, tags=["Health"])
app.include_router(roteirizacao_router, tags=["Roteirização"])


@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "service": "rec-roteirizador-motor-python",
        "version": "1.0.0",
        "docs": "/docs",
    }
