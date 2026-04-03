from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.roteirizacao import router as roteirizacao_router

app = FastAPI(
    title="REC Roteirizador - Sistema 2 (Motor Python)",
    description="API do motor de roteirização da REC Transportes",
    version="1.0.0",
)

# 🔥 CONFIGURAÇÃO DE CORS (ESSENCIAL)
origins = [
    "*",  # em desenvolvimento libera tudo
    # depois você pode restringir para o domínio do seu frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # permite OPTIONS, POST, GET, etc
    allow_headers=["*"],   # permite todos headers
)

# 🔌 ROTAS
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
