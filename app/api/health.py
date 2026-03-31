from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": "rec-roteirizador-motor-python",
        "message": "API funcionando corretamente"
    }
