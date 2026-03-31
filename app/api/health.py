from fastapi import APIRouter

from app.config import settings
from app.schemas import HealthResponse

router = APIRouter()


@router.get("", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", versao=settings.APP_VERSION)
