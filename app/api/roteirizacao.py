from fastapi import APIRouter, HTTPException

from app.schemas import RoteirizacaoRequest, RoteirizacaoResponse
from app.services import pipeline_service, validation_service

router = APIRouter()


@router.post("", response_model=RoteirizacaoResponse)
def roteirizar(request: RoteirizacaoRequest):
    erros = validation_service.validar(request.payload)
    if erros:
        raise HTTPException(status_code=422, detail={"errors": erros})

    resultado = pipeline_service.executar(request.payload)
    return RoteirizacaoResponse(sucesso=True, resultado=resultado)
