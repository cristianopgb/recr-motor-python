from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from app.schemas import RoteirizacaoRequest
from app.services.validation_service import validar_payload
from app.services.pipeline_service import executar_pipeline

router = APIRouter()


@router.post("/roteirizar")
def roteirizar(payload: RoteirizacaoRequest):
    try:
        # 1. validação do contrato de entrada
        validar_payload(payload)

        # 2. execução/orquestração do pipeline
        resultado_pipeline = executar_pipeline(payload)

        # 3. serialização segura para JSON
        return JSONResponse(
            status_code=200,
            content=jsonable_encoder(resultado_pipeline),
        )

    except ValueError as ve:
        return JSONResponse(
            status_code=200,
            content=jsonable_encoder({
                "status": "erro",
                "mensagem": str(ve),
                "tipo_erro": "VALIDACAO",
                "resumo": {
                    "total_manifestos": 0,
                    "total_manifestos_fechados": 0,
                    "total_manifestos_compostos": 0,
                    "total_nao_roteirizados": 0,
                },
                "manifestos_fechados": [],
                "manifestos_compostos": [],
                "nao_roteirizados": [],
                "logs": [
                    {
                        "modulo": "api_roteirizacao",
                        "status": "erro",
                        "mensagem": f"VALIDACAO: {str(ve)}",
                        "quantidade_entrada": None,
                        "quantidade_saida": None,
                    }
                ],
            }),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno no motor de roteirização: {str(e)}"
        )
