from fastapi import APIRouter, HTTPException
from app.schemas import RoteirizacaoRequest, RoteirizacaoResponse
from app.services.validation_service import validar_payload
from app.services.pipeline_service import executar_pipeline
from app.services.response_service import montar_resposta_sucesso, montar_resposta_erro

router = APIRouter()


@router.post("/roteirizar", response_model=RoteirizacaoResponse)
def roteirizar(payload: RoteirizacaoRequest):
    try:
        # =========================
        # 1. VALIDAÇÃO DO PAYLOAD
        # =========================
        validar_payload(payload)

        # =========================
        # 2. EXECUÇÃO DO PIPELINE
        # =========================
        resultado_pipeline = executar_pipeline(payload)

        # =========================
        # 3. MONTAGEM DA RESPOSTA
        # =========================
        response = montar_resposta_sucesso(resultado_pipeline)

        return response

    except ValueError as ve:
        # erro de validação controlado
        return montar_resposta_erro(
            mensagem=str(ve),
            tipo_erro="VALIDACAO"
        )

    except Exception as e:
        # erro inesperado
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno no motor de roteirização: {str(e)}"
        )
