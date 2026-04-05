from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import RoteirizacaoRequest
from app.services.pipeline_service import executar_pipeline
from app.services.validation_service import validar_payload

router = APIRouter()


def _sanitize_for_json(value: Any) -> Any:
    """
    Converte recursivamente valores não compatíveis com JSON:
    - np.nan -> None
    - pd.NaT -> None
    - inf / -inf -> None
    - tipos numpy -> tipos Python nativos
    """
    if value is None:
        return None

    if value is pd.NaT:
        return None

    if isinstance(value, (np.floating,)):
        value = float(value)

    if isinstance(value, (np.integer,)):
        value = int(value)

    if isinstance(value, (np.bool_,)):
        value = bool(value)

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]

    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]

    # fallback para objetos pandas/numpy estranhos
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


@router.post("/roteirizar")
def roteirizar(payload: RoteirizacaoRequest):
    try:
        # 1) validação do contrato de entrada
        validar_payload(payload)

        # 2) execução/orquestração do pipeline
        resultado_pipeline = executar_pipeline(payload)

        # 3) sanitização para JSON
        resultado_pipeline = _sanitize_for_json(resultado_pipeline)

        # 4) devolve resultado bruto estruturado
        return JSONResponse(
            status_code=200,
            content=resultado_pipeline,
        )

    except ValueError as ve:
        conteudo_erro = {
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
        }

        return JSONResponse(
            status_code=200,
            content=_sanitize_for_json(conteudo_erro),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno no motor de roteirização: {str(e)}"
        )
