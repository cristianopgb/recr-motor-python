from __future__ import annotations

import math
import traceback
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas import RoteirizacaoRequest
from app.services.pipeline_service import executar_pipeline
from app.services.validation_service import validar_payload

router = APIRouter()


def _sanitize_for_json(value: Any) -> Any:
    if value is None:
        return None

    if value is pd.NaT:
        return None

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value

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

    if isinstance(value, np.ndarray):
        return [_sanitize_for_json(v) for v in value.tolist()]

    if isinstance(value, pd.Series):
        return [_sanitize_for_json(v) for v in value.tolist()]

    if isinstance(value, pd.Index):
        return [_sanitize_for_json(v) for v in value.tolist()]

    try:
        resultado_isna = pd.isna(value)
        if isinstance(resultado_isna, (bool, np.bool_)) and bool(resultado_isna):
            return None
    except Exception:
        pass

    return value


def _resposta_erro(
    *,
    mensagem: str,
    tipo_erro: str,
    detalhe_tecnico: str | None = None,
    traceback_texto: str | None = None,
) -> JSONResponse:
    conteudo_erro = {
        "status": "erro",
        "mensagem": mensagem,
        "tipo_erro": tipo_erro,
        "detalhe_tecnico": detalhe_tecnico,
        "traceback": traceback_texto,
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
                "mensagem": mensagem,
                "quantidade_entrada": None,
                "quantidade_saida": None,
            }
        ],
    }

    return JSONResponse(
        status_code=200,
        content=_sanitize_for_json(conteudo_erro),
    )


@router.post("/roteirizar")
def roteirizar(payload: RoteirizacaoRequest):
    # 1) validação do contrato de entrada
    try:
        validar_payload(payload)
    except Exception as e:
        return _resposta_erro(
            mensagem=f"VALIDACAO_CONTRATO: {str(e)}",
            tipo_erro="VALIDACAO_CONTRATO",
            detalhe_tecnico=type(e).__name__,
            traceback_texto=traceback.format_exc(),
        )

    # 2) execução/orquestração do pipeline
    try:
        resultado_pipeline = executar_pipeline(payload)
    except Exception as e:
        return _resposta_erro(
            mensagem=f"ERRO_PIPELINE: {str(e)}",
            tipo_erro="ERRO_PIPELINE",
            detalhe_tecnico=type(e).__name__,
            traceback_texto=traceback.format_exc(),
        )

    # 3) sanitização para JSON
    try:
        resultado_pipeline = _sanitize_for_json(resultado_pipeline)
    except Exception as e:
        return _resposta_erro(
            mensagem=f"ERRO_SANITIZACAO: {str(e)}",
            tipo_erro="ERRO_SANITIZACAO",
            detalhe_tecnico=type(e).__name__,
            traceback_texto=traceback.format_exc(),
        )

    # 4) devolve resultado bruto estruturado
    return JSONResponse(
        status_code=200,
        content=resultado_pipeline,
    )
