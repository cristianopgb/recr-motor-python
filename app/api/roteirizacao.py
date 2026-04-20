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
from app.services.m8_contract_service import build_m8_contract
from app.services.callback_service import should_send_callback, send_callback

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
    try:
        validar_payload(payload)
    except Exception as e:
        return _resposta_erro(
            mensagem=f"VALIDACAO_CONTRATO: {str(e)}",
            tipo_erro="VALIDACAO_CONTRATO",
            detalhe_tecnico=type(e).__name__,
            traceback_texto=traceback.format_exc(),
        )

    try:
        resultado_pipeline = executar_pipeline(payload)
    except Exception as e:
        return _resposta_erro(
            mensagem=f"ERRO_PIPELINE: {str(e)}",
            tipo_erro="ERRO_PIPELINE",
            detalhe_tecnico=type(e).__name__,
            traceback_texto=traceback.format_exc(),
        )

    try:
        payload_dict = payload.model_dump(mode="python")
        contrato_m8 = build_m8_contract(resultado_pipeline)

        callback_resultado = {
            "callback_enviado": False,
            "callback_status": "desabilitado",
            "callback_http_status": None,
            "callback_url": "",
            "callback_mensagem": "Callback desabilitado.",
        }

        if should_send_callback(payload_dict):
            callback_resultado = send_callback(payload_dict, contrato_m8)

        contrato_m8["callback_resultado"] = callback_resultado
        contrato_m8 = _sanitize_for_json(contrato_m8)

    except Exception as e:
        return _resposta_erro(
            mensagem=f"ERRO_M8_CALLBACK: {str(e)}",
            tipo_erro="ERRO_M8_CALLBACK",
            detalhe_tecnico=type(e).__name__,
            traceback_texto=traceback.format_exc(),
        )

    return JSONResponse(
        status_code=200,
        content=contrato_m8,
    )
