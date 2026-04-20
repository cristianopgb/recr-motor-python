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


def _build_m8_error_response(
    *,
    payload: RoteirizacaoRequest | None,
    mensagem: str,
    tipo_erro: str,
    pipeline_real_ate: str,
    detalhe_tecnico: str | None = None,
    traceback_texto: str | None = None,
) -> dict[str, Any]:
    resumo_execucao: dict[str, Any] = {
        "rodada_id": "",
        "upload_id": "",
        "filial_id": "",
        "usuario_id": "",
        "data_execucao": None,
        "origem_sistema": None,
        "tipo_roteirizacao": None,
        "modelo_roteirizacao": None,
        "versao_motor": None,
        "tempos_ms": {
            "tempo_total_pipeline_ms": None,
            "tempo_leitura_ms": None,
            "tempo_geocodificacao_ms": None,
            "tempo_otimizacao_ms": None,
            "tempo_montagem_ms": None,
        },
    }

    contexto_rodada: dict[str, Any] = {
        "rodada_id": None,
        "upload_id": None,
        "filial_id": None,
        "usuario_id": None,
        "data_base_roteirizacao": None,
        "tipo_roteirizacao": None,
        "filtros_aplicados": None,
        "configuracao_frota": None,
    }

    if payload is not None:
        try:
            payload_dict = payload.model_dump(mode="python")

            resumo_execucao.update(
                {
                    "rodada_id": str(payload_dict.get("rodada_id") or ""),
                    "upload_id": str(payload_dict.get("upload_id") or ""),
                    "filial_id": str(payload_dict.get("filial_id") or ""),
                    "usuario_id": str(payload_dict.get("usuario_id") or ""),
                    "data_execucao": None,
                    "origem_sistema": (
                        (payload_dict.get("parametros") or {}).get("origem_sistema")
                        if isinstance(payload_dict.get("parametros"), dict)
                        else None
                    ),
                    "tipo_roteirizacao": payload_dict.get("tipo_roteirizacao"),
                    "modelo_roteirizacao": (
                        (payload_dict.get("parametros") or {}).get("modelo_roteirizacao")
                        if isinstance(payload_dict.get("parametros"), dict)
                        else None
                    ),
                }
            )

            contexto_rodada.update(
                {
                    "rodada_id": payload_dict.get("rodada_id"),
                    "upload_id": payload_dict.get("upload_id"),
                    "filial_id": payload_dict.get("filial_id"),
                    "usuario_id": payload_dict.get("usuario_id"),
                    "data_base_roteirizacao": payload_dict.get("data_base_roteirizacao"),
                    "tipo_roteirizacao": payload_dict.get("tipo_roteirizacao"),
                    "filtros_aplicados": (
                        (payload_dict.get("parametros") or {}).get("filtros_aplicados")
                        if isinstance(payload_dict.get("parametros"), dict)
                        else None
                    ),
                    "configuracao_frota": payload_dict.get("configuracao_frota"),
                }
            )
        except Exception:
            pass

    contrato_erro = {
        "status": "erro",
        "mensagem": mensagem,
        "pipeline_real_ate": pipeline_real_ate,
        "modo_resposta": "contrato_retorno_sistema_1_m8",
        "resumo_execucao": resumo_execucao,
        "contexto_rodada": contexto_rodada,
        "status_modulos": [
            {
                "modulo": pipeline_real_ate,
                "status": "erro",
                "mensagem": mensagem,
                "tempo_ms": None,
                "quantidade_entrada": None,
                "quantidade_saida": None,
            }
        ],
        "estatisticas_roteirizacao": {
            "carteira": {
                "total_carteira": 0,
                "total_roteirizavel": 0,
                "total_agendamento_futuro": 0,
                "total_agendas_vencidas": 0,
                "total_excecoes": 0,
                "total_sem_agenda": 0,
            },
            "cargas": {
                "total_manifestos_m7": 0,
                "total_itens_m7": 0,
                "cargas_fechadas_m4": 0,
                "cargas_compostas_m5": 0,
                "remanescente_m6_2": 0,
                "km_total_m7": 0.0,
                "km_medio_manifesto_m7": 0.0,
                "ocupacao_media_manifesto_m7": 0.0,
                "qtd_media_itens_por_manifesto": 0.0,
            },
            "por_veiculo": {
                "cargas_por_perfil": [],
                "ocupacao_por_perfil": [],
                "km_por_perfil": [],
            },
            "por_cidade": [],
            "por_leadtime_agenda": [],
        },
        "resultados": {
            "manifestos": [],
            "itens_manifestos": [],
            "tentativas_m7": [],
        },
        "auditoria": {
            "auditoria_m7": {},
            "logs": [
                {
                    "modulo": pipeline_real_ate,
                    "status": "erro",
                    "mensagem": mensagem,
                    "tempo_ms": None,
                    "quantidade_entrada": None,
                    "quantidade_saida": None,
                }
            ],
        },
        "motor_response_raw": {
            "status": "erro",
            "mensagem": mensagem,
            "tipo_erro": tipo_erro,
            "pipeline_real_ate": pipeline_real_ate,
            "detalhe_tecnico": detalhe_tecnico,
            "traceback": traceback_texto,
        },
        "callback_resultado": {
            "callback_enviado": False,
            "callback_status": "nao_aplicavel",
            "callback_http_status": None,
            "callback_url": "",
            "callback_mensagem": "Callback não executado porque a requisição falhou antes do retorno final.",
        },
    }

    return contrato_erro


def _resposta_erro(
    *,
    payload: RoteirizacaoRequest | None,
    mensagem: str,
    tipo_erro: str,
    pipeline_real_ate: str,
    detalhe_tecnico: str | None = None,
    traceback_texto: str | None = None,
) -> JSONResponse:
    conteudo_erro = _build_m8_error_response(
        payload=payload,
        mensagem=mensagem,
        tipo_erro=tipo_erro,
        pipeline_real_ate=pipeline_real_ate,
        detalhe_tecnico=detalhe_tecnico,
        traceback_texto=traceback_texto,
    )

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
            payload=payload,
            mensagem=f"VALIDACAO_CONTRATO: {str(e)}",
            tipo_erro="VALIDACAO_CONTRATO",
            pipeline_real_ate="VALIDACAO_CONTRATO",
            detalhe_tecnico=type(e).__name__,
            traceback_texto=traceback.format_exc(),
        )

    try:
        resultado_pipeline = executar_pipeline(payload)
    except Exception as e:
        return _resposta_erro(
            payload=payload,
            mensagem=f"ERRO_PIPELINE: {str(e)}",
            tipo_erro="ERRO_PIPELINE",
            pipeline_real_ate="ERRO_PIPELINE",
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
            payload=payload,
            mensagem=f"ERRO_M8_CALLBACK: {str(e)}",
            tipo_erro="ERRO_M8_CALLBACK",
            pipeline_real_ate="ERRO_M8_CALLBACK",
            detalhe_tecnico=type(e).__name__,
            traceback_texto=traceback.format_exc(),
        )

    return JSONResponse(
        status_code=200,
        content=contrato_m8,
    )
