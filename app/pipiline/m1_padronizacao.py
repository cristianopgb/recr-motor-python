from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import pandas as pd

from app.schemas import RoteirizacaoRequest
from app.services.payload_service import normalizar_payload_para_pipeline
from app.pipeline.m1_padronizacao import executar_m1_padronizacao


def _log(
    modulo: str,
    status: str,
    mensagem: str,
    quantidade_entrada: int | None = None,
    quantidade_saida: int | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    registro = {
        "modulo": modulo,
        "status": status,
        "mensagem": mensagem,
        "quantidade_entrada": quantidade_entrada,
        "quantidade_saida": quantidade_saida,
    }
    if extra:
        registro["extra"] = extra
    return registro


def _snapshot_dataframe(df: pd.DataFrame, nome: str, max_colunas: int = 30) -> Dict[str, Any]:
    return {
        "nome": nome,
        "linhas": int(len(df)),
        "qtd_colunas_total": int(len(df.columns)),
        "colunas": list(df.columns[:max_colunas]),
    }


def executar_pipeline(payload: RoteirizacaoRequest) -> Dict[str, Any]:
    logs = []
    data_execucao = datetime.now()

    # ============================================================
    # 1) NORMALIZA PAYLOAD DE ENTRADA
    # ============================================================
    contexto = normalizar_payload_para_pipeline(payload)

    logs.append(
        _log(
            modulo="payload_service",
            status="ok",
            mensagem="Payload normalizado para estruturas internas do pipeline.",
            extra={
                "rodada_id": contexto.rodada_id,
                "tipo_roteirizacao": contexto.tipo_roteirizacao,
            },
        )
    )

    # ============================================================
    # 2) M0 ADAPTADO - INVENTÁRIO DA RODADA
    # ============================================================
    df_carteira_raw = contexto.df_carteira_raw.copy()
    df_geo_raw = contexto.df_geo_raw.copy()
    df_parametros_raw = contexto.df_parametros_raw.copy()
    df_veiculos_raw = contexto.df_veiculos_raw.copy()

    inventario_entrada = {
        "rodada_id": contexto.rodada_id,
        "upload_id": contexto.upload_id,
        "usuario_id": contexto.usuario_id,
        "filial_id": contexto.filial_id,
        "tipo_roteirizacao": contexto.tipo_roteirizacao,
        "data_execucao": data_execucao.isoformat(),
        "data_base": str(contexto.data_base),
        "inputs": {
            "carteira": _snapshot_dataframe(df_carteira_raw, "df_carteira_raw"),
            "regionalidades": _snapshot_dataframe(df_geo_raw, "df_geo_raw"),
            "parametros": _snapshot_dataframe(df_parametros_raw, "df_parametros_raw"),
            "veiculos": _snapshot_dataframe(df_veiculos_raw, "df_veiculos_raw"),
        },
    }

    logs.append(
        _log(
            modulo="m0_adapter",
            status="ok",
            mensagem="Inventário inicial da rodada gerado com sucesso.",
            quantidade_entrada=len(df_carteira_raw),
            quantidade_saida=len(df_carteira_raw),
        )
    )

    # ============================================================
    # 3) M1 - PADRONIZAÇÃO
    # ============================================================
    resultado_m1 = executar_m1_padronizacao(
        df_carteira_raw=df_carteira_raw,
        df_geo_raw=df_geo_raw,
        df_parametros_raw=df_parametros_raw,
        df_veiculos_raw=df_veiculos_raw,
    )

    df_carteira_tratada = resultado_m1["df_carteira_tratada"]
    df_geo_tratado = resultado_m1["df_geo_tratado"]
    df_parametros_tratados = resultado_m1["df_parametros_tratados"]
    df_veiculos_tratados = resultado_m1["df_veiculos_tratados"]

    logs.append(
        _log(
            modulo="m1_padronizacao",
            status="ok",
            mensagem="M1 executado com sucesso.",
            quantidade_entrada=len(df_carteira_raw),
            quantidade_saida=len(df_carteira_tratada),
            extra={
                "carteira_colunas_finais": list(df_carteira_tratada.columns),
                "geo_colunas_finais": list(df_geo_tratado.columns),
                "parametros_colunas_finais": list(df_parametros_tratados.columns),
                "veiculos_colunas_finais": list(df_veiculos_tratados.columns),
            },
        )
    )

    # ============================================================
    # 4) RESUMO OPERACIONAL TEMPORÁRIO
    # ============================================================
    resumo = {
        "rodada_id": contexto.rodada_id,
        "tipo_roteirizacao": contexto.tipo_roteirizacao,
        "total_carteira_recebida": int(len(df_carteira_raw)),
        "total_carteira_tratada": int(len(df_carteira_tratada)),
        "total_regionalidades": int(len(df_geo_tratado)),
        "total_parametros": int(len(df_parametros_tratados)),
        "total_veiculos": int(len(df_veiculos_tratados)),
        "total_manifestos": 0,
        "total_manifestos_fechados": 0,
        "total_manifestos_compostos": 0,
        "total_nao_roteirizados": 0,
        "pipeline_real_ate": "M1",
    }

    # ============================================================
    # 5) RESPOSTA TEMPORÁRIA PARA TESTE PONTA A PONTA
    # ============================================================
    return {
        "status": "ok",
        "mensagem": "Motor executado com sucesso até o M1.",
        "resumo": resumo,
        "inventario_entrada": inventario_entrada,
        "snapshots": {
            "df_carteira_tratada": _snapshot_dataframe(df_carteira_tratada, "df_carteira_tratada"),
            "df_geo_tratado": _snapshot_dataframe(df_geo_tratado, "df_geo_tratado"),
            "df_parametros_tratados": _snapshot_dataframe(df_parametros_tratados, "df_parametros_tratados"),
            "df_veiculos_tratados": _snapshot_dataframe(df_veiculos_tratados, "df_veiculos_tratados"),
        },
        "manifestos_fechados": [],
        "manifestos_compostos": [],
        "nao_roteirizados": [],
        "logs": logs,
        "debug": {
            "amostra_carteira_tratada": df_carteira_tratada.head(5).fillna("").to_dict(orient="records"),
            "amostra_geo_tratado": df_geo_tratado.head(5).fillna("").to_dict(orient="records"),
            "amostra_parametros_tratados": df_parametros_tratados.head(5).fillna("").to_dict(orient="records"),
            "amostra_veiculos_tratados": df_veiculos_tratados.head(5).fillna("").to_dict(orient="records"),
        },
    }
