from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from app.schemas import RoteirizacaoRequest
from app.services.payload_service import PipelineContext, normalizar_payload_para_pipeline


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
        "colunas": list(df.columns[:max_colunas]),
        "qtd_colunas_total": int(len(df.columns)),
    }


def _df_to_records(df: pd.DataFrame, limit: int | None = None) -> List[Dict[str, Any]]:
    df2 = df.copy()
    if limit is not None:
        df2 = df2.head(limit)

    for col in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[col]):
            df2[col] = df2[col].astype(str)

    df2 = df2.where(pd.notnull(df2), None)
    return df2.to_dict(orient="records")


def _executar_m0_adapter(contexto: PipelineContext) -> Dict[str, Any]:
    inventario = {
        "rodada_id": contexto.rodada_id,
        "upload_id": contexto.upload_id,
        "usuario_id": contexto.usuario_id,
        "filial_id": contexto.filial_id,
        "tipo_roteirizacao": contexto.tipo_roteirizacao,
        "data_execucao": contexto.data_execucao.isoformat(),
        "data_base_roteirizacao": contexto.data_base.isoformat(),
        "filial": contexto.filial,
        "inputs": {
            "carteira": _snapshot_dataframe(contexto.df_carteira_raw, "df_carteira_raw"),
            "regionalidades": _snapshot_dataframe(contexto.df_geo_raw, "df_geo_raw"),
            "parametros": _snapshot_dataframe(contexto.df_parametros_raw, "df_parametros_raw"),
            "veiculos": _snapshot_dataframe(contexto.df_veiculos_raw, "df_veiculos_raw"),
        },
        "caminhos_pipeline": contexto.caminhos_pipeline,
    }

    return {
        "inventario": inventario,
        "df_carteira_raw": contexto.df_carteira_raw,
        "df_geo_raw": contexto.df_geo_raw,
        "df_parametros_raw": contexto.df_parametros_raw,
        "df_veiculos_raw": contexto.df_veiculos_raw,
        "DATA_BASE": contexto.data_base,
        "filial_rodada": contexto.filial,
        "parametros_rodada": contexto.parametros_rodada,
        "caminhos_pipeline": contexto.caminhos_pipeline,
        "metadados_rodada": contexto.metadados_rodada,
    }


def _executar_stub_pipeline(contexto: PipelineContext, resultado_m0: Dict[str, Any]) -> Dict[str, Any]:
    df_carteira_raw = resultado_m0["df_carteira_raw"]
    df_veiculos_raw = resultado_m0["df_veiculos_raw"]
    df_geo_raw = resultado_m0["df_geo_raw"]
    df_parametros_raw = resultado_m0["df_parametros_raw"]

    return {
        "status": "ok",
        "mensagem": "Motor recebeu o novo contrato da rodada e preparou o contexto até antes do M2.",
        "pipeline_real_ate": "M0_adapter",
        "resumo": {
            "total_carteira": int(len(df_carteira_raw)),
            "total_veiculos": int(len(df_veiculos_raw)),
            "total_regionalidades": int(len(df_geo_raw)),
            "total_parametros": int(len(df_parametros_raw)),
            "filial_id": contexto.filial_id,
            "tipo_roteirizacao": contexto.tipo_roteirizacao,
            "data_base_roteirizacao": contexto.data_base.isoformat(),
        },
        "contexto_rodada": {
            "filial": contexto.filial,
            "parametros_rodada": contexto.parametros_rodada,
        },
        "snapshots": {
            "carteira": _snapshot_dataframe(df_carteira_raw, "df_carteira_raw"),
            "regionalidades": _snapshot_dataframe(df_geo_raw, "df_geo_raw"),
            "parametros": _snapshot_dataframe(df_parametros_raw, "df_parametros_raw"),
            "veiculos": _snapshot_dataframe(df_veiculos_raw, "df_veiculos_raw"),
        },
        "amostras": {
            "carteira": _df_to_records(df_carteira_raw, limit=5),
            "regionalidades": _df_to_records(df_geo_raw, limit=5),
            "parametros": _df_to_records(df_parametros_raw, limit=15),
            "veiculos": _df_to_records(df_veiculos_raw, limit=5),
        },
        "manifestos_fechados": [],
        "manifestos_compostos": [],
        "nao_roteirizados": [],
    }


def executar_pipeline(payload: RoteirizacaoRequest) -> Dict[str, Any]:
    logs: List[Dict[str, Any]] = []

    contexto = normalizar_payload_para_pipeline(payload)
    logs.append(
        _log(
            modulo="payload_service",
            status="ok",
            mensagem="Payload normalizado para o contexto interno do pipeline",
            extra={
                "rodada_id": contexto.rodada_id,
                "filial_id": contexto.filial_id,
                "data_base_roteirizacao": contexto.data_base.isoformat(),
            },
        )
    )

    resultado_m0 = _executar_m0_adapter(contexto)
    logs.append(
        _log(
            modulo="m0_adapter",
            status="ok",
            mensagem="M0 adaptado executado com sucesso",
            quantidade_entrada=int(len(contexto.df_carteira_raw)),
            quantidade_saida=int(len(contexto.df_carteira_raw)),
            extra={
                "filial": contexto.filial,
                "data_base_roteirizacao": contexto.data_base.isoformat(),
            },
        )
    )

    resultado = _executar_stub_pipeline(contexto, resultado_m0)
    logs.append(
        _log(
            modulo="pipeline_service",
            status="ok",
            mensagem="Contexto pronto para encaixe do M1/M2 com contrato atualizado",
        )
    )

    resultado["logs"] = logs
    return resultado
