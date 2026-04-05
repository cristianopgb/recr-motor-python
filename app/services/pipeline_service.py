from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from app.pipeline.m1_padronizacao import executar_m1_padronizacao
from app.pipeline.m2_enriquecimento import executar_m2_enriquecimento
from app.pipeline.m3_triagem import executar_m3_triagem
from app.pipeline.m3_1_validacao_fronteira import executar_m3_1_validacao_fronteira
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

    resultado_m1 = executar_m1_padronizacao(
        df_carteira_raw=resultado_m0["df_carteira_raw"],
        df_geo_raw=resultado_m0["df_geo_raw"],
        df_parametros_raw=resultado_m0["df_parametros_raw"],
        df_veiculos_raw=resultado_m0["df_veiculos_raw"],
    )

    df_carteira_tratada = resultado_m1["df_carteira_tratada"]
    df_geo_tratado = resultado_m1["df_geo_tratado"]
    df_parametros_tratados = resultado_m1["df_parametros_tratados"]
    df_veiculos_tratados = resultado_m1["df_veiculos_tratados"]

    logs.append(
        _log(
            modulo="m1_padronizacao",
            status="ok",
            mensagem="M1 real executado com sucesso",
            quantidade_entrada=int(len(contexto.df_carteira_raw)),
            quantidade_saida=int(len(df_carteira_tratada)),
            extra={
                "carteira_colunas": int(len(df_carteira_tratada.columns)),
                "geo_colunas": int(len(df_geo_tratado.columns)),
                "parametros_colunas": int(len(df_parametros_tratados.columns)),
                "veiculos_colunas": int(len(df_veiculos_tratados.columns)),
            },
        )
    )

    df_carteira_enriquecida, resumo_m2 = executar_m2_enriquecimento(
        df_carteira_tratada=df_carteira_tratada,
        df_geo_tratado=df_geo_tratado,
        df_parametros_tratados=df_parametros_tratados,
        data_base_roteirizacao=contexto.data_base,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )

    logs.append(
        _log(
            modulo="m2_enriquecimento",
            status="ok",
            mensagem="M2 executado com sucesso",
            quantidade_entrada=int(len(df_carteira_tratada)),
            quantidade_saida=int(len(df_carteira_enriquecida)),
            extra=resumo_m2,
        )
    )

    df_carteira_triagem, meta_m3 = executar_m3_triagem(
        df_carteira_enriquecida=df_carteira_enriquecida,
        data_base_roteirizacao=contexto.data_base,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )

    outputs_m3 = meta_m3["outputs_m3"]
    resumo_m3 = meta_m3["resumo_m3"]

    df_carteira_roteirizavel = outputs_m3["df_carteira_roteirizavel"]
    df_carteira_entrega_futura = outputs_m3["df_carteira_entrega_futura"]
    df_carteira_aguardando_agendamento = outputs_m3["df_carteira_aguardando_agendamento"]
    df_carteira_excecoes_triagem = outputs_m3["df_carteira_excecoes_triagem"]

    logs.append(
        _log(
            modulo="m3_triagem",
            status="ok",
            mensagem="M3 executado com sucesso",
            quantidade_entrada=int(len(df_carteira_enriquecida)),
            quantidade_saida=int(len(df_carteira_triagem)),
            extra=resumo_m3,
        )
    )

    df_input_oficial_bloco_4, meta_m31 = executar_m3_1_validacao_fronteira(
        df_carteira_roteirizavel=df_carteira_roteirizavel,
        data_base_roteirizacao=contexto.data_base,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )
    resumo_m31 = meta_m31["resumo_m31"]

    logs.append(
        _log(
            modulo="m3_1_validacao_fronteira",
            status="ok",
            mensagem="M3.1 executado com sucesso e input oficial do bloco 4 foi consolidado",
            quantidade_entrada=int(len(df_carteira_roteirizavel)),
            quantidade_saida=int(len(df_input_oficial_bloco_4)),
            extra=resumo_m31,
        )
    )

    return {
        "status": "ok",
        "mensagem": "Motor executou com sucesso até o M3.1.",
        "pipeline_real_ate": "M3.1",
        "resumo": {
            "total_carteira": int(len(contexto.df_carteira_raw)),
            "total_veiculos": int(len(contexto.df_veiculos_raw)),
            "total_regionalidades": int(len(contexto.df_geo_raw)),
            "total_parametros": int(len(contexto.df_parametros_raw)),
            "filial_id": contexto.filial_id,
            "tipo_roteirizacao": contexto.tipo_roteirizacao,
            "data_base_roteirizacao": contexto.data_base.isoformat(),
            "resumo_m2": resumo_m2,
            "resumo_m3": resumo_m3,
            "resumo_m31": resumo_m31,
        },
        "contexto_rodada": {
            "filial": contexto.filial,
            "parametros_rodada": contexto.parametros_rodada,
        },
        "snapshots": {
            "carteira_raw": _snapshot_dataframe(contexto.df_carteira_raw, "df_carteira_raw"),
            "carteira_tratada": _snapshot_dataframe(df_carteira_tratada, "df_carteira_tratada"),
            "carteira_enriquecida": _snapshot_dataframe(df_carteira_enriquecida, "df_carteira_enriquecida"),
            "carteira_triagem": _snapshot_dataframe(df_carteira_triagem, "df_carteira_triagem"),
            "carteira_roteirizavel": _snapshot_dataframe(df_carteira_roteirizavel, "df_carteira_roteirizavel"),
            "input_oficial_bloco_4": _snapshot_dataframe(df_input_oficial_bloco_4, "df_input_oficial_bloco_4"),
            "carteira_entrega_futura": _snapshot_dataframe(df_carteira_entrega_futura, "df_carteira_entrega_futura"),
            "carteira_aguardando_agendamento": _snapshot_dataframe(
                df_carteira_aguardando_agendamento, "df_carteira_aguardando_agendamento"
            ),
            "carteira_excecoes_triagem": _snapshot_dataframe(
                df_carteira_excecoes_triagem, "df_carteira_excecoes_triagem"
            ),
            "regionalidades": _snapshot_dataframe(contexto.df_geo_raw, "df_geo_raw"),
            "parametros": _snapshot_dataframe(contexto.df_parametros_raw, "df_parametros_raw"),
            "veiculos": _snapshot_dataframe(contexto.df_veiculos_raw, "df_veiculos_raw"),
        },
        "amostras": {
            "carteira_tratada": _df_to_records(df_carteira_tratada, limit=5),
            "carteira_enriquecida": _df_to_records(df_carteira_enriquecida, limit=5),
            "carteira_roteirizavel": _df_to_records(df_carteira_roteirizavel, limit=5),
            "input_oficial_bloco_4": _df_to_records(df_input_oficial_bloco_4, limit=5),
            "entrega_futura": _df_to_records(df_carteira_entrega_futura, limit=5),
            "aguardando_agendamento": _df_to_records(df_carteira_aguardando_agendamento, limit=5),
            "excecoes_triagem": _df_to_records(df_carteira_excecoes_triagem, limit=5),
        },
        "outputs_intermediarios": {
            "df_input_oficial_bloco_4": _df_to_records(df_input_oficial_bloco_4),
        },
        "manifestos_fechados": [],
        "manifestos_compostos": [],
        "nao_roteirizados": [],
        "logs": logs,
    }

