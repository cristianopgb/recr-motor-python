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


def _snapshot_dataframe(df: pd.DataFrame, nome: str, max_colunas: int = 20) -> Dict[str, Any]:
    """
    Snapshot leve para auditoria.
    Não devolve a base inteira; devolve metadados úteis para diagnóstico.
    """
    return {
        "nome": nome,
        "linhas": int(len(df)),
        "colunas": list(df.columns[:max_colunas]),
        "qtd_colunas_total": int(len(df.columns)),
    }


def _executar_m0_adapter(contexto: PipelineContext) -> Dict[str, Any]:
    """
    M0 adaptado para API:
    - não lê Excel
    - recebe DataFrames já montados pela camada adaptadora
    - inventaria a rodada
    """
    df_carteira_raw = contexto.df_carteira_raw
    df_geo_raw = contexto.df_geo_raw
    df_parametros_raw = contexto.df_parametros_raw
    df_veiculos_raw = contexto.df_veiculos_raw

    inventario = {
        "rodada_id": contexto.rodada_id,
        "upload_id": contexto.upload_id,
        "usuario_id": contexto.usuario_id,
        "filial_id": contexto.filial_id,
        "tipo_roteirizacao": contexto.tipo_roteirizacao,
        "data_execucao": contexto.data_execucao.isoformat(),
        "data_base": contexto.data_base.isoformat(),
        "inputs": {
            "carteira": _snapshot_dataframe(df_carteira_raw, "df_carteira_raw"),
            "regionalidades": _snapshot_dataframe(df_geo_raw, "df_geo_raw"),
            "parametros": _snapshot_dataframe(df_parametros_raw, "df_parametros_raw"),
            "veiculos": _snapshot_dataframe(df_veiculos_raw, "df_veiculos_raw"),
        },
        "caminhos_pipeline": contexto.caminhos_pipeline,
    }

    return {
        "inventario": inventario,
        "df_carteira_raw": df_carteira_raw,
        "df_geo_raw": df_geo_raw,
        "df_parametros_raw": df_parametros_raw,
        "df_veiculos_raw": df_veiculos_raw,
        "DATA_BASE": contexto.data_base,
        "caminhos_pipeline": contexto.caminhos_pipeline,
        "metadados_rodada": contexto.metadados_rodada,
    }


def _executar_stub_pipeline(contexto: PipelineContext, resultado_m0: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stub estruturado do pipeline.

    Importante:
    - ainda NÃO roda M1..M9 reais
    - apenas encapsula a entrada de forma auditável
    - devolve estrutura bruta para, no próximo passo, encaixarmos os módulos reais
    """
    df_carteira_raw = resultado_m0["df_carteira_raw"]
    df_veiculos_raw = resultado_m0["df_veiculos_raw"]
    df_geo_raw = resultado_m0["df_geo_raw"]
    df_parametros_raw = resultado_m0["df_parametros_raw"]

    total_carteira = int(len(df_carteira_raw))
    total_veiculos = int(len(df_veiculos_raw))
    total_regionalidades = int(len(df_geo_raw))
    total_parametros = int(len(df_parametros_raw))

    # modo frota: extrai configuração para auditoria
    configuracao_frota = []
    if "parametro" in df_parametros_raw.columns and "valor" in df_parametros_raw.columns:
        for _, row in df_parametros_raw.iterrows():
            parametro = str(row["parametro"])
            valor = row["valor"]

            if parametro.startswith("configuracao_frota__"):
                configuracao_frota.append(
                    {
                        "parametro": parametro,
                        "valor": valor,
                    }
                )

    resumo = {
        "rodada_id": contexto.rodada_id,
        "upload_id": contexto.upload_id,
        "tipo_roteirizacao": contexto.tipo_roteirizacao,
        "data_base_roteirizacao": contexto.data_base.isoformat(),
        "total_carteira_recebida": total_carteira,
        "total_veiculos_recebidos": total_veiculos,
        "total_regionalidades_recebidas": total_regionalidades,
        "total_parametros_recebidos": total_parametros,
        "pipeline_real_executado": False,
        "fase_atual": "stub_estruturado",
        "total_manifestos_fechados": 0,
        "total_manifestos_compostos": 0,
        "total_nao_roteirizados": 0,
        "total_manifestos": 0,
    }

    pipeline_bruto = {
        "contexto": {
            "rodada_id": contexto.rodada_id,
            "upload_id": contexto.upload_id,
            "usuario_id": contexto.usuario_id,
            "filial_id": contexto.filial_id,
            "tipo_roteirizacao": contexto.tipo_roteirizacao,
            "data_execucao": contexto.data_execucao.isoformat(),
            "data_base": contexto.data_base.isoformat(),
        },
        "m0": resultado_m0["inventario"],
        "entrada_bruta": {
            "carteira": _snapshot_dataframe(df_carteira_raw, "df_carteira_raw"),
            "veiculos": _snapshot_dataframe(df_veiculos_raw, "df_veiculos_raw"),
            "regionalidades": _snapshot_dataframe(df_geo_raw, "df_geo_raw"),
            "parametros": _snapshot_dataframe(df_parametros_raw, "df_parametros_raw"),
        },
        "modo_frota_configuracao": configuracao_frota,
        "proximo_passo": "encaixar_modulos_reais_do_notebook",
    }

    return {
        "mensagem": (
            "Payload recebido, validado e encapsulado com sucesso. "
            "O pipeline real M0–M9 ainda não foi plugado nesta versão do serviço."
        ),
        "resumo": resumo,
        "manifestos_fechados": [],
        "manifestos_compostos": [],
        "nao_roteirizados": [],
        "pipeline_bruto": pipeline_bruto,
    }


def executar_pipeline(payload: RoteirizacaoRequest) -> Dict[str, Any]:
    """
    Orquestrador central do Sistema 2.

    Fluxo:
    1. recebe payload validado
    2. normaliza para o formato interno do pipeline
    3. executa M0 adaptado
    4. devolve estrutura bruta auditável
    """
    logs: List[Dict[str, Any]] = []

    logs.append(
        _log(
            modulo="pipeline_service",
            status="inicio",
            mensagem="Iniciando encapsulamento do pipeline a partir do payload da API.",
        )
    )

    contexto = normalizar_payload_para_pipeline(payload)

    logs.append(
        _log(
            modulo="payload_adapter",
            status="ok",
            mensagem="Payload convertido para contexto interno do pipeline.",
            quantidade_entrada=int(len(payload.carteira)),
            quantidade_saida=int(len(contexto.df_carteira_raw)),
            extra={
                "rodada_id": contexto.rodada_id,
                "tipo_roteirizacao": contexto.tipo_roteirizacao,
            },
        )
    )

    resultado_m0 = _executar_m0_adapter(contexto)

    logs.append(
        _log(
            modulo="m0_adapter",
            status="ok",
            mensagem="M0 adaptado executado com inventário da rodada a partir de DataFrames já recebidos.",
            quantidade_entrada=int(len(contexto.df_carteira_raw)),
            quantidade_saida=int(len(contexto.df_carteira_raw)),
            extra={
                "qtd_veiculos": int(len(contexto.df_veiculos_raw)),
                "qtd_regionalidades": int(len(contexto.df_geo_raw)),
                "qtd_parametros": int(len(contexto.df_parametros_raw)),
            },
        )
    )

    resultado_stub = _executar_stub_pipeline(contexto, resultado_m0)

    logs.append(
        _log(
            modulo="pipeline_stub",
            status="ok",
            mensagem=(
                "Stub estruturado executado. Próximo passo: acoplar os módulos reais "
                "do notebook ao serviço."
            ),
            quantidade_entrada=int(len(contexto.df_carteira_raw)),
            quantidade_saida=0,
        )
    )

    resultado_stub["logs"] = logs
    return resultado_stub
