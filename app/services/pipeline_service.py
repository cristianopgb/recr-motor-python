from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from app.schemas import RoteirizacaoRequest
from app.services.payload_service import PipelineContext, normalizar_payload_para_pipeline

from app.pipeline.m0_leitura import executar_m0_adapter
from app.pipeline.m1_padronizacao import executar_m1_padronizacao
from app.pipeline.m2_enriquecimento import executar_m2_enriquecimento
from app.pipeline.m3_triagem import executar_m3_triagem


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
    return {
        "nome": nome,
        "linhas": int(len(df)),
        "colunas": list(df.columns[:max_colunas]),
        "qtd_colunas_total": int(len(df.columns)),
    }


def executar_pipeline(payload: RoteirizacaoRequest) -> Dict[str, Any]:
    """
    Orquestra a execução real do pipeline no Sistema 2.

    Fluxo atual:
    - normalização do payload
    - M0 adaptado
    - M1 real
    - M2 real
    - M3 real

    Observação:
    - modo 'frota' ainda não foi implementado no pipeline real
    """

    logs: list[Dict[str, Any]] = []

    try:
        logs.append(_log("pipeline_service", "inicio", "Iniciando normalização do payload."))

        contexto: PipelineContext = normalizar_payload_para_pipeline(payload)

        logs.append(
            _log(
                "payload_service",
                "ok",
                "Payload normalizado com sucesso.",
                extra={
                    "rodada_id": contexto.rodada_id,
                    "upload_id": contexto.upload_id,
                    "usuario_id": contexto.usuario_id,
                    "filial_id": contexto.filial_id,
                    "tipo_roteirizacao": contexto.tipo_roteirizacao,
                    "data_base": contexto.data_base.isoformat(),
                },
            )
        )

        if str(contexto.tipo_roteirizacao).lower() == "frota":
            raise ValueError(
                "O modo 'frota' ainda não foi implementado no pipeline real do Sistema 2."
            )

        # ============================================================
        # M0 - ADAPTADOR / INVENTÁRIO DA RODADA
        # ============================================================
        logs.append(
            _log(
                "M0",
                "inicio",
                "Executando M0 adaptado.",
                quantidade_entrada=int(len(contexto.df_carteira_raw)),
            )
        )

        resultado_m0 = executar_m0_adapter(contexto)

        df_carteira_raw = resultado_m0["df_carteira_raw"]
        df_geo_raw = resultado_m0["df_geo_raw"]
        df_parametros_raw = resultado_m0["df_parametros_raw"]
        df_veiculos_raw = resultado_m0["df_veiculos_raw"]
        DATA_BASE = resultado_m0["DATA_BASE"]
        caminhos_pipeline = resultado_m0["caminhos_pipeline"]
        inventario = resultado_m0["inventario"]

        logs.append(
            _log(
                "M0",
                "ok",
                "M0 adaptado executado com sucesso.",
                quantidade_entrada=int(len(df_carteira_raw)),
                quantidade_saida=int(len(df_carteira_raw)),
            )
        )

        # ============================================================
        # M1 - PADRONIZAÇÃO
        # ============================================================
        logs.append(
            _log(
                "M1",
                "inicio",
                "Executando M1 padronização.",
                quantidade_entrada=int(len(df_carteira_raw)),
            )
        )

        df_carteira_tratada, meta_m1 = executar_m1_padronizacao(
            df_carteira_raw=df_carteira_raw,
            df_geo_raw=df_geo_raw,
            df_parametros_raw=df_parametros_raw,
            df_veiculos_raw=df_veiculos_raw,
            data_base_roteirizacao=DATA_BASE,
            caminhos_pipeline=caminhos_pipeline,
        )

        df_geo_tratado = meta_m1["outputs_m1"]["df_geo_tratado"]
        df_parametros_tratados = meta_m1["outputs_m1"]["df_parametros_tratados"]
        df_veiculos_tratados = meta_m1["outputs_m1"]["df_veiculos_tratados"]

        logs.append(
            _log(
                "M1",
                "ok",
                "M1 executado com sucesso.",
                quantidade_entrada=int(len(df_carteira_raw)),
                quantidade_saida=int(len(df_carteira_tratada)),
                extra={
                    "colunas_carteira_tratada": int(len(df_carteira_tratada.columns)),
                    "linhas_geo_tratado": int(len(df_geo_tratado)),
                    "linhas_parametros_tratados": int(len(df_parametros_tratados)),
                    "linhas_veiculos_tratados": int(len(df_veiculos_tratados)),
                },
            )
        )

        # ============================================================
        # M2 - ENRIQUECIMENTO
        # ============================================================
        logs.append(
            _log(
                "M2",
                "inicio",
                "Executando M2 enriquecimento geográfico e temporal.",
                quantidade_entrada=int(len(df_carteira_tratada)),
            )
        )

        df_carteira_enriquecida, meta_m2 = executar_m2_enriquecimento(
            df_carteira_tratada=df_carteira_tratada,
            df_geo_tratado=df_geo_tratado,
            df_parametros_tratados=df_parametros_tratados,
            data_base_roteirizacao=DATA_BASE,
            caminhos_pipeline=caminhos_pipeline,
        )

        logs.append(
            _log(
                "M2",
                "ok",
                "M2 executado com sucesso.",
                quantidade_entrada=int(len(df_carteira_tratada)),
                quantidade_saida=int(len(df_carteira_enriquecida)),
                extra={
                    "colunas_carteira_enriquecida": int(len(df_carteira_enriquecida.columns)),
                    "resumo_m2": meta_m2.get("resumo_m2", {}),
                },
            )
        )

        # ============================================================
        # M3 - TRIAGEM
        # ============================================================
        logs.append(
            _log(
                "M3",
                "inicio",
                "Executando M3 triagem operacional.",
                quantidade_entrada=int(len(df_carteira_enriquecida)),
            )
        )

        df_carteira_triagem, meta_m3 = executar_m3_triagem(
            df_carteira_enriquecida=df_carteira_enriquecida,
            data_base_roteirizacao=DATA_BASE,
            caminhos_pipeline=caminhos_pipeline,
        )

        outputs_m3 = meta_m3["outputs_m3"]
        df_carteira_roteirizavel = outputs_m3["df_carteira_roteirizavel"]
        df_carteira_entrega_futura = outputs_m3["df_carteira_entrega_futura"]
        df_carteira_aguardando_agendamento = outputs_m3["df_carteira_aguardando_agendamento"]
        df_carteira_excecoes_triagem = outputs_m3["df_carteira_excecoes_triagem"]

        logs.append(
            _log(
                "M3",
                "ok",
                "M3 executado com sucesso.",
                quantidade_entrada=int(len(df_carteira_enriquecida)),
                quantidade_saida=int(len(df_carteira_triagem)),
                extra={
                    "carteira_roteirizavel": int(len(df_carteira_roteirizavel)),
                    "carteira_entrega_futura": int(len(df_carteira_entrega_futura)),
                    "carteira_aguardando_agendamento": int(len(df_carteira_aguardando_agendamento)),
                    "carteira_excecoes_triagem": int(len(df_carteira_excecoes_triagem)),
                    "resumo_m3": meta_m3.get("resumo_m3", {}),
                },
            )
        )

        # ============================================================
        # RESPOSTA FINAL ATÉ M3
        # ============================================================
        resumo = {
            "pipeline_real_ate": "M3",
            "rodada_id": contexto.rodada_id,
            "upload_id": contexto.upload_id,
            "usuario_id": contexto.usuario_id,
            "filial_id": contexto.filial_id,
            "tipo_roteirizacao": contexto.tipo_roteirizacao,
            "data_base_roteirizacao": DATA_BASE.isoformat(),
            "total_linhas_input": int(len(df_carteira_raw)),
            "total_linhas_tratadas": int(len(df_carteira_tratada)),
            "total_linhas_enriquecidas": int(len(df_carteira_enriquecida)),
            "total_linhas_triagem": int(len(df_carteira_triagem)),
            "carteira_roteirizavel": int(len(df_carteira_roteirizavel)),
            "carteira_entrega_futura": int(len(df_carteira_entrega_futura)),
            "carteira_aguardando_agendamento": int(len(df_carteira_aguardando_agendamento)),
            "carteira_excecoes_triagem": int(len(df_carteira_excecoes_triagem)),
        }

        snapshots = {
            "df_carteira_raw": _snapshot_dataframe(df_carteira_raw, "df_carteira_raw"),
            "df_carteira_tratada": _snapshot_dataframe(df_carteira_tratada, "df_carteira_tratada"),
            "df_carteira_enriquecida": _snapshot_dataframe(df_carteira_enriquecida, "df_carteira_enriquecida"),
            "df_carteira_triagem": _snapshot_dataframe(df_carteira_triagem, "df_carteira_triagem"),
            "df_carteira_roteirizavel": _snapshot_dataframe(df_carteira_roteirizavel, "df_carteira_roteirizavel"),
            "df_carteira_entrega_futura": _snapshot_dataframe(df_carteira_entrega_futura, "df_carteira_entrega_futura"),
            "df_carteira_aguardando_agendamento": _snapshot_dataframe(
                df_carteira_aguardando_agendamento, "df_carteira_aguardando_agendamento"
            ),
            "df_carteira_excecoes_triagem": _snapshot_dataframe(
                df_carteira_excecoes_triagem, "df_carteira_excecoes_triagem"
            ),
            "df_geo_tratado": _snapshot_dataframe(df_geo_tratado, "df_geo_tratado"),
            "df_parametros_tratados": _snapshot_dataframe(df_parametros_tratados, "df_parametros_tratados"),
            "df_veiculos_tratados": _snapshot_dataframe(df_veiculos_tratados, "df_veiculos_tratados"),
        }

        amostras = {
            "carteira_roteirizavel": df_carteira_roteirizavel.head(10).to_dict(orient="records"),
            "entrega_futura": df_carteira_entrega_futura.head(10).to_dict(orient="records"),
            "aguardando_agendamento": df_carteira_aguardando_agendamento.head(10).to_dict(orient="records"),
            "excecoes_triagem": df_carteira_excecoes_triagem.head(10).to_dict(orient="records"),
        }

        logs.append(
            _log(
                "pipeline_service",
                "ok",
                "Pipeline executado com sucesso até o M3.",
                quantidade_entrada=int(len(df_carteira_raw)),
                quantidade_saida=int(len(df_carteira_triagem)),
            )
        )

        return {
            "status": "ok",
            "mensagem": "Motor executou o pipeline real até o M3.",
            "resumo": resumo,
            "inventario": inventario,
            "snapshots": snapshots,
            "amostras": amostras,
            "modulos": {
                "m1": meta_m1.get("resumo_m1", {}),
                "m2": meta_m2.get("resumo_m2", {}),
                "m3": meta_m3.get("resumo_m3", {}),
            },
            "logs": logs,
        }

    except Exception as e:
        logs.append(
            _log(
                "pipeline_service",
                "erro",
                f"Falha na execução do pipeline: {str(e)}",
            )
        )
        raise
