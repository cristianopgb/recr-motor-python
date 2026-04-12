from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd

from app.pipeline.m1_padronizacao import executar_m1_padronizacao
from app.pipeline.m2_enriquecimento import executar_m2_enriquecimento
from app.pipeline.m3_triagem import executar_m3_triagem
from app.pipeline.m3_1_validacao_fronteira import executar_m3_1_validacao_fronteira
from app.pipeline.m4_manifestos_fechados import executar_m4_manifestos_fechados
from app.pipeline.m5_1_triagem_cidades import executar_m5_1_triagem_cidades
from app.pipeline.m5_2_composicao_cidades import executar_m5_2_composicao_cidades
from app.pipeline.m5_3_triagem_subregioes import executar_m5_3_triagem_subregioes
from app.pipeline.m5_4a_preparacao_subregioes import executar_m5_4a_preparacao_subregioes
from app.pipeline.m5_4b_aderencia_frota_subregioes import executar_m5_4b_aderencia_frota_subregioes
from app.schemas import RoteirizacaoRequest
from app.services.payload_service import PipelineContext, normalizar_payload_para_pipeline


def _agora() -> float:
    return time.perf_counter()


def _duracao_ms(inicio: float) -> float:
    return round((time.perf_counter() - inicio) * 1000, 2)


def _safe_len(obj: Any) -> int:
    try:
        return int(len(obj))
    except Exception:
        return 0


def _is_debug(payload: RoteirizacaoRequest) -> bool:
    for attr in ("modo_debug", "debug", "retornar_debug", "incluir_debug"):
        try:
            valor = getattr(payload, attr, False)
            if isinstance(valor, bool):
                return valor
            if isinstance(valor, str):
                return valor.strip().lower() in {"1", "true", "sim", "yes"}
        except Exception:
            continue
    return False


def _log(
    modulo: str,
    status: str,
    mensagem: str,
    quantidade_entrada: int | None = None,
    quantidade_saida: int | None = None,
    tempo_ms: float | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    registro = {
        "modulo": modulo,
        "status": status,
        "mensagem": mensagem,
        "quantidade_entrada": quantidade_entrada,
        "quantidade_saida": quantidade_saida,
    }
    if tempo_ms is not None:
        registro["tempo_ms"] = tempo_ms
    if extra:
        registro["extra"] = extra
    return registro


def _snapshot_dataframe(df: pd.DataFrame, nome: str, max_colunas: int = 30) -> Dict[str, Any]:
    if df is None:
        return {
            "nome": nome,
            "linhas": 0,
            "colunas": [],
            "qtd_colunas_total": 0,
        }

    return {
        "nome": nome,
        "linhas": int(len(df)),
        "colunas": list(df.columns[:max_colunas]),
        "qtd_colunas_total": int(len(df.columns)),
    }


def _serializar_dataframe_para_records(
    df: pd.DataFrame,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    df2 = df.copy()

    if limit is not None:
        df2 = df2.head(limit)

    for col in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[col]):
            df2[col] = df2[col].astype(str)

    df2 = df2.where(pd.notnull(df2), None)
    return df2.to_dict(orient="records")


def _montar_resumo_dataframe(df: pd.DataFrame, nome: str) -> Dict[str, Any]:
    return {
        "nome": nome,
        "total_linhas": _safe_len(df),
        "qtd_colunas": int(len(df.columns)) if isinstance(df, pd.DataFrame) else 0,
    }


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
    }


def executar_pipeline(payload: RoteirizacaoRequest) -> Dict[str, Any]:
    inicio_total = _agora()
    logs: List[Dict[str, Any]] = []
    metricas_tempo: Dict[str, float] = {}
    debug = _is_debug(payload)

    # =========================================================================================
    # PAYLOAD -> CONTEXTO
    # =========================================================================================
    t0 = _agora()
    contexto = normalizar_payload_para_pipeline(payload)
    tempo_payload = _duracao_ms(t0)
    metricas_tempo["payload_service_ms"] = tempo_payload

    logs.append(
        _log(
            modulo="payload_service",
            status="ok",
            mensagem="Payload normalizado para o contexto interno do pipeline",
            quantidade_entrada=_safe_len(contexto.df_carteira_raw),
            quantidade_saida=_safe_len(contexto.df_carteira_raw),
            tempo_ms=tempo_payload,
            extra={
                "rodada_id": contexto.rodada_id,
                "filial_id": contexto.filial_id,
                "data_base_roteirizacao": contexto.data_base.isoformat(),
                "tipo_roteirizacao": contexto.tipo_roteirizacao,
            },
        )
    )

    # =========================================================================================
    # M0
    # =========================================================================================
    t0 = _agora()
    resultado_m0 = _executar_m0_adapter(contexto)
    tempo_m0 = _duracao_ms(t0)
    metricas_tempo["m0_adapter_ms"] = tempo_m0

    logs.append(
        _log(
            modulo="m0_adapter",
            status="ok",
            mensagem="M0 adaptado executado com sucesso",
            quantidade_entrada=_safe_len(contexto.df_carteira_raw),
            quantidade_saida=_safe_len(contexto.df_carteira_raw),
            tempo_ms=tempo_m0,
            extra={
                "filial": contexto.filial,
                "data_base_roteirizacao": contexto.data_base.isoformat(),
                "tipo_roteirizacao": contexto.tipo_roteirizacao,
            },
        )
    )

    # =========================================================================================
    # M1
    # =========================================================================================
    t0 = _agora()
    resultado_m1 = executar_m1_padronizacao(
        df_carteira_raw=resultado_m0["df_carteira_raw"],
        df_geo_raw=resultado_m0["df_geo_raw"],
        df_parametros_raw=resultado_m0["df_parametros_raw"],
        df_veiculos_raw=resultado_m0["df_veiculos_raw"],
    )
    tempo_m1 = _duracao_ms(t0)
    metricas_tempo["m1_padronizacao_ms"] = tempo_m1

    df_carteira_tratada = resultado_m1["df_carteira_tratada"]
    df_geo_tratado = resultado_m1["df_geo_tratado"]
    df_parametros_tratados = resultado_m1["df_parametros_tratados"]
    df_veiculos_tratados = resultado_m1["df_veiculos_tratados"]

    resumo_m1 = {
        "carteira_colunas": int(len(df_carteira_tratada.columns)),
        "geo_colunas": int(len(df_geo_tratado.columns)),
        "parametros_colunas": int(len(df_parametros_tratados.columns)),
        "veiculos_colunas": int(len(df_veiculos_tratados.columns)),
    }

    logs.append(
        _log(
            modulo="m1_padronizacao",
            status="ok",
            mensagem="M1 executado com sucesso",
            quantidade_entrada=_safe_len(contexto.df_carteira_raw),
            quantidade_saida=_safe_len(df_carteira_tratada),
            tempo_ms=tempo_m1,
            extra=resumo_m1,
        )
    )

    # =========================================================================================
    # M2
    # =========================================================================================
    t0 = _agora()
    df_carteira_enriquecida, resumo_m2 = executar_m2_enriquecimento(
        df_carteira_tratada=df_carteira_tratada,
        df_geo_tratado=df_geo_tratado,
        df_parametros_tratados=df_parametros_tratados,
        data_base_roteirizacao=contexto.data_base,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )
    tempo_m2 = _duracao_ms(t0)
    metricas_tempo["m2_enriquecimento_ms"] = tempo_m2

    logs.append(
        _log(
            modulo="m2_enriquecimento",
            status="ok",
            mensagem="M2 executado com sucesso",
            quantidade_entrada=_safe_len(df_carteira_tratada),
            quantidade_saida=_safe_len(df_carteira_enriquecida),
            tempo_ms=tempo_m2,
            extra=resumo_m2,
        )
    )

    # =========================================================================================
    # M3
    # =========================================================================================
    t0 = _agora()
    df_carteira_triagem, meta_m3 = executar_m3_triagem(
        df_carteira_enriquecida=df_carteira_enriquecida,
        data_base_roteirizacao=contexto.data_base,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )
    tempo_m3 = _duracao_ms(t0)
    metricas_tempo["m3_triagem_ms"] = tempo_m3

    outputs_m3 = meta_m3["outputs_m3"]
    resumo_m3 = meta_m3["resumo_m3"]

    df_carteira_roteirizavel = outputs_m3["df_carteira_roteirizavel"]
    df_carteira_agendamento_futuro = outputs_m3["df_carteira_agendamento_futuro"]
    df_carteira_agendas_vencidas = outputs_m3["df_carteira_agendas_vencidas"]
    df_carteira_excecoes_triagem = outputs_m3.get("df_carteira_excecoes_triagem", pd.DataFrame())

    logs.append(
        _log(
            modulo="m3_triagem",
            status="ok",
            mensagem="M3 executado com sucesso",
            quantidade_entrada=_safe_len(df_carteira_enriquecida),
            quantidade_saida=_safe_len(df_carteira_triagem),
            tempo_ms=tempo_m3,
            extra=resumo_m3,
        )
    )

    # =========================================================================================
    # M3.1
    # =========================================================================================
    t0 = _agora()
    df_input_oficial_bloco_4, meta_m31 = executar_m3_1_validacao_fronteira(
        df_carteira_roteirizavel=df_carteira_roteirizavel,
        data_base_roteirizacao=contexto.data_base,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )
    tempo_m31 = _duracao_ms(t0)
    metricas_tempo["m3_1_validacao_fronteira_ms"] = tempo_m31

    resumo_m31 = meta_m31["resumo_m31"]

    logs.append(
        _log(
            modulo="m3_1_validacao_fronteira",
            status="ok",
            mensagem="M3.1 executado com sucesso e input oficial do bloco 4 foi consolidado",
            quantidade_entrada=_safe_len(df_carteira_roteirizavel),
            quantidade_saida=_safe_len(df_input_oficial_bloco_4),
            tempo_ms=tempo_m31,
            extra=resumo_m31,
        )
    )

    # =========================================================================================
    # M4
    # =========================================================================================
    t0 = _agora()
    outputs_m4, meta_m4 = executar_m4_manifestos_fechados(
        df_input_oficial_bloco_4=df_input_oficial_bloco_4,
        df_veiculos_tratados=df_veiculos_tratados,
        rodada_id=contexto.rodada_id,
        data_base_roteirizacao=contexto.data_base,
        tipo_roteirizacao=contexto.tipo_roteirizacao,
        configuracao_frota=payload.configuracao_frota,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )
    tempo_m4 = _duracao_ms(t0)
    metricas_tempo["m4_manifestos_fechados_ms"] = tempo_m4

    resumo_m4 = meta_m4["resumo_m4"]

    df_remanescente_roteirizavel_bloco_4 = outputs_m4["df_remanescente_roteirizavel_bloco_4"]

    logs.append(
        _log(
            modulo="m4_manifestos_fechados",
            status="ok",
            mensagem="M4 executado com sucesso",
            quantidade_entrada=_safe_len(df_input_oficial_bloco_4),
            quantidade_saida=_safe_len(df_remanescente_roteirizavel_bloco_4),
            tempo_ms=tempo_m4,
            extra=resumo_m4,
        )
    )

    # =========================================================================================
    # M5.1
    # =========================================================================================
    t0 = _agora()
    outputs_m5_1, meta_m5_1 = executar_m5_1_triagem_cidades(
        df_remanescente_roteirizavel_bloco_4=df_remanescente_roteirizavel_bloco_4,
        df_veiculos_tratados=df_veiculos_tratados,
        rodada_id=contexto.rodada_id,
        data_base_roteirizacao=contexto.data_base,
        tipo_roteirizacao=contexto.tipo_roteirizacao,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )
    tempo_m5_1 = _duracao_ms(t0)
    metricas_tempo["m5_1_triagem_cidades_ms"] = tempo_m5_1

    resumo_m5_1 = meta_m5_1["resumo_m5_1_triagem"]

    df_saldo_elegivel_composicao_m5_1 = outputs_m5_1["df_saldo_elegivel_composicao_m5_1"]
    df_saldo_excluido_triagem_m5_1 = outputs_m5_1["df_saldo_excluido_triagem_m5_1"]
    df_perfis_viaveis_por_cidade_m5_1 = outputs_m5_1["df_perfis_viaveis_por_cidade_m5_1"]

    logs.append(
        _log(
            modulo="m5_1_triagem_cidades",
            status="ok",
            mensagem="M5.1 executado com sucesso",
            quantidade_entrada=_safe_len(df_remanescente_roteirizavel_bloco_4),
            quantidade_saida=_safe_len(df_saldo_elegivel_composicao_m5_1),
            tempo_ms=tempo_m5_1,
            extra={
                **resumo_m5_1,
                "total_linhas_elegiveis_composicao": _safe_len(df_saldo_elegivel_composicao_m5_1),
                "total_linhas_excluidas_triagem": _safe_len(df_saldo_excluido_triagem_m5_1),
                "total_perfis_viaveis_por_cidade": _safe_len(df_perfis_viaveis_por_cidade_m5_1),
            },
        )
    )

    # =========================================================================================
    # M5.2
    # =========================================================================================
    t0 = _agora()
    outputs_m5_2, meta_m5_2 = executar_m5_2_composicao_cidades(
        df_saldo_elegivel_composicao_m5_1=df_saldo_elegivel_composicao_m5_1,
        df_perfis_viaveis_por_cidade_m5_1=df_perfis_viaveis_por_cidade_m5_1,
        rodada_id=contexto.rodada_id,
        data_base_roteirizacao=contexto.data_base,
        tipo_roteirizacao=contexto.tipo_roteirizacao,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )
    tempo_m5_2 = _duracao_ms(t0)
    metricas_tempo["m5_2_composicao_cidades_ms"] = tempo_m5_2

    resumo_m5_2 = meta_m5_2["resumo_m5_2"]
    df_remanescente_m5_2 = outputs_m5_2["df_remanescente_m5_2"]

    logs.append(
        _log(
            modulo="m5_2_composicao_cidades",
            status="ok",
            mensagem="M5.2 executado com sucesso",
            quantidade_entrada=_safe_len(df_saldo_elegivel_composicao_m5_1),
            quantidade_saida=_safe_len(df_remanescente_m5_2),
            tempo_ms=tempo_m5_2,
            extra={
                **resumo_m5_2,
                "total_remanescente_m5_2": _safe_len(df_remanescente_m5_2),
            },
        )
    )

    # =========================================================================================
    # SALDO GLOBAL
    # =========================================================================================
    if _safe_len(df_saldo_excluido_triagem_m5_1) > 0 and _safe_len(df_remanescente_m5_2) > 0:
        df_saldo_global_pos_cidade_m5 = pd.concat(
            [df_saldo_excluido_triagem_m5_1, df_remanescente_m5_2],
            ignore_index=True,
            sort=False,
        )
    elif _safe_len(df_saldo_excluido_triagem_m5_1) > 0:
        df_saldo_global_pos_cidade_m5 = df_saldo_excluido_triagem_m5_1.copy()
    elif _safe_len(df_remanescente_m5_2) > 0:
        df_saldo_global_pos_cidade_m5 = df_remanescente_m5_2.copy()
    else:
        df_saldo_global_pos_cidade_m5 = pd.DataFrame()

    # =========================================================================================
    # M5.3
    # =========================================================================================
    t0 = _agora()
    outputs_m5_3, meta_m5_3 = executar_m5_3_triagem_subregioes(
        df_saldo_global_pos_cidade_m5=df_saldo_global_pos_cidade_m5,
        df_perfis_base_m5=df_perfis_viaveis_por_cidade_m5_1,
        rodada_id=contexto.rodada_id,
        data_base_roteirizacao=contexto.data_base,
        tipo_roteirizacao=contexto.tipo_roteirizacao,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )
    tempo_m5_3 = _duracao_ms(t0)
    metricas_tempo["m5_3_triagem_subregioes_ms"] = tempo_m5_3

    resumo_m5_3 = meta_m5_3["resumo_m5_3_triagem"]
    df_saldo_elegivel_composicao_m5_3 = outputs_m5_3["df_saldo_elegivel_composicao_m5_3"]

    logs.append(
        _log(
            modulo="m5_3_triagem_subregioes",
            status="ok",
            mensagem="M5.3 executado com sucesso",
            quantidade_entrada=_safe_len(df_saldo_global_pos_cidade_m5),
            quantidade_saida=_safe_len(df_saldo_elegivel_composicao_m5_3),
            tempo_ms=tempo_m5_3,
            extra={
                **resumo_m5_3,
                "total_linhas_elegiveis_composicao": _safe_len(df_saldo_elegivel_composicao_m5_3),
            },
        )
    )

    # =========================================================================================
    # M5.4A
    # =========================================================================================
    t0 = _agora()
    outputs_m5_4a, meta_m5_4a = executar_m5_4a_preparacao_subregioes(
        df_saldo_elegivel_composicao_m5_3=df_saldo_elegivel_composicao_m5_3,
        rodada_id=contexto.rodada_id,
        data_base_roteirizacao=contexto.data_base,
        tipo_roteirizacao=contexto.tipo_roteirizacao,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )
    tempo_m5_4a = _duracao_ms(t0)
    metricas_tempo["m5_4a_preparacao_subregioes_ms"] = tempo_m5_4a

    resumo_m5_4a = meta_m5_4a["resumo_m5_4a"]
    df_base_preparada_m5_4a = outputs_m5_4a["df_base_preparada_m5_4a"]

    logs.append(
        _log(
            modulo="m5_4a_preparacao_subregioes",
            status="ok",
            mensagem="M5.4A executado com sucesso",
            quantidade_entrada=_safe_len(df_saldo_elegivel_composicao_m5_3),
            quantidade_saida=_safe_len(df_base_preparada_m5_4a),
            tempo_ms=tempo_m5_4a,
            extra=resumo_m5_4a,
        )
    )

    # =========================================================================================
    # M5.4B
    # =========================================================================================
    t0 = _agora()
    outputs_m5_4b, meta_m5_4b = executar_m5_4b_aderencia_frota_subregioes(
        df_base_preparada_m5_4a=df_base_preparada_m5_4a,
        df_veiculos_tratados=df_veiculos_tratados,
        rodada_id=contexto.rodada_id,
        data_base_roteirizacao=contexto.data_base,
        tipo_roteirizacao=contexto.tipo_roteirizacao,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )
    tempo_m5_4b = _duracao_ms(t0)
    metricas_tempo["m5_4b_aderencia_frota_subregioes_ms"] = tempo_m5_4b

    resumo_m5_4b = meta_m5_4b["resumo_m5_4b"]

    df_frota_aderente_m5_4b = outputs_m5_4b["df_frota_aderente_m5_4b"]
    df_frota_nao_aderente_m5_4b = outputs_m5_4b["df_frota_nao_aderente_m5_4b"]
    df_base_tentativas_m5_4b = outputs_m5_4b["df_base_tentativas_m5_4b"]
    df_cidades_candidatas_m5_4b = outputs_m5_4b["df_cidades_candidatas_m5_4b"]

    logs.append(
        _log(
            modulo="m5_4b_aderencia_frota_subregioes",
            status="ok",
            mensagem="M5.4B executado com sucesso",
            quantidade_entrada=_safe_len(df_base_preparada_m5_4a),
            quantidade_saida=_safe_len(df_base_tentativas_m5_4b),
            tempo_ms=tempo_m5_4b,
            extra={
                **resumo_m5_4b,
                "total_frota_aderente_m5_4b": _safe_len(df_frota_aderente_m5_4b),
                "total_frota_nao_aderente_m5_4b": _safe_len(df_frota_nao_aderente_m5_4b),
                "total_base_tentativas_m5_4b": _safe_len(df_base_tentativas_m5_4b),
                "total_cidades_candidatas_m5_4b": _safe_len(df_cidades_candidatas_m5_4b),
            },
        )
    )

    # =========================================================================================
    # SERIALIZAÇÃO FINAL - SOMENTE ETAPA ATUAL
    # =========================================================================================
    t0 = _agora()

    base_preparada_m5_4a = _serializar_dataframe_para_records(df_base_preparada_m5_4a, limit=None)
    frota_aderente_m5_4b = _serializar_dataframe_para_records(df_frota_aderente_m5_4b, limit=None)
    frota_nao_aderente_m5_4b = _serializar_dataframe_para_records(df_frota_nao_aderente_m5_4b, limit=None)
    base_tentativas_m5_4b = _serializar_dataframe_para_records(df_base_tentativas_m5_4b, limit=None)
    cidades_candidatas_m5_4b = _serializar_dataframe_para_records(df_cidades_candidatas_m5_4b, limit=None)

    tempo_serializacao = _duracao_ms(t0)
    metricas_tempo["serializacao_resposta_ms"] = tempo_serializacao

    tempo_total = _duracao_ms(inicio_total)
    metricas_tempo["tempo_total_pipeline_ms"] = tempo_total

    resposta: Dict[str, Any] = {
        "status": "ok",
        "mensagem": "Motor executou com sucesso até o M5.4B aderência de frota por subregiões.",
        "pipeline_real_ate": "M5.4B",
        "modo_resposta": "validacao_manual_m5_4b_aderencia_frota",
        "resposta_truncada": False,
        "resumo_execucao": {
            "rodada_id": contexto.rodada_id,
            "upload_id": contexto.upload_id,
            "usuario_id": contexto.usuario_id,
            "filial_id": contexto.filial_id,
            "tipo_roteirizacao": contexto.tipo_roteirizacao,
            "data_base_roteirizacao": contexto.data_base.isoformat(),
            "tempos_ms": metricas_tempo,
        },
        "resumo_negocio": {
            "total_carteira": _safe_len(contexto.df_carteira_raw),
            "total_roteirizavel": _safe_len(df_carteira_roteirizavel),
            "total_remanescentes_m4": _safe_len(df_remanescente_roteirizavel_bloco_4),
            "total_elegiveis_m5_3": _safe_len(df_saldo_elegivel_composicao_m5_3),
            "total_base_preparada_m5_4a": _safe_len(df_base_preparada_m5_4a),
            "total_frota_aderente_m5_4b": _safe_len(df_frota_aderente_m5_4b),
            "total_frota_nao_aderente_m5_4b": _safe_len(df_frota_nao_aderente_m5_4b),
            "total_base_tentativas_m5_4b": _safe_len(df_base_tentativas_m5_4b),
            "total_cidades_candidatas_m5_4b": _safe_len(df_cidades_candidatas_m5_4b),
            "resumo_m4": resumo_m4,
            "resumo_m5_3": resumo_m5_3,
            "resumo_m5_4a": resumo_m5_4a,
            "resumo_m5_4b": resumo_m5_4b,
        },
        "contexto_rodada": {
            "filial": contexto.filial,
            "parametros_rodada": contexto.parametros_rodada,
        },
        "base_preparada_m5_4a": base_preparada_m5_4a,
        "frota_aderente_m5_4b": frota_aderente_m5_4b,
        "frota_nao_aderente_m5_4b": frota_nao_aderente_m5_4b,
        "base_tentativas_m5_4b": base_tentativas_m5_4b,
        "cidades_candidatas_m5_4b": cidades_candidatas_m5_4b,
        "auditoria_serializacao": {
            "base_preparada_m5_4a_total": _safe_len(df_base_preparada_m5_4a),
            "base_preparada_m5_4a_retornado": len(base_preparada_m5_4a),
            "frota_aderente_m5_4b_total": _safe_len(df_frota_aderente_m5_4b),
            "frota_aderente_m5_4b_retornado": len(frota_aderente_m5_4b),
            "frota_nao_aderente_m5_4b_total": _safe_len(df_frota_nao_aderente_m5_4b),
            "frota_nao_aderente_m5_4b_retornado": len(frota_nao_aderente_m5_4b),
            "base_tentativas_m5_4b_total": _safe_len(df_base_tentativas_m5_4b),
            "base_tentativas_m5_4b_retornado": len(base_tentativas_m5_4b),
            "cidades_candidatas_m5_4b_total": _safe_len(df_cidades_candidatas_m5_4b),
            "cidades_candidatas_m5_4b_retornado": len(cidades_candidatas_m5_4b),
        },
        "logs": logs,
    }

    if debug:
        resposta["debug"] = {
            "snapshots": {
                "base_preparada_m5_4a": _snapshot_dataframe(df_base_preparada_m5_4a, "df_base_preparada_m5_4a"),
                "frota_aderente_m5_4b": _snapshot_dataframe(df_frota_aderente_m5_4b, "df_frota_aderente_m5_4b"),
                "frota_nao_aderente_m5_4b": _snapshot_dataframe(df_frota_nao_aderente_m5_4b, "df_frota_nao_aderente_m5_4b"),
                "base_tentativas_m5_4b": _snapshot_dataframe(df_base_tentativas_m5_4b, "df_base_tentativas_m5_4b"),
                "cidades_candidatas_m5_4b": _snapshot_dataframe(df_cidades_candidatas_m5_4b, "df_cidades_candidatas_m5_4b"),
            },
            "resumos_dataframes": {
                "base_preparada_m5_4a": _montar_resumo_dataframe(df_base_preparada_m5_4a, "base_preparada_m5_4a"),
                "frota_aderente_m5_4b": _montar_resumo_dataframe(df_frota_aderente_m5_4b, "frota_aderente_m5_4b"),
                "frota_nao_aderente_m5_4b": _montar_resumo_dataframe(df_frota_nao_aderente_m5_4b, "frota_nao_aderente_m5_4b"),
                "base_tentativas_m5_4b": _montar_resumo_dataframe(df_base_tentativas_m5_4b, "base_tentativas_m5_4b"),
                "cidades_candidatas_m5_4b": _montar_resumo_dataframe(df_cidades_candidatas_m5_4b, "cidades_candidatas_m5_4b"),
            },
        }

    return resposta
