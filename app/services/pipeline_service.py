from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd

from app.pipeline.m1_padronizacao import executar_m1_padronizacao
from app.pipeline.m2_enriquecimento import executar_m2_enriquecimento
from app.pipeline.m3_triagem import executar_m3_triagem
from app.pipeline.m3_1_validacao_fronteira import executar_m3_1_validacao_fronteira
from app.pipeline.m4_manifestos_fechados import executar_m4_manifestos_fechados
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
    """
    Não exige mudança imediata no schema.
    Se no futuro existir payload.modo_debug ou payload.debug, ele passa a funcionar.
    """
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
    """
    Só usar quando realmente precisar devolver linhas no JSON.
    """
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
    # M0 ADAPTER
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

    logs.append(
        _log(
            modulo="m1_padronizacao",
            status="ok",
            mensagem="M1 executado com sucesso",
            quantidade_entrada=_safe_len(contexto.df_carteira_raw),
            quantidade_saida=_safe_len(df_carteira_tratada),
            tempo_ms=tempo_m1,
            extra={
                "carteira_colunas": int(len(df_carteira_tratada.columns)),
                "geo_colunas": int(len(df_geo_tratado.columns)),
                "parametros_colunas": int(len(df_parametros_tratados.columns)),
                "veiculos_colunas": int(len(df_veiculos_tratados.columns)),
            },
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

    df_manifestos_fechados_bloco_4 = outputs_m4["df_manifestos_fechados_bloco_4"]
    df_itens_manifestos_fechados_bloco_4 = outputs_m4["df_itens_manifestos_fechados_bloco_4"]
    df_tentativas_fechamento_bloco_4 = outputs_m4["df_tentativas_fechamento_bloco_4"]
    df_remanescente_roteirizavel_bloco_4 = outputs_m4["df_remanescente_roteirizavel_bloco_4"]
    df_uso_frota_m4 = outputs_m4.get("df_uso_frota_m4", pd.DataFrame())

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
    # SERIALIZAÇÃO FINAL
    # =========================================================================================
    t0 = _agora()

    manifestos_fechados = _serializar_dataframe_para_records(df_manifestos_fechados_bloco_4)
    itens_manifestos_fechados = _serializar_dataframe_para_records(df_itens_manifestos_fechados_bloco_4)

    # Resumo enxuto de remanescentes: evita devolver carteira inteira sem necessidade.
    colunas_preferenciais_remanescente = [
        "Nro Doc.",
        "Destinatário",
        "Cidade Dest.",
        "Sub-Região",
        "Mesoregião",
        "Peso",
        "Peso Calculado",
        "D.L.E.",
        "Agendam.",
        "Prioridade",
    ]
    colunas_existentes_remanescente = [
        c for c in colunas_preferenciais_remanescente if c in df_remanescente_roteirizavel_bloco_4.columns
    ]
    if colunas_existentes_remanescente:
        df_remanescente_resumido = df_remanescente_roteirizavel_bloco_4[colunas_existentes_remanescente].copy()
    else:
        df_remanescente_resumido = df_remanescente_roteirizavel_bloco_4.head(0).copy()

    remanescentes_resumidos = _serializar_dataframe_para_records(df_remanescente_resumido)

    # Auditoria enxuta do M4
    auditoria_m4 = {
        "total_tentativas": _safe_len(df_tentativas_fechamento_bloco_4),
        "total_manifestos_fechados": _safe_len(df_manifestos_fechados_bloco_4),
        "total_itens_manifestados": _safe_len(df_itens_manifestos_fechados_bloco_4),
        "total_remanescentes": _safe_len(df_remanescente_roteirizavel_bloco_4),
        "total_uso_frota_registros": _safe_len(df_uso_frota_m4),
    }

    # Se vier enriquecido do M4 no futuro, aproveita automaticamente
    if isinstance(meta_m4, dict):
        if "auditoria_m4" in meta_m4 and isinstance(meta_m4["auditoria_m4"], dict):
            auditoria_m4.update(meta_m4["auditoria_m4"])

        if "metricas_m4" in meta_m4 and isinstance(meta_m4["metricas_m4"], dict):
            auditoria_m4["metricas_m4"] = meta_m4["metricas_m4"]

    tempo_serializacao = _duracao_ms(t0)
    metricas_tempo["serializacao_resposta_ms"] = tempo_serializacao

    tempo_total = _duracao_ms(inicio_total)
    metricas_tempo["tempo_total_pipeline_ms"] = tempo_total

    # =========================================================================================
    # RESPOSTA PADRÃO (LEVE)
    # =========================================================================================
    resposta: Dict[str, Any] = {
        "status": "ok",
        "mensagem": "Motor executou com sucesso até o M4.",
        "pipeline_real_ate": "M4",
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
            "total_veiculos": _safe_len(contexto.df_veiculos_raw),
            "total_regionalidades": _safe_len(contexto.df_geo_raw),
            "total_parametros": _safe_len(contexto.df_parametros_raw),
            "total_carteira_tratada": _safe_len(df_carteira_tratada),
            "total_carteira_enriquecida": _safe_len(df_carteira_enriquecida),
            "total_carteira_triagem": _safe_len(df_carteira_triagem),
            "total_roteirizavel": _safe_len(df_carteira_roteirizavel),
            "total_agendamento_futuro": _safe_len(df_carteira_agendamento_futuro),
            "total_agendas_vencidas": _safe_len(df_carteira_agendas_vencidas),
            "total_excecoes_triagem": _safe_len(df_carteira_excecoes_triagem),
            "total_input_oficial_bloco_4": _safe_len(df_input_oficial_bloco_4),
            "total_manifestos_fechados_m4": _safe_len(df_manifestos_fechados_bloco_4),
            "total_itens_manifestados_m4": _safe_len(df_itens_manifestos_fechados_bloco_4),
            "total_remanescentes_m4": _safe_len(df_remanescente_roteirizavel_bloco_4),
            "resumo_m2": resumo_m2,
            "resumo_m3": resumo_m3,
            "resumo_m31": resumo_m31,
            "resumo_m4": resumo_m4,
        },
        "contexto_rodada": {
            "filial": contexto.filial,
            "parametros_rodada": contexto.parametros_rodada,
        },
        "manifestos_fechados": manifestos_fechados,
        "itens_manifestos_fechados": itens_manifestos_fechados,
        "remanescente_roteirizavel_resumido": remanescentes_resumidos,
        "auditoria_m4": auditoria_m4,
        "manifestos_compostos": [],
        "nao_roteirizados": [],
        "logs": logs,
    }

    # =========================================================================================
    # RESPOSTA DEBUG (OPCIONAL)
    # =========================================================================================
    if debug:
        resposta["debug"] = {
            "snapshots": {
                "carteira_raw": _snapshot_dataframe(contexto.df_carteira_raw, "df_carteira_raw"),
                "carteira_tratada": _snapshot_dataframe(df_carteira_tratada, "df_carteira_tratada"),
                "carteira_enriquecida": _snapshot_dataframe(df_carteira_enriquecida, "df_carteira_enriquecida"),
                "carteira_triagem": _snapshot_dataframe(df_carteira_triagem, "df_carteira_triagem"),
                "carteira_roteirizavel": _snapshot_dataframe(df_carteira_roteirizavel, "df_carteira_roteirizavel"),
                "input_oficial_bloco_4": _snapshot_dataframe(df_input_oficial_bloco_4, "df_input_oficial_bloco_4"),
                "carteira_agendamento_futuro": _snapshot_dataframe(
                    df_carteira_agendamento_futuro, "df_carteira_agendamento_futuro"
                ),
                "carteira_agendas_vencidas": _snapshot_dataframe(
                    df_carteira_agendas_vencidas, "df_carteira_agendas_vencidas"
                ),
                "carteira_excecoes_triagem": _snapshot_dataframe(
                    df_carteira_excecoes_triagem, "df_carteira_excecoes_triagem"
                ),
                "manifestos_fechados_bloco_4": _snapshot_dataframe(
                    df_manifestos_fechados_bloco_4, "df_manifestos_fechados_bloco_4"
                ),
                "itens_manifestos_fechados_bloco_4": _snapshot_dataframe(
                    df_itens_manifestos_fechados_bloco_4, "df_itens_manifestos_fechados_bloco_4"
                ),
                "tentativas_fechamento_bloco_4": _snapshot_dataframe(
                    df_tentativas_fechamento_bloco_4, "df_tentativas_fechamento_bloco_4"
                ),
                "remanescente_roteirizavel_bloco_4": _snapshot_dataframe(
                    df_remanescente_roteirizavel_bloco_4, "df_remanescente_roteirizavel_bloco_4"
                ),
                "uso_frota_m4": _snapshot_dataframe(df_uso_frota_m4, "df_uso_frota_m4"),
                "regionalidades": _snapshot_dataframe(contexto.df_geo_raw, "df_geo_raw"),
                "parametros": _snapshot_dataframe(contexto.df_parametros_raw, "df_parametros_raw"),
                "veiculos": _snapshot_dataframe(contexto.df_veiculos_raw, "df_veiculos_raw"),
            },
            "amostras": {
                "carteira_tratada": _serializar_dataframe_para_records(df_carteira_tratada, limit=5),
                "carteira_enriquecida": _serializar_dataframe_para_records(df_carteira_enriquecida, limit=5),
                "carteira_roteirizavel": _serializar_dataframe_para_records(df_carteira_roteirizavel, limit=5),
                "input_oficial_bloco_4": _serializar_dataframe_para_records(df_input_oficial_bloco_4, limit=5),
                "agendamento_futuro": _serializar_dataframe_para_records(df_carteira_agendamento_futuro, limit=5),
                "agendas_vencidas": _serializar_dataframe_para_records(df_carteira_agendas_vencidas, limit=5),
                "excecoes_triagem": _serializar_dataframe_para_records(df_carteira_excecoes_triagem, limit=5),
                "manifestos_fechados_bloco_4": _serializar_dataframe_para_records(
                    df_manifestos_fechados_bloco_4, limit=10
                ),
                "itens_manifestos_fechados_bloco_4": _serializar_dataframe_para_records(
                    df_itens_manifestos_fechados_bloco_4, limit=10
                ),
                "remanescente_roteirizavel_bloco_4": _serializar_dataframe_para_records(
                    df_remanescente_roteirizavel_bloco_4, limit=10
                ),
                "uso_frota_m4": _serializar_dataframe_para_records(df_uso_frota_m4, limit=10),
            },
            "outputs_intermediarios": {
                "df_input_oficial_bloco_4": _serializar_dataframe_para_records(df_input_oficial_bloco_4),
                "df_manifestos_fechados_bloco_4": _serializar_dataframe_para_records(
                    df_manifestos_fechados_bloco_4
                ),
                "df_itens_manifestos_fechados_bloco_4": _serializar_dataframe_para_records(
                    df_itens_manifestos_fechados_bloco_4
                ),
                "df_tentativas_fechamento_bloco_4": _serializar_dataframe_para_records(
                    df_tentativas_fechamento_bloco_4
                ),
                "df_remanescente_roteirizavel_bloco_4": _serializar_dataframe_para_records(
                    df_remanescente_roteirizavel_bloco_4
                ),
                "df_uso_frota_m4": _serializar_dataframe_para_records(df_uso_frota_m4),
            },
        }

    return resposta
