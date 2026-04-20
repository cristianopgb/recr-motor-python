from __future__ import annotations

from typing import Any, Dict, List


def _safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _sum_field(records: List[Dict[str, Any]], field: str) -> float:
    total = 0.0
    for row in records or []:
        try:
            total += float(row.get(field) or 0)
        except Exception:
            continue
    return total


def _avg_field(records: List[Dict[str, Any]], field: str) -> float:
    values = []
    for row in records or []:
        try:
            values.append(float(row.get(field) or 0))
        except Exception:
            continue
    if not values:
        return 0.0
    return sum(values) / len(values)


def _group_count(records: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    acc: Dict[str, int] = {}
    for row in records or []:
        chave = str(row.get(field) or "NAO_INFORMADO").strip()
        acc[chave] = acc.get(chave, 0) + 1
    return [
        {"chave": chave, "quantidade": quantidade}
        for chave, quantidade in sorted(acc.items(), key=lambda x: (-x[1], x[0]))
    ]


def _group_sum(records: List[Dict[str, Any]], group_field: str, sum_field: str, out_name: str) -> List[Dict[str, Any]]:
    acc: Dict[str, float] = {}
    for row in records or []:
        chave = str(row.get(group_field) or "NAO_INFORMADO").strip()
        try:
            valor = float(row.get(sum_field) or 0)
        except Exception:
            valor = 0.0
        acc[chave] = acc.get(chave, 0.0) + valor

    return [
        {"chave": chave, out_name: valor}
        for chave, valor in sorted(acc.items(), key=lambda x: (-x[1], x[0]))
    ]


def _build_resumo_execucao(resultado_pipeline: Dict[str, Any], contexto_rodada: Dict[str, Any]) -> Dict[str, Any]:
    resumo_execucao = resultado_pipeline.get("resumo_execucao", {}) or {}

    tempos_ms = resumo_execucao.get("tempos_ms", {}) if isinstance(resumo_execucao, dict) else {}

    return {
        "rodada_id": _safe_str(
            resumo_execucao.get("rodada_id")
            or contexto_rodada.get("rodada_id")
        ),
        "upload_id": _safe_str(
            resumo_execucao.get("upload_id")
            or contexto_rodada.get("upload_id")
        ),
        "filial_id": _safe_str(
            resumo_execucao.get("filial_id")
            or contexto_rodada.get("filial_id")
        ),
        "usuario_id": _safe_str(
            resumo_execucao.get("usuario_id")
            or contexto_rodada.get("usuario_id")
        ),
        "data_execucao": resumo_execucao.get("data_execucao"),
        "origem_sistema": resumo_execucao.get("origem_sistema"),
        "tipo_roteirizacao": resumo_execucao.get("tipo_roteirizacao") or contexto_rodada.get("tipo_roteirizacao"),
        "modelo_roteirizacao": resumo_execucao.get("modelo_roteirizacao"),
        "versao_motor": resumo_execucao.get("versao_motor"),
        "tempos_ms": {
            "tempo_total_pipeline_ms": tempos_ms.get("tempo_total_pipeline_ms"),
            "tempo_leitura_ms": tempos_ms.get("tempo_leitura_ms"),
            "tempo_geocodificacao_ms": tempos_ms.get("tempo_geocodificacao_ms"),
            "tempo_otimizacao_ms": tempos_ms.get("tempo_otimizacao_ms"),
            "tempo_montagem_ms": tempos_ms.get("tempo_montagem_ms"),
        },
    }


def _build_contexto_rodada(contexto_rodada: Dict[str, Any]) -> Dict[str, Any]:
    parametros_rodada = contexto_rodada.get("parametros_rodada", {}) if isinstance(contexto_rodada, dict) else {}
    filial = contexto_rodada.get("filial", {}) if isinstance(contexto_rodada, dict) else {}

    return {
        "rodada_id": contexto_rodada.get("rodada_id"),
        "upload_id": contexto_rodada.get("upload_id"),
        "filial_id": contexto_rodada.get("filial_id"),
        "usuario_id": contexto_rodada.get("usuario_id"),
        "data_base_roteirizacao": contexto_rodada.get("data_base_roteirizacao"),
        "tipo_roteirizacao": contexto_rodada.get("tipo_roteirizacao"),
        "filtros_aplicados": (
            parametros_rodada.get("filtros_aplicados")
            if isinstance(parametros_rodada, dict)
            else None
        ),
        "configuracao_frota": contexto_rodada.get("configuracao_frota"),
        "filial": filial if isinstance(filial, dict) else {},
        "parametros_rodada": parametros_rodada if isinstance(parametros_rodada, dict) else {},
    }


def build_m8_contract(resultado_pipeline: Dict[str, Any]) -> Dict[str, Any]:
    resumo_negocio = resultado_pipeline.get("resumo_negocio", {}) or {}
    contexto_rodada = resultado_pipeline.get("contexto_rodada", {}) or {}
    logs = resultado_pipeline.get("logs", []) or []

    manifestos_m7 = resultado_pipeline.get("manifestos_m7", []) or []
    itens_m7 = resultado_pipeline.get("itens_manifestos_sequenciados_m7", []) or []
    tentativas_m7 = resultado_pipeline.get("tentativas_sequenciamento_m7", []) or []

    total_manifestos = len(manifestos_m7)
    total_itens = len(itens_m7)

    cargas_por_veiculo = _group_count(manifestos_m7, "veiculo_perfil")
    ocupacao_por_veiculo = _group_sum(
        manifestos_m7,
        "veiculo_perfil",
        "ocupacao_final_m6_2",
        "ocupacao_total",
    )
    km_por_veiculo = _group_sum(
        manifestos_m7,
        "veiculo_perfil",
        "km_total_sequencia_paradas_m7",
        "km_total",
    )

    estatisticas_cidade = []
    cidade_qtd = _group_count(itens_m7, "cidade")
    cidade_peso = {
        x["chave"]: x["peso_total"]
        for x in _group_sum(itens_m7, "cidade", "peso_seq_m7", "peso_total")
    }
    for item in cidade_qtd:
        estatisticas_cidade.append(
            {
                "cidade": item["chave"],
                "quantidade_entregas": item["quantidade"],
                "peso_total": cidade_peso.get(item["chave"], 0.0),
            }
        )

    leadtime_stats = []
    for faixa in [
        "Agendada com folga vencida/zero",
        "Agendada com folga de 1 dia",
        "Agendada com folga acima de 1 dia",
        "Não agendada urgente",
        "Não agendada com folga de 1 dia",
        "Não agendada normal",
    ]:
        qtd = 0
        for row in itens_m7:
            txt = str(row.get("justificativa_ordem_entrega_m7") or "")
            if faixa in txt:
                qtd += 1
        leadtime_stats.append({"faixa": faixa, "quantidade": qtd})

    modulos_status = []
    for row in logs:
        modulos_status.append(
            {
                "modulo": row.get("modulo"),
                "status": row.get("status"),
                "mensagem": row.get("mensagem"),
                "tempo_ms": row.get("tempo_ms"),
                "quantidade_entrada": row.get("quantidade_entrada"),
                "quantidade_saida": row.get("quantidade_saida"),
            }
        )

    contrato = {
        "status": resultado_pipeline.get("status", "erro"),
        "mensagem": resultado_pipeline.get("mensagem", ""),
        "pipeline_real_ate": resultado_pipeline.get("pipeline_real_ate", "M7"),
        "modo_resposta": "contrato_retorno_sistema_1_m8",
        "resumo_execucao": _build_resumo_execucao(resultado_pipeline, contexto_rodada),
        "contexto_rodada": _build_contexto_rodada(contexto_rodada),
        "status_modulos": modulos_status,
        "estatisticas_roteirizacao": {
            "carteira": {
                "total_carteira": resumo_negocio.get("total_carteira", 0),
                "total_roteirizavel": resumo_negocio.get("total_roteirizavel_m3", 0),
                "total_agendamento_futuro": resumo_negocio.get("total_agendamento_futuro_m3", 0),
                "total_agendas_vencidas": resumo_negocio.get("total_agendas_vencidas_m3", 0),
                "total_excecoes": _safe_get(
                    resumo_negocio,
                    "resumo_m3",
                    "carteira_excecoes_triagem",
                    default=0,
                ),
                "total_sem_agenda": 0,
            },
            "cargas": {
                "total_manifestos_m7": total_manifestos,
                "total_itens_m7": total_itens,
                "cargas_fechadas_m4": resumo_negocio.get("total_manifestos_m4", 0),
                "cargas_compostas_m5": (
                    resumo_negocio.get("total_premanifestos_m5_2", 0)
                    + resumo_negocio.get("total_premanifestos_m5_3", 0)
                    + resumo_negocio.get("total_premanifestos_m5_4", 0)
                ),
                "remanescente_m6_2": resumo_negocio.get("total_remanescente_m6_2", 0),
                "km_total_m7": _sum_field(manifestos_m7, "km_total_sequencia_paradas_m7"),
                "km_medio_manifesto_m7": _avg_field(manifestos_m7, "km_total_sequencia_paradas_m7"),
                "ocupacao_media_manifesto_m7": _avg_field(manifestos_m7, "ocupacao_final_m6_2"),
                "qtd_media_itens_por_manifesto": (total_itens / total_manifestos) if total_manifestos else 0.0,
            },
            "por_veiculo": {
                "cargas_por_perfil": cargas_por_veiculo,
                "ocupacao_por_perfil": ocupacao_por_veiculo,
                "km_por_perfil": km_por_veiculo,
            },
            "por_cidade": estatisticas_cidade,
            "por_leadtime_agenda": leadtime_stats,
        },
        "resultados": {
            "manifestos": manifestos_m7,
            "itens_manifestos": itens_m7,
            "tentativas_m7": tentativas_m7,
        },
        "auditoria": {
            "auditoria_m7": resultado_pipeline.get("auditoria_m7", {}),
            "logs": logs,
        },
        "motor_response_raw": resultado_pipeline,
    }

    return contrato
