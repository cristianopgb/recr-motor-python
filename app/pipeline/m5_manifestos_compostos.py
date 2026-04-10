from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import math
import pandas as pd


# =========================================================================================
# M5.1 - MANIFESTOS COMPOSTOS
# -----------------------------------------------------------------------------------------
# CONTRATO:
# - compatível com pipeline_service.py
# - função principal: executar_m5_manifestos_compostos(...)
# - retorno: outputs_m5_1, meta_m5_1
#
# LÓGICA:
# 1) entrada dura: só remanescente do M4
# 2) fase 1: consolidação por mesmo cliente
# 3) fase 2: saldo -> cidade -> subregião -> mesorregião
# 4) regionalidade é UNIVERSO ELEGÍVEL, não bloco indivisível
# 5) quando grupo cheio não fecha, tenta subconjunto viável
#
# OTIMIZAÇÕES DESTA VERSÃO:
# - pré-cálculo de colunas de ordenação e elegibilidade
# - eliminação de apply(axis=1) em loops críticos
# - subconjunto viável incremental, sem recriar DataFrame a cada linha
# - normalização de chaves e restrições uma vez só
# - reutilização de métricas em validação, auditoria e manifesto
# - ocupação calculada somente por Peso Calculo
# =========================================================================================


# -----------------------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------------------
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "sim", "s", "yes", "y"}


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _safe_text_upper(value: Any) -> str:
    return _safe_text(value).upper()


def _ensure_column(df: pd.DataFrame, col: str, default: Any) -> None:
    if col not in df.columns:
        df[col] = default


def _first_existing(row: pd.Series, candidates: List[str], default: Any = None) -> Any:
    for col in candidates:
        if col in row.index:
            value = row[col]
            if not pd.isna(value):
                return value
    return default


# -----------------------------------------------------------------------------------------
# Normalização
# -----------------------------------------------------------------------------------------
def _normalizar_inputs(
    df_remanescente_roteirizavel_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = (
        df_remanescente_roteirizavel_bloco_4.copy()
        if df_remanescente_roteirizavel_bloco_4 is not None
        else pd.DataFrame()
    )
    veic = df_veiculos_tratados.copy() if df_veiculos_tratados is not None else pd.DataFrame()

    if df.empty:
        return df, veic

    rename_map: Dict[str, str] = {}
    if "sub_regiao" in df.columns and "subregiao" not in df.columns:
        rename_map["sub_regiao"] = "subregiao"
    if "mesoregiao" in df.columns and "mesorregiao" not in df.columns:
        rename_map["mesoregiao"] = "mesorregiao"
    if rename_map:
        df = df.rename(columns=rename_map)

    defaults = {
        "id_linha_pipeline": None,
        "destinatario": "",
        "cidade": "",
        "subregiao": "",
        "mesorregiao": "",
        "peso_calculado": 0.0,
        "peso_kg": 0.0,
        "vol_m3": 0.0,
        "distancia_rodoviaria_est_km": 0.0,
        "restricao_veiculo": None,
        "agendada": False,
        "folga_dias": 999,
        "prioridade_embarque_num": pd.NA,
        "prioridade_embarque": pd.NA,
        "ranking_prioridade_operacional": pd.NA,
        "cte": pd.NA,
        "nro_documento": pd.NA,
    }
    for col, default in defaults.items():
        _ensure_column(df, col, default)

    if df["id_linha_pipeline"].isna().any():
        raise ValueError("M5.1 exige id_linha_pipeline em todas as linhas do remanescente do M4.")

    if "peso_calculado" not in df.columns and "peso_c" in df.columns:
        df["peso_calculado"] = df["peso_c"]

    if "peso_kg" not in df.columns:
        df["peso_kg"] = df["peso_calculado"]

    if "distancia_rodoviaria_est_km" not in df.columns and "distancia_km" in df.columns:
        df["distancia_rodoviaria_est_km"] = df["distancia_km"]

    numeric_cols = [
        "peso_calculado",
        "peso_kg",
        "vol_m3",
        "distancia_rodoviaria_est_km",
        "folga_dias",
        "prioridade_embarque_num",
        "prioridade_embarque",
        "ranking_prioridade_operacional",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = ["agendada", "veiculo_exclusivo", "veiculo_exclusivo_flag"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(_safe_bool)

    if veic.empty:
        raise ValueError("M5.1 exige df_veiculos_tratados.")

    if "tipo" not in veic.columns and "perfil" in veic.columns:
        veic["tipo"] = veic["perfil"]
    if "perfil" not in veic.columns and "tipo" in veic.columns:
        veic["perfil"] = veic["tipo"]

    for col in [
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ocupacao_minima_perc",
        "ocupacao_maxima_perc",
    ]:
        if col not in veic.columns:
            veic[col] = pd.NA
        veic[col] = pd.to_numeric(veic[col], errors="coerce")

    for col in ["tipo", "perfil"]:
        if col in veic.columns:
            veic[col] = veic[col].astype(str)

    return df, veic


# -----------------------------------------------------------------------------------------
# Pré-cálculo de colunas auxiliares
# -----------------------------------------------------------------------------------------
def _precompute_df_inputs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    out["_id_str_m5_1"] = out["id_linha_pipeline"].astype(str)
    out["_cliente_chave_m5_1"] = out["destinatario"].fillna("").astype(str).str.strip()
    out["_cidade_key_m5_1"] = out["cidade"].fillna("").astype(str).str.strip()
    out["_subregiao_key_m5_1"] = out["subregiao"].fillna("").astype(str).str.strip()
    out["_mesorregiao_key_m5_1"] = out["mesorregiao"].fillna("").astype(str).str.strip()
    out["_destinatario_key_m5_1"] = out["destinatario"].fillna("").astype(str).str.strip()
    out["_restricao_norm_m5_1"] = out["restricao_veiculo"].fillna("").astype(str).str.strip().str.upper()

    out["_peso_calc_m5_1"] = pd.to_numeric(out["peso_calculado"], errors="coerce").fillna(0.0)
    out["_vol_m3_m5_1"] = pd.to_numeric(out["vol_m3"], errors="coerce").fillna(0.0)
    out["_km_ref_m5_1"] = pd.to_numeric(out["distancia_rodoviaria_est_km"], errors="coerce").fillna(0.0)
    out["_folga_m5_1"] = pd.to_numeric(out["folga_dias"], errors="coerce").fillna(999.0)

    prioridade_num = pd.to_numeric(out["prioridade_embarque_num"], errors="coerce")
    prioridade_raw = pd.to_numeric(out["prioridade_embarque"], errors="coerce")
    out["_prioridade_embarque_ord_m5_1"] = prioridade_num.fillna(prioridade_raw)

    ranking_oper = pd.to_numeric(out["ranking_prioridade_operacional"], errors="coerce")
    out["_ranking_oper_ord_m5_1"] = ranking_oper.fillna(999.0)

    out["_agendada_bool_m5_1"] = out["agendada"].apply(_safe_bool)

    cond_prioridade_embarque = out["_prioridade_embarque_ord_m5_1"].notna() & (
        out["_prioridade_embarque_ord_m5_1"] > 0
    )
    cond_agendada = out["_agendada_bool_m5_1"]
    cond_folga_0 = out["_folga_m5_1"] == 0
    cond_folga_1 = out["_folga_m5_1"] == 1

    out["_bucket_m5_1"] = 99
    out.loc[cond_prioridade_embarque, "_bucket_m5_1"] = 0
    out.loc[~cond_prioridade_embarque & cond_agendada & cond_folga_0, "_bucket_m5_1"] = 1
    out.loc[~cond_prioridade_embarque & cond_agendada & cond_folga_1, "_bucket_m5_1"] = 2
    out.loc[~cond_prioridade_embarque & ~cond_agendada & cond_folga_0, "_bucket_m5_1"] = 3
    out.loc[~cond_prioridade_embarque & ~cond_agendada & cond_folga_1, "_bucket_m5_1"] = 4

    out["_prioridade_ord_m5_1"] = out["_prioridade_embarque_ord_m5_1"].fillna(999.0)
    out["_folga_ord_m5_1"] = out["_folga_m5_1"]
    out["_km_ord_m5_1"] = out["_km_ref_m5_1"]
    out["_peso_ord_m5_1"] = -out["_peso_calc_m5_1"]
    out["_elegivel_fase2_m5_1"] = out["_bucket_m5_1"] < 99

    return out


def _precompute_df_veiculos(df_veiculos: pd.DataFrame) -> pd.DataFrame:
    if df_veiculos.empty:
        return df_veiculos.copy()

    out = df_veiculos.copy()
    out["_tipo_upper_m5_1"] = out["tipo"].fillna("").astype(str).str.strip().str.upper()
    out["_perfil_upper_m5_1"] = out["perfil"].fillna("").astype(str).str.strip().str.upper()
    out["_cap_peso_m5_1"] = pd.to_numeric(out["capacidade_peso_kg"], errors="coerce").fillna(0.0)
    out["_cap_vol_m5_1"] = pd.to_numeric(out["capacidade_vol_m3"], errors="coerce").fillna(0.0)
    out["_max_entregas_m5_1"] = pd.to_numeric(out["max_entregas"], errors="coerce").fillna(0).astype(int)
    out["_max_km_m5_1"] = pd.to_numeric(out["max_km_distancia"], errors="coerce").fillna(0.0)
    out["_ocup_min_m5_1"] = pd.to_numeric(out["ocupacao_minima_perc"], errors="coerce").fillna(70.0)
    out["_ocup_max_m5_1"] = pd.to_numeric(out["ocupacao_maxima_perc"], errors="coerce").fillna(100.0)
    return out


# -----------------------------------------------------------------------------------------
# Prioridade operacional
# -----------------------------------------------------------------------------------------
def _ordenar_operacional(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    sort_cols = [
        "_bucket_m5_1",
        "_prioridade_ord_m5_1",
        "_folga_ord_m5_1",
        "_ranking_oper_ord_m5_1",
        "_km_ord_m5_1",
        "_peso_ord_m5_1",
        "_id_str_m5_1",
    ]
    presentes = [c for c in sort_cols if c in df.columns]
    temp = df.sort_values(presentes, ascending=[True] * len(presentes), kind="mergesort")
    return temp.reset_index(drop=True)


# -----------------------------------------------------------------------------------------
# Veículos
# -----------------------------------------------------------------------------------------
def _veiculos_maior_para_menor(df_veiculos: pd.DataFrame) -> pd.DataFrame:
    temp = df_veiculos.copy()
    temp = temp.sort_values(
        ["_cap_peso_m5_1", "_cap_vol_m5_1"],
        ascending=[False, False],
        kind="mergesort",
    )
    return temp.reset_index(drop=True)


def _restricao_compatível(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> bool:
    if "restricao_veiculo" not in df_itens.columns:
        return True

    if "_restricao_norm_m5_1" in df_itens.columns:
        restricoes = {
            v for v in df_itens["_restricao_norm_m5_1"].tolist() if isinstance(v, str) and v.strip()
        }
    else:
        restricoes = set()
        for value in df_itens["restricao_veiculo"].tolist():
            txt = _safe_text_upper(value)
            if txt:
                restricoes.add(txt)

    if not restricoes:
        return True

    tipo = _safe_text_upper(vehicle_row.get("tipo"))
    perfil = _safe_text_upper(vehicle_row.get("perfil"))

    for restr in restricoes:
        if restr not in {tipo, perfil}:
            return False
    return True


# -----------------------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------------------
def _qtd_paradas(df_itens: pd.DataFrame) -> int:
    if df_itens.empty:
        return 0
    if "_destinatario_key_m5_1" in df_itens.columns:
        return int(df_itens["_destinatario_key_m5_1"].nunique())
    return int(df_itens["destinatario"].fillna("").astype(str).nunique())


def _peso_total(df_itens: pd.DataFrame) -> float:
    if df_itens.empty:
        return 0.0
    if "_peso_calc_m5_1" in df_itens.columns:
        return float(df_itens["_peso_calc_m5_1"].sum())
    return float(df_itens["peso_calculado"].fillna(0).sum())


def _volume_total(df_itens: pd.DataFrame) -> float:
    if df_itens.empty:
        return 0.0
    if "_vol_m3_m5_1" in df_itens.columns:
        return float(df_itens["_vol_m3_m5_1"].sum())
    return float(df_itens["vol_m3"].fillna(0).sum())


def _km_referencia(df_itens: pd.DataFrame) -> float:
    if df_itens.empty:
        return 0.0
    if "_km_ref_m5_1" in df_itens.columns:
        return float(df_itens["_km_ref_m5_1"].max())
    return float(df_itens["distancia_rodoviaria_est_km"].fillna(0).max())


def _ocupacao_perc(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> float:
    capacidade = _safe_float(
        vehicle_row.get("_cap_peso_m5_1", vehicle_row.get("capacidade_peso_kg", 0.0)),
        0.0,
    )
    if capacidade <= 0:
        return 0.0
    return (_peso_total(df_itens) / capacidade) * 100.0


def _extrair_metricas_df(
    df_itens: pd.DataFrame,
    vehicle_row: Optional[pd.Series] = None,
) -> Dict[str, float]:
    peso_total = _peso_total(df_itens)
    volume_total = _volume_total(df_itens)
    qtd_paradas = _qtd_paradas(df_itens)
    km_ref = _km_referencia(df_itens)

    ocupacao_perc = 0.0
    if vehicle_row is not None:
        capacidade_peso = _safe_float(
            vehicle_row.get("_cap_peso_m5_1", vehicle_row.get("capacidade_peso_kg", 0.0)),
            0.0,
        )
        if capacidade_peso > 0:
            ocupacao_perc = (peso_total / capacidade_peso) * 100.0

    return {
        "peso_total": peso_total,
        "volume_total": volume_total,
        "qtd_paradas": float(qtd_paradas),
        "km_ref": km_ref,
        "ocupacao_perc": ocupacao_perc,
    }


# -----------------------------------------------------------------------------------------
# Validação
# -----------------------------------------------------------------------------------------
def _validar_hard_constraints_metricas(
    *,
    peso_total: float,
    volume_total: float,
    qtd_paradas: int,
    km_ref: float,
    ocupacao_perc: float,
    restricoes_ok: bool,
    vehicle_row: pd.Series,
) -> Tuple[bool, str]:
    if not restricoes_ok:
        return False, "restricao_veiculo_incompativel"

    cap_peso = _safe_float(
        vehicle_row.get("_cap_peso_m5_1", vehicle_row.get("capacidade_peso_kg")),
        0.0,
    )
    cap_vol = _safe_float(
        vehicle_row.get("_cap_vol_m5_1", vehicle_row.get("capacidade_vol_m3")),
        0.0,
    )
    max_entregas = _safe_int(
        vehicle_row.get("_max_entregas_m5_1", vehicle_row.get("max_entregas")),
        0,
    )
    max_km = _safe_float(
        vehicle_row.get("_max_km_m5_1", vehicle_row.get("max_km_distancia")),
        0.0,
    )
    ocup_max = _safe_float(
        vehicle_row.get("_ocup_max_m5_1", vehicle_row.get("ocupacao_maxima_perc")),
        100.0,
    )

    if cap_peso > 0 and peso_total > cap_peso:
        return False, "excede_capacidade_peso"
    if cap_vol > 0 and volume_total > cap_vol:
        return False, "excede_capacidade_volume"
    if max_entregas > 0 and qtd_paradas > max_entregas:
        return False, "excede_max_entregas"
    if max_km > 0 and km_ref > max_km:
        return False, "excede_max_km"
    if ocupacao_perc > ocup_max:
        return False, "excede_ocupacao_maxima"

    return True, "ok"


def _validar_hard_constraints(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> Tuple[bool, str]:
    if df_itens.empty:
        return False, "grupo_vazio"

    restricoes_ok = _restricao_compatível(df_itens, vehicle_row)
    metricas = _extrair_metricas_df(df_itens, vehicle_row)

    return _validar_hard_constraints_metricas(
        peso_total=metricas["peso_total"],
        volume_total=metricas["volume_total"],
        qtd_paradas=int(metricas["qtd_paradas"]),
        km_ref=metricas["km_ref"],
        ocupacao_perc=metricas["ocupacao_perc"],
        restricoes_ok=restricoes_ok,
        vehicle_row=vehicle_row,
    )


def _validar_fechamento(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> Tuple[bool, str]:
    if df_itens.empty:
        return False, "grupo_vazio"

    restricoes_ok = _restricao_compatível(df_itens, vehicle_row)
    metricas = _extrair_metricas_df(df_itens, vehicle_row)

    ok_hard, motivo = _validar_hard_constraints_metricas(
        peso_total=metricas["peso_total"],
        volume_total=metricas["volume_total"],
        qtd_paradas=int(metricas["qtd_paradas"]),
        km_ref=metricas["km_ref"],
        ocupacao_perc=metricas["ocupacao_perc"],
        restricoes_ok=restricoes_ok,
        vehicle_row=vehicle_row,
    )
    if not ok_hard:
        return False, motivo

    ocup_min = _safe_float(
        vehicle_row.get("_ocup_min_m5_1", vehicle_row.get("ocupacao_minima_perc")),
        70.0,
    )
    if metricas["ocupacao_perc"] < ocup_min:
        return False, "abaixo_ocupacao_minima"

    return True, "ok"


# -----------------------------------------------------------------------------------------
# Auditoria
# -----------------------------------------------------------------------------------------
def _tentativa_dict(
    fase: str,
    camada: str,
    anchor_id: Optional[str],
    vehicle_row: Optional[pd.Series],
    resultado: str,
    motivo: str,
    df_candidato: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    candidato = df_candidato if df_candidato is not None else pd.DataFrame()

    if not candidato.empty:
        metricas = _extrair_metricas_df(candidato, vehicle_row)
    else:
        metricas = {
            "peso_total": 0.0,
            "volume_total": 0.0,
            "qtd_paradas": 0.0,
            "km_ref": 0.0,
            "ocupacao_perc": 0.0,
        }

    return {
        "fase": fase,
        "camada": camada,
        "anchor_id_linha_pipeline": anchor_id,
        "veiculo_tipo_tentado": None if vehicle_row is None else _safe_text(vehicle_row.get("tipo")),
        "resultado": resultado,
        "motivo": motivo,
        "qtd_itens_candidato": int(len(candidato)),
        "qtd_paradas_candidato": int(metricas["qtd_paradas"]),
        "peso_total_candidato": round(metricas["peso_total"], 3),
        "volume_total_candidato": round(metricas["volume_total"], 3),
        "km_referencia_candidato": round(metricas["km_ref"], 2),
        "ocupacao_perc_candidato": round(metricas["ocupacao_perc"], 2),
    }


# -----------------------------------------------------------------------------------------
# Manifesto
# -----------------------------------------------------------------------------------------
def _build_manifesto_id(seq: int) -> str:
    return f"PM51_{seq:04d}"


def _build_manifesto(
    df_itens: pd.DataFrame,
    vehicle_row: pd.Series,
    manifesto_id: str,
    fase: str,
    camada: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    qtd_itens = int(len(df_itens))
    qtd_ctes = int(df_itens["cte"].nunique(dropna=True)) if "cte" in df_itens.columns else qtd_itens
    metricas = _extrair_metricas_df(df_itens, vehicle_row)

    manifesto = {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": "pre_manifesto_bloco_5_1",
        "veiculo_tipo": _safe_text(vehicle_row.get("tipo")),
        "qtd_itens": qtd_itens,
        "qtd_ctes": qtd_ctes,
        "qtd_paradas": int(metricas["qtd_paradas"]),
        "base_carga_oficial": round(metricas["peso_total"], 3),
        "peso_total_kg": round(metricas["peso_total"], 3),
        "vol_total_m3": round(metricas["volume_total"], 3),
        "km_referencia": round(metricas["km_ref"], 2),
        "ocupacao_oficial_perc": round(metricas["ocupacao_perc"], 2),
        "capacidade_peso_kg_veiculo": _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0),
        "capacidade_vol_m3_veiculo": _safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0),
        "max_entregas_veiculo": _safe_int(vehicle_row.get("max_entregas"), 0),
        "max_km_distancia_veiculo": _safe_float(vehicle_row.get("max_km_distancia"), 0.0),
        "ignorar_ocupacao_minima": False,
        "origem_modulo": 5,
        "origem_etapa": f"{fase}_{camada}",
    }

    df_manifesto = pd.DataFrame([manifesto])
    df_itens_saida = df_itens.copy()
    for k, v in manifesto.items():
        df_itens_saida[k] = v

    return df_manifesto, df_itens_saida


# -----------------------------------------------------------------------------------------
# Fase 1 - mesmo cliente
# -----------------------------------------------------------------------------------------
def _tentar_fechar_grupo_inteiro(
    df_grupo: pd.DataFrame,
    veiculos_ordenados: pd.DataFrame,
    tentativas: List[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], str]:
    melhor_motivo = "nenhum_veiculo_compativel"

    for _, vehicle_row in veiculos_ordenados.iterrows():
        ok, motivo = _validar_fechamento(df_grupo, vehicle_row)
        tentativas.append(
            _tentativa_dict(
                fase="fase_1_mesmo_cliente",
                camada="mesmo_cliente",
                anchor_id=None,
                vehicle_row=vehicle_row,
                resultado="fechado" if ok else "falhou",
                motivo=motivo,
                df_candidato=df_grupo,
            )
        )
        melhor_motivo = motivo
        if ok:
            return df_grupo.copy(), vehicle_row.copy(), "ok"

    return None, None, melhor_motivo


# -----------------------------------------------------------------------------------------
# Fase 2 - composição regional com subconjunto viável
# -----------------------------------------------------------------------------------------
def _restricao_item_compativel_com_veiculo(
    restricao_norm: str,
    vehicle_row: pd.Series,
) -> bool:
    if not restricao_norm:
        return True

    tipo = _safe_text_upper(vehicle_row.get("tipo"))
    perfil = _safe_text_upper(vehicle_row.get("perfil"))
    return restricao_norm in {tipo, perfil}


def _validar_metricas_incrementais_hard(
    *,
    peso_total: float,
    volume_total: float,
    qtd_paradas: int,
    km_ref: float,
    vehicle_row: pd.Series,
) -> Tuple[bool, str]:
    cap_peso = _safe_float(
        vehicle_row.get("_cap_peso_m5_1", vehicle_row.get("capacidade_peso_kg")),
        0.0,
    )
    cap_vol = _safe_float(
        vehicle_row.get("_cap_vol_m5_1", vehicle_row.get("capacidade_vol_m3")),
        0.0,
    )
    max_entregas = _safe_int(
        vehicle_row.get("_max_entregas_m5_1", vehicle_row.get("max_entregas")),
        0,
    )
    max_km = _safe_float(
        vehicle_row.get("_max_km_m5_1", vehicle_row.get("max_km_distancia")),
        0.0,
    )
    ocup_max = _safe_float(
        vehicle_row.get("_ocup_max_m5_1", vehicle_row.get("ocupacao_maxima_perc")),
        100.0,
    )

    if cap_peso > 0 and peso_total > cap_peso:
        return False, "excede_capacidade_peso"
    if cap_vol > 0 and volume_total > cap_vol:
        return False, "excede_capacidade_volume"
    if max_entregas > 0 and qtd_paradas > max_entregas:
        return False, "excede_max_entregas"
    if max_km > 0 and km_ref > max_km:
        return False, "excede_max_km"

    if cap_peso > 0:
        ocupacao_perc = (peso_total / cap_peso) * 100.0
        if ocupacao_perc > ocup_max:
            return False, "excede_ocupacao_maxima"

    return True, "ok"


def _montar_subconjunto_regional_para_veiculo(
    pool_regional: pd.DataFrame,
    anchor_row: pd.Series,
    vehicle_row: pd.Series,
) -> pd.DataFrame:
    """
    Regionalidade = pool elegível.
    O algoritmo monta subconjunto viável dentro do pool.
    Otimizado para acumular métricas sem recriar DataFrame a cada iteração.
    """
    if pool_regional.empty:
        return pd.DataFrame(columns=pool_regional.columns)

    anchor_id = str(
        anchor_row["_id_str_m5_1"] if "_id_str_m5_1" in anchor_row.index else anchor_row["id_linha_pipeline"]
    )

    temp = _ordenar_operacional(pool_regional.copy())
    temp["_anchor_first_m5_1"] = temp["_id_str_m5_1"] == anchor_id
    temp = temp.sort_values(["_anchor_first_m5_1"], ascending=[False], kind="mergesort").drop(
        columns=["_anchor_first_m5_1"]
    ).reset_index(drop=True)

    selected_indices: List[int] = []
    selected_ids: Set[str] = set()
    selected_dests: Set[str] = set()

    peso_total = 0.0
    volume_total = 0.0
    km_ref = 0.0

    for row in temp.itertuples(index=True):
        row_id = getattr(row, "_id_str_m5_1")
        if row_id in selected_ids:
            continue

        restr_norm = getattr(row, "_restricao_norm_m5_1")
        if not _restricao_item_compativel_com_veiculo(restr_norm, vehicle_row):
            continue

        novo_peso = peso_total + _safe_float(getattr(row, "_peso_calc_m5_1"), 0.0)
        novo_volume = volume_total + _safe_float(getattr(row, "_vol_m3_m5_1"), 0.0)
        novo_km = max(km_ref, _safe_float(getattr(row, "_km_ref_m5_1"), 0.0))

        dest_key = getattr(row, "_destinatario_key_m5_1")
        novo_qtd_paradas = len(selected_dests | {dest_key})

        ok_hard, _ = _validar_metricas_incrementais_hard(
            peso_total=novo_peso,
            volume_total=novo_volume,
            qtd_paradas=novo_qtd_paradas,
            km_ref=novo_km,
            vehicle_row=vehicle_row,
        )
        if not ok_hard:
            continue

        selected_indices.append(row.Index)
        selected_ids.add(row_id)
        selected_dests.add(dest_key)
        peso_total = novo_peso
        volume_total = novo_volume
        km_ref = novo_km

    if not selected_indices:
        return pd.DataFrame(columns=pool_regional.columns)

    df_sel = temp.loc[selected_indices].copy().reset_index(drop=True)

    if anchor_id not in set(df_sel["_id_str_m5_1"].tolist()):
        df_anchor = pool_regional[pool_regional["_id_str_m5_1"] == anchor_id].copy()
        if not df_anchor.empty:
            tentativa = pd.concat([df_anchor, df_sel], ignore_index=True).drop_duplicates(
                subset=["id_linha_pipeline"], keep="first"
            )
            ok_hard, _ = _validar_hard_constraints(tentativa, vehicle_row)
            if ok_hard:
                df_sel = tentativa.copy()

    return _ordenar_operacional(df_sel.reset_index(drop=True))


def _tentar_pool_regional(
    pool_regional: pd.DataFrame,
    anchor_row: pd.Series,
    veiculos_ordenados: pd.DataFrame,
    camada: str,
    tentativas: List[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], str]:
    melhor_motivo = "nenhum_veiculo_compativel"
    anchor_id = str(
        anchor_row["_id_str_m5_1"] if "_id_str_m5_1" in anchor_row.index else anchor_row["id_linha_pipeline"]
    )

    for _, vehicle_row in veiculos_ordenados.iterrows():
        ok_cheio, motivo_cheio = _validar_fechamento(pool_regional, vehicle_row)
        tentativas.append(
            _tentativa_dict(
                fase="fase_2_regional",
                camada=f"{camada}_grupo_cheio",
                anchor_id=anchor_id,
                vehicle_row=vehicle_row,
                resultado="fechado" if ok_cheio else "falhou",
                motivo=motivo_cheio,
                df_candidato=pool_regional,
            )
        )
        if ok_cheio:
            return pool_regional.copy(), vehicle_row.copy(), "ok"

    for _, vehicle_row in veiculos_ordenados.iterrows():
        candidato = _montar_subconjunto_regional_para_veiculo(
            pool_regional=pool_regional,
            anchor_row=anchor_row,
            vehicle_row=vehicle_row,
        )

        if candidato.empty:
            tentativas.append(
                _tentativa_dict(
                    fase="fase_2_regional",
                    camada=f"{camada}_subconjunto",
                    anchor_id=anchor_id,
                    vehicle_row=vehicle_row,
                    resultado="falhou",
                    motivo="sem_subconjunto_viavel_hard_constraints",
                    df_candidato=candidato,
                )
            )
            melhor_motivo = "sem_subconjunto_viavel_hard_constraints"
            continue

        ok, motivo = _validar_fechamento(candidato, vehicle_row)
        tentativas.append(
            _tentativa_dict(
                fase="fase_2_regional",
                camada=f"{camada}_subconjunto",
                anchor_id=anchor_id,
                vehicle_row=vehicle_row,
                resultado="fechado" if ok else "falhou",
                motivo=motivo,
                df_candidato=candidato,
            )
        )
        melhor_motivo = motivo

        if ok:
            return candidato.copy(), vehicle_row.copy(), "ok"

    return None, None, melhor_motivo


# -----------------------------------------------------------------------------------------
# Limpeza de colunas auxiliares
# -----------------------------------------------------------------------------------------
def _remover_colunas_auxiliares_m5(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    cols_aux_remover = [
        "_anchor_attempted_m5_1",
        "_id_str_m5_1",
        "_cliente_chave_m5_1",
        "_cidade_key_m5_1",
        "_subregiao_key_m5_1",
        "_mesorregiao_key_m5_1",
        "_destinatario_key_m5_1",
        "_restricao_norm_m5_1",
        "_peso_calc_m5_1",
        "_vol_m3_m5_1",
        "_km_ref_m5_1",
        "_folga_m5_1",
        "_prioridade_embarque_ord_m5_1",
        "_ranking_oper_ord_m5_1",
        "_agendada_bool_m5_1",
        "_bucket_m5_1",
        "_prioridade_ord_m5_1",
        "_folga_ord_m5_1",
        "_km_ord_m5_1",
        "_peso_ord_m5_1",
        "_elegivel_fase2_m5_1",
    ]
    return df.drop(columns=[c for c in cols_aux_remover if c in df.columns]).reset_index(drop=True)


# -----------------------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------------------
def executar_m5_manifestos_compostos(
    df_remanescente_roteirizavel_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
    rodada_id: Optional[str] = None,
    data_base_roteirizacao: Optional[Any] = None,
    tipo_roteirizacao: str = "carteira",
    configuracao_frota: Optional[Any] = None,
    df_uso_frota_m4: Optional[pd.DataFrame] = None,
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    del configuracao_frota, df_uso_frota_m4, kwargs, rodada_id

    df_input, df_veic = _normalizar_inputs(
        df_remanescente_roteirizavel_bloco_4=df_remanescente_roteirizavel_bloco_4,
        df_veiculos_tratados=df_veiculos_tratados,
    )

    if df_input.empty:
        outputs_vazio = {
            "df_premanifestos_m5_1": pd.DataFrame(),
            "df_itens_premanifestos_m5_1": pd.DataFrame(),
            "df_tentativas_m5_1": pd.DataFrame(),
            "df_remanescente_m5_1": pd.DataFrame(),
            "df_nao_roteirizados_bloco_5_1": pd.DataFrame(),
            "df_uso_frota_m5_1": pd.DataFrame(),
        }
        meta_vazio = {
            "resumo_m5_1": {
                "modulo": "M5.1",
                "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
                "tipo_roteirizacao": tipo_roteirizacao,
                "remanescente_entrada_m5_1": 0,
                "pre_manifestos_gerados_m5_1": 0,
                "itens_pre_manifestados_m5_1": 0,
                "remanescente_saida_m5_1": 0,
                "nao_roteirizados_bloco_5_1": 0,
                "coluna_tipo_veiculo_utilizada": "tipo",
                "estrategia_m5_1": [
                    "fase_1_mesmo_cliente",
                    "saldo_para_cidade_subregiao_mesorregiao",
                    "regionalidade_como_pool_elegivel",
                    "subconjunto_viavel_ativo",
                    "VERSAO_M5_1_2026_04_10_OTIMIZADA_PLUS",
                ],
                "ocupacao_minima_padrao_perc": 70,
                "ocupacao_maxima_padrao_perc": 100,
                "anchors_geradas_m5_1": 0,
                "persistiu_artefatos": False,
                "caminhos_pipeline": caminhos_pipeline or {},
            },
            "auditoria_m5_1": {
                "total_tentativas": 0,
                "total_pre_manifestos": 0,
                "total_itens_pre_manifestados": 0,
                "total_remanescentes": 0,
                "anchors_geradas_m5_1": 0,
                "regionalidade_tratada_como_pool": True,
                "subconjunto_viavel_ativo": True,
            },
        }
        return outputs_vazio, meta_vazio

    df_input = _precompute_df_inputs(df_input)
    df_veic = _precompute_df_veiculos(df_veic)

    veiculos_ordenados = _veiculos_maior_para_menor(df_veic)
    saldo = _ordenar_operacional(df_input.copy())
    tentativas: List[Dict[str, Any]] = []
    manifestos_list: List[pd.DataFrame] = []
    itens_manifestados_list: List[pd.DataFrame] = []
    manifest_seq = 1
    anchors_geradas = 0

    # =====================================================================================
    # FASE 1 - MESMO CLIENTE
    # =====================================================================================
    ids_consumidos_fase1: Set[str] = set()

    for _, grp in saldo.groupby("_cliente_chave_m5_1", sort=False):
        df_grupo = _ordenar_operacional(grp.copy())
        ids_grupo = set(df_grupo["_id_str_m5_1"].tolist())
        if ids_consumidos_fase1 & ids_grupo:
            continue

        cliente_chave = df_grupo["_cliente_chave_m5_1"].iloc[0] if not df_grupo.empty else ""

        candidato, vehicle_row, motivo = _tentar_fechar_grupo_inteiro(
            df_grupo=df_grupo,
            veiculos_ordenados=veiculos_ordenados,
            tentativas=tentativas,
        )

        if candidato is not None and vehicle_row is not None:
            manifesto_id = _build_manifesto_id(manifest_seq)
            manifest_seq += 1
            df_manifesto, df_itens = _build_manifesto(
                df_itens=candidato,
                vehicle_row=vehicle_row,
                manifesto_id=manifesto_id,
                fase="fase_1_mesmo_cliente",
                camada="mesmo_cliente",
            )
            manifestos_list.append(df_manifesto)
            itens_manifestados_list.append(df_itens)
            ids_consumidos_fase1.update(ids_grupo)
        else:
            tentativas.append(
                {
                    "fase": "fase_1_mesmo_cliente",
                    "camada": "mesmo_cliente",
                    "anchor_id_linha_pipeline": None,
                    "veiculo_tipo_tentado": None,
                    "resultado": "saldo",
                    "motivo": motivo,
                    "qtd_itens_candidato": int(len(df_grupo)),
                    "qtd_paradas_candidato": _qtd_paradas(df_grupo),
                    "peso_total_candidato": round(_peso_total(df_grupo), 3),
                    "volume_total_candidato": round(_volume_total(df_grupo), 3),
                    "km_referencia_candidato": round(_km_referencia(df_grupo), 2),
                    "ocupacao_perc_candidato": 0.0,
                    "cliente_chave": cliente_chave,
                }
            )

    if ids_consumidos_fase1:
        saldo = saldo[~saldo["_id_str_m5_1"].isin(ids_consumidos_fase1)].copy()
        saldo = _ordenar_operacional(saldo)

    # =====================================================================================
    # FASE 2 - SALDO PARA CIDADE / SUB / MESO
    # =====================================================================================
    saldo["_anchor_attempted_m5_1"] = False

    while True:
        elegiveis = saldo[
            (saldo["_anchor_attempted_m5_1"] == False)  # noqa: E712
            & (saldo["_elegivel_fase2_m5_1"] == True)  # noqa: E712
        ].copy()

        if elegiveis.empty:
            break

        elegiveis = _ordenar_operacional(elegiveis)
        anchor_row = elegiveis.iloc[0].copy()
        anchor_id = str(anchor_row["_id_str_m5_1"])
        anchors_geradas += 1

        saldo.loc[saldo["_id_str_m5_1"] == anchor_id, "_anchor_attempted_m5_1"] = True

        fechou_anchor = False
        melhor_motivo_anchor = "sem_pool_regional"

        camada_to_col = {
            "cidade": "_cidade_key_m5_1",
            "subregiao": "_subregiao_key_m5_1",
            "mesorregiao": "_mesorregiao_key_m5_1",
        }

        for camada in ["cidade", "subregiao", "mesorregiao"]:
            col_key = camada_to_col[camada]
            valor = _safe_text(anchor_row.get(col_key))
            if not valor:
                tentativas.append(
                    _tentativa_dict(
                        fase="fase_2_regional",
                        camada=camada,
                        anchor_id=anchor_id,
                        vehicle_row=None,
                        resultado="falhou",
                        motivo=f"{camada}_vazia",
                        df_candidato=None,
                    )
                )
                melhor_motivo_anchor = f"{camada}_vazia"
                continue

            pool = saldo[saldo[col_key] == valor].copy()
            pool = _ordenar_operacional(pool)

            if pool.empty:
                tentativas.append(
                    _tentativa_dict(
                        fase="fase_2_regional",
                        camada=camada,
                        anchor_id=anchor_id,
                        vehicle_row=None,
                        resultado="falhou",
                        motivo="pool_regional_vazio",
                        df_candidato=None,
                    )
                )
                melhor_motivo_anchor = "pool_regional_vazio"
                continue

            candidato, vehicle_row, motivo = _tentar_pool_regional(
                pool_regional=pool,
                anchor_row=anchor_row,
                veiculos_ordenados=veiculos_ordenados,
                camada=camada,
                tentativas=tentativas,
            )

            melhor_motivo_anchor = motivo

            if candidato is not None and vehicle_row is not None:
                manifesto_id = _build_manifesto_id(manifest_seq)
                manifest_seq += 1
                df_manifesto, df_itens = _build_manifesto(
                    df_itens=candidato,
                    vehicle_row=vehicle_row,
                    manifesto_id=manifesto_id,
                    fase="fase_2_regional",
                    camada=camada,
                )
                manifestos_list.append(df_manifesto)
                itens_manifestados_list.append(df_itens)

                consumed_ids = set(candidato["_id_str_m5_1"].tolist())
                saldo = saldo[~saldo["_id_str_m5_1"].isin(consumed_ids)].copy()
                saldo = _ordenar_operacional(saldo)
                fechou_anchor = True
                break

        if not fechou_anchor:
            tentativas.append(
                {
                    "fase": "fase_2_regional",
                    "camada": "fim_anchor",
                    "anchor_id_linha_pipeline": anchor_id,
                    "veiculo_tipo_tentado": None,
                    "resultado": "saldo",
                    "motivo": melhor_motivo_anchor,
                    "qtd_itens_candidato": 1,
                    "qtd_paradas_candidato": 1,
                    "peso_total_candidato": round(_safe_float(anchor_row.get("_peso_calc_m5_1"), 0.0), 3),
                    "volume_total_candidato": round(_safe_float(anchor_row.get("_vol_m3_m5_1"), 0.0), 3),
                    "km_referencia_candidato": round(_safe_float(anchor_row.get("_km_ref_m5_1"), 0.0), 2),
                    "ocupacao_perc_candidato": 0.0,
                }
            )

    # =====================================================================================
    # SAÍDAS
    # =====================================================================================
    df_premanifestos_m5_1 = pd.concat(manifestos_list, ignore_index=True) if manifestos_list else pd.DataFrame()
    df_itens_premanifestos_m5_1 = (
        pd.concat(itens_manifestados_list, ignore_index=True) if itens_manifestados_list else pd.DataFrame()
    )
    df_tentativas_m5_1 = pd.DataFrame(tentativas)

    saldo = _remover_colunas_auxiliares_m5(saldo)
    df_itens_premanifestos_m5_1 = _remover_colunas_auxiliares_m5(df_itens_premanifestos_m5_1)

    df_remanescente_m5_1 = saldo.copy()
    df_nao_roteirizados_bloco_5_1 = saldo.copy()

    if not df_itens_premanifestos_m5_1.empty:
        df_uso_frota_m5_1 = (
            df_itens_premanifestos_m5_1.groupby("veiculo_tipo", dropna=False)
            .agg(
                pre_manifestos=("manifesto_id", "nunique"),
                itens=("id_linha_pipeline", "count"),
                peso_total_kg=("peso_calculado", "sum"),
                paradas=("destinatario", "nunique"),
            )
            .reset_index()
        )
    else:
        df_uso_frota_m5_1 = pd.DataFrame(
            columns=["veiculo_tipo", "pre_manifestos", "itens", "peso_total_kg", "paradas"]
        )

    resumo_m5_1 = {
        "modulo": "M5.1",
        "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
        "tipo_roteirizacao": tipo_roteirizacao,
        "remanescente_entrada_m5_1": int(len(df_input)),
        "pre_manifestos_gerados_m5_1": int(df_premanifestos_m5_1["manifesto_id"].nunique())
        if not df_premanifestos_m5_1.empty
        else 0,
        "itens_pre_manifestados_m5_1": int(len(df_itens_premanifestos_m5_1)),
        "remanescente_saida_m5_1": int(len(df_remanescente_m5_1)),
        "nao_roteirizados_bloco_5_1": int(len(df_nao_roteirizados_bloco_5_1)),
        "coluna_tipo_veiculo_utilizada": "tipo",
        "estrategia_m5_1": [
            "fase_1_mesmo_cliente",
            "saldo_para_cidade_subregiao_mesorregiao",
            "regionalidade_como_pool_elegivel",
            "subconjunto_viavel_ativo",
            "VERSAO_M5_1_2026_04_10_OTIMIZADA_PLUS",
        ],
        "ocupacao_minima_padrao_perc": 70,
        "ocupacao_maxima_padrao_perc": 100,
        "anchors_geradas_m5_1": int(anchors_geradas),
        "persistiu_artefatos": False,
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    auditoria_m5_1 = {
        "total_tentativas": int(len(df_tentativas_m5_1)),
        "total_pre_manifestos": int(df_premanifestos_m5_1["manifesto_id"].nunique())
        if not df_premanifestos_m5_1.empty
        else 0,
        "total_itens_pre_manifestados": int(len(df_itens_premanifestos_m5_1)),
        "total_remanescentes": int(len(df_remanescente_m5_1)),
        "anchors_geradas_m5_1": int(anchors_geradas),
        "regionalidade_tratada_como_pool": True,
        "subconjunto_viavel_ativo": True,
    }

    outputs_m5_1 = {
        "df_premanifestos_m5_1": df_premanifestos_m5_1,
        "df_itens_premanifestos_m5_1": df_itens_premanifestos_m5_1,
        "df_tentativas_m5_1": df_tentativas_m5_1,
        "df_remanescente_m5_1": df_remanescente_m5_1,
        "df_nao_roteirizados_bloco_5_1": df_nao_roteirizados_bloco_5_1,
        "df_uso_frota_m5_1": df_uso_frota_m5_1,
    }

    meta_m5_1 = {
        "resumo_m5_1": resumo_m5_1,
        "auditoria_m5_1": auditoria_m5_1,
    }

    return outputs_m5_1, meta_m5_1


# Aliases defensivos
def executar_m5_1(*args: Any, **kwargs: Any):
    return executar_m5_manifestos_compostos(*args, **kwargs)


def processar_m5_1(*args: Any, **kwargs: Any):
    return executar_m5_manifestos_compostos(*args, **kwargs)


def rodar_m5_1(*args: Any, **kwargs: Any):
    return executar_m5_manifestos_compostos(*args, **kwargs)
