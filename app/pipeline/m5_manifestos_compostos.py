from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

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
# OTIMIZAÇÕES:
# - pré-cálculo de colunas auxiliares
# - eliminação de apply(axis=1) em loops críticos
# - subconjunto incremental sem recriar DataFrame a cada linha
# - reutilização de métricas
# - ocupação calculada somente por Peso Calculo
# - correção do erro com itertuples() + colunas iniciadas por "_"
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
