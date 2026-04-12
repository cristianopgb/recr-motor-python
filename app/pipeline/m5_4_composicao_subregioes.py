from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import pandas as pd


# =========================================================================================
# M5.4 - COMPOSIÇÃO DE SUBREGIÕES
# -----------------------------------------------------------------------------------------
# OBJETIVO
# - receber os elegíveis do M5.3
# - compor por subregião
# - tentar do maior perfil para o menor
# - ancorar por cliente
# - tentar máxima composição possível dentro da subregião
# - priorizar mesma cidade da âncora
# - depois expandir para cidades próximas da mesma subregião
# - respeitar ocupação mínima, ocupação máxima, raio e quantidade de paradas
#
# REGRAS IMPORTANTES
# - NÃO compor só dentro da mesma cidade
# - cidade já foi tratada no M5.2
# - aqui a unidade territorial é a SUBREGIÃO
# - perfil é otimização, não trava
# - se um perfil não fechar, tenta o próximo
# - se uma combinação não fechar, ajusta pela seleção de blocos e segue
#
# SAÍDA
# - pré-manifestos compostos por subregião
# - remanescente oficial do M5.4
# - auditoria de subregiões e perfis
# =========================================================================================


# -----------------------------------------------------------------------------------------
# Helpers básicos
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


def _ensure_column(df: pd.DataFrame, col: str, default: Any) -> None:
    if col not in df.columns:
        df[col] = default


# -----------------------------------------------------------------------------------------
# Geometria básica
# -----------------------------------------------------------------------------------------
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return 999999.0

    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


# -----------------------------------------------------------------------------------------
# Normalização
# -----------------------------------------------------------------------------------------
def _normalizar_inputs(
    df_saldo_elegivel_composicao_m5_3: pd.DataFrame,
    df_perfis_base_m5: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    saldo = (
        df_saldo_elegivel_composicao_m5_3.copy()
        if df_saldo_elegivel_composicao_m5_3 is not None
        else pd.DataFrame()
    )
    perfis = df_perfis_base_m5.copy() if df_perfis_base_m5 is not None else pd.DataFrame()

    if saldo.empty:
        return saldo, perfis

    rename_map: Dict[str, str] = {}
    if "sub_regiao" in saldo.columns and "subregiao" not in saldo.columns:
        rename_map["sub_regiao"] = "subregiao"
    if "mesoregiao" in saldo.columns and "mesorregiao" not in saldo.columns:
        rename_map["mesoregiao"] = "mesorregiao"
    if rename_map:
        saldo = saldo.rename(columns=rename_map)

    defaults = {
        "id_linha_pipeline": None,
        "cidade": "",
        "uf": "",
        "subregiao": "",
        "mesorregiao": "",
        "destinatario": "",
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
        "veiculo_exclusivo": False,
        "veiculo_exclusivo_flag": False,
        "latitude_destinatario": pd.NA,
        "longitude_destinatario": pd.NA,
        "origem_latitude": pd.NA,
        "origem_longitude": pd.NA,
        "perfil_base_triagem_m5_3": "",
        "capacidade_peso_kg_perfil_base_m5_3": pd.NA,
        "ocupacao_minima_perc_perfil_base_m5_3": pd.NA,
        "piso_minimo_kg_perfil_base_m5_3": pd.NA,
    }
    for col, default in defaults.items():
        _ensure_column(saldo, col, default)

    if saldo["id_linha_pipeline"].isna().any():
        raise ValueError("M5.4 exige id_linha_pipeline em todas as linhas elegíveis do M5.3.")

    if "peso_calculado" not in saldo.columns and "peso_c" in saldo.columns:
        saldo["peso_calculado"] = saldo["peso_c"]

    if "peso_kg" not in saldo.columns:
        saldo["peso_kg"] = saldo["peso_calculado"]

    if "distancia_rodoviaria_est_km" not in saldo.columns and "distancia_km" in saldo.columns:
        saldo["distancia_rodoviaria_est_km"] = saldo["distancia_km"]

    numeric_cols = [
        "peso_calculado",
        "peso_kg",
        "vol_m3",
        "distancia_rodoviaria_est_km",
        "folga_dias",
        "prioridade_embarque_num",
        "prioridade_embarque",
        "ranking_prioridade_operacional",
        "latitude_destinatario",
        "longitude_destinatario",
        "origem_latitude",
        "origem_longitude",
        "capacidade_peso_kg_perfil_base_m5_3",
        "ocupacao_minima_perc_perfil_base_m5_3",
        "piso_minimo_kg_perfil_base_m5_3",
    ]
    for col in numeric_cols:
        if col in saldo.columns:
            saldo[col] = pd.to_numeric(saldo[col], errors="coerce")

    for col in ["agendada", "veiculo_exclusivo", "veiculo_exclusivo_flag"]:
        if col in saldo.columns:
            saldo[col] = saldo[col].apply(_safe_bool)

    for col in ["cidade", "uf", "subregiao", "mesorregiao", "destinatario"]:
        saldo[col] = saldo[col].fillna("").astype(str).str.strip()

    if perfis.empty:
        raise ValueError("M5.4 exige df_perfis_base_m5.")

    for col in [
        "perfil",
        "tipo",
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ocupacao_minima_perc",
        "ocupacao_maxima_perc",
    ]:
        _ensure_column(perfis, col, pd.NA if col not in ["perfil", "tipo"] else "")

    for col in [
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ocupacao_minima_perc",
        "ocupacao_maxima_perc",
    ]:
        perfis[col] = pd.to_numeric(perfis[col], errors="coerce")

    for col in ["perfil", "tipo"]:
        perfis[col] = perfis[col].fillna("").astype(str).str.strip()

    perfis = (
        perfis[
            [
                "perfil",
                "tipo",
                "capacidade_peso_kg",
                "capacidade_vol_m3",
                "max_entregas",
                "max_km_distancia",
                "ocupacao_minima_perc",
                "ocupacao_maxima_perc",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return saldo, perfis


# -----------------------------------------------------------------------------------------
# Ordenação operacional
# -----------------------------------------------------------------------------------------
def _fase_bucket(row: pd.Series) -> int:
    prioridade_embarque = pd.to_numeric(
        row.get("prioridade_embarque_num", row.get("prioridade_embarque", pd.NA)),
        errors="coerce",
    )
    agendada = _safe_bool(row.get("agendada"))
    folga = _safe_float(row.get("folga_dias"), 999)

    if not pd.isna(prioridade_embarque) and _safe_float(prioridade_embarque, 0) > 0:
        return 0
    if agendada and folga == 0:
        return 1
    if agendada and folga == 1:
        return 2
    if (not agendada) and folga == 0:
        return 3
    if (not agendada) and folga == 1:
        return 4
    return 99


def _precalcular_ordenacao(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    temp = df.copy()
    temp["_id_str_m5_4"] = temp["id_linha_pipeline"].astype(str)
    temp["_subregiao_key_m5_4"] = temp["subregiao"].fillna("").astype(str).str.strip()
    temp["_uf_key_m5_4"] = temp["uf"].fillna("").astype(str).str.strip()
    temp["_cidade_key_m5_4"] = temp["cidade"].fillna("").astype(str).str.strip()
    temp["_cliente_key_m5_4"] = temp["destinatario"].fillna("").astype(str).str.strip()

    folga = pd.to_numeric(temp["folga_dias"], errors="coerce").fillna(999)
    ranking = pd.to_numeric(temp["ranking_prioridade_operacional"], errors="coerce").fillna(999)
    km = pd.to_numeric(temp["distancia_rodoviaria_est_km"], errors="coerce").fillna(999999)
    peso = pd.to_numeric(temp["peso_calculado"], errors="coerce").fillna(0.0)

    buckets: List[int] = []
    prioridade_ord: List[float] = []

    for _, row in temp.iterrows():
        buckets.append(_fase_bucket(row))
        p = pd.to_numeric(
            row.get("prioridade_embarque_num", row.get("prioridade_embarque", pd.NA)),
            errors="coerce",
        )
        prioridade_ord.append(_safe_float(p, 999.0) if not pd.isna(p) else 999.0)

    temp["_bucket_m5_4"] = buckets
    temp["_prioridade_ord_m5_4"] = prioridade_ord
    temp["_folga_ord_m5_4"] = folga
    temp["_ranking_ord_m5_4"] = ranking
    temp["_km_ord_m5_4"] = km
    temp["_peso_ord_m5_4"] = -peso

    return temp


def _ordenar_operacional(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "_bucket_m5_4" not in df.columns:
        df = _precalcular_ordenacao(df)

    return (
        df.sort_values(
            by=[
                "_bucket_m5_4",
                "_prioridade_ord_m5_4",
                "_folga_ord_m5_4",
                "_ranking_ord_m5_4",
                "_km_ord_m5_4",
                "_peso_ord_m5_4",
                "_id_str_m5_4",
            ],
            ascending=[True, True, True, True, True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )


# -----------------------------------------------------------------------------------------
# Veículos
# -----------------------------------------------------------------------------------------
def _veiculos_maior_para_menor(df_perfis: pd.DataFrame) -> pd.DataFrame:
    temp = df_perfis.copy()
    temp["_cap_peso_tmp"] = pd.to_numeric(temp["capacidade_peso_kg"], errors="coerce").fillna(0)
    temp["_cap_vol_tmp"] = pd.to_numeric(temp["capacidade_vol_m3"], errors="coerce").fillna(0)

    return (
        temp.sort_values(
            ["_cap_peso_tmp", "_cap_vol_tmp", "tipo", "perfil"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        .drop(columns=["_cap_peso_tmp", "_cap_vol_tmp"], errors="ignore")
        .reset_index(drop=True)
        .copy()
    )


# -----------------------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------------------
def _peso_total(df_itens: pd.DataFrame) -> float:
    if df_itens.empty:
        return 0.0
    return float(pd.to_numeric(df_itens["peso_calculado"], errors="coerce").fillna(0).sum())


def _volume_total(df_itens: pd.DataFrame) -> float:
    if df_itens.empty:
        return 0.0
    return float(pd.to_numeric(df_itens["vol_m3"], errors="coerce").fillna(0).sum())


def _km_referencia(df_itens: pd.DataFrame) -> float:
    if df_itens.empty:
        return 0.0
    return float(pd.to_numeric(df_itens["distancia_rodoviaria_est_km"], errors="coerce").fillna(0).max())


def _qtd_paradas(df_itens: pd.DataFrame) -> int:
    if df_itens.empty:
        return 0
    return int(df_itens["destinatario"].fillna("").astype(str).nunique())


def _ocupacao_minima_kg(vehicle_row: pd.Series) -> float:
    cap_peso = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    ocup_min = _safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0)
    return cap_peso * (ocup_min / 100.0)


def _ocupacao_maxima_kg(vehicle_row: pd.Series) -> float:
    cap_peso = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    ocup_max = _safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0)
    if ocup_max <= 0:
        ocup_max = 100.0
    return cap_peso * (ocup_max / 100.0)


# -----------------------------------------------------------------------------------------
# Blocos por cliente
# -----------------------------------------------------------------------------------------
def _montar_blocos_cliente(df_subregiao: pd.DataFrame) -> pd.DataFrame:
    if df_subregiao.empty:
        return pd.DataFrame()

    group_cols = [
        "_subregiao_key_m5_4",
        "_uf_key_m5_4",
        "_cidade_key_m5_4",
        "_cliente_key_m5_4",
    ]

    records: List[Dict[str, Any]] = []

    for keys, grupo in df_subregiao.groupby(group_cols, dropna=False, sort=False):
        subregiao_key, uf_key, cidade_key, cliente_key = keys
        grupo = _ordenar_operacional(grupo.copy())

        lat_media = pd.to_numeric(grupo["latitude_destinatario"], errors="coerce").dropna()
        lon_media = pd.to_numeric(grupo["longitude_destinatario"], errors="coerce").dropna()

        record = {
            "subregiao": subregiao_key,
            "uf": uf_key,
            "cidade": cidade_key,
            "destinatario": cliente_key,
            "bloco_id": f"{subregiao_key}|{uf_key}|{cidade_key}|{cliente_key}",
            "qtd_linhas": int(len(grupo)),
            "qtd_paradas": int(grupo["destinatario"].fillna("").astype(str).nunique()),
            "peso_total": round(_peso_total(grupo), 3),
            "volume_total": round(_volume_total(grupo), 3),
            "km_referencia": round(_km_referencia(grupo), 2),
            "bucket_min": int(pd.to_numeric(grupo["_bucket_m5_4"], errors="coerce").fillna(99).min()),
            "prioridade_min": float(pd.to_numeric(grupo["_prioridade_ord_m5_4"], errors="coerce").fillna(999).min()),
            "ranking_min": float(pd.to_numeric(grupo["_ranking_ord_m5_4"], errors="coerce").fillna(999).min()),
            "lat_centroide": float(lat_media.mean()) if not lat_media.empty else pd.NA,
            "lon_centroide": float(lon_media.mean()) if not lon_media.empty else pd.NA,
        }
        records.append(record)

    blocos = pd.DataFrame(records)
    if blocos.empty:
        return blocos

    return (
        blocos.sort_values(
            by=["bucket_min", "prioridade_min", "ranking_min", "peso_total", "bloco_id"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )


def _filtrar_itens_bloco(df_subregiao: pd.DataFrame, bloco_row: pd.Series) -> pd.DataFrame:
    return df_subregiao[
        (df_subregiao["_subregiao_key_m5_4"] == _safe_text(bloco_row.get("subregiao")))
        & (df_subregiao["_uf_key_m5_4"] == _safe_text(bloco_row.get("uf")))
        & (df_subregiao["_cidade_key_m5_4"] == _safe_text(bloco_row.get("cidade")))
        & (df_subregiao["_cliente_key_m5_4"] == _safe_text(bloco_row.get("destinatario")))
    ].copy()


# -----------------------------------------------------------------------------------------
# Regras de compatibilidade
# -----------------------------------------------------------------------------------------
def _bloco_respeita_restricao_veiculo(df_bloco: pd.DataFrame, vehicle_row: pd.Series) -> bool:
    tipo_veiculo = _safe_text(vehicle_row.get("tipo")) or _safe_text(vehicle_row.get("perfil"))
    if not tipo_veiculo:
        return True

    if "restricao_veiculo" not in df_bloco.columns:
        return True

    restricoes = (
        df_bloco["restricao_veiculo"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    if restricoes.empty:
        return True

    for restr in restricoes.unique().tolist():
        if restr and restr.lower() not in {"nan", "none", ""}:
            if restr.lower() != tipo_veiculo.lower():
                return False

    return True


def _conjunto_respeita_limites(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> Tuple[bool, str]:
    if df_itens.empty:
        return False, "conjunto_vazio"

    peso_total = _peso_total(df_itens)
    vol_total = _volume_total(df_itens)
    km_ref = _km_referencia(df_itens)
    qtd_paradas = _qtd_paradas(df_itens)

    cap_peso = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    cap_vol = _safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0)
    max_entregas = _safe_int(vehicle_row.get("max_entregas"), 0)
    max_km = _safe_float(vehicle_row.get("max_km_distancia"), 0.0)

    if cap_peso > 0 and peso_total > cap_peso:
        return False, "excesso_capacidade_peso"

    if cap_vol > 0 and vol_total > cap_vol:
        return False, "excesso_capacidade_volume"

    if max_entregas > 0 and qtd_paradas > max_entregas:
        return False, "excesso_paradas"

    if max_km > 0 and km_ref > max_km:
        return False, "raio_excedido"

    return True, "ok"


def _conjunto_bate_ocupacao_minima(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> bool:
    peso_total = _peso_total(df_itens)
    return peso_total >= _ocupacao_minima_kg(vehicle_row)


def _conjunto_bate_ocupacao_maxima(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> bool:
    peso_total = _peso_total(df_itens)
    return peso_total <= _ocupacao_maxima_kg(vehicle_row)


# -----------------------------------------------------------------------------------------
# Priorização territorial dentro da subregião
# -----------------------------------------------------------------------------------------
def _cidade_mais_proxima_ancora(bloco_row: pd.Series, anchor_row: pd.Series) -> float:
    if _safe_text(bloco_row.get("cidade")) == _safe_text(anchor_row.get("cidade")):
        return 0.0

    return _haversine_km(
        _safe_float(bloco_row.get("lat_centroide"), math.nan),
        _safe_float(bloco_row.get("lon_centroide"), math.nan),
        _safe_float(anchor_row.get("lat_centroide"), math.nan),
        _safe_float(anchor_row.get("lon_centroide"), math.nan),
    )


def _ordenar_candidatos(anchor_row: pd.Series, blocos_restantes: pd.DataFrame) -> pd.DataFrame:
    if blocos_restantes.empty:
        return blocos_restantes.copy()

    temp = blocos_restantes.copy()
    cidade_anchor = _safe_text(anchor_row.get("cidade"))

    temp["_mesma_cidade_anchor"] = (
        temp["cidade"].fillna("").astype(str).str.strip() == cidade_anchor
    )
    temp["_dist_cidade_anchor"] = temp.apply(
        lambda row: _cidade_mais_proxima_ancora(row, anchor_row), axis=1
    )

    return (
        temp.sort_values(
            by=[
                "_mesma_cidade_anchor",
                "_dist_cidade_anchor",
                "bucket_min",
                "prioridade_min",
                "ranking_min",
                "peso_total",
                "bloco_id",
            ],
            ascending=[False, True, True, True, True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )


# -----------------------------------------------------------------------------------------
# Montagem de manifesto candidato
# -----------------------------------------------------------------------------------------
def _tentar_montar_manifesto_subregiao(
    df_subregiao_restante: pd.DataFrame,
    blocos_restantes: pd.DataFrame,
    vehicle_row: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if df_subregiao_restante.empty or blocos_restantes.empty:
        return pd.DataFrame(), blocos_restantes.copy(), {
            "status_tentativa": "sem_base",
            "motivo_tentativa": "subregiao_sem_blocos",
        }

    anchor_row = blocos_restantes.iloc[0].copy()
    anchor_itens = _filtrar_itens_bloco(df_subregiao_restante, anchor_row)

    if anchor_itens.empty:
        blocos_sem_anchor = blocos_restantes.iloc[1:].reset_index(drop=True).copy()
        return pd.DataFrame(), blocos_sem_anchor, {
            "status_tentativa": "falha",
            "motivo_tentativa": "bloco_anchor_sem_itens",
        }

    if not _bloco_respeita_restricao_veiculo(anchor_itens, vehicle_row):
        blocos_sem_anchor = blocos_restantes.iloc[1:].reset_index(drop=True).copy()
        return pd.DataFrame(), blocos_sem_anchor, {
            "status_tentativa": "falha",
            "motivo_tentativa": "restricao_veiculo_anchor_incompativel",
        }

    ok_anchor, motivo_anchor = _conjunto_respeita_limites(anchor_itens, vehicle_row)
    if not ok_anchor:
        blocos_sem_anchor = blocos_restantes.iloc[1:].reset_index(drop=True).copy()
        return pd.DataFrame(), blocos_sem_anchor, {
            "status_tentativa": "falha",
            "motivo_tentativa": motivo_anchor,
        }

    manifesto_df = anchor_itens.copy()
    blocos_usados = { _safe_text(anchor_row.get("bloco_id")) }

    candidatos = _ordenar_candidatos(anchor_row, blocos_restantes.iloc[1:].copy())

    for _, bloco_row in candidatos.iterrows():
        bloco_id = _safe_text(bloco_row.get("bloco_id"))
        if bloco_id in blocos_usados:
            continue

        bloco_itens = _filtrar_itens_bloco(df_subregiao_restante, bloco_row)
        if bloco_itens.empty:
            continue

        if not _bloco_respeita_restricao_veiculo(bloco_itens, vehicle_row):
            continue

        tentativa_df = pd.concat([manifesto_df, bloco_itens], ignore_index=True)

        ok_limites, _ = _conjunto_respeita_limites(tentativa_df, vehicle_row)
        ok_max = _conjunto_bate_ocupacao_maxima(tentativa_df, vehicle_row)

        if ok_limites and ok_max:
            manifesto_df = tentativa_df
            blocos_usados.add(bloco_id)

    if not _conjunto_bate_ocupacao_minima(manifesto_df, vehicle_row):
        return pd.DataFrame(), blocos_restantes.copy(), {
            "status_tentativa": "falha",
            "motivo_tentativa": "abaixo_ocupacao_minima",
            "bloco_anchor": _safe_text(anchor_row.get("bloco_id")),
        }

    blocos_restantes_saida = blocos_restantes[
        ~blocos_restantes["bloco_id"].astype(str).isin(list(blocos_usados))
    ].reset_index(drop=True).copy()

    return manifesto_df.reset_index(drop=True).copy(), blocos_restantes_saida, {
        "status_tentativa": "sucesso",
        "motivo_tentativa": "ok",
        "bloco_anchor": _safe_text(anchor_row.get("bloco_id")),
        "blocos_usados_qtd": len(blocos_usados),
    }


# -----------------------------------------------------------------------------------------
# Auditoria
# -----------------------------------------------------------------------------------------
def _registro_manifesto(
    manifesto_id: str,
    subregiao_key: str,
    uf_key: str,
    vehicle_row: pd.Series,
    df_itens: pd.DataFrame,
    origem_etapa: str,
) -> Dict[str, Any]:
    return {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": "composto_bloco_5_4_subregiao",
        "origem_modulo": 5.4,
        "origem_etapa": origem_etapa,
        "subregiao": subregiao_key,
        "uf": uf_key,
        "veiculo_tipo": _safe_text(vehicle_row.get("tipo")) or _safe_text(vehicle_row.get("perfil")),
        "perfil": _safe_text(vehicle_row.get("perfil")),
        "qtd_itens": int(len(df_itens)),
        "qtd_ctes": int(df_itens["cte"].nunique() if "cte" in df_itens.columns else len(df_itens)),
        "qtd_paradas": int(_qtd_paradas(df_itens)),
        "peso_total_kg": round(_peso_total(df_itens), 3),
        "vol_total_m3": round(_volume_total(df_itens), 3),
        "km_referencia": round(_km_referencia(df_itens), 2),
        "ocupacao_minima_kg": round(_ocupacao_minima_kg(vehicle_row), 3),
        "ocupacao_maxima_kg": round(_ocupacao_maxima_kg(vehicle_row), 3),
        "capacidade_peso_kg_veiculo": round(_safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0), 3),
        "capacidade_vol_m3_veiculo": round(_safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0), 3),
        "max_entregas_veiculo": _safe_int(vehicle_row.get("max_entregas"), 0),
        "max_km_distancia_veiculo": round(_safe_float(vehicle_row.get("max_km_distancia"), 0.0), 3),
        "ocupacao_minima_perc_veiculo": round(_safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0), 3),
        "ocupacao_maxima_perc_veiculo": round(_safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0), 3),
        "cidade_anchor": _safe_text(df_itens.iloc[0].get("cidade")) if not df_itens.empty else "",
        "destinatario_anchor": _safe_text(df_itens.iloc[0].get("destinatario")) if not df_itens.empty else "",
    }


def _limpar_cols_auxiliares(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    cols_drop = [
        "_id_str_m5_4",
        "_subregiao_key_m5_4",
        "_uf_key_m5_4",
        "_cidade_key_m5_4",
        "_cliente_key_m5_4",
        "_bucket_m5_4",
        "_prioridade_ord_m5_4",
        "_folga_ord_m5_4",
        "_ranking_ord_m5_4",
        "_km_ord_m5_4",
        "_peso_ord_m5_4",
    ]
    return df.drop(columns=cols_drop, errors="ignore").reset_index(drop=True).copy()


# -----------------------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------------------
def executar_m5_4_composicao_subregioes(
    df_saldo_elegivel_composicao_m5_3: pd.DataFrame,
    df_perfis_base_m5: pd.DataFrame,
    rodada_id: Optional[str] = None,
    data_base_roteirizacao: Optional[Any] = None,
    tipo_roteirizacao: str = "carteira",
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    del kwargs

    saldo, perfis = _normalizar_inputs(
        df_saldo_elegivel_composicao_m5_3=df_saldo_elegivel_composicao_m5_3,
        df_perfis_base_m5=df_perfis_base_m5,
    )

    if saldo.empty:
        outputs_vazio = {
            "df_premanifestos_m5_4": pd.DataFrame(),
            "df_itens_premanifestados_m5_4": pd.DataFrame(),
            "df_remanescente_m5_4": pd.DataFrame(),
            "df_subregioes_processadas_m5_4": pd.DataFrame(),
            "df_tentativas_perfis_m5_4": pd.DataFrame(),
        }
        meta_vazio = {
            "resumo_m5_4": {
                "modulo": "M5.4",
                "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
                "tipo_roteirizacao": tipo_roteirizacao,
                "linhas_entrada_m5_4": 0,
                "subregioes_processadas_m5_4": 0,
                "pre_manifestos_gerados_m5_4": 0,
                "itens_pre_manifestados_m5_4": 0,
                "remanescente_saida_m5_4": 0,
                "estrategia_m5_4": [
                    "subregiao_por_subregiao",
                    "ancora_por_cliente",
                    "mesma_cidade_primeiro",
                    "expansao_para_outras_cidades_da_subregiao",
                    "maior_perfil_para_menor",
                    "perfil_e_otimizacao_nao_trava",
                    "VERSAO_M5_4_2026_04_11",
                ],
                "caminhos_pipeline": caminhos_pipeline or {},
            }
        }
        return outputs_vazio, meta_vazio

    saldo = _precalcular_ordenacao(saldo)
    saldo = _ordenar_operacional(saldo)
    perfis_maior_menor = _veiculos_maior_para_menor(perfis)

    manifestos_rows: List[Dict[str, Any]] = []
    itens_manifestados_list: List[pd.DataFrame] = []
    tentativas_rows: List[Dict[str, Any]] = []
    subregioes_rows: List[Dict[str, Any]] = []

    manifesto_seq = 1
    remanescente_global_list: List[pd.DataFrame] = []

    sub_keys = (
        saldo[["_subregiao_key_m5_4", "_uf_key_m5_4"]]
        .drop_duplicates()
        .sort_values(["_subregiao_key_m5_4", "_uf_key_m5_4"], kind="mergesort")
        .values.tolist()
    )

    for subregiao_key, uf_key in sub_keys:
        df_sub = saldo[
            (saldo["_subregiao_key_m5_4"] == subregiao_key)
            & (saldo["_uf_key_m5_4"] == uf_key)
        ].copy()

        if df_sub.empty:
            continue

        df_sub = _ordenar_operacional(df_sub)
        blocos_restantes = _montar_blocos_cliente(df_sub)
        processados_sub = 0
        manifestos_sub = 0

        for _, vehicle_row in perfis_maior_menor.iterrows():
            if blocos_restantes.empty:
                break

            houve_fechamento_no_perfil = True

            while houve_fechamento_no_perfil and not blocos_restantes.empty:
                houve_fechamento_no_perfil = False

                manifesto_df, novos_blocos_restantes, info_tentativa = _tentar_montar_manifesto_subregiao(
                    df_subregiao_restante=df_sub,
                    blocos_restantes=blocos_restantes,
                    vehicle_row=vehicle_row,
                )

                tentativas_rows.append(
                    {
                        "subregiao": subregiao_key,
                        "uf": uf_key,
                        "perfil": _safe_text(vehicle_row.get("perfil")),
                        "tipo": _safe_text(vehicle_row.get("tipo")),
                        "status_tentativa": _safe_text(info_tentativa.get("status_tentativa")),
                        "motivo_tentativa": _safe_text(info_tentativa.get("motivo_tentativa")),
                        "bloco_anchor": _safe_text(info_tentativa.get("bloco_anchor")),
                        "blocos_restantes_antes": int(len(blocos_restantes)),
                        "blocos_restantes_depois": int(len(novos_blocos_restantes)),
                    }
                )

                processados_sub += 1

                if manifesto_df.empty:
                    break

                manifesto_id = f"PM54_{manifesto_seq:04d}"
                manifesto_seq += 1

                manifesto_row = _registro_manifesto(
                    manifesto_id=manifesto_id,
                    subregiao_key=subregiao_key,
                    uf_key=uf_key,
                    vehicle_row=vehicle_row,
                    df_itens=manifesto_df,
                    origem_etapa="5_4_subregiao_multicidade",
                )
                manifestos_rows.append(manifesto_row)

                manifesto_df = manifesto_df.copy()
                manifesto_df["manifesto_id"] = manifesto_id
                manifesto_df["tipo_manifesto"] = "composto_bloco_5_4_subregiao"
                manifesto_df["origem_modulo"] = 5.4
                manifesto_df["origem_etapa"] = "5_4_subregiao_multicidade"
                manifesto_df["veiculo_tipo"] = _safe_text(vehicle_row.get("tipo")) or _safe_text(vehicle_row.get("perfil"))
                manifesto_df["perfil"] = _safe_text(vehicle_row.get("perfil"))
                manifesto_df["capacidade_peso_kg_veiculo"] = round(
                    _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0), 3
                )
                manifesto_df["capacidade_vol_m3_veiculo"] = round(
                    _safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0), 3
                )
                manifesto_df["max_entregas_veiculo"] = _safe_int(vehicle_row.get("max_entregas"), 0)
                manifesto_df["max_km_distancia_veiculo"] = round(
                    _safe_float(vehicle_row.get("max_km_distancia"), 0.0), 3
                )
                itens_manifestados_list.append(manifesto_df)

                used_ids = set(manifesto_df["id_linha_pipeline"].astype(str).tolist())
                df_sub = df_sub[~df_sub["id_linha_pipeline"].astype(str).isin(used_ids)].copy()
                df_sub = _ordenar_operacional(df_sub)

                blocos_restantes = _montar_blocos_cliente(df_sub)
                houve_fechamento_no_perfil = True
                manifestos_sub += 1

        if not df_sub.empty:
            df_sub["status_m5_4"] = "remanescente_m5_4"
            df_sub["motivo_m5_4"] = "nao_composto_na_subregiao"
            remanescente_global_list.append(df_sub)

        subregioes_rows.append(
            {
                "subregiao": subregiao_key,
                "uf": uf_key,
                "linhas_entrada_subregiao": int(
                    len(
                        saldo[
                            (saldo["_subregiao_key_m5_4"] == subregiao_key)
                            & (saldo["_uf_key_m5_4"] == uf_key)
                        ]
                    )
                ),
                "manifestos_gerados_subregiao": int(manifestos_sub),
                "linhas_remanescentes_subregiao": int(len(df_sub)),
                "tentativas_processadas_subregiao": int(processados_sub),
            }
        )

    df_premanifestos_m5_4 = pd.DataFrame(manifestos_rows)
    df_itens_premanifestados_m5_4 = (
        pd.concat(itens_manifestados_list, ignore_index=True) if itens_manifestados_list else pd.DataFrame()
    )
    df_remanescente_m5_4 = (
        pd.concat(remanescente_global_list, ignore_index=True) if remanescente_global_list else pd.DataFrame()
    )
    df_subregioes_processadas_m5_4 = pd.DataFrame(subregioes_rows)
    df_tentativas_perfis_m5_4 = pd.DataFrame(tentativas_rows)

    if not df_itens_premanifestados_m5_4.empty:
        df_itens_premanifestados_m5_4 = _limpar_cols_auxiliares(df_itens_premanifestados_m5_4)

    if not df_remanescente_m5_4.empty:
        df_remanescente_m5_4 = _limpar_cols_auxiliares(df_remanescente_m5_4)

    resumo_m5_4 = {
        "modulo": "M5.4",
        "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
        "tipo_roteirizacao": tipo_roteirizacao,
        "linhas_entrada_m5_4": int(len(saldo)),
        "subregioes_processadas_m5_4": int(len(df_subregioes_processadas_m5_4)),
        "pre_manifestos_gerados_m5_4": int(len(df_premanifestos_m5_4)),
        "itens_pre_manifestados_m5_4": int(len(df_itens_premanifestados_m5_4)),
        "remanescente_saida_m5_4": int(len(df_remanescente_m5_4)),
        "estrategia_m5_4": [
            "subregiao_por_subregiao",
            "ancora_por_cliente",
            "mesma_cidade_primeiro",
            "expansao_para_outras_cidades_da_subregiao",
            "maior_perfil_para_menor",
            "perfil_e_otimizacao_nao_trava",
            "VERSAO_M5_4_2026_04_11",
        ],
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    meta_m5_4 = {
        "resumo_m5_4": resumo_m5_4,
    }

    outputs_m5_4 = {
        "df_premanifestos_m5_4": df_premanifestos_m5_4,
        "df_itens_premanifestados_m5_4": df_itens_premanifestados_m5_4,
        "df_remanescente_m5_4": df_remanescente_m5_4,
        "df_subregioes_processadas_m5_4": df_subregioes_processadas_m5_4,
        "df_tentativas_perfis_m5_4": df_tentativas_perfis_m5_4,
    }

    return outputs_m5_4, meta_m5_4


# Aliases defensivos
def executar_m5_composicao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4_composicao_subregioes(*args, **kwargs)


def processar_m5_4_composicao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4_composicao_subregioes(*args, **kwargs)


def rodar_m5_4_composicao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4_composicao_subregioes(*args, **kwargs)
