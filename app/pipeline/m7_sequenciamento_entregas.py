from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd


TIME_LIMIT_SECONDS_PADRAO = 5
FATOR_KM_RODOVIARIO_M7_PADRAO = 1.20


# =========================================================================================
# HELPERS
# =========================================================================================
def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    txt = str(value).strip().lower()
    return txt in {"1", "true", "sim", "s", "yes", "y", "verdadeiro"}


def _resolver_coluna_existente(
    df: pd.DataFrame,
    candidatos: List[str],
    nome_logico: str,
    obrigatoria: bool = True,
) -> str:
    for c in candidatos:
        if c in df.columns:
            return c
    if obrigatoria:
        raise Exception(
            f"M7 não encontrou a coluna obrigatória '{nome_logico}'. "
            f"Esperado um destes nomes: {candidatos}."
        )
    return ""


def _garantir_colunas(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in colunas:
        if col not in out.columns:
            out[col] = None
    return out


def _validar_colunas(df: pd.DataFrame, obrigatorias: List[str], nome_df: str) -> None:
    faltando = [c for c in obrigatorias if c not in df.columns]
    if faltando:
        raise Exception(f"M7 encontrou colunas obrigatórias ausentes em {nome_df}: {faltando}")


# =========================================================================================
# DISTÂNCIA
# =========================================================================================
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return 999999.0

    r = 6371.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _distancia_operacional_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    fator_km_rodoviario: float,
) -> float:
    dist_hav = _haversine_km(lat1, lon1, lat2, lon2)
    if pd.isna(dist_hav):
        return 999999.0

    fator = _safe_float(fator_km_rodoviario, FATOR_KM_RODOVIARIO_M7_PADRAO)
    if fator <= 0:
        fator = FATOR_KM_RODOVIARIO_M7_PADRAO

    return float(dist_hav) * fator


def _inferir_fator_rodoviario_real_manifesto(
    df_manifesto: pd.DataFrame,
    fallback: float,
) -> float:
    if "distancia_rodoviaria_est_km" not in df_manifesto.columns:
        return float(fallback)

    ratios: List[float] = []

    for _, row in df_manifesto.iterrows():
        lat_o = pd.to_numeric(row.get("latitude_filial_m7"), errors="coerce")
        lon_o = pd.to_numeric(row.get("longitude_filial_m7"), errors="coerce")
        lat_d = pd.to_numeric(row.get("latitude_dest_m7"), errors="coerce")
        lon_d = pd.to_numeric(row.get("longitude_dest_m7"), errors="coerce")
        dist_est = pd.to_numeric(row.get("distancia_rodoviaria_est_km"), errors="coerce")

        dist_hav = _haversine_km(lat_o, lon_o, lat_d, lon_d)
        if pd.isna(dist_est) or pd.isna(dist_hav) or dist_est <= 0 or dist_hav <= 0:
            continue

        ratio = float(dist_est) / float(dist_hav)
        if 0.8 <= ratio <= 3.0:
            ratios.append(ratio)

    if not ratios:
        return float(fallback)

    return float(np.median(ratios))


# =========================================================================================
# PRIORIDADE
# =========================================================================================
def _classificar_prioridade_negocio(row: pd.Series) -> Tuple[int, float, float]:
    agendada = bool(row.get("agendada_norm", False))
    folga = row.get("folga_dias_norm", np.nan)
    peso = row.get("peso_seq_m7", 0.0)

    if pd.isna(folga):
        folga = 9999.0
    if pd.isna(peso):
        peso = 0.0

    if agendada:
        if folga <= 0:
            bucket = 0
        elif folga <= 1:
            bucket = 1
        else:
            bucket = 2
    else:
        if folga <= 0:
            bucket = 3
        elif folga <= 1:
            bucket = 4
        else:
            bucket = 5

    return (bucket, float(folga), -float(peso))


def _calcular_score_parada(df_parada: pd.DataFrame) -> Dict[str, Any]:
    buckets: List[int] = []
    folgas: List[float] = []
    pesos: List[float] = []

    for _, row in df_parada.iterrows():
        b, f, pneg = _classificar_prioridade_negocio(row)
        buckets.append(b)
        folgas.append(f)
        pesos.append(-pneg)

    return {
        "bucket_prioridade": min(buckets) if buckets else 9,
        "folga_min": min(folgas) if folgas else 9999.0,
        "peso_total": sum(pesos) if pesos else 0.0,
    }


def _montar_justificativa_doc(row: pd.Series) -> str:
    bucket, folga, _ = _classificar_prioridade_negocio(row)

    if bucket == 0:
        prioridade_txt = "Agendada com folga vencida/zero"
    elif bucket == 1:
        prioridade_txt = "Agendada com folga de 1 dia"
    elif bucket == 2:
        prioridade_txt = "Agendada com folga acima de 1 dia"
    elif bucket == 3:
        prioridade_txt = "Não agendada urgente"
    elif bucket == 4:
        prioridade_txt = "Não agendada com folga de 1 dia"
    else:
        prioridade_txt = "Não agendada normal"

    return (
        f"{prioridade_txt}; "
        f"folga={folga if not pd.isna(folga) else 'NA'}; "
        f"peso={_safe_float(row.get('peso_seq_m7', 0.0), 0.0):.2f}kg"
    )


def _ordenar_docs_por_prioridade(df_docs: pd.DataFrame, col_doc: str) -> pd.DataFrame:
    dfp = df_docs.copy()

    prioridades = dfp.apply(_classificar_prioridade_negocio, axis=1)
    dfp["bucket_prioridade_doc_m7"] = [x[0] for x in prioridades]
    dfp["folga_prioridade_doc_m7"] = [x[1] for x in prioridades]
    dfp["peso_prioridade_doc_m7"] = [(-x[2]) for x in prioridades]

    dfp = dfp.sort_values(
        by=[
            "bucket_prioridade_doc_m7",
            "folga_prioridade_doc_m7",
            "peso_prioridade_doc_m7",
            col_doc,
        ],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return dfp


# =========================================================================================
# NORMALIZAÇÃO
# =========================================================================================
def _normalizar_manifestos(df_manifestos_m6_2: pd.DataFrame) -> pd.DataFrame:
    out = df_manifestos_m6_2.copy()
    _validar_colunas(out, ["manifesto_id"], "df_manifestos_m6_2")
    out["manifesto_id"] = out["manifesto_id"].astype(str).str.strip()
    out = out[out["manifesto_id"] != ""].copy()
    return out.reset_index(drop=True)


def _normalizar_itens(df_itens_m6_2: pd.DataFrame) -> pd.DataFrame:
    out = df_itens_m6_2.copy()

    colunas_minimas = [
        "manifesto_id",
        "id_linha_pipeline",
        "nro_documento",
        "destinatario",
        "cidade",
        "uf",
        "peso_kg",
        "peso_calculado",
        "agendada",
        "folga_dias",
        "distancia_rodoviaria_est_km",
    ]
    out = _garantir_colunas(out, colunas_minimas)

    _validar_colunas(
        out,
        ["manifesto_id", "id_linha_pipeline", "destinatario", "cidade", "uf"],
        "df_itens_manifestos_m6_2",
    )

    col_lat_filial = _resolver_coluna_existente(
        out,
        ["latitude_filial", "origem_latitude"],
        "latitude_filial",
        obrigatoria=False,
    )
    if col_lat_filial == "":
        out["latitude_filial"] = np.nan
        col_lat_filial = "latitude_filial"

    col_lon_filial = _resolver_coluna_existente(
        out,
        ["longitude_filial", "origem_longitude"],
        "longitude_filial",
        obrigatoria=False,
    )
    if col_lon_filial == "":
        out["longitude_filial"] = np.nan
        col_lon_filial = "longitude_filial"

    col_lat_dest = _resolver_coluna_existente(
        out,
        ["latitude_destinatario", "latitude_destino", "latitude"],
        "latitude_destinatario",
        obrigatoria=False,
    )
    if col_lat_dest == "":
        out["latitude_destinatario"] = np.nan
        col_lat_dest = "latitude_destinatario"

    col_lon_dest = _resolver_coluna_existente(
        out,
        ["longitude_destinatario", "longitude_destino", "longitude"],
        "longitude_destinatario",
        obrigatoria=False,
    )
    if col_lon_dest == "":
        out["longitude_destinatario"] = np.nan
        col_lon_dest = "longitude_destinatario"

    out["manifesto_id"] = out["manifesto_id"].fillna("").astype(str).str.strip()
    out["id_linha_pipeline"] = out["id_linha_pipeline"].fillna("").astype(str).str.strip()
    out["nro_documento"] = out["nro_documento"].fillna("").astype(str).str.strip()
    out["destinatario"] = out["destinatario"].fillna("").astype(str).str.strip()
    out["cidade"] = out["cidade"].fillna("").astype(str).str.strip()
    out["uf"] = out["uf"].fillna("").astype(str).str.strip()

    for c in [
        "peso_kg",
        "peso_calculado",
        "folga_dias",
        "distancia_rodoviaria_est_km",
        col_lat_filial,
        col_lon_filial,
        col_lat_dest,
        col_lon_dest,
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["agendada_norm"] = out["agendada"].apply(_to_bool)
    out["folga_dias_norm"] = pd.to_numeric(out["folga_dias"], errors="coerce")

    out["peso_seq_m7"] = pd.to_numeric(out["peso_calculado"], errors="coerce").fillna(
        pd.to_numeric(out["peso_kg"], errors="coerce")
    )

    out["latitude_filial_m7"] = out[col_lat_filial]
    out["longitude_filial_m7"] = out[col_lon_filial]
    out["latitude_dest_m7"] = out[col_lat_dest]
    out["longitude_dest_m7"] = out[col_lon_dest]

    out = out[(out["manifesto_id"] != "") & (out["id_linha_pipeline"] != "")].copy()

    if out["id_linha_pipeline"].duplicated().any():
        duplicados = out.loc[
            out["id_linha_pipeline"].duplicated(), "id_linha_pipeline"
        ].astype(str).tolist()[:20]
        raise Exception(
            f"M7 recebeu id_linha_pipeline duplicado em df_itens_manifestos_m6_2: {duplicados}"
        )

    return out.reset_index(drop=True)


# =========================================================================================
# PREPARAÇÃO GEO
# =========================================================================================
def _preparar_coordenadas_contrato(
    df_itens: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df_itens.copy()

    out["status_coord_filial_m7"] = np.where(
        out["latitude_filial_m7"].notna() & out["longitude_filial_m7"].notna(),
        "ok",
        "sem_coordenada_filial",
    )

    out["status_coord_dest_m7"] = np.where(
        out["latitude_dest_m7"].notna() & out["longitude_dest_m7"].notna(),
        "ok",
        "sem_coordenada_destino",
    )

    out["coord_dest_origem_m7"] = np.where(
        out["latitude_dest_m7"].notna() & out["longitude_dest_m7"].notna(),
        "contrato_carteira",
        "ausente_no_contrato_recebido",
    )

    diagnostico = pd.DataFrame(
        [
            {"indicador": "linhas_filial_ok", "valor": int((out["status_coord_filial_m7"] == "ok").sum())},
            {"indicador": "linhas_filial_nula", "valor": int((out["status_coord_filial_m7"] != "ok").sum())},
            {"indicador": "linhas_destino_ok", "valor": int((out["status_coord_dest_m7"] == "ok").sum())},
            {"indicador": "linhas_destino_nula", "valor": int((out["status_coord_dest_m7"] != "ok").sum())},
        ]
    )

    return out.reset_index(drop=True), diagnostico.reset_index(drop=True)


# =========================================================================================
# GEOMETRIA / EIXO
# =========================================================================================
def _geo_para_xy_km(
    lat_base: float,
    lon_base: float,
    lat: float,
    lon: float,
) -> Tuple[float, float]:
    lat_base_rad = math.radians(float(lat_base))
    x = (float(lon) - float(lon_base)) * 111.320 * math.cos(lat_base_rad)
    y = (float(lat) - float(lat_base)) * 110.574
    return float(x), float(y)


def _norma_xy(x: float, y: float) -> float:
    return float(math.sqrt((x * x) + (y * y)))


def _projecao_no_eixo(
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    lat_p: float,
    lon_p: float,
) -> float:
    ax, ay = _geo_para_xy_km(lat_a, lon_a, lat_b, lon_b)
    px, py = _geo_para_xy_km(lat_a, lon_a, lat_p, lon_p)

    norma_a = _norma_xy(ax, ay)
    if norma_a <= 1e-9:
        return 0.0

    ux = ax / norma_a
    uy = ay / norma_a

    return float((px * ux) + (py * uy))


# =========================================================================================
# AGRUPAMENTOS
# =========================================================================================
def _agrupar_paradas(grupo: pd.DataFrame, fator_km_rodoviario_m7: float) -> pd.DataFrame:
    registros: List[Dict[str, Any]] = []

    lat_o = pd.to_numeric(grupo["latitude_filial_m7"], errors="coerce").dropna()
    lon_o = pd.to_numeric(grupo["longitude_filial_m7"], errors="coerce").dropna()
    if len(lat_o) == 0 or len(lon_o) == 0:
        raise Exception(
            f"Manifesto {grupo['manifesto_id'].iloc[0]} sem coordenada de filial no contrato."
        )

    origem = (float(lat_o.iloc[0]), float(lon_o.iloc[0]))

    grupo["chave_parada_seq_m7"] = (
        grupo["destinatario"].fillna("").astype(str).str.strip()
        + "|"
        + grupo["cidade"].fillna("").astype(str).str.strip()
        + "|"
        + grupo["uf"].fillna("").astype(str).str.strip()
    )

    for chave_parada, gpar in grupo.groupby("chave_parada_seq_m7", dropna=False):
        score = _calcular_score_parada(gpar)
        lat_ref = pd.to_numeric(gpar["latitude_dest_m7"], errors="coerce").mean()
        lon_ref = pd.to_numeric(gpar["longitude_dest_m7"], errors="coerce").mean()

        if pd.isna(lat_ref) or pd.isna(lon_ref):
            raise Exception(
                f"Manifesto {grupo['manifesto_id'].iloc[0]} possui parada sem coordenada de destino."
            )

        dist_origem = _distancia_operacional_km(
            origem[0], origem[1], float(lat_ref), float(lon_ref), fator_km_rodoviario_m7
        )

        registros.append(
            {
                "chave_parada_seq_m7": chave_parada,
                "destinatario_ref_m7": _safe_text(gpar["destinatario"].iloc[0]),
                "cidade_ref_m7": _safe_text(gpar["cidade"].iloc[0]),
                "uf_ref_m7": _safe_text(gpar["uf"].iloc[0]),
                "lat_ref_m7": float(lat_ref),
                "lon_ref_m7": float(lon_ref),
                "bucket_prioridade_m7": score["bucket_prioridade"],
                "folga_min_m7": score["folga_min"],
                "peso_total_m7": score["peso_total"],
                "qtd_docs_parada_m7": int(len(gpar)),
                "distancia_origem_parada_km_m7": float(dist_origem),
            }
        )

    return pd.DataFrame(registros).reset_index(drop=True)


def _agrupar_cidades(grupo: pd.DataFrame, fator_km_rodoviario_m7: float) -> pd.DataFrame:
    registros: List[Dict[str, Any]] = []

    lat_o = pd.to_numeric(grupo["latitude_filial_m7"], errors="coerce").dropna()
    lon_o = pd.to_numeric(grupo["longitude_filial_m7"], errors="coerce").dropna()
    if len(lat_o) == 0 or len(lon_o) == 0:
        raise Exception(
            f"Manifesto {grupo['manifesto_id'].iloc[0]} sem coordenada de filial no contrato."
        )

    origem = (float(lat_o.iloc[0]), float(lon_o.iloc[0]))

    grupo["chave_cidade_seq_m7"] = (
        grupo["cidade"].fillna("").astype(str).str.strip()
        + "|"
        + grupo["uf"].fillna("").astype(str).str.strip()
    )

    for chave_cidade, gcid in grupo.groupby("chave_cidade_seq_m7", dropna=False):
        lat_ref = pd.to_numeric(gcid["latitude_dest_m7"], errors="coerce").mean()
        lon_ref = pd.to_numeric(gcid["longitude_dest_m7"], errors="coerce").mean()

        if pd.isna(lat_ref) or pd.isna(lon_ref):
            raise Exception(
                f"Manifesto {grupo['manifesto_id'].iloc[0]} possui cidade sem coordenada válida."
            )

        dist_origem = _distancia_operacional_km(
            origem[0], origem[1], float(lat_ref), float(lon_ref), fator_km_rodoviario_m7
        )

        pesos = pd.to_numeric(gcid["peso_seq_m7"], errors="coerce").fillna(0.0)
        buckets = gcid.apply(lambda r: _classificar_prioridade_negocio(r)[0], axis=1).tolist()
        folgas = gcid.apply(lambda r: _classificar_prioridade_negocio(r)[1], axis=1).tolist()

        registros.append(
            {
                "chave_cidade_seq_m7": chave_cidade,
                "cidade_ref_m7": _safe_text(gcid["cidade"].iloc[0]),
                "uf_ref_m7": _safe_text(gcid["uf"].iloc[0]),
                "lat_ref_cidade_m7": float(lat_ref),
                "lon_ref_cidade_m7": float(lon_ref),
                "qtd_docs_cidade_m7": int(len(gcid)),
                "qtd_paradas_cidade_m7": int(gcid["chave_parada_seq_m7"].nunique()),
                "peso_total_cidade_m7": float(pesos.sum()),
                "bucket_prioridade_cidade_m7": min(buckets) if buckets else 9,
                "folga_min_cidade_m7": min(folgas) if folgas else 9999.0,
                "distancia_origem_cidade_km_m7": float(dist_origem),
            }
        )

    return pd.DataFrame(registros).reset_index(drop=True)


# =========================================================================================
# APOIO PARA CIDADE DA ORIGEM
# =========================================================================================
def _detectar_chave_cidade_origem(
    df_cidades: pd.DataFrame,
    origem_lat: float,
    origem_lon: float,
    tolerancia_km: float = 0.001,
) -> str:
    if df_cidades.empty:
        return ""

    candidatos: List[str] = []

    for _, row in df_cidades.iterrows():
        dist = _distancia_operacional_km(
            origem_lat,
            origem_lon,
            float(row["lat_ref_cidade_m7"]),
            float(row["lon_ref_cidade_m7"]),
            1.0,
        )
        if dist <= tolerancia_km:
            candidatos.append(str(row["chave_cidade_seq_m7"]))

    if len(candidatos) == 1:
        return candidatos[0]

    if len(candidatos) > 1:
        return candidatos[0]

    return ""


def _montar_ordem_com_cidade_origem_especial(
    ordem_base: List[str],
    chave_cidade_origem: str,
    posicao: str,
) -> List[str]:
    ordem_limpa = [x for x in ordem_base if x != chave_cidade_origem]
    if not chave_cidade_origem:
        return ordem_limpa

    if posicao == "fim":
        return ordem_limpa + [chave_cidade_origem]

    return [chave_cidade_origem] + ordem_limpa


# =========================================================================================
# SEQUÊNCIA DE CIDADES POR VARREDURA ENTRE EXTREMOS
# =========================================================================================
def _calcular_km_ordem_cidades(
    df_cidades: pd.DataFrame,
    ordem_chaves: List[str],
    origem_lat: float,
    origem_lon: float,
    fator_km_rodoviario_m7: float,
) -> Tuple[List[Dict[str, Any]], float]:
    idx_por_chave = {
        str(row["chave_cidade_seq_m7"]): i for i, row in df_cidades.iterrows()
    }

    trilha: List[Dict[str, Any]] = []
    km_total = 0.0

    atual_lat = float(origem_lat)
    atual_lon = float(origem_lon)
    origem_label = "ORIGEM"

    for pos, chave in enumerate(ordem_chaves, start=1):
        row = df_cidades.iloc[idx_por_chave[chave]]
        dist = _distancia_operacional_km(
            atual_lat,
            atual_lon,
            float(row["lat_ref_cidade_m7"]),
            float(row["lon_ref_cidade_m7"]),
            fator_km_rodoviario_m7,
        )
        km_total += float(dist)

        trilha.append(
            {
                "ordem_cidade_m7": int(pos),
                "chave_cidade_seq_m7": chave,
                "cidade_ref_m7": _safe_text(row["cidade_ref_m7"]),
                "uf_ref_m7": _safe_text(row["uf_ref_m7"]),
                "origem_anterior_cidade_m7": origem_label,
                "distancia_no_anterior_km_m7": float(dist),
            }
        )

        atual_lat = float(row["lat_ref_cidade_m7"])
        atual_lon = float(row["lon_ref_cidade_m7"])
        origem_label = chave

    return trilha, float(km_total)


def _sequenciar_cidades(
    df_cidades: pd.DataFrame,
    origem_lat: float,
    origem_lon: float,
    fator_km_rodoviario
