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


def _ordenar_docs_dentro_parada(df_parada: pd.DataFrame, col_doc: str) -> pd.DataFrame:
    dfp = df_parada.copy()

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

    dfp["justificativa_ordem_entrega_m7"] = dfp.apply(_montar_justificativa_doc, axis=1)

    return dfp.drop(
        columns=[
            "bucket_prioridade_doc_m7",
            "folga_prioridade_doc_m7",
            "peso_prioridade_doc_m7",
        ],
        errors="ignore",
    )


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
        duplicados = out.loc[out["id_linha_pipeline"].duplicated(), "id_linha_pipeline"].astype(str).tolist()[:20]
        raise Exception(
            f"M7 recebeu id_linha_pipeline duplicado em df_itens_manifestos_m6_2: {duplicados}"
        )

    return out.reset_index(drop=True)


# =========================================================================================
# PREPARAÇÃO GEO
# =========================================================================================
def _preparar_coordenadas_contrato(df_itens: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
# AGREGAÇÃO DE PARADAS
# =========================================================================================
def _agrupar_paradas(
    grupo: pd.DataFrame,
    fator_km_rodoviario_m7: float,
) -> pd.DataFrame:
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


# =========================================================================================
# GEOMETRIA OPERACIONAL PARA EXTREMOS DINÂMICOS
# =========================================================================================
def _geo_para_xy_km(
    lat_base: float,
    lon_base: float,
    lat: float,
    lon: float,
) -> Tuple[float, float]:
    """
    Converte lat/lon para coordenadas XY aproximadas em km usando projeção equiretangular
    local em torno do ponto base.
    """
    lat_base_rad = math.radians(float(lat_base))
    x = (float(lon) - float(lon_base)) * 111.320 * math.cos(lat_base_rad)
    y = (float(lat) - float(lat_base)) * 110.574
    return float(x), float(y)


def _norma_xy(x: float, y: float) -> float:
    return float(math.sqrt((x * x) + (y * y)))


def _metricas_candidato_extremos(
    origem_lat: float,
    origem_lon: float,
    cand_lat: float,
    cand_lon: float,
    extremo_lat: float,
    extremo_lon: float,
    fator_km_rodoviario_m7: float,
) -> Dict[str, float]:
    """
    Calcula métricas operacionais do candidato em relação à origem atual e ao extremo mais longe.
    """
    dist_atual = _distancia_operacional_km(
        origem_lat,
        origem_lon,
        cand_lat,
        cand_lon,
        fator_km_rodoviario_m7,
    )

    dist_ate_extremo = _distancia_operacional_km(
        origem_lat,
        origem_lon,
        extremo_lat,
        extremo_lon,
        fator_km_rodoviario_m7,
    )

    dist_cand_extremo = _distancia_operacional_km(
        cand_lat,
        cand_lon,
        extremo_lat,
        extremo_lon,
        fator_km_rodoviario_m7,
    )

    ox_e, oy_e = _geo_para_xy_km(origem_lat, origem_lon, extremo_lat, extremo_lon)
    ox_c, oy_c = _geo_para_xy_km(origem_lat, origem_lon, cand_lat, cand_lon)

    norma_e = _norma_xy(ox_e, oy_e)
    norma_c = _norma_xy(ox_c, oy_c)

    if norma_e <= 1e-9:
        progresso = 0.0
        lateral = 0.0
        alinhamento = 0.0
    else:
        ux = ox_e / norma_e
        uy = oy_e / norma_e

        progresso = (ox_c * ux) + (oy_c * uy)
        lateral_sq = max(0.0, (norma_c * norma_c) - (progresso * progresso))
        lateral = math.sqrt(lateral_sq)

        if norma_c <= 1e-9:
            alinhamento = 0.0
        else:
            alinhamento = max(-1.0, min(1.0, progresso / norma_c))

    # Detour: quanto o candidato se afasta do "corredor" origem -> extremo
    detour = max(0.0, (dist_atual + dist_cand_extremo) - dist_ate_extremo)

    # Regressão: candidato "anda para trás" em relação ao eixo
    regressao = abs(min(0.0, progresso))

    return {
        "dist_atual_km": float(dist_atual),
        "dist_origem_extremo_km": float(dist_ate_extremo),
        "dist_candidato_extremo_km": float(dist_cand_extremo),
        "progresso_km": float(max(0.0, progresso)),
        "lateral_km": float(lateral),
        "detour_km": float(detour),
        "regressao_km": float(regressao),
        "alinhamento": float(alinhamento),
    }


def _score_operacional_candidato(
    metricas: Dict[str, float],
    bucket_prioridade: int,
    folga_min: float,
    peso_total: float,
    eh_mais_perto: bool,
    eh_mais_longe: bool,
) -> Tuple[Any, ...]:
    """
    Score determinístico:
    - prioriza proximidade operacional
    - pune desvio lateral
    - pune detour
    - pune regressão
    - dá leve preferência ao mais perto
    - dá leve proteção ao extremo mais longe quando ele é um bom fechamento territorial
    """
    dist_atual = _safe_float(metricas.get("dist_atual_km"), 999999.0)
    lateral = _safe_float(metricas.get("lateral_km"), 999999.0)
    detour = _safe_float(metricas.get("detour_km"), 999999.0)
    regressao = _safe_float(metricas.get("regressao_km"), 999999.0)
    alinhamento = _safe_float(metricas.get("alinhamento"), 0.0)
    dist_cand_extremo = _safe_float(metricas.get("dist_candidato_extremo_km"), 999999.0)

    bonus_mais_perto = -1.20 if eh_mais_perto else 0.0
    bonus_extremo = -0.60 if eh_mais_longe else 0.0

    score_num = (
        dist_atual
        + (0.55 * lateral)
        + (0.75 * detour)
        + (1.10 * regressao)
        + (0.10 * dist_cand_extremo)
        - (1.50 * max(0.0, alinhamento))
        + bonus_mais_perto
        + bonus_extremo
    )

    return (
        round(score_num, 6),
        _safe_int(bucket_prioridade, 9),
        round(_safe_float(folga_min, 9999.0), 6),
        -round(_safe_float(peso_total, 0.0), 6),
    )


# =========================================================================================
# SEQUENCIAMENTO POR EXTREMOS DINÂMICOS
# =========================================================================================
def _resolver_ordem_extremos_dinamicos(
    df_paradas: pd.DataFrame,
    origem_lat: float,
    origem_lon: float,
    fator_km_rodoviario_m7: float,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], float]:
    if df_paradas.empty:
        return df_paradas.copy(), [], 0.0

    work = df_paradas.copy().reset_index(drop=True)

    chaves_restantes: List[str] = work["chave_parada_seq_m7"].astype(str).tolist()
    idx_por_chave: Dict[str, int] = {
        str(row["chave_parada_seq_m7"]): i for i, row in work.iterrows()
    }

    ordem_chaves: List[str] = []
    trilha_auditoria: List[Dict[str, Any]] = []

    origem_atual_lat = float(origem_lat)
    origem_atual_lon = float(origem_lon)
    chave_anterior = "ORIGEM"
    km_total = 0.0

    ordem = 1

    while chaves_restantes:
        candidatos_idx = [idx_por_chave[ch] for ch in chaves_restantes]
        df_restante = work.loc[candidatos_idx].copy().reset_index(drop=True)

        # Distância da origem atual para todos os restantes
        df_restante["dist_atual_km_tmp"] = df_restante.apply(
            lambda r: _distancia_operacional_km(
                origem_atual_lat,
                origem_atual_lon,
                float(r["lat_ref_m7"]),
                float(r["lon_ref_m7"]),
                fator_km_rodoviario_m7,
            ),
            axis=1,
        )

        df_restante = df_restante.sort_values(
            by=["dist_atual_km_tmp", "bucket_prioridade_m7", "folga_min_m7", "chave_parada_seq_m7"],
            ascending=[True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        idx_mais_perto_local = 0
        chave_mais_perto = str(df_restante.iloc[idx_mais_perto_local]["chave_parada_seq_m7"])

        idx_mais_longe_local = int(df_restante["dist_atual_km_tmp"].astype(float).idxmax())
        chave_mais_longe = str(df_restante.iloc[idx_mais_longe_local]["chave_parada_seq_m7"])

        # Candidatos avaliados:
        # - top N mais próximos
        # - sempre inclui o mais longe
        # - determinístico
        top_n = min(5, len(df_restante))
        chaves_candidatas = set(df_restante.head(top_n)["chave_parada_seq_m7"].astype(str).tolist())
        chaves_candidatas.add(chave_mais_longe)

        avaliacoes: List[Dict[str, Any]] = []

        row_extremo = df_restante.loc[df_restante["chave_parada_seq_m7"] == chave_mais_longe].iloc[0]

        for chave_cand in sorted(chaves_candidatas):
            row_cand = df_restante.loc[df_restante["chave_parada_seq_m7"] == chave_cand].iloc[0]

            metricas = _metricas_candidato_extremos(
                origem_lat=origem_atual_lat,
                origem_lon=origem_atual_lon,
                cand_lat=float(row_cand["lat_ref_m7"]),
                cand_lon=float(row_cand["lon_ref_m7"]),
                extremo_lat=float(row_extremo["lat_ref_m7"]),
                extremo_lon=float(row_extremo["lon_ref_m7"]),
                fator_km_rodoviario_m7=fator_km_rodoviario_m7,
            )

            score_ord = _score_operacional_candidato(
                metricas=metricas,
                bucket_prioridade=_safe_int(row_cand["bucket_prioridade_m7"], 9),
                folga_min=_safe_float(row_cand["folga_min_m7"], 9999.0),
                peso_total=_safe_float(row_cand["peso_total_m7"], 0.0),
                eh_mais_perto=(chave_cand == chave_mais_perto),
                eh_mais_longe=(chave_cand == chave_mais_longe),
            )

            avaliacoes.append(
                {
                    "chave_parada_seq_m7": chave_cand,
                    "score_ord": score_ord,
                    "score_num_m7": float(score_ord[0]),
                    "dist_atual_km_m7": float(metricas["dist_atual_km"]),
                    "dist_origem_extremo_km_m7": float(metricas["dist_origem_extremo_km"]),
                    "dist_candidato_extremo_km_m7": float(metricas["dist_candidato_extremo_km"]),
                    "progresso_km_m7": float(metricas["progresso_km"]),
                    "lateral_km_m7": float(metricas["lateral_km"]),
                    "detour_km_m7": float(metricas["detour_km"]),
                    "regressao_km_m7": float(metricas["regressao_km"]),
                    "alinhamento_m7": float(metricas["alinhamento"]),
                    "foi_mais_perto_m7": chave_cand == chave_mais_perto,
                    "foi_mais_longe_m7": chave_cand == chave_mais_longe,
                }
            )

        df_av = pd.DataFrame(avaliacoes).sort_values(
            by=["score_ord", "chave_parada_seq_m7"],
            ascending=[True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        escolhido = df_av.iloc[0]
        chave_escolhida = str(escolhido["chave_parada_seq_m7"])
        idx_escolhido = idx_por_chave[chave_escolhida]
        row_escolhida = work.iloc[idx_escolhido]

        ordem_chaves.append(chave_escolhida)
        km_trecho = float(escolhido["dist_atual_km_m7"])
        km_total += km_trecho

        criterio_escolha = (
            "extremos_dinamicos_origem_recalculada"
            if ordem == 1
            else "extremos_dinamicos_recalculados_na_nova_origem"
        )

        trilha_auditoria.append(
            {
                "ordem_entrega_parada_m7": int(ordem),
                "chave_parada_seq_m7": chave_escolhida,
                "chave_no_anterior_m7": chave_anterior,
                "distancia_no_anterior_km_m7": float(km_trecho),
                "score_total_projetado_km_m7": float(escolhido["score_num_m7"]),
                "criterio_escolha_m7": criterio_escolha,
                "chave_mais_perto_na_iteracao_m7": chave_mais_perto,
                "chave_mais_longe_na_iteracao_m7": chave_mais_longe,
                "dist_origem_extremo_km_m7": float(escolhido["dist_origem_extremo_km_m7"]),
                "dist_candidato_extremo_km_m7": float(escolhido["dist_candidato_extremo_km_m7"]),
                "progresso_km_m7": float(escolhido["progresso_km_m7"]),
                "lateral_km_m7": float(escolhido["lateral_km_m7"]),
                "detour_km_m7": float(escolhido["detour_km_m7"]),
                "regressao_km_m7": float(escolhido["regressao_km_m7"]),
                "alinhamento_m7": float(escolhido["alinhamento_m7"]),
                "foi_mais_perto_m7": bool(escolhido["foi_mais_perto_m7"]),
                "foi_mais_longe_m7": bool(escolhido["foi_mais_longe_m7"]),
            }
        )

        origem_atual_lat = float(row_escolhida["lat_ref_m7"])
        origem_atual_lon = float(row_escolhida["lon_ref_m7"])
        chave_anterior = chave_escolhida
        chaves_restantes.remove(chave_escolhida)
        ordem += 1

    # Grava ordem final e links entre paradas
    ordem_map = {ch: i + 1 for i, ch in enumerate(ordem_chaves)}
    proximo_map: Dict[str, str] = {}
    dist_proximo_map: Dict[str, float] = {}

    for pos, chave in enumerate(ordem_chaves):
        if pos == len(ordem_chaves) - 1:
            proximo_map[chave] = ""
            dist_proximo_map[chave] = np.nan
        else:
            chave_prox = ordem_chaves[pos + 1]
            row_atual = work.loc[work["chave_parada_seq_m7"] == chave].iloc[0]
            row_prox = work.loc[work["chave_parada_seq_m7"] == chave_prox].iloc[0]

            dist_prox = _distancia_operacional_km(
                float(row_atual["lat_ref_m7"]),
                float(row_atual["lon_ref_m7"]),
                float(row_prox["lat_ref_m7"]),
                float(row_prox["lon_ref_m7"]),
                fator_km_rodoviario_m7,
            )
            proximo_map[chave] = chave_prox
            dist_proximo_map[chave] = float(dist_prox)

    df_aud = pd.DataFrame(trilha_auditoria)

    work = work.merge(df_aud, on="chave_parada_seq_m7", how="left")
    work["ordem_entrega_parada_m7"] = pd.to_numeric(work["ordem_entrega_parada_m7"], errors="coerce").astype(int)
    work["metodo_sequenciamento_parada_m7"] = "extremos_dinamicos_origem_recalculada"
    work["chave_proximo_no_m7"] = work["chave_parada_seq_m7"].map(proximo_map)
    work["distancia_proximo_no_km_m7"] = work["chave_parada_seq_m7"].map(dist_proximo_map)

    work = work.sort_values(
        by=["ordem_entrega_parada_m7", "chave_parada_seq_m7"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return work, trilha_auditoria, float(km_total)


# =========================================================================================
# SEQUENCIAMENTO DE UM MANIFESTO
# =========================================================================================
def _sequenciar_manifesto(
    df_manifesto: pd.DataFrame,
    col_doc: str,
    fator_km_rodoviario_m7: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    grupo = df_manifesto.copy().reset_index(drop=True)

    lat_origem = pd.to_numeric(grupo["latitude_filial_m7"], errors="coerce").dropna()
    lon_origem = pd.to_numeric(grupo["longitude_filial_m7"], errors="coerce").dropna()

    if len(lat_origem) == 0 or len(lon_origem) == 0:
        raise Exception(
            f"Manifesto {grupo['manifesto_id'].iloc[0]} sem coordenada de filial no contrato."
        )

    origem_lat = float(lat_origem.iloc[0])
    origem_lon = float(lon_origem.iloc[0])

    df_paradas = _agrupar_paradas(
        grupo=grupo,
        fator_km_rodoviario_m7=fator_km_rodoviario_m7,
    )

    if len(df_paradas) == 1:
        df_paradas["ordem_entrega_parada_m7"] = 1
        df_paradas["chave_no_anterior_m7"] = "ORIGEM"
        df_paradas["distancia_no_anterior_km_m7"] = df_paradas["distancia_origem_parada_km_m7"]
        df_paradas["score_total_projetado_km_m7"] = df_paradas["distancia_origem_parada_km_m7"]
        df_paradas["chave_mais_perto_na_iteracao_m7"] = df_paradas["chave_parada_seq_m7"]
        df_paradas["chave_mais_longe_na_iteracao_m7"] = df_paradas["chave_parada_seq_m7"]
        df_paradas["dist_origem_extremo_km_m7"] = df_paradas["distancia_origem_parada_km_m7"]
        df_paradas["dist_candidato_extremo_km_m7"] = 0.0
        df_paradas["progresso_km_m7"] = 0.0
        df_paradas["lateral_km_m7"] = 0.0
        df_paradas["detour_km_m7"] = 0.0
        df_paradas["regressao_km_m7"] = 0.0
        df_paradas["alinhamento_m7"] = 1.0
        df_paradas["foi_mais_perto_m7"] = True
        df_paradas["foi_mais_longe_m7"] = True
        df_paradas["chave_proximo_no_m7"] = ""
        df_paradas["distancia_proximo_no_km_m7"] = np.nan
        df_paradas["criterio_escolha_m7"] = "parada_unica"
        df_paradas["metodo_sequenciamento_parada_m7"] = "parada_unica"

        trilha_auditoria = [
            {
                "ordem_entrega_parada_m7": 1,
                "chave_parada_seq_m7": str(df_paradas.iloc[0]["chave_parada_seq_m7"]),
                "chave_no_anterior_m7": "ORIGEM",
                "distancia_no_anterior_km_m7": float(df_paradas.iloc[0]["distancia_origem_parada_km_m7"]),
                "score_total_projetado_km_m7": float(df_paradas.iloc[0]["distancia_origem_parada_km_m7"]),
                "criterio_escolha_m7": "parada_unica",
                "chave_mais_perto_na_iteracao_m7": str(df_paradas.iloc[0]["chave_parada_seq_m7"]),
                "chave_mais_longe_na_iteracao_m7": str(df_paradas.iloc[0]["chave_parada_seq_m7"]),
                "dist_origem_extremo_km_m7": float(df_paradas.iloc[0]["distancia_origem_parada_km_m7"]),
                "dist_candidato_extremo_km_m7": 0.0,
                "progresso_km_m7": 0.0,
                "lateral_km_m7": 0.0,
                "detour_km_m7": 0.0,
                "regressao_km_m7": 0.0,
                "alinhamento_m7": 1.0,
                "foi_mais_perto_m7": True,
                "foi_mais_longe_m7": True,
            }
        ]
        km_total = float(df_paradas.iloc[0]["distancia_origem_parada_km_m7"])
    else:
        df_paradas, trilha_auditoria, km_total = _resolver_ordem_extremos_dinamicos(
            df_paradas=df_paradas,
            origem_lat=origem_lat,
            origem_lon=origem_lon,
            fator_km_rodoviario_m7=float(fator_km_rodoviario_m7),
        )

    grupo["chave_parada_seq_m7"] = (
        grupo["destinatario"].fillna("").astype(str).str.strip()
        + "|"
        + grupo["cidade"].fillna("").astype(str).str.strip()
        + "|"
        + grupo["uf"].fillna("").astype(str).str.strip()
    )

    grupo = grupo.merge(
        df_paradas[
            [
                "chave_parada_seq_m7",
                "ordem_entrega_parada_m7",
                "bucket_prioridade_m7",
                "folga_min_m7",
                "peso_total_m7",
                "distancia_origem_parada_km_m7",
                "chave_no_anterior_m7",
                "distancia_no_anterior_km_m7",
                "score_total_projetado_km_m7",
                "chave_mais_perto_na_iteracao_m7",
                "chave_mais_longe_na_iteracao_m7",
                "dist_origem_extremo_km_m7",
                "dist_candidato_extremo_km_m7",
                "progresso_km_m7",
                "lateral_km_m7",
                "detour_km_m7",
                "regressao_km_m7",
                "alinhamento_m7",
                "foi_mais_perto_m7",
                "foi_mais_longe_m7",
                "chave_proximo_no_m7",
                "distancia_proximo_no_km_m7",
                "criterio_escolha_m7",
                "metodo_sequenciamento_parada_m7",
                "lat_ref_m7",
                "lon_ref_m7",
            ]
        ],
        on="chave_parada_seq_m7",
        how="left",
    )

    partes_ordenadas: List[pd.DataFrame] = []
    for _, df_parada in grupo.groupby("chave_parada_seq_m7", sort=False):
        partes_ordenadas.append(_ordenar_docs_dentro_parada(df_parada, col_doc))

    grupo = pd.concat(partes_ordenadas, ignore_index=True)

    grupo = grupo.sort_values(
        by=[
            "ordem_entrega_parada_m7",
            "bucket_prioridade_m7",
            "folga_min_m7",
            "peso_total_m7",
            col_doc,
        ],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    grupo["ordem_entrega_doc_m7"] = np.arange(1, len(grupo) + 1)
    grupo["ordem_carregamento_doc_m7"] = (
        grupo["ordem_entrega_doc_m7"].max() - grupo["ordem_entrega_doc_m7"] + 1
    )

    grupo["justificativa_ordem_entrega_m7"] = grupo.apply(
        lambda row: (
            f"Parada={int(_safe_int(row.get('ordem_entrega_parada_m7'), 0))}; "
            f"metodo_parada={_safe_text(row.get('metodo_sequenciamento_parada_m7'))}; "
            f"criterio_escolha={_safe_text(row.get('criterio_escolha_m7'))}; "
            f"mais_perto_iteracao={_safe_text(row.get('chave_mais_perto_na_iteracao_m7'))}; "
            f"mais_longe_iteracao={_safe_text(row.get('chave_mais_longe_na_iteracao_m7'))}; "
            f"dist_origem_parada_km={_safe_float(row.get('distancia_origem_parada_km_m7'), 999999.0):.2f}; "
            f"dist_no_anterior_km={_safe_float(row.get('distancia_no_anterior_km_m7'), 999999.0):.2f}; "
            f"score_operacional={_safe_float(row.get('score_total_projetado_km_m7'), 999999.0):.2f}; "
            f"dist_origem_extremo_km={_safe_float(row.get('dist_origem_extremo_km_m7'), 999999.0):.2f}; "
            f"dist_candidato_extremo_km={_safe_float(row.get('dist_candidato_extremo_km_m7'), 999999.0):.2f}; "
            f"progresso_km={_safe_float(row.get('progresso_km_m7'), 0.0):.2f}; "
            f"lateral_km={_safe_float(row.get('lateral_km_m7'), 0.0):.2f}; "
            f"detour_km={_safe_float(row.get('detour_km_m7'), 0.0):.2f}; "
            f"regressao_km={_safe_float(row.get('regressao_km_m7'), 0.0):.2f}; "
            f"alinhamento={_safe_float(row.get('alinhamento_m7'), 0.0):.4f}; "
            f"foi_mais_perto={bool(row.get('foi_mais_perto_m7', False))}; "
            f"foi_mais_longe={bool(row.get('foi_mais_longe_m7', False))}; "
            f"prioridade_parada_bucket={_safe_int(row.get('bucket_prioridade_m7'), 9)}; "
            f"folga_min_parada={_safe_float(row.get('folga_min_m7'), 9999.0):.2f}; "
            f"peso_total_parada={_safe_float(row.get('peso_total_m7'), 0.0):.2f}; "
            f"criterio_doc={_montar_justificativa_doc(row)}"
        ),
        axis=1,
    )

    auditoria_local = {
        "trilha_sequenciamento_paradas_m7": trilha_auditoria,
        "km_total_sequencia_paradas_m7": float(km_total),
    }

    return grupo.reset_index(drop=True), df_paradas.reset_index(drop=True), auditoria_local


# =========================================================================================
# FUNÇÃO PRINCIPAL
# =========================================================================================
def executar_m7_sequenciamento_entregas(
    df_manifestos_m6_2: pd.DataFrame,
    df_itens_manifestos_m6_2: pd.DataFrame,
    df_geo_tratado: Optional[pd.DataFrame] = None,
    df_geo_raw: Optional[pd.DataFrame] = None,
    data_base_roteirizacao: Optional[datetime] = None,
    tipo_roteirizacao: str = "carteira",
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    time_limit_seconds: int = TIME_LIMIT_SECONDS_PADRAO,
    fator_km_rodoviario_m7: float = FATOR_KM_RODOVIARIO_M7_PADRAO,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    del df_geo_tratado
    del df_geo_raw
    del time_limit_seconds

    if not isinstance(df_manifestos_m6_2, pd.DataFrame) or df_manifestos_m6_2.empty:
        raise Exception("M7 recebeu df_manifestos_m6_2 vazio.")

    if not isinstance(df_itens_manifestos_m6_2, pd.DataFrame) or df_itens_manifestos_m6_2.empty:
        raise Exception("M7 recebeu df_itens_manifestos_m6_2 vazio.")

    df_manifestos = _normalizar_manifestos(df_manifestos_m6_2)
    df_itens = _normalizar_itens(df_itens_manifestos_m6_2)

    manifestos_validos = set(df_manifestos["manifesto_id"].astype(str))
    df_itens = df_itens.loc[df_itens["manifesto_id"].astype(str).isin(manifestos_validos)].copy()

    df_itens, df_diagnostico_recuperacao_coordenadas_m7 = _preparar_coordenadas_contrato(df_itens)

    resultados: List[pd.DataFrame] = []
    resumos_manifestos: List[Dict[str, Any]] = []
    tentativas: List[Dict[str, Any]] = []
    auditorias_manifestos: List[Dict[str, Any]] = []

    for manifesto_id, grupo in df_itens.groupby("manifesto_id", dropna=False):
        grupo = grupo.copy().reset_index(drop=True)

        try:
            if grupo["latitude_dest_m7"].isna().any() or grupo["longitude_dest_m7"].isna().any():
                raise Exception(
                    f"Manifesto {manifesto_id} ainda possui coordenada de destino nula no contrato recebido."
                )

            if grupo["latitude_filial_m7"].isna().any() or grupo["longitude_filial_m7"].isna().any():
                raise Exception(
                    f"Manifesto {manifesto_id} ainda possui coordenada de filial nula no contrato recebido."
                )

            fator_real_manifesto = _inferir_fator_rodoviario_real_manifesto(
                df_manifesto=grupo,
                fallback=fator_km_rodoviario_m7,
            )

            grupo_seq, df_paradas_seq, auditoria_local = _sequenciar_manifesto(
                df_manifesto=grupo,
                col_doc="id_linha_pipeline",
                fator_km_rodoviario_m7=float(fator_real_manifesto),
            )

            grupo_seq["status_sequenciamento_m7"] = "ok"
            grupo_seq["motivo_status_sequenciamento_m7"] = "sequenciamento_realizado"

            resultados.append(grupo_seq)

            resumos_manifestos.append(
                {
                    "manifesto_id": manifesto_id,
                    "qtd_docs_manifesto_m7": int(len(grupo_seq)),
                    "qtd_paradas_manifesto_m7": int(grupo_seq["chave_parada_seq_m7"].nunique()),
                    "primeira_entrega_parada_m7": grupo_seq.sort_values("ordem_entrega_doc_m7")["chave_parada_seq_m7"].iloc[0],
                    "ultima_entrega_parada_m7": grupo_seq.sort_values("ordem_entrega_doc_m7")["chave_parada_seq_m7"].iloc[-1],
                    "status_sequenciamento_m7": "ok",
                    "metodo_predominante_m7": (
                        df_paradas_seq["metodo_sequenciamento_parada_m7"].mode().iloc[0]
                        if not df_paradas_seq.empty
                        else "na"
                    ),
                    "fator_km_rodoviario_real_m7": float(fator_real_manifesto),
                    "km_total_sequencia_paradas_m7": float(auditoria_local["km_total_sequencia_paradas_m7"]),
                }
            )

            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "resultado": "ok",
                    "motivo": "sequenciamento_realizado",
                    "qtd_docs": int(len(grupo_seq)),
                    "qtd_paradas": int(df_paradas_seq.shape[0]),
                    "km_total_sequencia_paradas_m7": float(auditoria_local["km_total_sequencia_paradas_m7"]),
                }
            )

            auditorias_manifestos.append(
                {
                    "manifesto_id": manifesto_id,
                    **auditoria_local,
                }
            )

        except Exception as e:
            grupo_fallback = grupo.copy()

            prioridades_fb = grupo_fallback.apply(_classificar_prioridade_negocio, axis=1)
            grupo_fallback["bucket_prioridade_fb_m7"] = [x[0] for x in prioridades_fb]
            grupo_fallback["folga_prioridade_fb_m7"] = [x[1] for x in prioridades_fb]
            grupo_fallback["peso_prioridade_fb_m7"] = [(-x[2]) for x in prioridades_fb]

            grupo_fallback["chave_parada_seq_m7"] = (
                grupo_fallback["destinatario"].fillna("").astype(str).str.strip()
                + "|"
                + grupo_fallback["cidade"].fillna("").astype(str).str.strip()
                + "|"
                + grupo_fallback["uf"].fillna("").astype(str).str.strip()
            )

            grupo_fallback = grupo_fallback.sort_values(
                by=[
                    "bucket_prioridade_fb_m7",
                    "folga_prioridade_fb_m7",
                    "peso_prioridade_fb_m7",
                    "id_linha_pipeline",
                ],
                ascending=[True, True, False, True],
                kind="mergesort",
            ).reset_index(drop=True)

            grupo_fallback["ordem_entrega_parada_m7"] = np.nan
            grupo_fallback["ordem_entrega_doc_m7"] = np.arange(1, len(grupo_fallback) + 1)
            grupo_fallback["ordem_carregamento_doc_m7"] = (
                grupo_fallback["ordem_entrega_doc_m7"].max() - grupo_fallback["ordem_entrega_doc_m7"] + 1
            )
            grupo_fallback["status_sequenciamento_m7"] = "fallback"
            grupo_fallback["motivo_status_sequenciamento_m7"] = str(e)
            grupo_fallback["metodo_sequenciamento_parada_m7"] = "fallback_regra"
            grupo_fallback["justificativa_ordem_entrega_m7"] = grupo_fallback.apply(
                lambda row: f"Fallback por exceção; criterio_doc={_montar_justificativa_doc(row)}; motivo={str(e)}",
                axis=1,
            )

            grupo_fallback = grupo_fallback.drop(
                columns=[
                    "bucket_prioridade_fb_m7",
                    "folga_prioridade_fb_m7",
                    "peso_prioridade_fb_m7",
                ],
                errors="ignore",
            )

            resultados.append(grupo_fallback)

            resumos_manifestos.append(
                {
                    "manifesto_id": manifesto_id,
                    "qtd_docs_manifesto_m7": int(len(grupo_fallback)),
                    "qtd_paradas_manifesto_m7": int(grupo_fallback["chave_parada_seq_m7"].nunique()),
                    "primeira_entrega_parada_m7": "",
                    "ultima_entrega_parada_m7": "",
                    "status_sequenciamento_m7": "fallback",
                    "metodo_predominante_m7": "fallback_regra",
                    "fator_km_rodoviario_real_m7": None,
                    "km_total_sequencia_paradas_m7": None,
                }
            )

            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "resultado": "fallback",
                    "motivo": str(e),
                    "qtd_docs": int(len(grupo_fallback)),
                    "qtd_paradas": int(grupo_fallback["chave_parada_seq_m7"].nunique()),
                    "km_total_sequencia_paradas_m7": None,
                }
            )

    df_itens_manifestos_sequenciados_m7 = (
        pd.concat(resultados, ignore_index=True) if resultados else pd.DataFrame()
    )
    df_manifestos_sequenciamento_resumo_m7 = pd.DataFrame(resumos_manifestos)
    df_tentativas_sequenciamento_m7 = pd.DataFrame(tentativas)

    if not df_itens_manifestos_sequenciados_m7.empty:
        df_manifestos_m7 = df_manifestos.merge(
            df_manifestos_sequenciamento_resumo_m7,
            on="manifesto_id",
            how="left",
        )
    else:
        df_manifestos_m7 = df_manifestos.copy()

    resumo_m7 = {
        "modulo": "M7",
        "data_base_roteirizacao": (
            data_base_roteirizacao.isoformat()
            if isinstance(data_base_roteirizacao, datetime)
            else str(data_base_roteirizacao)
            if data_base_roteirizacao is not None
            else None
        ),
        "tipo_roteirizacao": tipo_roteirizacao,
        "fonte_geo_m7": "contrato_itens_e_filial",
        "metodo_m7": "extremos_dinamicos_origem_recalculada",
        "fator_km_rodoviario_param_m7": float(fator_km_rodoviario_m7),
        "manifestos_entrada_m7": int(df_manifestos["manifesto_id"].nunique()),
        "itens_entrada_m7": int(len(df_itens)),
        "manifestos_saida_m7": int(df_itens_manifestos_sequenciados_m7["manifesto_id"].nunique())
        if not df_itens_manifestos_sequenciados_m7.empty
        else 0,
        "itens_saida_m7": int(len(df_itens_manifestos_sequenciados_m7)),
        "fallbacks_m7": int(
            (df_tentativas_sequenciamento_m7["resultado"] == "fallback").sum()
        ) if not df_tentativas_sequenciamento_m7.empty else 0,
        "linhas_filial_nula_m7": int(
            (df_itens_manifestos_sequenciados_m7["status_coord_filial_m7"] != "ok").sum()
        ) if not df_itens_manifestos_sequenciados_m7.empty else 0,
        "linhas_destino_nula_m7": int(
            (df_itens_manifestos_sequenciados_m7["status_coord_dest_m7"] != "ok").sum()
        ) if not df_itens_manifestos_sequenciados_m7.empty else 0,
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    auditoria_m7 = {
        "manifestos_fallback_m7": (
            df_tentativas_sequenciamento_m7.loc[
                df_tentativas_sequenciamento_m7["resultado"] == "fallback", "manifesto_id"
            ].astype(str).tolist()
            if not df_tentativas_sequenciamento_m7.empty
            else []
        ),
        "auditoria_manifestos_m7": auditorias_manifestos,
        "amostra_justificativas_ordem_m7": (
            df_itens_manifestos_sequenciados_m7[
                [
                    "manifesto_id",
                    "id_linha_pipeline",
                    "ordem_entrega_doc_m7",
                    "ordem_carregamento_doc_m7",
                    "justificativa_ordem_entrega_m7",
                ]
            ]
            .head(50)
            .to_dict(orient="records")
            if not df_itens_manifestos_sequenciados_m7.empty
            else []
        ),
    }

    outputs = {
        "df_manifestos_m7": df_manifestos_m7.reset_index(drop=True),
        "df_itens_manifestos_sequenciados_m7": df_itens_manifestos_sequenciados_m7.reset_index(drop=True),
        "df_manifestos_sequenciamento_resumo_m7": df_manifestos_sequenciamento_resumo_m7.reset_index(drop=True),
        "df_tentativas_sequenciamento_m7": df_tentativas_sequenciamento_m7.reset_index(drop=True),
        "df_diagnostico_recuperacao_coordenadas_m7": df_diagnostico_recuperacao_coordenadas_m7.reset_index(drop=True),
    }

    meta = {
        "resumo_m7": resumo_m7,
        "auditoria_m7": auditoria_m7,
    }

    return outputs, meta


def executar_m7(*args: Any, **kwargs: Any):
    return executar_m7_sequenciamento_entregas(*args, **kwargs)


def processar_m7_sequenciamento_entregas(*args: Any, **kwargs: Any):
    return executar_m7_sequenciamento_entregas(*args, **kwargs)


def rodar_m7_sequenciamento_entregas(*args: Any, **kwargs: Any):
    return executar_m7_sequenciamento_entregas(*args, **kwargs)
