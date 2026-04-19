from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import math

FATOR_KM_PADRAO = 1.2


# =========================================================
# HELPERS
# =========================================================
def _safe_float(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except:
        return default


def _dist(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _dist_km(lat1, lon1, lat2, lon2):
    return _dist(lat1, lon1, lat2, lon2) * FATOR_KM_PADRAO


# =========================================================
# CORE CORRIGIDO
# =========================================================
def _sequenciar_cidades(
    df: pd.DataFrame,
    origem_lat: float,
    origem_lon: float,
) -> Tuple[pd.DataFrame, List[Dict], float]:

    if df.empty:
        return df.copy(), [], 0.0

    df = df.copy()

    # -----------------------------------------------------
    # DISTÂNCIA DA ORIGEM
    # -----------------------------------------------------
    df["dist_origem"] = df.apply(
        lambda r: _dist_km(
            origem_lat,
            origem_lon,
            r["lat_ref_cidade_m7"],
            r["lon_ref_cidade_m7"],
        ),
        axis=1,
    )

    # -----------------------------------------------------
    # IDENTIFICAR CIDADE DA FILIAL
    # -----------------------------------------------------
    df["is_origem"] = df["dist_origem"] <= 0.5

    cidades_origem = df[df["is_origem"]].copy()
    cidades_normais = df[~df["is_origem"]].copy()

    # -----------------------------------------------------
    # CASO 1: TODAS SÃO ORIGEM
    # -----------------------------------------------------
    if cidades_normais.empty:

        df = df.sort_values("dist_origem").reset_index(drop=True)
        df["ordem"] = range(1, len(df) + 1)

        return df, [], df["dist_origem"].sum()

    # -----------------------------------------------------
    # CASO 2: EXISTE MACRO-ROTA
    # -----------------------------------------------------

    # pega mais distante da origem
    cidades_normais = cidades_normais.sort_values("dist_origem", ascending=False)
    A = cidades_normais.iloc[0]

    # pega mais distante de A
    cidades_normais["dist_A"] = cidades_normais.apply(
        lambda r: _dist_km(
            A["lat_ref_cidade_m7"],
            A["lon_ref_cidade_m7"],
            r["lat_ref_cidade_m7"],
            r["lon_ref_cidade_m7"],
        ),
        axis=1,
    )

    cidades_normais = cidades_normais.sort_values("dist_A", ascending=False)
    B = cidades_normais.iloc[0]

    # eixo A -> B
    def proj(r):
        ax = B["lat_ref_cidade_m7"] - A["lat_ref_cidade_m7"]
        ay = B["lon_ref_cidade_m7"] - A["lon_ref_cidade_m7"]

        px = r["lat_ref_cidade_m7"] - A["lat_ref_cidade_m7"]
        py = r["lon_ref_cidade_m7"] - A["lon_ref_cidade_m7"]

        norma = math.sqrt(ax * ax + ay * ay)
        if norma == 0:
            return 0

        return (px * ax + py * ay) / norma

    cidades_normais["proj"] = cidades_normais.apply(proj, axis=1)

    rota1 = cidades_normais.sort_values("proj", ascending=False)
    rota2 = cidades_normais.sort_values("proj", ascending=True)

    # -----------------------------------------------------
    # TESTAR POSIÇÃO DA ORIGEM (INICIO vs FIM)
    # -----------------------------------------------------
    origem_list = cidades_origem["chave_cidade_seq_m7"].tolist()

    def calc_rota(ordem):
        km = 0
        lat = origem_lat
        lon = origem_lon

        for _, r in ordem.iterrows():
            km += _dist_km(lat, lon, r["lat_ref_cidade_m7"], r["lon_ref_cidade_m7"])
            lat = r["lat_ref_cidade_m7"]
            lon = r["lon_ref_cidade_m7"]

        return km

    candidatos = []

    for rota in [rota1, rota2]:

        # origem no começo
        df1 = pd.concat([cidades_origem, rota])
        km1 = calc_rota(df1)

        # origem no fim
        df2 = pd.concat([rota, cidades_origem])
        km2 = calc_rota(df2)

        candidatos.append((df1, km1))
        candidatos.append((df2, km2))

    melhor_df, melhor_km = sorted(candidatos, key=lambda x: x[1])[0]

    melhor_df = melhor_df.reset_index(drop=True)
    melhor_df["ordem"] = range(1, len(melhor_df) + 1)

    return melhor_df, [], melhor_km


# =========================================================
# ENTRY POINT
# =========================================================
def executar_m7_sequenciamento_entregas(
    df_manifestos: pd.DataFrame,
    df_itens: pd.DataFrame,
    parametros: Dict[str, Any],
):

    if df_itens.empty:
        return df_manifestos, df_itens, {}

    origem_lat = parametros.get("origem_latitude")
    origem_lon = parametros.get("origem_longitude")

    resultado = []

    for manifesto_id, grupo in df_itens.groupby("manifesto_id"):

        cidades = (
            grupo.groupby(["cidade", "uf"])
            .agg(
                lat_ref_cidade_m7=("latitude_dest_m7", "mean"),
                lon_ref_cidade_m7=("longitude_dest_m7", "mean"),
            )
            .reset_index()
        )

        cidades["chave_cidade_seq_m7"] = (
            cidades["cidade"] + "|" + cidades["uf"]
        )

        cidades_seq, _, _ = _sequenciar_cidades(
            cidades,
            origem_lat,
            origem_lon,
        )

        mapa_ordem = dict(
            zip(
                cidades_seq["chave_cidade_seq_m7"],
                cidades_seq["ordem"],
            )
        )

        grupo["ordem_cidade"] = grupo.apply(
            lambda r: mapa_ordem.get(r["cidade"] + "|" + r["uf"], 999),
            axis=1,
        )

        grupo = grupo.sort_values(["ordem_cidade"])

        resultado.append(grupo)

    df_final = pd.concat(resultado).reset_index(drop=True)

    return df_manifestos, df_final, {}
