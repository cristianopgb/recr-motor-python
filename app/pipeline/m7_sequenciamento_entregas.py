from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd


TIME_LIMIT_SECONDS_PADRAO = 5
FATOR_KM_RODOVIARIO_M7_PADRAO = 1.20
FATOR_KM_RODOVIARIO_M7_MIN = 1.00
FATOR_KM_RODOVIARIO_M7_MAX = 3.00

# =========================================================================================
# HELPERS BÁSICOS
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


def _txt_norm(value: Any) -> str:
    return _safe_text(value).upper()


# =========================================================================================
# DISTÂNCIA
# =========================================================================================
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return np.nan

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

    fator = float(fator_km_rodoviario)
    if pd.isna(fator) or fator <= 0:
        fator = FATOR_KM_RODOVIARIO_M7_PADRAO

    return float(dist_hav) * fator


def _inferir_fator_rodoviario_do_manifesto(
    df_manifesto: pd.DataFrame,
    fallback: float,
) -> float:
    amostra = df_manifesto.copy()

    if "distancia_rodoviaria_est_km" not in amostra.columns:
        return float(fallback)

    ratios: List[float] = []

    for _, row in amostra.iterrows():
        dist_est = pd.to_numeric(row.get("distancia_rodoviaria_est_km"), errors="coerce")
        lat_o = pd.to_numeric(row.get("latitude_filial_m7"), errors="coerce")
        lon_o = pd.to_numeric(row.get("longitude_filial_m7"), errors="coerce")
        lat_d = pd.to_numeric(row.get("latitude_dest_m7"), errors="coerce")
        lon_d = pd.to_numeric(row.get("longitude_dest_m7"), errors="coerce")

        dist_hav = _haversine_km(lat_o, lon_o, lat_d, lon_d)
        if pd.isna(dist_est) or pd.isna(dist_hav) or dist_est <= 0 or dist_hav <= 0:
            continue

        ratio = float(dist_est) / float(dist_hav)
        if FATOR_KM_RODOVIARIO_M7_MIN <= ratio <= FATOR_KM_RODOVIARIO_M7_MAX:
            ratios.append(ratio)

    if not ratios:
        return float(fallback)

    return float(np.median(ratios))


# =========================================================================================
# PRIORIDADE OPERACIONAL
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

    # M7 sequencia com peso calculado como fonte principal, fallback para peso_kg
    out["peso_seq_m7"] = pd.to_numeric(out["peso_calculado"], errors="coerce").fillna(
        pd.to_numeric(out["peso_kg"], errors="coerce")
    )

    out["latitude_filial_m7"] = out[col_lat_filial]
    out["longitude_filial_m7"] = out[col_lon_filial]
    out["latitude_dest_m7"] = out[col_lat_dest]
    out["longitude_dest_m7"] = out[col_lon_dest]

    out = out[(out["manifesto_id"] != "") & (out["id_linha_pipeline"] != "")].copy()

    if out["id_linha_pipeline"].duplicated().any():
        duplicados = (
            out.loc[out["id_linha_pipeline"].duplicated(), "id_linha_pipeline"]
            .astype(str)
            .tolist()[:20]
        )
        raise Exception(
            f"M7 recebeu id_linha_pipeline duplicado em df_itens_manifestos_m6_2: {duplicados}"
        )

    return out.reset_index(drop=True)


# =========================================================================================
# PREPARAÇÃO GEO DIRETA DO CONTRATO
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
# AGREGAÇÃO DE PARADAS / CIDADES
# =========================================================================================
def _agrupar_paradas(
    grupo: pd.DataFrame,
    fator_km_rodoviario_m7: float,
) -> pd.DataFrame:
    registros: List[Dict[str, Any]] = []

    for chave_parada, gpar in grupo.groupby("chave_parada_seq_m7", dropna=False):
        score = _calcular_score_parada(gpar)
        lat_ref = pd.to_numeric(gpar["latitude_dest_m7"], errors="coerce").mean()
        lon_ref = pd.to_numeric(gpar["longitude_dest_m7"], errors="coerce").mean()

        dist_origem_ref = pd.to_numeric(gpar["distancia_rodoviaria_est_km"], errors="coerce").min()
        if pd.isna(dist_origem_ref):
            lat_o = pd.to_numeric(gpar["latitude_filial_m7"], errors="coerce").dropna()
            lon_o = pd.to_numeric(gpar["longitude_filial_m7"], errors="coerce").dropna()
            if len(lat_o) > 0 and len(lon_o) > 0 and pd.notna(lat_ref) and pd.notna(lon_ref):
                dist_origem_ref = _distancia_operacional_km(
                    float(lat_o.iloc[0]),
                    float(lon_o.iloc[0]),
                    float(lat_ref),
                    float(lon_ref),
                    fator_km_rodoviario_m7,
                )
            else:
                dist_origem_ref = 999999.0

        registros.append(
            {
                "chave_parada_seq_m7": chave_parada,
                "destinatario_ref_m7": _safe_text(gpar["destinatario"].dropna().iloc[0] if len(gpar["destinatario"].dropna()) > 0 else ""),
                "cidade_ref_m7": _safe_text(gpar["cidade"].dropna().iloc[0] if len(gpar["cidade"].dropna()) > 0 else ""),
                "uf_ref_m7": _safe_text(gpar["uf"].dropna().iloc[0] if len(gpar["uf"].dropna()) > 0 else ""),
                "lat_ref_m7": lat_ref,
                "lon_ref_m7": lon_ref,
                "bucket_prioridade_m7": score["bucket_prioridade"],
                "folga_min_m7": score["folga_min"],
                "peso_total_m7": score["peso_total"],
                "qtd_docs_parada_m7": int(len(gpar)),
                "distancia_origem_ref_m7": float(dist_origem_ref),
            }
        )

    df_paradas = pd.DataFrame(registros).reset_index(drop=True)

    if df_paradas.empty:
        return df_paradas

    df_paradas["cidade_bloco_m7"] = (
        df_paradas["cidade_ref_m7"].fillna("").astype(str).str.strip()
        + "|"
        + df_paradas["uf_ref_m7"].fillna("").astype(str).str.strip()
    )

    return df_paradas.reset_index(drop=True)


def _agrupar_cidades(df_paradas: pd.DataFrame) -> pd.DataFrame:
    registros: List[Dict[str, Any]] = []

    for cidade_bloco, g in df_paradas.groupby("cidade_bloco_m7", dropna=False):
        registros.append(
            {
                "cidade_bloco_m7": cidade_bloco,
                "cidade_ref_m7": _safe_text(g["cidade_ref_m7"].dropna().iloc[0] if len(g["cidade_ref_m7"].dropna()) > 0 else ""),
                "uf_ref_m7": _safe_text(g["uf_ref_m7"].dropna().iloc[0] if len(g["uf_ref_m7"].dropna()) > 0 else ""),
                "lat_centro_m7": pd.to_numeric(g["lat_ref_m7"], errors="coerce").mean(),
                "lon_centro_m7": pd.to_numeric(g["lon_ref_m7"], errors="coerce").mean(),
                "peso_total_cidade_m7": float(pd.to_numeric(g["peso_total_m7"], errors="coerce").fillna(0).sum()),
                "bucket_min_cidade_m7": int(pd.to_numeric(g["bucket_prioridade_m7"], errors="coerce").fillna(9).min()),
                "folga_min_cidade_m7": float(pd.to_numeric(g["folga_min_m7"], errors="coerce").fillna(9999).min()),
                "qtd_paradas_cidade_m7": int(len(g)),
                "distancia_origem_min_cidade_m7": float(pd.to_numeric(g["distancia_origem_ref_m7"], errors="coerce").min()),
            }
        )

    return pd.DataFrame(registros).reset_index(drop=True)


# =========================================================================================
# HEURÍSTICA HIERÁRQUICA
# =========================================================================================
def _distancia_entre_pontos_km(
    ponto_a: Tuple[float, float],
    ponto_b: Tuple[float, float],
    fator_km_rodoviario_m7: float,
) -> float:
    return _distancia_operacional_km(
        ponto_a[0], ponto_a[1], ponto_b[0], ponto_b[1], fator_km_rodoviario_m7
    )


def _ordenar_cidades_hierarquico(
    df_cidades: pd.DataFrame,
    df_paradas: pd.DataFrame,
    origem: Tuple[float, float],
    fator_km_rodoviario_m7: float,
) -> List[str]:
    if df_cidades.empty:
        return []

    restantes = set(df_cidades["cidade_bloco_m7"].astype(str).tolist())
    ordem_cidades: List[str] = []
    ponto_atual = origem

    while restantes:
        melhor_cidade = None
        melhor_chave = None

        for cidade_bloco in restantes:
            stops_city = df_paradas.loc[df_paradas["cidade_bloco_m7"] == cidade_bloco].copy()
            if stops_city.empty:
                continue

            dist_min = 999999.0
            for _, row in stops_city.iterrows():
                d = _distancia_entre_pontos_km(
                    ponto_atual,
                    (float(row["lat_ref_m7"]), float(row["lon_ref_m7"])),
                    fator_km_rodoviario_m7,
                )
                if d < dist_min:
                    dist_min = d

            row_city = df_cidades.loc[df_cidades["cidade_bloco_m7"] == cidade_bloco].iloc[0]

            chave = (
                round(float(dist_min), 6),
                -round(float(row_city["peso_total_cidade_m7"]), 6),
                int(row_city["bucket_min_cidade_m7"]),
                round(float(row_city["folga_min_cidade_m7"]), 6),
                str(cidade_bloco),
            )

            if melhor_chave is None or chave < melhor_chave:
                melhor_chave = chave
                melhor_cidade = cidade_bloco

        if melhor_cidade is None:
            break

        ordem_cidades.append(str(melhor_cidade))
        restantes.remove(melhor_cidade)

        row_city_sel = df_cidades.loc[df_cidades["cidade_bloco_m7"] == melhor_cidade].iloc[0]
        ponto_atual = (
            float(row_city_sel["lat_centro_m7"]),
            float(row_city_sel["lon_centro_m7"]),
        )

    return ordem_cidades


def _ordenar_paradas_dentro_cidade(
    df_paradas_cidade: pd.DataFrame,
    ponto_inicio: Tuple[float, float],
    fator_km_rodoviario_m7: float,
) -> List[str]:
    restantes = df_paradas_cidade.copy().reset_index(drop=True)
    ordem: List[str] = []
    ponto_atual = ponto_inicio

    while not restantes.empty:
        chaves: List[Tuple[Any, ...]] = []

        for idx, row in restantes.iterrows():
            dist = _distancia_entre_pontos_km(
                ponto_atual,
                (float(row["lat_ref_m7"]), float(row["lon_ref_m7"])),
                fator_km_rodoviario_m7,
            )

            chave = (
                round(float(dist), 6),
                -round(float(row["peso_total_m7"]), 6),
                int(row["bucket_prioridade_m7"]),
                round(float(row["folga_min_m7"]), 6),
                round(float(row["distancia_origem_ref_m7"]), 6),
                str(row["chave_parada_seq_m7"]),
            )
            chaves.append((idx, chave))

        idx_escolhido = sorted(chaves, key=lambda x: x[1])[0][0]
        row_sel = restantes.loc[idx_escolhido]

        ordem.append(str(row_sel["chave_parada_seq_m7"]))
        ponto_atual = (float(row_sel["lat_ref_m7"]), float(row_sel["lon_ref_m7"]))

        restantes = restantes.drop(index=idx_escolhido).reset_index(drop=True)

    return ordem


def _montar_ordem_inicial_paradas(
    df_paradas: pd.DataFrame,
    origem: Tuple[float, float],
    fator_km_rodoviario_m7: float,
) -> List[str]:
    if df_paradas.empty:
        return []

    df_cidades = _agrupar_cidades(df_paradas)
    ordem_cidades = _ordenar_cidades_hierarquico(
        df_cidades=df_cidades,
        df_paradas=df_paradas,
        origem=origem,
        fator_km_rodoviario_m7=fator_km_rodoviario_m7,
    )

    ordem_final: List[str] = []
    ponto_atual = origem

    for cidade_bloco in ordem_cidades:
        df_city = df_paradas.loc[df_paradas["cidade_bloco_m7"] == cidade_bloco].copy().reset_index(drop=True)
        ordem_city = _ordenar_paradas_dentro_cidade(
            df_paradas_cidade=df_city,
            ponto_inicio=ponto_atual,
            fator_km_rodoviario_m7=fator_km_rodoviario_m7,
        )
        ordem_final.extend(ordem_city)

        if ordem_city:
            ultima = df_city.loc[df_city["chave_parada_seq_m7"] == ordem_city[-1]].iloc[0]
            ponto_atual = (float(ultima["lat_ref_m7"]), float(ultima["lon_ref_m7"]))

    return ordem_final


# =========================================================================================
# VALIDAÇÃO DE FRONTEIRA
# -----------------------------------------------------------------------------------------
# Compara sequência atual vs sequência trocada localmente:
# atual:   A -> B -> C -> D
# troca:   A -> C -> B -> D
# Aceita a troca se o custo total cair.
# =========================================================================================
def _custo_trecho_local_trocado(
    prev_coords: Tuple[float, float],
    a_coords: Tuple[float, float],
    b_coords: Tuple[float, float],
    next_coords: Optional[Tuple[float, float]],
    fator_km_rodoviario_m7: float,
) -> float:
    custo = _distancia_entre_pontos_km(prev_coords, b_coords, fator_km_rodoviario_m7)
    custo += _distancia_entre_pontos_km(b_coords, a_coords, fator_km_rodoviario_m7)
    if next_coords is not None:
        custo += _distancia_entre_pontos_km(a_coords, next_coords, fator_km_rodoviario_m7)
    return custo


def _custo_trecho_local_atual(
    prev_coords: Tuple[float, float],
    a_coords: Tuple[float, float],
    b_coords: Tuple[float, float],
    next_coords: Optional[Tuple[float, float]],
    fator_km_rodoviario_m7: float,
) -> float:
    custo = _distancia_entre_pontos_km(prev_coords, a_coords, fator_km_rodoviario_m7)
    custo += _distancia_entre_pontos_km(a_coords, b_coords, fator_km_rodoviario_m7)
    if next_coords is not None:
        custo += _distancia_entre_pontos_km(b_coords, next_coords, fator_km_rodoviario_m7)
    return custo


def _aplicar_validacao_local_fronteira(
    ordem_paradas: List[str],
    df_paradas: pd.DataFrame,
    origem: Tuple[float, float],
    fator_km_rodoviario_m7: float,
) -> Tuple[List[str], int]:
    if len(ordem_paradas) <= 2:
        return ordem_paradas, 0

    mapa = {
        str(row["chave_parada_seq_m7"]): row
        for _, row in df_paradas.iterrows()
    }

    ordem = ordem_paradas.copy()
    trocas_aceitas = 0
    houve_mudanca = True
    limite_passadas = max(3, len(ordem) * 2)
    passada = 0

    while houve_mudanca and passada < limite_passadas:
        houve_mudanca = False
        passada += 1

        for i in range(len(ordem) - 1):
            chave_a = ordem[i]
            chave_b = ordem[i + 1]

            row_a = mapa[chave_a]
            row_b = mapa[chave_b]

            cidade_a = _txt_norm(row_a["cidade_ref_m7"])
            cidade_b = _txt_norm(row_b["cidade_ref_m7"])

            if cidade_a == cidade_b:
                continue

            if i == 0:
                prev_coords = origem
            else:
                row_prev = mapa[ordem[i - 1]]
                prev_coords = (float(row_prev["lat_ref_m7"]), float(row_prev["lon_ref_m7"]))

            if i + 2 < len(ordem):
                row_next = mapa[ordem[i + 2]]
                next_coords = (float(row_next["lat_ref_m7"]), float(row_next["lon_ref_m7"]))
            else:
                next_coords = None

            a_coords = (float(row_a["lat_ref_m7"]), float(row_a["lon_ref_m7"]))
            b_coords = (float(row_b["lat_ref_m7"]), float(row_b["lon_ref_m7"]))

            custo_atual = _custo_trecho_local_atual(
                prev_coords=prev_coords,
                a_coords=a_coords,
                b_coords=b_coords,
                next_coords=next_coords,
                fator_km_rodoviario_m7=fator_km_rodoviario_m7,
            )
            custo_trocado = _custo_trecho_local_trocado(
                prev_coords=prev_coords,
                a_coords=a_coords,
                b_coords=b_coords,
                next_coords=next_coords,
                fator_km_rodoviario_m7=fator_km_rodoviario_m7,
            )

            if custo_trocado + 1e-9 < custo_atual:
                ordem[i], ordem[i + 1] = ordem[i + 1], ordem[i]
                trocas_aceitas += 1
                houve_mudanca = True

    return ordem, trocas_aceitas


# =========================================================================================
# ORDEM DAS PARADAS
# =========================================================================================
def _ordenar_paradas_hierarquico_com_fronteira(
    df_manifesto: pd.DataFrame,
    col_manifesto: str,
    col_doc: str,
    fator_km_rodoviario_m7: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    grupo = df_manifesto.copy().reset_index(drop=True)

    grupo["chave_parada_seq_m7"] = (
        grupo["destinatario"].fillna("").astype(str).str.strip()
        + "|"
        + grupo["cidade"].fillna("").astype(str).str.strip()
        + "|"
        + grupo["uf"].fillna("").astype(str).str.strip()
    )

    lat_origem = pd.to_numeric(grupo["latitude_filial_m7"], errors="coerce").dropna()
    lon_origem = pd.to_numeric(grupo["longitude_filial_m7"], errors="coerce").dropna()

    if len(lat_origem) == 0 or len(lon_origem) == 0:
        raise Exception(
            f"Manifesto {grupo[col_manifesto].iloc[0]} sem coordenada de filial no contrato."
        )

    origem = (float(lat_origem.iloc[0]), float(lon_origem.iloc[0]))

    df_paradas = _agrupar_paradas(
        grupo=grupo,
        fator_km_rodoviario_m7=fator_km_rodoviario_m7,
    )

    if df_paradas.empty:
        raise Exception(f"Manifesto {grupo[col_manifesto].iloc[0]} sem paradas válidas para sequenciamento.")

    if df_paradas["lat_ref_m7"].isna().any() or df_paradas["lon_ref_m7"].isna().any():
        raise Exception(
            f"Manifesto {grupo[col_manifesto].iloc[0]} possui parada sem coordenada de destino no contrato."
        )

    if len(df_paradas) == 1:
        df_paradas["ordem_entrega_parada_m7"] = 1
        df_paradas["metodo_sequenciamento_parada_m7"] = "parada_unica"
        df_paradas["ajuste_fronteira_m7"] = False
        df_paradas["trocas_fronteira_m7"] = 0
        ordem_inicial = df_paradas["chave_parada_seq_m7"].astype(str).tolist()
        ordem_final = ordem_inicial.copy()
        trocas_fronteira = 0
    else:
        ordem_inicial = _montar_ordem_inicial_paradas(
            df_paradas=df_paradas,
            origem=origem,
            fator_km_rodoviario_m7=fator_km_rodoviario_m7,
        )

        ordem_final, trocas_fronteira = _aplicar_validacao_local_fronteira(
            ordem_paradas=ordem_inicial,
            df_paradas=df_paradas,
            origem=origem,
            fator_km_rodoviario_m7=fator_km_rodoviario_m7,
        )

        mapa_ordem = {chave: pos + 1 for pos, chave in enumerate(ordem_final)}
        df_paradas["ordem_entrega_parada_m7"] = df_paradas["chave_parada_seq_m7"].map(mapa_ordem).astype(int)
        df_paradas["metodo_sequenciamento_parada_m7"] = "heuristica_hierarquica_fronteira"
        df_paradas["ajuste_fronteira_m7"] = trocas_fronteira > 0
        df_paradas["trocas_fronteira_m7"] = int(trocas_fronteira)

    grupo = grupo.merge(
        df_paradas[
            [
                "chave_parada_seq_m7",
                "ordem_entrega_parada_m7",
                "bucket_prioridade_m7",
                "folga_min_m7",
                "peso_total_m7",
                "cidade_ref_m7",
                "metodo_sequenciamento_parada_m7",
                "ajuste_fronteira_m7",
                "trocas_fronteira_m7",
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
            f"prioridade_parada_bucket={_safe_int(row.get('bucket_prioridade_m7'), 9)}; "
            f"folga_min_parada={_safe_float(row.get('folga_min_m7'), 9999.0):.2f}; "
            f"peso_total_parada={_safe_float(row.get('peso_total_m7'), 0.0):.2f}; "
            f"cidade_parada={_safe_text(row.get('cidade_ref_m7'))}; "
            f"ajuste_fronteira={str(bool(row.get('ajuste_fronteira_m7', False))).lower()}; "
            f"trocas_fronteira={_safe_int(row.get('trocas_fronteira_m7'), 0)}; "
            f"criterio_doc={_montar_justificativa_doc(row)}"
        ),
        axis=1,
    )

    auditoria_local = {
        "ordem_inicial_paradas_m7": ordem_inicial,
        "ordem_final_paradas_m7": ordem_final,
        "trocas_fronteira_m7": int(trocas_fronteira),
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
    del time_limit_seconds  # M7 novo é determinístico e não usa solver com timeout

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

            fator_real_manifesto = _inferir_fator_rodoviario_do_manifesto(
                df_manifesto=grupo,
                fallback=fator_km_rodoviario_m7,
            )

            grupo_seq, df_paradas_seq, auditoria_local = _ordenar_paradas_hierarquico_com_fronteira(
                df_manifesto=grupo,
                col_manifesto="manifesto_id",
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
                    "trocas_fronteira_m7": int(auditoria_local["trocas_fronteira_m7"]),
                }
            )

            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "resultado": "ok",
                    "motivo": "sequenciamento_realizado",
                    "qtd_docs": int(len(grupo_seq)),
                    "qtd_paradas": int(df_paradas_seq.shape[0]),
                    "trocas_fronteira_m7": int(auditoria_local["trocas_fronteira_m7"]),
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
                    "trocas_fronteira_m7": 0,
                }
            )

            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "resultado": "fallback",
                    "motivo": str(e),
                    "qtd_docs": int(len(grupo_fallback)),
                    "qtd_paradas": int(grupo_fallback["chave_parada_seq_m7"].nunique()),
                    "trocas_fronteira_m7": 0,
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
        "metodo_m7": "heuristica_hierarquica_cidades_com_validacao_local_fronteira",
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
        "total_trocas_fronteira_m7": int(df_manifestos_sequenciamento_resumo_m7["trocas_fronteira_m7"].fillna(0).sum())
        if not df_manifestos_sequenciamento_resumo_m7.empty and "trocas_fronteira_m7" in df_manifestos_sequenciamento_resumo_m7.columns
        else 0,
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
