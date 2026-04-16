from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except Exception as e:
    raise Exception(
        "OR-Tools não está disponível no ambiente.\n"
        f"Erro: {e}\n"
        "Instale com: pip install ortools"
    )


TIME_LIMIT_SECONDS_PADRAO = 5

# =========================================================================================
# AJUSTE FINO DO ORTOOLS
# -----------------------------------------------------------------------------------------
# REGRA AJUSTADA:
# - distância operacional deve mandar na rota
# - bucket / folga / peso entram apenas como ajuste fino
# - rota deve evitar sair de uma cidade e voltar depois
#
# ESCALA:
# - distância entra em metros (km * 1000)
# - penalidades abaixo são leves/moderadas
# =========================================================================================
PESO_BUCKET_ORTO = 60
PESO_FOLGA_ORTO = 6
PESO_PESO_PARADA_ORTO = 0.02

# Distância operacional do M7 deve se aproximar do M2:
# km_rodoviario_estimado = haversine * fator_rodoviario
FATOR_KM_RODOVIARIO_M7_PADRAO = 1.20

# Penalidades operacionais
PENALIDADE_TROCA_CIDADE_M7 = 2500        # 2,5 km equivalentes
PENALIDADE_REENTRADA_CIDADE_M7 = 4000    # 4,0 km equivalentes


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


def _construir_matriz_distancias(
    coords: List[Tuple[float, float]],
    fator_km_rodoviario: float,
) -> np.ndarray:
    n = len(coords)
    matriz = np.zeros((n, n), dtype=float)

    fator = float(fator_km_rodoviario)
    if pd.isna(fator) or fator <= 0:
        fator = FATOR_KM_RODOVIARIO_M7_PADRAO

    for i in range(n):
        for j in range(n):
            if i == j:
                matriz[i, j] = 0.0
            else:
                dist_hav = _haversine_km(
                    coords[i][0], coords[i][1],
                    coords[j][0], coords[j][1],
                )
                matriz[i, j] = dist_hav * fator
    return matriz


# =========================================================================================
# PRIORIDADE OPERACIONAL
# =========================================================================================
def _classificar_prioridade_negocio(row: pd.Series) -> Tuple[int, float, float]:
    agendada = bool(row.get("agendada_norm", False))
    folga = row.get("folga_dias_norm", np.nan)
    peso = row.get("peso_kg_norm", 0.0)

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
        f"peso={_safe_float(row.get('peso_kg_norm', 0.0), 0.0):.2f}kg"
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
        col_lat_filial,
        col_lon_filial,
        col_lat_dest,
        col_lon_dest,
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["agendada_norm"] = out["agendada"].apply(_to_bool)
    out["folga_dias_norm"] = pd.to_numeric(out["folga_dias"], errors="coerce")
    out["peso_kg_norm"] = pd.to_numeric(out["peso_kg"], errors="coerce").fillna(
        pd.to_numeric(out["peso_calculado"], errors="coerce")
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
# AJUSTE OPERACIONAL PÓS-ORTO
# -----------------------------------------------------------------------------------------
# Regra:
# - respeita a ordem-base do OR-Tools
# - dentro de "blocos próximos", prefere não reentrar em cidade já abandonada
# - não muda radicalmente a rota; apenas corrige reentradas feias
# =========================================================================================
def _ajustar_ordem_para_evitar_reentrada_cidade(df_paradas: pd.DataFrame) -> pd.DataFrame:
    if df_paradas.empty or len(df_paradas) <= 2:
        return df_paradas.copy()

    work = df_paradas.copy().reset_index(drop=True)
    work["cidade_ref_m7"] = work["cidade_ref_m7"].fillna("").astype(str).str.strip().str.upper()

    visitadas: List[str] = []
    saidas_definitivas: set[str] = set()
    ordem_final: List[int] = []
    restantes = list(work.index)

    atual = restantes.pop(0)
    ordem_final.append(atual)
    cidade_atual = work.loc[atual, "cidade_ref_m7"]
    visitadas.append(cidade_atual)

    while restantes:
        melhor_idx = None
        melhor_custo = None

        for idx in restantes:
            cidade_candidata = work.loc[idx, "cidade_ref_m7"]
            custo = float(work.loc[idx, "ordem_entrega_parada_m7"]) * 1000.0

            if cidade_candidata == cidade_atual:
                custo -= 500.0

            if cidade_candidata in saidas_definitivas:
                custo += PENALIDADE_REENTRADA_CIDADE_M7

            if melhor_custo is None or custo < melhor_custo:
                melhor_custo = custo
                melhor_idx = idx

        if melhor_idx is None:
            melhor_idx = restantes[0]

        proxima_cidade = work.loc[melhor_idx, "cidade_ref_m7"]

        if proxima_cidade != cidade_atual:
            ainda_tem_mesma_cidade = any(
                work.loc[idx, "cidade_ref_m7"] == cidade_atual for idx in restantes if idx != melhor_idx
            )
            if not ainda_tem_mesma_cidade and cidade_atual != "":
                saidas_definitivas.add(cidade_atual)

        ordem_final.append(melhor_idx)
        restantes.remove(melhor_idx)
        cidade_atual = proxima_cidade
        visitadas.append(cidade_atual)

    mapa_nova_ordem = {idx: pos + 1 for pos, idx in enumerate(ordem_final)}
    work["ordem_entrega_parada_m7"] = work.index.map(mapa_nova_ordem).astype(int)
    work["ajuste_reentrada_cidade_m7"] = True

    return work.sort_values("ordem_entrega_parada_m7").reset_index(drop=True)


# =========================================================================================
# ORDEM DAS PARADAS
# =========================================================================================
def _ordenar_paradas_por_regra_e_orto(
    df_manifesto: pd.DataFrame,
    col_manifesto: str,
    col_doc: str,
    time_limit_seconds: int,
    fator_km_rodoviario_m7: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grupo = df_manifesto.copy().reset_index(drop=True)

    grupo["chave_parada_seq_m7"] = (
        grupo["destinatario"].fillna("").astype(str).str.strip()
        + "|"
        + grupo["cidade"].fillna("").astype(str).str.strip()
        + "|"
        + grupo["uf"].fillna("").astype(str).str.strip()
    )

    registros_paradas: List[Dict[str, Any]] = []

    for chave_parada, gpar in grupo.groupby("chave_parada_seq_m7", dropna=False):
        score = _calcular_score_parada(gpar)
        lat_ref = pd.to_numeric(gpar["latitude_dest_m7"], errors="coerce").mean()
        lon_ref = pd.to_numeric(gpar["longitude_dest_m7"], errors="coerce").mean()
        cidade_ref = _safe_text(gpar["cidade"].dropna().iloc[0] if len(gpar["cidade"].dropna()) > 0 else "")
        uf_ref = _safe_text(gpar["uf"].dropna().iloc[0] if len(gpar["uf"].dropna()) > 0 else "")

        registros_paradas.append(
            {
                "chave_parada_seq_m7": chave_parada,
                "lat_ref_m7": lat_ref,
                "lon_ref_m7": lon_ref,
                "cidade_ref_m7": cidade_ref,
                "uf_ref_m7": uf_ref,
                "bucket_prioridade_m7": score["bucket_prioridade"],
                "folga_min_m7": score["folga_min"],
                "peso_total_m7": score["peso_total"],
                "qtd_docs_parada_m7": int(len(gpar)),
            }
        )

    df_paradas = pd.DataFrame(registros_paradas).reset_index(drop=True)

    if df_paradas["lat_ref_m7"].isna().any() or df_paradas["lon_ref_m7"].isna().any():
        raise Exception(
            f"Manifesto {grupo[col_manifesto].iloc[0]} possui parada sem coordenada de destino no contrato."
        )

    if len(df_paradas) == 1:
        df_paradas["ordem_entrega_parada_m7"] = 1
        df_paradas["metodo_sequenciamento_parada_m7"] = "parada_unica"
        df_paradas["ajuste_reentrada_cidade_m7"] = False
    else:
        lat_origem = pd.to_numeric(grupo["latitude_filial_m7"], errors="coerce").dropna()
        lon_origem = pd.to_numeric(grupo["longitude_filial_m7"], errors="coerce").dropna()

        if len(lat_origem) == 0 or len(lon_origem) == 0:
            raise Exception(
                f"Manifesto {grupo[col_manifesto].iloc[0]} sem coordenada de filial no contrato."
            )

        origem = (float(lat_origem.iloc[0]), float(lon_origem.iloc[0]))
        coords_clientes = list(zip(df_paradas["lat_ref_m7"], df_paradas["lon_ref_m7"]))
        coords = [origem] + coords_clientes

        matriz = _construir_matriz_distancias(
            coords=coords,
            fator_km_rodoviario=fator_km_rodoviario_m7,
        )

        manager = pywrapcp.RoutingIndexManager(len(coords), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        prioridade_por_no = {0: 0}
        cidade_por_no = {0: "__ORIGEM__"}

        for idx, row in df_paradas.reset_index(drop=True).iterrows():
            bucket_pen = int(row["bucket_prioridade_m7"]) * PESO_BUCKET_ORTO

            folga_val = float(row["folga_min_m7"])
            if pd.isna(folga_val) or folga_val >= 9999:
                folga_pen = 0
            else:
                folga_pen = int(max(0.0, 15.0 - folga_val) * PESO_FOLGA_ORTO)

            peso_bonus = int(float(row["peso_total_m7"]) * PESO_PESO_PARADA_ORTO)

            prioridade_por_no[idx + 1] = bucket_pen + folga_pen - peso_bonus
            cidade_por_no[idx + 1] = _safe_text(row["cidade_ref_m7"]).upper()

        def distance_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)

            distancia_base_km = matriz[from_node][to_node]
            distancia_base_m = int(distancia_base_km * 1000)

            penalidade_prioridade = int(prioridade_por_no.get(to_node, 0))

            cidade_from = cidade_por_no.get(from_node, "")
            cidade_to = cidade_por_no.get(to_node, "")
            penalidade_troca_cidade = 0
            if from_node != 0 and cidade_from != "" and cidade_to != "" and cidade_from != cidade_to:
                penalidade_troca_cidade = PENALIDADE_TROCA_CIDADE_M7

            custo = distancia_base_m + penalidade_prioridade + penalidade_troca_cidade
            return max(int(custo), 0)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = int(time_limit_seconds)

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            ordem_nodes: List[int] = []

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                ordem_nodes.append(node)
                index = solution.Value(routing.NextVar(index))

            ordem_clientes = [n for n in ordem_nodes if n != 0]

            mapa_ordem: Dict[str, int] = {}
            for ordem, node in enumerate(ordem_clientes, start=1):
                chave = df_paradas.iloc[node - 1]["chave_parada_seq_m7"]
                mapa_ordem[chave] = ordem

            df_paradas["ordem_entrega_parada_m7"] = df_paradas["chave_parada_seq_m7"].map(mapa_ordem)
            df_paradas["metodo_sequenciamento_parada_m7"] = "ortools"
            df_paradas["ajuste_reentrada_cidade_m7"] = False

            df_paradas = _ajustar_ordem_para_evitar_reentrada_cidade(df_paradas)
            if bool(df_paradas.get("ajuste_reentrada_cidade_m7", pd.Series([False])).any()):
                df_paradas["metodo_sequenciamento_parada_m7"] = "ortools_ajuste_cidade"
        else:
            df_paradas = df_paradas.sort_values(
                by=[
                    "bucket_prioridade_m7",
                    "folga_min_m7",
                    "peso_total_m7",
                ],
                ascending=[True, True, False],
                kind="mergesort",
            ).reset_index(drop=True)

            df_paradas["ordem_entrega_parada_m7"] = np.arange(1, len(df_paradas) + 1)
            df_paradas["metodo_sequenciamento_parada_m7"] = "fallback_regra"
            df_paradas["ajuste_reentrada_cidade_m7"] = False

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
                "ajuste_reentrada_cidade_m7",
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
            f"ajuste_reentrada_cidade={str(bool(row.get('ajuste_reentrada_cidade_m7', False))).lower()}; "
            f"criterio_doc={_montar_justificativa_doc(row)}"
        ),
        axis=1,
    )

    return grupo.reset_index(drop=True), df_paradas.reset_index(drop=True)


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

    if not isinstance(df_manifestos_m6_2, pd.DataFrame) or df_manifestos_m6_2.empty:
        raise Exception("M7 recebeu df_manifestos_m6_2 vazio.")

    if not isinstance(df_itens_manifestos_m6_2, pd.DataFrame) or df_itens_manifestos_m6_2.empty:
        raise Exception("M7 recebeu df_itens_manifestos_m6_2 vazio.")

    if pd.isna(fator_km_rodoviario_m7) or float(fator_km_rodoviario_m7) <= 0:
        fator_km_rodoviario_m7 = FATOR_KM_RODOVIARIO_M7_PADRAO

    df_manifestos = _normalizar_manifestos(df_manifestos_m6_2)
    df_itens = _normalizar_itens(df_itens_manifestos_m6_2)

    manifestos_validos = set(df_manifestos["manifesto_id"].astype(str))
    df_itens = df_itens.loc[df_itens["manifesto_id"].astype(str).isin(manifestos_validos)].copy()

    df_itens, df_diagnostico_recuperacao_coordenadas_m7 = _preparar_coordenadas_contrato(df_itens)

    resultados: List[pd.DataFrame] = []
    resumos_manifestos: List[Dict[str, Any]] = []
    tentativas: List[Dict[str, Any]] = []

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

            grupo_seq, df_paradas_seq = _ordenar_paradas_por_regra_e_orto(
                df_manifesto=grupo,
                col_manifesto="manifesto_id",
                col_doc="id_linha_pipeline",
                time_limit_seconds=time_limit_seconds,
                fator_km_rodoviario_m7=float(fator_km_rodoviario_m7),
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
                }
            )

            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "resultado": "ok",
                    "motivo": "sequenciamento_realizado",
                    "qtd_docs": int(len(grupo_seq)),
                    "qtd_paradas": int(df_paradas_seq.shape[0]),
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
                }
            )

            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "resultado": "fallback",
                    "motivo": str(e),
                    "qtd_docs": int(len(grupo_fallback)),
                    "qtd_paradas": int(grupo_fallback["chave_parada_seq_m7"].nunique()),
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
        "time_limit_seconds_m7": int(time_limit_seconds),
        "fator_km_rodoviario_m7": float(fator_km_rodoviario_m7),
        "pesos_ortools_m7": {
            "peso_bucket": PESO_BUCKET_ORTO,
            "peso_folga": PESO_FOLGA_ORTO,
            "peso_peso_parada": PESO_PESO_PARADA_ORTO,
            "penalidade_troca_cidade": PENALIDADE_TROCA_CIDADE_M7,
            "penalidade_reentrada_cidade": PENALIDADE_REENTRADA_CIDADE_M7,
            "distancia_dominante": True,
        },
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
