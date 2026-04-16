from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import math
import unicodedata

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


# =========================================================================================
# M7 - SEQUENCIAMENTO DE ENTREGAS COM OR-TOOLS
# -----------------------------------------------------------------------------------------
# OBJETIVO
# - sequenciar as entregas dentro de cada manifesto fechado do M6.2
# - preservar a composição do manifesto
# - reduzir deslocamento entre paradas sem quebrar prioridade operacional
# - gerar ordem de entrega e ordem reversa de carregamento
#
# PRINCÍPIOS
# - não altera composição do manifesto
# - não puxa remanescente
# - não cria novo manifesto
# - mesma parada permanece agrupada
# - prioridade operacional entra antes do solver
# - OR-Tools organiza o caminho entre as paradas
#
# ENTRADAS ESPERADAS
# - df_manifestos_m6_2
# - df_itens_manifestos_m6_2
# - df_geo_tratado ou df_geo_raw
# - data_base_roteirizacao
# - tipo_roteirizacao
# - caminhos_pipeline
#
# SAÍDAS
# - df_itens_manifestos_sequenciados_m7
# - df_manifestos_sequenciamento_resumo_m7
# - df_tentativas_sequenciamento_m7
# - df_diagnostico_recuperacao_coordenadas_m7
# =========================================================================================


TIME_LIMIT_SECONDS_PADRAO = 5


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


def _normalizar_num(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce")


def _normalizar_texto(valor: Any) -> str:
    if pd.isna(valor):
        return ""
    return str(valor).strip()


def _remover_acentos(texto: Any) -> str:
    if pd.isna(texto):
        return ""
    texto = str(texto)
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )


def _chave_texto(valor: Any) -> str:
    return _remover_acentos(_normalizar_texto(valor)).upper()


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
            f"Esperado um destes nomes: {candidatos}. "
            f"Corrija o contrato do módulo anterior."
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
        raise Exception(
            f"M7 encontrou colunas obrigatórias ausentes em {nome_df}: {faltando}"
        )


# =========================================================================================
# HELPERS GEO
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


def _construir_matriz_distancias(coords: List[Tuple[float, float]]) -> np.ndarray:
    n = len(coords)
    matriz = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                matriz[i, j] = 0.0
            else:
                matriz[i, j] = _haversine_km(
                    coords[i][0], coords[i][1],
                    coords[j][0], coords[j][1]
                )

    return matriz


# =========================================================================================
# PRIORIDADE OPERACIONAL
# =========================================================================================
def _classificar_prioridade_negocio(row: pd.Series) -> Tuple[int, float, float]:
    """
    Bucket menor = maior prioridade operacional.

    Regras:
    1) agendadas
    2) dentro das agendadas: menor folga primeiro
    3) depois não agendadas urgentes
    4) por fim não agendadas normais
    5) maior peso ajuda no desempate
    """
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
# NORMALIZAÇÃO DAS BASES
# =========================================================================================
def _normalizar_manifestos(df_manifestos_m6_2: pd.DataFrame) -> pd.DataFrame:
    out = df_manifestos_m6_2.copy()

    obrigatorias = [
        "manifesto_id",
    ]
    _validar_colunas(out, obrigatorias, "df_manifestos_m6_2")

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
        "latitude_filial",
        "longitude_filial",
    ]
    out = _garantir_colunas(out, colunas_minimas)

    _validar_colunas(
        out,
        ["manifesto_id", "id_linha_pipeline", "destinatario", "cidade", "uf"],
        "df_itens_manifestos_m6_2",
    )

    col_lat_dest = _resolver_coluna_existente(
        out,
        ["latitude_destinatario", "latitude_destino"],
        "latitude_destinatario",
        obrigatoria=False,
    )
    if col_lat_dest == "":
        out["latitude_destinatario"] = np.nan
        col_lat_dest = "latitude_destinatario"

    col_lon_dest = _resolver_coluna_existente(
        out,
        ["longitude_destinatario", "longitude_destino"],
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
        "latitude_filial",
        "longitude_filial",
        col_lat_dest,
        col_lon_dest,
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["agendada_norm"] = out["agendada"].apply(_to_bool)
    out["folga_dias_norm"] = pd.to_numeric(out["folga_dias"], errors="coerce")
    out["peso_kg_norm"] = pd.to_numeric(out["peso_kg"], errors="coerce").fillna(
        pd.to_numeric(out["peso_calculado"], errors="coerce")
    )

    out["latitude_dest_norm_m7"] = out[col_lat_dest]
    out["longitude_dest_norm_m7"] = out[col_lon_dest]

    out = out[(out["manifesto_id"] != "") & (out["id_linha_pipeline"] != "")].copy()

    if out["id_linha_pipeline"].duplicated().any():
        duplicados = out.loc[out["id_linha_pipeline"].duplicated(), "id_linha_pipeline"].astype(str).tolist()[:20]
        raise Exception(
            f"M7 recebeu id_linha_pipeline duplicado em df_itens_manifestos_m6_2: {duplicados}"
        )

    return out.reset_index(drop=True)


def _normalizar_geo(df_geo: pd.DataFrame) -> pd.DataFrame:
    geo = df_geo.copy()
    geo.columns = [str(c).strip().lower() for c in geo.columns]

    col_cidade = _resolver_coluna_existente(
        geo,
        ["nome", "cidade", "mun_uf"],
        "cidade na base geo",
        obrigatoria=True,
    )
    col_uf = _resolver_coluna_existente(
        geo,
        ["uf"],
        "uf na base geo",
        obrigatoria=True,
    )
    col_lat = _resolver_coluna_existente(
        geo,
        ["latitude", "lat"],
        "latitude na base geo",
        obrigatoria=True,
    )
    col_lon = _resolver_coluna_existente(
        geo,
        ["longitude", "lon"],
        "longitude na base geo",
        obrigatoria=True,
    )

    geo[col_lat] = pd.to_numeric(geo[col_lat], errors="coerce")
    geo[col_lon] = pd.to_numeric(geo[col_lon], errors="coerce")

    if col_cidade == "mun_uf":
        geo["cidade_geo_base_m7"] = geo["mun_uf"].astype(str).str.split("-").str[0].str.strip()
    else:
        geo["cidade_geo_base_m7"] = geo[col_cidade].astype(str).str.strip()

    geo["cidade_geo_chave_m7"] = geo["cidade_geo_base_m7"].apply(_chave_texto)
    geo["uf_geo_chave_m7"] = geo[col_uf].apply(_chave_texto)

    geo_lookup = (
        geo[
            [
                "cidade_geo_chave_m7",
                "uf_geo_chave_m7",
                col_lat,
                col_lon,
            ]
        ]
        .dropna(subset=[col_lat, col_lon])
        .drop_duplicates(subset=["cidade_geo_chave_m7", "uf_geo_chave_m7"])
        .rename(
            columns={
                col_lat: "latitude_rec_geo_m7",
                col_lon: "longitude_rec_geo_m7",
            }
        )
        .reset_index(drop=True)
    )

    return geo_lookup


# =========================================================================================
# RECUPERAÇÃO DE COORDENADAS
# =========================================================================================
def _recuperar_coordenadas_destino(
    df_itens: pd.DataFrame,
    geo_lookup: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df_itens.copy()

    out["cidade_chave_seq7"] = out["cidade"].apply(_chave_texto)
    out["uf_chave_seq7"] = out["uf"].apply(_chave_texto)

    lat_nulos_antes = int(out["latitude_dest_norm_m7"].isna().sum())
    lon_nulos_antes = int(out["longitude_dest_norm_m7"].isna().sum())

    out = out.merge(
        geo_lookup,
        left_on=["cidade_chave_seq7", "uf_chave_seq7"],
        right_on=["cidade_geo_chave_m7", "uf_geo_chave_m7"],
        how="left",
    )

    mask_lat = out["latitude_dest_norm_m7"].isna() & out["latitude_rec_geo_m7"].notna()
    mask_lon = out["longitude_dest_norm_m7"].isna() & out["longitude_rec_geo_m7"].notna()

    out.loc[mask_lat, "latitude_dest_norm_m7"] = out.loc[mask_lat, "latitude_rec_geo_m7"]
    out.loc[mask_lon, "longitude_dest_norm_m7"] = out.loc[mask_lon, "longitude_rec_geo_m7"]

    lat_nulos_depois = int(out["latitude_dest_norm_m7"].isna().sum())
    lon_nulos_depois = int(out["longitude_dest_norm_m7"].isna().sum())

    out["coord_dest_origem_m7"] = "original"
    out.loc[mask_lat | mask_lon, "coord_dest_origem_m7"] = "recuperada_via_geo"

    out["status_coord_dest_m7"] = np.where(
        out["latitude_rec_geo_m7"].notna() & out["longitude_rec_geo_m7"].notna(),
        "recuperavel_geo",
        "nao_encontrado_geo",
    )

    diagnostico = pd.DataFrame(
        [
            {"indicador": "lat_nulos_antes", "valor": lat_nulos_antes},
            {"indicador": "lon_nulos_antes", "valor": lon_nulos_antes},
            {"indicador": "lat_nulos_depois", "valor": lat_nulos_depois},
            {"indicador": "lon_nulos_depois", "valor": lon_nulos_depois},
            {
                "indicador": "linhas_recuperadas_via_geo",
                "valor": int((out["coord_dest_origem_m7"] == "recuperada_via_geo").sum()),
            },
        ]
    )

    return out.reset_index(drop=True), diagnostico.reset_index(drop=True)


# =========================================================================================
# ORDENAÇÃO DE PARADAS POR REGRA + ORTOOLS
# =========================================================================================
def _ordenar_paradas_por_regra_e_orto(
    df_manifesto: pd.DataFrame,
    col_manifesto: str,
    col_doc: str,
    time_limit_seconds: int,
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
        lat_ref = pd.to_numeric(gpar["latitude_dest_norm_m7"], errors="coerce").mean()
        lon_ref = pd.to_numeric(gpar["longitude_dest_norm_m7"], errors="coerce").mean()

        registros_paradas.append(
            {
                "chave_parada_seq_m7": chave_parada,
                "lat_ref_m7": lat_ref,
                "lon_ref_m7": lon_ref,
                "bucket_prioridade_m7": score["bucket_prioridade"],
                "folga_min_m7": score["folga_min"],
                "peso_total_m7": score["peso_total"],
                "qtd_docs_parada_m7": int(len(gpar)),
            }
        )

    df_paradas = pd.DataFrame(registros_paradas).reset_index(drop=True)

    if df_paradas["lat_ref_m7"].isna().any() or df_paradas["lon_ref_m7"].isna().any():
        raise Exception(
            f"Manifesto {grupo[col_manifesto].iloc[0]} possui parada sem coordenada após recuperação."
        )

    if len(df_paradas) == 1:
        df_paradas["ordem_entrega_parada_m7"] = 1
        df_paradas["metodo_sequenciamento_parada_m7"] = "parada_unica"
    else:
        lat_origem = pd.to_numeric(grupo["latitude_filial"], errors="coerce").dropna()
        lon_origem = pd.to_numeric(grupo["longitude_filial"], errors="coerce").dropna()

        if len(lat_origem) == 0 or len(lon_origem) == 0:
            raise Exception(
                f"Manifesto {grupo[col_manifesto].iloc[0]} sem coordenada de filial válida."
            )

        origem = (float(lat_origem.iloc[0]), float(lon_origem.iloc[0]))
        coords_clientes = list(zip(df_paradas["lat_ref_m7"], df_paradas["lon_ref_m7"]))
        coords = [origem] + coords_clientes

        matriz = _construir_matriz_distancias(coords)

        manager = pywrapcp.RoutingIndexManager(len(coords), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        prioridade_por_no = {0: 0}
        for idx, row in df_paradas.reset_index(drop=True).iterrows():
            prioridade_por_no[idx + 1] = (
                int(row["bucket_prioridade_m7"]) * 10000
                + int(float(row["folga_min_m7"]) * 100)
                - int(float(row["peso_total_m7"]) / 10.0)
            )

        def distance_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)

            distancia_base = matriz[from_node][to_node]
            penalidade = prioridade_por_no.get(to_node, 0)

            custo = int(distancia_base * 1000) + penalidade
            return max(custo, 0)

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

    grupo = grupo.merge(
        df_paradas[
            [
                "chave_parada_seq_m7",
                "ordem_entrega_parada_m7",
                "bucket_prioridade_m7",
                "folga_min_m7",
                "peso_total_m7",
                "metodo_sequenciamento_parada_m7",
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
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    if not isinstance(df_manifestos_m6_2, pd.DataFrame) or df_manifestos_m6_2.empty:
        raise Exception("M7 recebeu df_manifestos_m6_2 vazio.")

    if not isinstance(df_itens_manifestos_m6_2, pd.DataFrame) or df_itens_manifestos_m6_2.empty:
        raise Exception("M7 recebeu df_itens_manifestos_m6_2 vazio.")

    if isinstance(df_geo_tratado, pd.DataFrame) and not df_geo_tratado.empty:
        df_geo_base = df_geo_tratado.copy()
        nome_base_geo = "df_geo_tratado"
    elif isinstance(df_geo_raw, pd.DataFrame) and not df_geo_raw.empty:
        df_geo_base = df_geo_raw.copy()
        nome_base_geo = "df_geo_raw"
    else:
        raise Exception("M7 não encontrou base geo válida. Esperado df_geo_tratado ou df_geo_raw.")

    df_manifestos = _normalizar_manifestos(df_manifestos_m6_2)
    df_itens = _normalizar_itens(df_itens_manifestos_m6_2)
    geo_lookup = _normalizar_geo(df_geo_base)

    manifestos_validos = set(df_manifestos["manifesto_id"].astype(str))
    df_itens = df_itens.loc[df_itens["manifesto_id"].astype(str).isin(manifestos_validos)].copy()

    df_itens, df_diagnostico_recuperacao_coordenadas_m7 = _recuperar_coordenadas_destino(
        df_itens=df_itens,
        geo_lookup=geo_lookup,
    )

    resultados: List[pd.DataFrame] = []
    resumos_manifestos: List[Dict[str, Any]] = []
    tentativas: List[Dict[str, Any]] = []

    for manifesto_id, grupo in df_itens.groupby("manifesto_id", dropna=False):
        grupo = grupo.copy().reset_index(drop=True)

        try:
            if grupo["latitude_dest_norm_m7"].isna().any() or grupo["longitude_dest_norm_m7"].isna().any():
                raise Exception(
                    f"Manifesto {manifesto_id} ainda possui coordenada de destino nula após recuperação."
                )

            grupo_seq, df_paradas_seq = _ordenar_paradas_por_regra_e_orto(
                df_manifesto=grupo,
                col_manifesto="manifesto_id",
                col_doc="id_linha_pipeline",
                time_limit_seconds=time_limit_seconds,
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
        "base_geo_utilizada_m7": nome_base_geo,
        "time_limit_seconds_m7": int(time_limit_seconds),
        "manifestos_entrada_m7": int(df_manifestos["manifesto_id"].nunique()),
        "itens_entrada_m7": int(len(df_itens)),
        "manifestos_saida_m7": int(df_itens_manifestos_sequenciados_m7["manifesto_id"].nunique())
        if not df_itens_manifestos_sequenciados_m7.empty
        else 0,
        "itens_saida_m7": int(len(df_itens_manifestos_sequenciados_m7)),
        "fallbacks_m7": int(
            (df_tentativas_sequenciamento_m7["resultado"] == "fallback").sum()
        ) if not df_tentativas_sequenciamento_m7.empty else 0,
        "linhas_recuperadas_via_geo_m7": int(
            (df_itens_manifestos_sequenciados_m7["coord_dest_origem_m7"] == "recuperada_via_geo").sum()
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


# =========================================================================================
# ALIASES DEFENSIVOS
# =========================================================================================
def executar_m7(*args: Any, **kwargs: Any):
    return executar_m7_sequenciamento_entregas(*args, **kwargs)


def processar_m7_sequenciamento_entregas(*args: Any, **kwargs: Any):
    return executar_m7_sequenciamento_entregas(*args, **kwargs)


def rodar_m7_sequenciamento_entregas(*args: Any, **kwargs: Any):
    return executar_m7_sequenciamento_entregas(*args, **kwargs)
