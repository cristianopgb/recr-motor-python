from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

import pandas as pd


# =========================================================================================
# M5.4B - ADERÊNCIA DE FROTA POR SUBREGIÃO
# -----------------------------------------------------------------------------------------
# OBJETIVO
# - receber a base preparada do M5.4A
# - testar veículos do maior para o menor
# - reduzir o universo de tentativa para o M5.4C
#
# ESTA ETAPA NÃO:
# - não fecha pré-manifesto
# - não remove itens do pool
# - não compõe definitivamente
#
# ELA DEVE:
# - definir cidade âncora
# - testar aderência de cada perfil
# - montar uma base teórica de tentativa
# - devolver veículos aderentes e não aderentes, com motivo
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
# Geometria
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
    df_base_preparada_m5_4a: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = df_base_preparada_m5_4a.copy() if df_base_preparada_m5_4a is not None else pd.DataFrame()
    veiculos = df_veiculos_tratados.copy() if df_veiculos_tratados is not None else pd.DataFrame()

    if base.empty:
        return base, veiculos

    defaults_base = {
        "id_linha_pipeline": None,
        "subregiao": "",
        "uf": "",
        "cidade": "",
        "destinatario": "",
        "peso_calculado": 0.0,
        "vol_m3": 0.0,
        "distancia_rodoviaria_est_km": 0.0,
        "restricao_veiculo": None,
        "latitude_destinatario": pd.NA,
        "longitude_destinatario": pd.NA,
        "ordem_subregiao_m5_4a": 0,
        "ordem_cidade_na_subregiao_m5_4a": 0,
        "ordem_cliente_na_cidade_m5_4a": 0,
        "peso_total_subregiao_m5_4a": 0.0,
        "peso_total_cidade_m5_4a": 0.0,
        "peso_total_cliente_m5_4a": 0.0,
    }
    for col, default in defaults_base.items():
        _ensure_column(base, col, default)

    if base["id_linha_pipeline"].isna().any():
        raise ValueError("M5.4B exige id_linha_pipeline em todas as linhas da base preparada do M5.4A.")

    for col in [
        "peso_calculado",
        "vol_m3",
        "distancia_rodoviaria_est_km",
        "latitude_destinatario",
        "longitude_destinatario",
        "ordem_subregiao_m5_4a",
        "ordem_cidade_na_subregiao_m5_4a",
        "ordem_cliente_na_cidade_m5_4a",
        "peso_total_subregiao_m5_4a",
        "peso_total_cidade_m5_4a",
        "peso_total_cliente_m5_4a",
    ]:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    for col in ["subregiao", "uf", "cidade", "destinatario"]:
        base[col] = base[col].fillna("").astype(str).str.strip()

    _ensure_column(veiculos, "perfil", "")
    _ensure_column(veiculos, "tipo", "")
    for col in [
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ocupacao_minima_perc",
        "ocupacao_maxima_perc",
    ]:
        _ensure_column(veiculos, col, pd.NA)
        veiculos[col] = pd.to_numeric(veiculos[col], errors="coerce")

    for col in ["perfil", "tipo"]:
        veiculos[col] = veiculos[col].fillna("").astype(str).str.strip()

    veiculos = (
        veiculos[
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

    return base, veiculos


# -----------------------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------------------
def _peso_total(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(pd.to_numeric(df["peso_calculado"], errors="coerce").fillna(0).sum())


def _volume_total(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(pd.to_numeric(df["vol_m3"], errors="coerce").fillna(0).sum())


def _km_referencia(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(pd.to_numeric(df["distancia_rodoviaria_est_km"], errors="coerce").fillna(0).max())


def _qtd_paradas(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df["destinatario"].fillna("").astype(str).nunique())


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
# Veículos
# -----------------------------------------------------------------------------------------
def _veiculos_maior_para_menor(df_veiculos: pd.DataFrame) -> pd.DataFrame:
    temp = df_veiculos.copy()
    temp["_cap_peso"] = pd.to_numeric(temp["capacidade_peso_kg"], errors="coerce").fillna(0)
    temp["_cap_vol"] = pd.to_numeric(temp["capacidade_vol_m3"], errors="coerce").fillna(0)

    return (
        temp.sort_values(
            by=["_cap_peso", "_cap_vol", "tipo", "perfil"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        .drop(columns=["_cap_peso", "_cap_vol"], errors="ignore")
        .reset_index(drop=True)
        .copy()
    )


# -----------------------------------------------------------------------------------------
# Cliente / restrição
# -----------------------------------------------------------------------------------------
def _cliente_respeita_restricao_veiculo(df_cliente: pd.DataFrame, vehicle_row: pd.Series) -> bool:
    tipo_veiculo = _safe_text(vehicle_row.get("tipo")) or _safe_text(vehicle_row.get("perfil"))
    if not tipo_veiculo:
        return True

    if "restricao_veiculo" not in df_cliente.columns:
        return True

    restricoes = df_cliente["restricao_veiculo"].dropna().astype(str).str.strip()
    if restricoes.empty:
        return True

    for restr in restricoes.unique().tolist():
        if restr and restr.lower() not in {"", "nan", "none"}:
            if restr.lower() != tipo_veiculo.lower():
                return False

    return True


# -----------------------------------------------------------------------------------------
# Distância entre cidades destino
# -----------------------------------------------------------------------------------------
def _centroide_cidade(df_cidade: pd.DataFrame) -> Tuple[float, float]:
    lat = pd.to_numeric(df_cidade["latitude_destinatario"], errors="coerce").dropna()
    lon = pd.to_numeric(df_cidade["longitude_destinatario"], errors="coerce").dropna()
    if lat.empty or lon.empty:
        return (math.nan, math.nan)
    return (float(lat.mean()), float(lon.mean()))


def _ordenar_cidades_da_subregiao_por_expansao(
    df_sub: pd.DataFrame,
    cidade_anchor: str,
) -> List[str]:
    cidades = []
    for cidade, grupo in df_sub.groupby("cidade", dropna=False, sort=False):
        lat, lon = _centroide_cidade(grupo)
        cidades.append(
            {
                "cidade": _safe_text(cidade),
                "peso_total_cidade": _peso_total(grupo),
                "lat": lat,
                "lon": lon,
            }
        )

    df_cidades = pd.DataFrame(cidades)
    if df_cidades.empty:
        return []

    if cidade_anchor not in df_cidades["cidade"].tolist():
        return df_cidades.sort_values(
            by=["peso_total_cidade", "cidade"],
            ascending=[False, True],
            kind="mergesort",
        )["cidade"].tolist()

    anchor_row = df_cidades[df_cidades["cidade"] == cidade_anchor].iloc[0]
    lat_anchor = _safe_float(anchor_row.get("lat"), math.nan)
    lon_anchor = _safe_float(anchor_row.get("lon"), math.nan)

    def _dist(row: pd.Series) -> float:
        if _safe_text(row.get("cidade")) == cidade_anchor:
            return 0.0
        return _haversine_km(
            _safe_float(row.get("lat"), math.nan),
            _safe_float(row.get("lon"), math.nan),
            lat_anchor,
            lon_anchor,
        )

    df_cidades["_dist_ancora"] = df_cidades.apply(_dist, axis=1)

    df_cidades = df_cidades.sort_values(
        by=["_dist_ancora", "peso_total_cidade", "cidade"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return df_cidades["cidade"].tolist()


# -----------------------------------------------------------------------------------------
# Base teórica de tentativa por veículo
# -----------------------------------------------------------------------------------------
def _montar_base_teorica_para_veiculo(
    df_sub: pd.DataFrame,
    cidade_anchor: str,
    vehicle_row: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cap_peso = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    cap_vol = _safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0)
    max_entregas = _safe_int(vehicle_row.get("max_entregas"), 0)
    max_km = _safe_float(vehicle_row.get("max_km_distancia"), 0.0)
    ocup_min_kg = _ocupacao_minima_kg(vehicle_row)
    ocup_max_kg = _ocupacao_maxima_kg(vehicle_row)

    if df_sub.empty:
        return pd.DataFrame(), {
            "aderente": False,
            "motivo": "subregiao_vazia",
        }

    if cidade_anchor not in df_sub["cidade"].astype(str).tolist():
        return pd.DataFrame(), {
            "aderente": False,
            "motivo": "cidade_anchor_nao_encontrada",
        }

    cidades_ordenadas = _ordenar_cidades_da_subregiao_por_expansao(df_sub, cidade_anchor)

    blocos_clientes: List[pd.DataFrame] = []
    for cidade in cidades_ordenadas:
        df_cidade = df_sub[df_sub["cidade"] == cidade].copy()
        if df_cidade.empty:
            continue

        clientes = (
            df_cidade.groupby("destinatario", dropna=False, sort=False)["peso_total_cliente_m5_4a"]
            .max()
            .reset_index()
            .sort_values(by=["peso_total_cliente_m5_4a", "destinatario"], ascending=[False, True], kind="mergesort")
        )

        for _, row_cli in clientes.iterrows():
            destinatario = _safe_text(row_cli.get("destinatario"))
            df_cliente = df_cidade[df_cidade["destinatario"] == destinatario].copy()
            if df_cliente.empty:
                continue
            blocos_clientes.append(df_cliente)

    if not blocos_clientes:
        return pd.DataFrame(), {
            "aderente": False,
            "motivo": "sem_blocos_cliente",
        }

    selecionados: List[pd.DataFrame] = []

    for df_cliente in blocos_clientes:
        if not _cliente_respeita_restricao_veiculo(df_cliente, vehicle_row):
            continue

        tentativa = pd.concat(selecionados + [df_cliente], ignore_index=True) if selecionados else df_cliente.copy()

        peso_tent = _peso_total(tentativa)
        vol_tent = _volume_total(tentativa)
        km_tent = _km_referencia(tentativa)
        paradas_tent = _qtd_paradas(tentativa)

        if cap_peso > 0 and peso_tent > cap_peso:
            continue
        if ocup_max_kg > 0 and peso_tent > ocup_max_kg:
            continue
        if cap_vol > 0 and vol_tent > cap_vol:
            continue
        if max_entregas > 0 and paradas_tent > max_entregas:
            continue
        if max_km > 0 and km_tent > max_km:
            continue

        selecionados.append(df_cliente)

    if not selecionados:
        return pd.DataFrame(), {
            "aderente": False,
            "motivo": "nenhum_cliente_aderente",
        }

    base_tentativa = pd.concat(selecionados, ignore_index=True)

    peso_final = _peso_total(base_tentativa)
    vol_final = _volume_total(base_tentativa)
    km_final = _km_referencia(base_tentativa)
    paradas_final = _qtd_paradas(base_tentativa)

    aderente = peso_final >= ocup_min_kg
    motivo = "ok" if aderente else "peso_montavel_abaixo_ocupacao_minima"

    return base_tentativa, {
        "aderente": aderente,
        "motivo": motivo,
        "peso_montavel_teorico": round(peso_final, 3),
        "volume_montavel_teorico": round(vol_final, 3),
        "km_referencia_teorico": round(km_final, 2),
        "qtd_paradas_teorico": int(paradas_final),
        "ocupacao_minima_kg": round(ocup_min_kg, 3),
        "ocupacao_maxima_kg": round(ocup_max_kg, 3),
    }


# -----------------------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------------------
def executar_m5_4b_aderencia_frota_subregioes(
    df_base_preparada_m5_4a: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
    rodada_id: Optional[str] = None,
    data_base_roteirizacao: Optional[Any] = None,
    tipo_roteirizacao: str = "carteira",
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    del rodada_id, kwargs

    base, veiculos = _normalizar_inputs(
        df_base_preparada_m5_4a=df_base_preparada_m5_4a,
        df_veiculos_tratados=df_veiculos_tratados,
    )

    if base.empty:
        outputs_vazio = {
            "df_frota_aderente_m5_4b": pd.DataFrame(),
            "df_frota_nao_aderente_m5_4b": pd.DataFrame(),
            "df_base_tentativas_m5_4b": pd.DataFrame(),
            "df_cidades_candidatas_m5_4b": pd.DataFrame(),
        }
        meta_vazio = {
            "resumo_m5_4b": {
                "modulo": "M5.4B",
                "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
                "tipo_roteirizacao": tipo_roteirizacao,
                "linhas_entrada_m5_4b": 0,
                "total_registros_frota_aderente_m5_4b": 0,
                "total_registros_frota_nao_aderente_m5_4b": 0,
                "total_linhas_base_tentativas_m5_4b": 0,
                "total_cidades_candidatas_m5_4b": 0,
                "estrategia_m5_4b": [
                    "teste_de_veiculos_do_maior_para_o_menor",
                    "base_teorica_sem_composicao_final",
                    "cidade_ancora_por_maior_massa",
                    "expansao_por_menor_distancia_entre_destinos",
                    "VERSAO_M5_4B_2026_04_12",
                ],
                "caminhos_pipeline": caminhos_pipeline or {},
            }
        }
        return outputs_vazio, meta_vazio

    veiculos_ord = _veiculos_maior_para_menor(veiculos)

    frota_aderente_rows: List[Dict[str, Any]] = []
    frota_nao_aderente_rows: List[Dict[str, Any]] = []
    base_tentativas_list: List[pd.DataFrame] = []
    cidades_candidatas_rows: List[Dict[str, Any]] = []

    # Processa uma cidade âncora por vez dentro de cada subregião
    subregioes = (
        base[["subregiao", "uf", "ordem_subregiao_m5_4a"]]
        .drop_duplicates()
        .sort_values(by=["ordem_subregiao_m5_4a", "subregiao", "uf"], ascending=[True, True, True], kind="mergesort")
    )

    for _, sub_row in subregioes.iterrows():
        subregiao = _safe_text(sub_row.get("subregiao"))
        uf = _safe_text(sub_row.get("uf"))

        df_sub = base[(base["subregiao"] == subregiao) & (base["uf"] == uf)].copy()
        if df_sub.empty:
            continue

        cidades_anchor = (
            df_sub[["cidade", "ordem_cidade_na_subregiao_m5_4a", "peso_total_cidade_m5_4a"]]
            .drop_duplicates()
            .sort_values(
                by=["ordem_cidade_na_subregiao_m5_4a", "peso_total_cidade_m5_4a", "cidade"],
                ascending=[True, False, True],
                kind="mergesort",
            )
        )

        for _, anchor_row in cidades_anchor.iterrows():
            cidade_anchor = _safe_text(anchor_row.get("cidade"))
            ordem_cidade = _safe_int(anchor_row.get("ordem_cidade_na_subregiao_m5_4a"), 0)

            cidades_expansao = _ordenar_cidades_da_subregiao_por_expansao(df_sub, cidade_anchor)
            for ordem_expansao, cidade_cand in enumerate(cidades_expansao, start=1):
                cidades_candidatas_rows.append(
                    {
                        "subregiao": subregiao,
                        "uf": uf,
                        "cidade_anchor": cidade_anchor,
                        "ordem_cidade_anchor_m5_4b": ordem_cidade,
                        "cidade_candidata": cidade_cand,
                        "ordem_expansao_m5_4b": ordem_expansao,
                    }
                )

            for ordem_perfil, (_, vehicle_row) in enumerate(veiculos_ord.iterrows(), start=1):
                base_tentativa, info = _montar_base_teorica_para_veiculo(
                    df_sub=df_sub,
                    cidade_anchor=cidade_anchor,
                    vehicle_row=vehicle_row,
                )

                registro = {
                    "subregiao": subregiao,
                    "uf": uf,
                    "cidade_anchor": cidade_anchor,
                    "ordem_cidade_anchor_m5_4b": ordem_cidade,
                    "perfil": _safe_text(vehicle_row.get("perfil")),
                    "tipo": _safe_text(vehicle_row.get("tipo")),
                    "ordem_tentativa_perfil_m5_4b": ordem_perfil,
                    "capacidade_peso_kg": round(_safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0), 3),
                    "capacidade_vol_m3": round(_safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0), 3),
                    "max_entregas": _safe_int(vehicle_row.get("max_entregas"), 0),
                    "max_km_distancia": round(_safe_float(vehicle_row.get("max_km_distancia"), 0.0), 3),
                    "ocupacao_minima_perc": round(_safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0), 3),
                    "ocupacao_maxima_perc": round(_safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0), 3),
                    "ocupacao_minima_kg": round(_ocupacao_minima_kg(vehicle_row), 3),
                    "ocupacao_maxima_kg": round(_ocupacao_maxima_kg(vehicle_row), 3),
                    "peso_montavel_teorico": round(_safe_float(info.get("peso_montavel_teorico"), 0.0), 3),
                    "volume_montavel_teorico": round(_safe_float(info.get("volume_montavel_teorico"), 0.0), 3),
                    "km_referencia_teorico": round(_safe_float(info.get("km_referencia_teorico"), 0.0), 2),
                    "qtd_paradas_teorico": _safe_int(info.get("qtd_paradas_teorico"), 0),
                    "aderente_m5_4b": bool(info.get("aderente", False)),
                    "motivo_aderencia_m5_4b": _safe_text(info.get("motivo")),
                }

                if bool(info.get("aderente", False)):
                    frota_aderente_rows.append(registro)

                    if not base_tentativa.empty:
                        df_temp = base_tentativa.copy()
                        df_temp["subregiao"] = subregiao
                        df_temp["uf"] = uf
                        df_temp["cidade_anchor_m5_4b"] = cidade_anchor
                        df_temp["perfil_m5_4b"] = _safe_text(vehicle_row.get("perfil"))
                        df_temp["tipo_m5_4b"] = _safe_text(vehicle_row.get("tipo"))
                        df_temp["ordem_tentativa_perfil_m5_4b"] = ordem_perfil
                        base_tentativas_list.append(df_temp)
                else:
                    frota_nao_aderente_rows.append(registro)

    df_frota_aderente_m5_4b = pd.DataFrame(frota_aderente_rows)
    df_frota_nao_aderente_m5_4b = pd.DataFrame(frota_nao_aderente_rows)
    df_base_tentativas_m5_4b = (
        pd.concat(base_tentativas_list, ignore_index=True) if base_tentativas_list else pd.DataFrame()
    )
    df_cidades_candidatas_m5_4b = pd.DataFrame(cidades_candidatas_rows)

    resumo_m5_4b = {
        "modulo": "M5.4B",
        "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
        "tipo_roteirizacao": tipo_roteirizacao,
        "linhas_entrada_m5_4b": int(len(base)),
        "total_registros_frota_aderente_m5_4b": int(len(df_frota_aderente_m5_4b)),
        "total_registros_frota_nao_aderente_m5_4b": int(len(df_frota_nao_aderente_m5_4b)),
        "total_linhas_base_tentativas_m5_4b": int(len(df_base_tentativas_m5_4b)),
        "total_cidades_candidatas_m5_4b": int(len(df_cidades_candidatas_m5_4b)),
        "estrategia_m5_4b": [
            "teste_de_veiculos_do_maior_para_o_menor",
            "base_teorica_sem_composicao_final",
            "cidade_ancora_por_maior_massa",
            "expansao_por_menor_distancia_entre_destinos",
            "VERSAO_M5_4B_2026_04_12",
        ],
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    outputs_m5_4b = {
        "df_frota_aderente_m5_4b": df_frota_aderente_m5_4b,
        "df_frota_nao_aderente_m5_4b": df_frota_nao_aderente_m5_4b,
        "df_base_tentativas_m5_4b": df_base_tentativas_m5_4b,
        "df_cidades_candidatas_m5_4b": df_cidades_candidatas_m5_4b,
    }

    meta_m5_4b = {
        "resumo_m5_4b": resumo_m5_4b,
    }

    return outputs_m5_4b, meta_m5_4b


# Aliases defensivos
def executar_m5_4b_frota_aderente_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4b_aderencia_frota_subregioes(*args, **kwargs)


def processar_m5_4b_aderencia_frota_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4b_aderencia_frota_subregioes(*args, **kwargs)


def rodar_m5_4b_aderencia_frota_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4b_aderencia_frota_subregioes(*args, **kwargs)
