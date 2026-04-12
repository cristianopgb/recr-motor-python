from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =========================================================================================
# M5.4A - PREPARAÇÃO DE SUBREGIÕES
# -----------------------------------------------------------------------------------------
# OBJETIVO
# - receber os elegíveis do M5.3
# - ordenar subregiões por massa
# - ordenar cidades por massa dentro da subregião
# - ordenar clientes por massa dentro da cidade
# - devolver base auditável e pronta para as próximas etapas
#
# ESTA ETAPA NÃO:
# - escolhe veículo final
# - testa composição
# - gera pré-manifesto
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
# Normalização
# -----------------------------------------------------------------------------------------
def _normalizar_inputs(
    df_saldo_elegivel_composicao_m5_3: pd.DataFrame,
) -> pd.DataFrame:
    saldo = (
        df_saldo_elegivel_composicao_m5_3.copy()
        if df_saldo_elegivel_composicao_m5_3 is not None
        else pd.DataFrame()
    )

    if saldo.empty:
        return saldo

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
    }
    for col, default in defaults.items():
        _ensure_column(saldo, col, default)

    if saldo["id_linha_pipeline"].isna().any():
        raise ValueError("M5.4A exige id_linha_pipeline em todas as linhas elegíveis do M5.3.")

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
    ]
    for col in numeric_cols:
        if col in saldo.columns:
            saldo[col] = pd.to_numeric(saldo[col], errors="coerce")

    for col in ["agendada", "veiculo_exclusivo", "veiculo_exclusivo_flag"]:
        if col in saldo.columns:
            saldo[col] = saldo[col].apply(_safe_bool)

    for col in ["cidade", "uf", "subregiao", "mesorregiao", "destinatario"]:
        if col in saldo.columns:
            saldo[col] = saldo[col].fillna("").astype(str).str.strip()

    return saldo


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
    temp["_id_str_m5_4a"] = temp["id_linha_pipeline"].astype(str)
    temp["_subregiao_key_m5_4a"] = temp["subregiao"].fillna("").astype(str).str.strip()
    temp["_uf_key_m5_4a"] = temp["uf"].fillna("").astype(str).str.strip()
    temp["_cidade_key_m5_4a"] = temp["cidade"].fillna("").astype(str).str.strip()
    temp["_cliente_key_m5_4a"] = temp["destinatario"].fillna("").astype(str).str.strip()

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

    temp["_bucket_m5_4a"] = buckets
    temp["_prioridade_ord_m5_4a"] = prioridade_ord
    temp["_folga_ord_m5_4a"] = folga
    temp["_ranking_ord_m5_4a"] = ranking
    temp["_km_ord_m5_4a"] = km
    temp["_peso_ord_m5_4a"] = -peso

    return temp


def _ordenar_operacional(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "_bucket_m5_4a" not in df.columns:
        df = _precalcular_ordenacao(df)

    return (
        df.sort_values(
            by=[
                "_bucket_m5_4a",
                "_prioridade_ord_m5_4a",
                "_folga_ord_m5_4a",
                "_ranking_ord_m5_4a",
                "_km_ord_m5_4a",
                "_peso_ord_m5_4a",
                "_id_str_m5_4a",
            ],
            ascending=[True, True, True, True, True, True, True],
            kind="mergesort",
        )
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


# -----------------------------------------------------------------------------------------
# Agregações preparatórias
# -----------------------------------------------------------------------------------------
def _montar_subregioes_ordenadas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    registros: List[Dict[str, Any]] = []

    for (subregiao, uf), grupo in df.groupby(["_subregiao_key_m5_4a", "_uf_key_m5_4a"], dropna=False, sort=False):
        registros.append(
            {
                "subregiao": _safe_text(subregiao),
                "uf": _safe_text(uf),
                "peso_total_subregiao": round(_peso_total(grupo), 3),
                "volume_total_subregiao": round(_volume_total(grupo), 3),
                "km_referencia_subregiao": round(_km_referencia(grupo), 2),
                "qtd_linhas_subregiao": int(len(grupo)),
                "qtd_paradas_subregiao": int(_qtd_paradas(grupo)),
                "qtd_cidades_subregiao": int(grupo["_cidade_key_m5_4a"].nunique()),
                "qtd_clientes_subregiao": int(grupo["_cliente_key_m5_4a"].nunique()),
            }
        )

    out = pd.DataFrame(registros)
    if out.empty:
        return out

    out = (
        out.sort_values(
            by=["peso_total_subregiao", "subregiao", "uf"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )
    out["ordem_subregiao_m5_4a"] = range(1, len(out) + 1)
    return out


def _montar_cidades_ordenadas(df: pd.DataFrame, df_subregioes: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    ordem_sub_map = {
        (_safe_text(row["subregiao"]), _safe_text(row["uf"])): _safe_int(row["ordem_subregiao_m5_4a"], 0)
        for _, row in df_subregioes.iterrows()
    }

    registros: List[Dict[str, Any]] = []

    for (subregiao, uf, cidade), grupo in df.groupby(
        ["_subregiao_key_m5_4a", "_uf_key_m5_4a", "_cidade_key_m5_4a"],
        dropna=False,
        sort=False,
    ):
        registros.append(
            {
                "subregiao": _safe_text(subregiao),
                "uf": _safe_text(uf),
                "cidade": _safe_text(cidade),
                "peso_total_cidade": round(_peso_total(grupo), 3),
                "volume_total_cidade": round(_volume_total(grupo), 3),
                "km_referencia_cidade": round(_km_referencia(grupo), 2),
                "qtd_linhas_cidade": int(len(grupo)),
                "qtd_paradas_cidade": int(_qtd_paradas(grupo)),
                "qtd_clientes_cidade": int(grupo["_cliente_key_m5_4a"].nunique()),
                "ordem_subregiao_m5_4a": ordem_sub_map.get((_safe_text(subregiao), _safe_text(uf)), 0),
            }
        )

    out = pd.DataFrame(registros)
    if out.empty:
        return out

    out = (
        out.sort_values(
            by=["ordem_subregiao_m5_4a", "peso_total_cidade", "cidade"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    out["ordem_cidade_na_subregiao_m5_4a"] = (
        out.groupby(["subregiao", "uf"]).cumcount() + 1
    )

    return out


def _montar_clientes_ordenados(df: pd.DataFrame, df_subregioes: pd.DataFrame, df_cidades: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    ordem_sub_map = {
        (_safe_text(row["subregiao"]), _safe_text(row["uf"])): _safe_int(row["ordem_subregiao_m5_4a"], 0)
        for _, row in df_subregioes.iterrows()
    }
    ordem_cidade_map = {
        (_safe_text(row["subregiao"]), _safe_text(row["uf"]), _safe_text(row["cidade"])): _safe_int(row["ordem_cidade_na_subregiao_m5_4a"], 0)
        for _, row in df_cidades.iterrows()
    }

    registros: List[Dict[str, Any]] = []

    for (subregiao, uf, cidade, cliente), grupo in df.groupby(
        ["_subregiao_key_m5_4a", "_uf_key_m5_4a", "_cidade_key_m5_4a", "_cliente_key_m5_4a"],
        dropna=False,
        sort=False,
    ):
        registros.append(
            {
                "subregiao": _safe_text(subregiao),
                "uf": _safe_text(uf),
                "cidade": _safe_text(cidade),
                "destinatario": _safe_text(cliente),
                "peso_total_cliente": round(_peso_total(grupo), 3),
                "volume_total_cliente": round(_volume_total(grupo), 3),
                "km_referencia_cliente": round(_km_referencia(grupo), 2),
                "qtd_linhas_cliente": int(len(grupo)),
                "qtd_paradas_cliente": int(_qtd_paradas(grupo)),
                "agendada_tem_alguma": bool(grupo["agendada"].fillna(False).any()),
                "agendada_todas": bool(grupo["agendada"].fillna(False).all()),
                "ordem_subregiao_m5_4a": ordem_sub_map.get((_safe_text(subregiao), _safe_text(uf)), 0),
                "ordem_cidade_na_subregiao_m5_4a": ordem_cidade_map.get(
                    (_safe_text(subregiao), _safe_text(uf), _safe_text(cidade)),
                    0,
                ),
            }
        )

    out = pd.DataFrame(registros)
    if out.empty:
        return out

    out = (
        out.sort_values(
            by=[
                "ordem_subregiao_m5_4a",
                "ordem_cidade_na_subregiao_m5_4a",
                "peso_total_cliente",
                "destinatario",
            ],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    out["ordem_cliente_na_cidade_m5_4a"] = (
        out.groupby(["subregiao", "uf", "cidade"]).cumcount() + 1
    )

    return out


def _montar_base_preparada(
    df: pd.DataFrame,
    df_subregioes: pd.DataFrame,
    df_cidades: pd.DataFrame,
    df_clientes: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    base = df.copy()

    ordem_sub_map = {
        (_safe_text(row["subregiao"]), _safe_text(row["uf"])): _safe_int(row["ordem_subregiao_m5_4a"], 0)
        for _, row in df_subregioes.iterrows()
    }
    ordem_cidade_map = {
        (_safe_text(row["subregiao"]), _safe_text(row["uf"]), _safe_text(row["cidade"])): _safe_int(row["ordem_cidade_na_subregiao_m5_4a"], 0)
        for _, row in df_cidades.iterrows()
    }
    ordem_cliente_map = {
        (_safe_text(row["subregiao"]), _safe_text(row["uf"]), _safe_text(row["cidade"]), _safe_text(row["destinatario"])): _safe_int(row["ordem_cliente_na_cidade_m5_4a"], 0)
        for _, row in df_clientes.iterrows()
    }
    peso_sub_map = {
        (_safe_text(row["subregiao"]), _safe_text(row["uf"])): _safe_float(row["peso_total_subregiao"], 0.0)
        for _, row in df_subregioes.iterrows()
    }
    peso_cidade_map = {
        (_safe_text(row["subregiao"]), _safe_text(row["uf"]), _safe_text(row["cidade"])): _safe_float(row["peso_total_cidade"], 0.0)
        for _, row in df_cidades.iterrows()
    }
    peso_cliente_map = {
        (_safe_text(row["subregiao"]), _safe_text(row["uf"]), _safe_text(row["cidade"]), _safe_text(row["destinatario"])): _safe_float(row["peso_total_cliente"], 0.0)
        for _, row in df_clientes.iterrows()
    }

    base["ordem_subregiao_m5_4a"] = base.apply(
        lambda row: ordem_sub_map.get((_safe_text(row["subregiao"]), _safe_text(row["uf"])), 0),
        axis=1,
    )
    base["ordem_cidade_na_subregiao_m5_4a"] = base.apply(
        lambda row: ordem_cidade_map.get(
            (_safe_text(row["subregiao"]), _safe_text(row["uf"]), _safe_text(row["cidade"])),
            0,
        ),
        axis=1,
    )
    base["ordem_cliente_na_cidade_m5_4a"] = base.apply(
        lambda row: ordem_cliente_map.get(
            (
                _safe_text(row["subregiao"]),
                _safe_text(row["uf"]),
                _safe_text(row["cidade"]),
                _safe_text(row["destinatario"]),
            ),
            0,
        ),
        axis=1,
    )
    base["peso_total_subregiao_m5_4a"] = base.apply(
        lambda row: peso_sub_map.get((_safe_text(row["subregiao"]), _safe_text(row["uf"])), 0.0),
        axis=1,
    )
    base["peso_total_cidade_m5_4a"] = base.apply(
        lambda row: peso_cidade_map.get(
            (_safe_text(row["subregiao"]), _safe_text(row["uf"]), _safe_text(row["cidade"])),
            0.0,
        ),
        axis=1,
    )
    base["peso_total_cliente_m5_4a"] = base.apply(
        lambda row: peso_cliente_map.get(
            (
                _safe_text(row["subregiao"]),
                _safe_text(row["uf"]),
                _safe_text(row["cidade"]),
                _safe_text(row["destinatario"]),
            ),
            0.0,
        ),
        axis=1,
    )

    base = (
        base.sort_values(
            by=[
                "ordem_subregiao_m5_4a",
                "ordem_cidade_na_subregiao_m5_4a",
                "ordem_cliente_na_cidade_m5_4a",
                "_bucket_m5_4a",
                "_prioridade_ord_m5_4a",
                "_folga_ord_m5_4a",
                "_ranking_ord_m5_4a",
                "_km_ord_m5_4a",
                "_peso_ord_m5_4a",
                "_id_str_m5_4a",
            ],
            ascending=[True, True, True, True, True, True, True, True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    return base


def _limpar_cols_auxiliares(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    cols_drop = [
        "_id_str_m5_4a",
        "_subregiao_key_m5_4a",
        "_uf_key_m5_4a",
        "_cidade_key_m5_4a",
        "_cliente_key_m5_4a",
        "_bucket_m5_4a",
        "_prioridade_ord_m5_4a",
        "_folga_ord_m5_4a",
        "_ranking_ord_m5_4a",
        "_km_ord_m5_4a",
        "_peso_ord_m5_4a",
    ]
    return df.drop(columns=cols_drop, errors="ignore").reset_index(drop=True).copy()


# -----------------------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------------------
def executar_m5_4a_preparacao_subregioes(
    df_saldo_elegivel_composicao_m5_3: pd.DataFrame,
    rodada_id: Optional[str] = None,
    data_base_roteirizacao: Optional[Any] = None,
    tipo_roteirizacao: str = "carteira",
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    del rodada_id, kwargs

    saldo = _normalizar_inputs(
        df_saldo_elegivel_composicao_m5_3=df_saldo_elegivel_composicao_m5_3,
    )

    if saldo.empty:
        outputs_vazio = {
            "df_base_preparada_m5_4a": pd.DataFrame(),
            "df_subregioes_ordenadas_m5_4a": pd.DataFrame(),
            "df_cidades_ordenadas_m5_4a": pd.DataFrame(),
            "df_clientes_ordenados_m5_4a": pd.DataFrame(),
        }
        meta_vazio = {
            "resumo_m5_4a": {
                "modulo": "M5.4A",
                "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
                "tipo_roteirizacao": tipo_roteirizacao,
                "linhas_entrada_m5_4a": 0,
                "subregioes_preparadas_m5_4a": 0,
                "cidades_preparadas_m5_4a": 0,
                "clientes_preparados_m5_4a": 0,
                "linhas_base_preparada_m5_4a": 0,
                "estrategia_m5_4a": [
                    "ordenacao_subregioes_por_massa",
                    "ordenacao_cidades_por_massa",
                    "ordenacao_clientes_por_massa",
                    "preparacao_base_auditavel",
                    "VERSAO_M5_4A_2026_04_12",
                ],
                "caminhos_pipeline": caminhos_pipeline or {},
            }
        }
        return outputs_vazio, meta_vazio

    saldo = _precalcular_ordenacao(saldo)
    saldo = _ordenar_operacional(saldo)

    df_subregioes_ordenadas_m5_4a = _montar_subregioes_ordenadas(saldo)
    df_cidades_ordenadas_m5_4a = _montar_cidades_ordenadas(saldo, df_subregioes_ordenadas_m5_4a)
    df_clientes_ordenados_m5_4a = _montar_clientes_ordenados(
        saldo,
        df_subregioes_ordenadas_m5_4a,
        df_cidades_ordenadas_m5_4a,
    )
    df_base_preparada_m5_4a = _montar_base_preparada(
        saldo,
        df_subregioes_ordenadas_m5_4a,
        df_cidades_ordenadas_m5_4a,
        df_clientes_ordenados_m5_4a,
    )

    df_base_preparada_m5_4a = _limpar_cols_auxiliares(df_base_preparada_m5_4a)

    resumo_m5_4a = {
        "modulo": "M5.4A",
        "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
        "tipo_roteirizacao": tipo_roteirizacao,
        "linhas_entrada_m5_4a": int(len(saldo)),
        "subregioes_preparadas_m5_4a": int(len(df_subregioes_ordenadas_m5_4a)),
        "cidades_preparadas_m5_4a": int(len(df_cidades_ordenadas_m5_4a)),
        "clientes_preparados_m5_4a": int(len(df_clientes_ordenados_m5_4a)),
        "linhas_base_preparada_m5_4a": int(len(df_base_preparada_m5_4a)),
        "estrategia_m5_4a": [
            "ordenacao_subregioes_por_massa",
            "ordenacao_cidades_por_massa",
            "ordenacao_clientes_por_massa",
            "preparacao_base_auditavel",
            "VERSAO_M5_4A_2026_04_12",
        ],
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    outputs_m5_4a = {
        "df_base_preparada_m5_4a": df_base_preparada_m5_4a,
        "df_subregioes_ordenadas_m5_4a": df_subregioes_ordenadas_m5_4a,
        "df_cidades_ordenadas_m5_4a": df_cidades_ordenadas_m5_4a,
        "df_clientes_ordenados_m5_4a": df_clientes_ordenados_m5_4a,
    }

    meta_m5_4a = {
        "resumo_m5_4a": resumo_m5_4a,
    }

    return outputs_m5_4a, meta_m5_4a


# Aliases defensivos
def executar_m5_preparacao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4a_preparacao_subregioes(*args, **kwargs)


def processar_m5_4a_preparacao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4a_preparacao_subregioes(*args, **kwargs)


def rodar_m5_4a_preparacao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4a_preparacao_subregioes(*args, **kwargs)
