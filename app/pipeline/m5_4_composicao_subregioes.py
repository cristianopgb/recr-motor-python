from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

import pandas as pd


# =========================================================================================
# M5.4 - COMPOSIÇÃO DE SUBREGIÕES
# -----------------------------------------------------------------------------------------
# REGRAS DE NEGÓCIO CONSOLIDADAS
#
# 1. Trabalha somente com os elegíveis do M5.3
# 2. Processa subregião por subregião
# 3. Não mistura subregiões
# 4. Dentro da subregião:
#    - ordena cidades por maior massa
#    - usa a cidade de maior massa como âncora
#    - usa clientes da âncora por maior massa
# 5. Testa perfis do maior para o menor
# 6. Dentro da tentativa:
#    - o perfil fica fixo
#    - tenta compor primeiro na cidade âncora
#    - se não fechar, expande para outras cidades da subregião
#      pela menor distância entre destinos (e não pela origem)
# 7. Um cliente ruim não elimina a cidade
# 8. A cidade candidata pode contribuir com vários clientes
# 9. Se fechar:
#    - remove os itens do pool
#    - recomeça a subregião com o saldo atualizado
# 10. Só vira remanescente no fim da subregião
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
    ]
    for col in numeric_cols:
        saldo[col] = pd.to_numeric(saldo[col], errors="coerce")

    for col in ["agendada", "veiculo_exclusivo", "veiculo_exclusivo_flag"]:
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
# Compatibilidades e validações
# -----------------------------------------------------------------------------------------
def _cliente_respeita_restricao_veiculo(df_cliente: pd.DataFrame, vehicle_row: pd.Series) -> bool:
    tipo_veiculo = _safe_text(vehicle_row.get("tipo")) or _safe_text(vehicle_row.get("perfil"))
    if not tipo_veiculo or "restricao_veiculo" not in df_cliente.columns:
        return True

    restricoes = df_cliente["restricao_veiculo"].dropna().astype(str).str.strip()
    if restricoes.empty:
        return True

    for restr in restricoes.unique().tolist():
        if restr and restr.lower() not in {"", "nan", "none"}:
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

    if peso_total > _ocupacao_maxima_kg(vehicle_row):
        return False, "ocupacao_acima_maxima"

    return True, "ok"


def _conjunto_bate_ocupacao_minima(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> bool:
    return _peso_total(df_itens) >= _ocupacao_minima_kg(vehicle_row)


def _perfil_tem_chance_por_raio_e_paradas(
    df_subregiao: pd.DataFrame,
    vehicle_row: pd.Series,
) -> bool:
    if df_subregiao.empty:
        return False

    max_km = _safe_float(vehicle_row.get("max_km_distancia"), 0.0)
    max_entregas = _safe_int(vehicle_row.get("max_entregas"), 0)

    if max_km > 0:
        if pd.to_numeric(df_subregiao["distancia_rodoviaria_est_km"], errors="coerce").fillna(0).min() > max_km:
            return False

    if max_entregas > 0:
        menor_parada_possivel = 1
        if menor_parada_possivel > max_entregas:
            return False

    return True


# -----------------------------------------------------------------------------------------
# Agregações por cidade e cliente
# -----------------------------------------------------------------------------------------
def _agrupar_cidades(df_subregiao: pd.DataFrame) -> pd.DataFrame:
    if df_subregiao.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []

    for cidade_key, grupo in df_subregiao.groupby("_cidade_key_m5_4", dropna=False, sort=False):
        lat_media = pd.to_numeric(grupo["latitude_destinatario"], errors="coerce").dropna()
        lon_media = pd.to_numeric(grupo["longitude_destinatario"], errors="coerce").dropna()

        records.append(
            {
                "cidade": cidade_key,
                "peso_total_cidade": round(_peso_total(grupo), 3),
                "qtd_linhas": int(len(grupo)),
                "qtd_paradas": int(_qtd_paradas(grupo)),
                "lat_centroide": float(lat_media.mean()) if not lat_media.empty else pd.NA,
                "lon_centroide": float(lon_media.mean()) if not lon_media.empty else pd.NA,
            }
        )

    cidades = pd.DataFrame(records)
    if cidades.empty:
        return cidades

    return (
        cidades.sort_values(
            by=["peso_total_cidade", "qtd_linhas", "cidade"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )


def _agrupar_clientes_da_cidade(df_cidade: pd.DataFrame) -> pd.DataFrame:
    if df_cidade.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []

    for cliente_key, grupo in df_cidade.groupby("_cliente_key_m5_4", dropna=False, sort=False):
        lat_media = pd.to_numeric(grupo["latitude_destinatario"], errors="coerce").dropna()
        lon_media = pd.to_numeric(grupo["longitude_destinatario"], errors="coerce").dropna()

        records.append(
            {
                "cidade": _safe_text(grupo.iloc[0].get("_cidade_key_m5_4")),
                "destinatario": cliente_key,
                "peso_total_cliente": round(_peso_total(grupo), 3),
                "qtd_linhas": int(len(grupo)),
                "qtd_paradas": int(_qtd_paradas(grupo)),
                "lat_centroide": float(lat_media.mean()) if not lat_media.empty else pd.NA,
                "lon_centroide": float(lon_media.mean()) if not lon_media.empty else pd.NA,
            }
        )

    clientes = pd.DataFrame(records)
    if clientes.empty:
        return clientes

    return (
        clientes.sort_values(
            by=["peso_total_cliente", "qtd_linhas", "destinatario"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )


def _filtrar_itens_cliente(df_base: pd.DataFrame, cidade: str, destinatario: str) -> pd.DataFrame:
    return df_base[
        (df_base["_cidade_key_m5_4"] == _safe_text(cidade))
        & (df_base["_cliente_key_m5_4"] == _safe_text(destinatario))
    ].copy()


# -----------------------------------------------------------------------------------------
# Distância entre cidades destino
# -----------------------------------------------------------------------------------------
def _distancia_cidade_para_composicao(
    cidade_row: pd.Series,
    composicao_df: pd.DataFrame,
) -> float:
    if composicao_df.empty:
        return 999999.0

    lat_comp = pd.to_numeric(composicao_df["latitude_destinatario"], errors="coerce").dropna()
    lon_comp = pd.to_numeric(composicao_df["longitude_destinatario"], errors="coerce").dropna()

    if lat_comp.empty or lon_comp.empty:
        return 999999.0

    lat_ref = float(lat_comp.mean())
    lon_ref = float(lon_comp.mean())

    return _haversine_km(
        _safe_float(cidade_row.get("lat_centroide"), math.nan),
        _safe_float(cidade_row.get("lon_centroide"), math.nan),
        lat_ref,
        lon_ref,
    )


def _ordenar_cidades_candidatas(
    cidades_df: pd.DataFrame,
    composicao_df: pd.DataFrame,
    cidade_anchor: str,
) -> pd.DataFrame:
    if cidades_df.empty:
        return cidades_df.copy()

    temp = cidades_df.copy()
    temp["_mesma_cidade_anchor"] = temp["cidade"].astype(str).str.strip() == _safe_text(cidade_anchor)
    temp["_dist_para_composicao"] = temp.apply(
        lambda row: _distancia_cidade_para_composicao(row, composicao_df),
        axis=1,
    )

    return (
        temp.sort_values(
            by=[
                "_mesma_cidade_anchor",
                "_dist_para_composicao",
                "peso_total_cidade",
                "cidade",
            ],
            ascending=[False, True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )


# -----------------------------------------------------------------------------------------
# Tentativa de composição dentro de um perfil fixo
# -----------------------------------------------------------------------------------------
def _tentar_enriquecer_com_clientes_da_cidade(
    composicao_df: pd.DataFrame,
    df_pool_subregiao: pd.DataFrame,
    cidade: str,
    vehicle_row: pd.Series,
    clientes_ja_testados: set[tuple[str, str]],
    auditoria_clientes: List[Dict[str, Any]],
) -> pd.DataFrame:
    df_cidade = df_pool_subregiao[df_pool_subregiao["_cidade_key_m5_4"] == _safe_text(cidade)].copy()
    if df_cidade.empty:
        return composicao_df

    clientes_df = _agrupar_clientes_da_cidade(df_cidade)

    for _, cliente_row in clientes_df.iterrows():
        cidade_key = _safe_text(cliente_row.get("cidade"))
        destinatario_key = _safe_text(cliente_row.get("destinatario"))
        chave_cliente = (cidade_key, destinatario_key)

        if chave_cliente in clientes_ja_testados:
            continue

        clientes_ja_testados.add(chave_cliente)

        df_cliente = _filtrar_itens_cliente(df_pool_subregiao, cidade_key, destinatario_key)
        if df_cliente.empty:
            auditoria_clientes.append(
                {
                    "cidade": cidade_key,
                    "destinatario": destinatario_key,
                    "status_cliente": "rejeitado",
                    "motivo_cliente": "cliente_sem_itens",
                }
            )
            continue

        if not _cliente_respeita_restricao_veiculo(df_cliente, vehicle_row):
            auditoria_clientes.append(
                {
                    "cidade": cidade_key,
                    "destinatario": destinatario_key,
                    "status_cliente": "rejeitado",
                    "motivo_cliente": "restricao_veiculo_incompativel",
                }
            )
            continue

        tentativa_df = pd.concat([composicao_df, df_cliente], ignore_index=True)
        ok_limites, motivo_limites = _conjunto_respeita_limites(tentativa_df, vehicle_row)

        if not ok_limites:
            auditoria_clientes.append(
                {
                    "cidade": cidade_key,
                    "destinatario": destinatario_key,
                    "status_cliente": "rejeitado",
                    "motivo_cliente": motivo_limites,
                }
            )
            continue

        composicao_df = tentativa_df
        auditoria_clientes.append(
            {
                "cidade": cidade_key,
                "destinatario": destinatario_key,
                "status_cliente": "aceito",
                "motivo_cliente": "ok",
                "peso_composicao_pos_cliente": round(_peso_total(composicao_df), 3),
                "qtd_paradas_pos_cliente": int(_qtd_paradas(composicao_df)),
            }
        )

    return composicao_df


def _tentar_compor_com_perfil(
    df_pool_subregiao: pd.DataFrame,
    cidade_anchor: str,
    vehicle_row: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    auditoria_clientes: List[Dict[str, Any]] = []
    clientes_ja_testados: set[tuple[str, str]] = set()

    composicao_df = pd.DataFrame(columns=df_pool_subregiao.columns)

    # 1. Cidade âncora primeiro
    composicao_df = _tentar_enriquecer_com_clientes_da_cidade(
        composicao_df=composicao_df,
        df_pool_subregiao=df_pool_subregiao,
        cidade=cidade_anchor,
        vehicle_row=vehicle_row,
        clientes_ja_testados=clientes_ja_testados,
        auditoria_clientes=auditoria_clientes,
    )

    # 2. Se ainda não bateu ocupação mínima, expande para outras cidades
    if not composicao_df.empty and not _conjunto_bate_ocupacao_minima(composicao_df, vehicle_row):
        cidades_df = _agrupar_cidades(df_pool_subregiao)
        cidades_ordenadas = _ordenar_cidades_candidatas(
            cidades_df=cidades_df,
            composicao_df=composicao_df,
            cidade_anchor=cidade_anchor,
        )

        for _, cidade_row in cidades_ordenadas.iterrows():
            cidade_key = _safe_text(cidade_row.get("cidade"))
            if cidade_key == _safe_text(cidade_anchor):
                continue

            composicao_df = _tentar_enriquecer_com_clientes_da_cidade(
                composicao_df=composicao_df,
                df_pool_subregiao=df_pool_subregiao,
                cidade=cidade_key,
                vehicle_row=vehicle_row,
                clientes_ja_testados=clientes_ja_testados,
                auditoria_clientes=auditoria_clientes,
            )

            if not composicao_df.empty and _conjunto_bate_ocupacao_minima(composicao_df, vehicle_row):
                break

    # 3. Resultado final da tentativa no perfil
    if composicao_df.empty:
        return (
            pd.DataFrame(),
            {
                "status_tentativa": "falha",
                "motivo_tentativa": "nenhum_cliente_aceito",
            },
            auditoria_clientes,
        )

    ok_limites, motivo_limites = _conjunto_respeita_limites(composicao_df, vehicle_row)
    bate_min = _conjunto_bate_ocupacao_minima(composicao_df, vehicle_row)

    if ok_limites and bate_min:
        return (
            composicao_df.reset_index(drop=True).copy(),
            {
                "status_tentativa": "sucesso",
                "motivo_tentativa": "ok",
                "peso_total": round(_peso_total(composicao_df), 3),
                "qtd_paradas": int(_qtd_paradas(composicao_df)),
                "km_referencia": round(_km_referencia(composicao_df), 2),
            },
            auditoria_clientes,
        )

    return (
        pd.DataFrame(),
        {
            "status_tentativa": "falha",
            "motivo_tentativa": motivo_limites if not ok_limites else "abaixo_ocupacao_minima",
            "peso_total_tentado": round(_peso_total(composicao_df), 3),
            "qtd_paradas_tentado": int(_qtd_paradas(composicao_df)),
            "km_referencia_tentado": round(_km_referencia(composicao_df), 2),
        },
        auditoria_clientes,
    )


# -----------------------------------------------------------------------------------------
# Serialização interna de auditoria
# -----------------------------------------------------------------------------------------
def _registro_manifesto(
    manifesto_id: str,
    subregiao_key: str,
    uf_key: str,
    vehicle_row: pd.Series,
    df_itens: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": "composto_bloco_5_4_subregiao",
        "origem_modulo": 5.4,
        "origem_etapa": "5_4_subregiao_multicidade",
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
        "capacidade_peso_kg_veiculo": round(_safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0), 3),
        "capacidade_vol_m3_veiculo": round(_safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0), 3),
        "max_entregas_veiculo": _safe_int(vehicle_row.get("max_entregas"), 0),
        "max_km_distancia_veiculo": round(_safe_float(vehicle_row.get("max_km_distancia"), 0.0), 3),
        "ocupacao_minima_perc_veiculo": round(_safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0), 3),
        "ocupacao_maxima_perc_veiculo": round(_safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0), 3),
        "ocupacao_minima_kg_veiculo": round(_ocupacao_minima_kg(vehicle_row), 3),
        "ocupacao_maxima_kg_veiculo": round(_ocupacao_maxima_kg(vehicle_row), 3),
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
    del rodada_id, kwargs

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
            "df_tentativas_clientes_m5_4": pd.DataFrame(),
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
                    "cidade_ancora_por_maior_massa",
                    "clientes_por_maior_massa",
                    "perfil_do_maior_para_o_menor",
                    "perfil_fixo_na_tentativa",
                    "expansao_por_menor_distancia_entre_destinos",
                    "saldo_vivo_ate_final_da_subregiao",
                    "VERSAO_M5_4_2026_04_12",
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
    subregioes_rows: List[Dict[str, Any]] = []
    tentativas_perfis_rows: List[Dict[str, Any]] = []
    tentativas_clientes_rows: List[Dict[str, Any]] = []
    remanescente_global_list: List[pd.DataFrame] = []

    manifesto_seq = 1

    # Fila de subregiões por maior massa
    subregioes_fila = (
        saldo.groupby(["_subregiao_key_m5_4", "_uf_key_m5_4"], dropna=False)["peso_calculado"]
        .sum()
        .reset_index(name="peso_total_subregiao")
        .sort_values(
            by=["peso_total_subregiao", "_subregiao_key_m5_4", "_uf_key_m5_4"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    for _, sub_row in subregioes_fila.iterrows():
        subregiao_key = _safe_text(sub_row.get("_subregiao_key_m5_4"))
        uf_key = _safe_text(sub_row.get("_uf_key_m5_4"))

        pool_subregiao = saldo[
            (saldo["_subregiao_key_m5_4"] == subregiao_key)
            & (saldo["_uf_key_m5_4"] == uf_key)
        ].copy()

        pool_subregiao = _ordenar_operacional(pool_subregiao)

        if pool_subregiao.empty:
            continue

        total_manifestos_sub = 0
        total_tentativas_sub = 0

        while True:
            cidades_df = _agrupar_cidades(pool_subregiao)
            if cidades_df.empty:
                break

            houve_fechamento = False

            for _, cidade_anchor_row in cidades_df.iterrows():
                cidade_anchor = _safe_text(cidade_anchor_row.get("cidade"))
                if not cidade_anchor:
                    continue

                # perfis do maior para o menor
                perfis_viaveis_tentativa = perfis_maior_menor[
                    perfis_maior_menor.apply(
                        lambda row: _perfil_tem_chance_por_raio_e_paradas(pool_subregiao, row),
                        axis=1,
                    )
                ].reset_index(drop=True)

                if perfis_viaveis_tentativa.empty:
                    continue

                for _, vehicle_row in perfis_viaveis_tentativa.iterrows():
                    total_tentativas_sub += 1

                    manifesto_df, info_tentativa, auditoria_clientes = _tentar_compor_com_perfil(
                        df_pool_subregiao=pool_subregiao,
                        cidade_anchor=cidade_anchor,
                        vehicle_row=vehicle_row,
                    )

                    tentativas_perfis_rows.append(
                        {
                            "subregiao": subregiao_key,
                            "uf": uf_key,
                            "cidade_anchor": cidade_anchor,
                            "perfil": _safe_text(vehicle_row.get("perfil")),
                            "tipo": _safe_text(vehicle_row.get("tipo")),
                            "status_tentativa": _safe_text(info_tentativa.get("status_tentativa")),
                            "motivo_tentativa": _safe_text(info_tentativa.get("motivo_tentativa")),
                            "peso_total_tentativa": round(_safe_float(info_tentativa.get("peso_total", info_tentativa.get("peso_total_tentado", 0.0)), 0.0), 3),
                            "qtd_paradas_tentativa": _safe_int(info_tentativa.get("qtd_paradas", info_tentativa.get("qtd_paradas_tentado", 0)), 0),
                            "km_referencia_tentativa": round(_safe_float(info_tentativa.get("km_referencia", info_tentativa.get("km_referencia_tentado", 0.0)), 0.0), 2),
                        }
                    )

                    for reg in auditoria_clientes:
                        tentativas_clientes_rows.append(
                            {
                                "subregiao": subregiao_key,
                                "uf": uf_key,
                                "cidade_anchor": cidade_anchor,
                                "perfil": _safe_text(vehicle_row.get("perfil")),
                                "tipo": _safe_text(vehicle_row.get("tipo")),
                                **reg,
                            }
                        )

                    if manifesto_df.empty:
                        continue

                    manifesto_id = f"PM54_{manifesto_seq:04d}"
                    manifesto_seq += 1
                    total_manifestos_sub += 1

                    manifestos_rows.append(
                        _registro_manifesto(
                            manifesto_id=manifesto_id,
                            subregiao_key=subregiao_key,
                            uf_key=uf_key,
                            vehicle_row=vehicle_row,
                            df_itens=manifesto_df,
                        )
                    )

                    manifesto_df = manifesto_df.copy()
                    manifesto_df["manifesto_id"] = manifesto_id
                    manifesto_df["tipo_manifesto"] = "composto_bloco_5_4_subregiao"
                    manifesto_df["origem_modulo"] = 5.4
                    manifesto_df["origem_etapa"] = "5_4_subregiao_multicidade"
                    manifesto_df["veiculo_tipo"] = _safe_text(vehicle_row.get("tipo")) or _safe_text(vehicle_row.get("perfil"))
                    manifesto_df["perfil"] = _safe_text(vehicle_row.get("perfil"))
                    manifesto_df["capacidade_peso_kg_veiculo"] = round(_safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0), 3)
                    manifesto_df["capacidade_vol_m3_veiculo"] = round(_safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0), 3)
                    manifesto_df["max_entregas_veiculo"] = _safe_int(vehicle_row.get("max_entregas"), 0)
                    manifesto_df["max_km_distancia_veiculo"] = round(_safe_float(vehicle_row.get("max_km_distancia"), 0.0), 3)

                    itens_manifestados_list.append(manifesto_df)

                    used_ids = set(manifesto_df["id_linha_pipeline"].astype(str).tolist())
                    pool_subregiao = pool_subregiao[
                        ~pool_subregiao["id_linha_pipeline"].astype(str).isin(used_ids)
                    ].copy()
                    pool_subregiao = _ordenar_operacional(pool_subregiao)

                    houve_fechamento = True
                    break

                if houve_fechamento:
                    break

            if not houve_fechamento:
                break

        if not pool_subregiao.empty:
            pool_subregiao["status_m5_4"] = "remanescente_m5_4"
            pool_subregiao["motivo_m5_4"] = "nao_composto_apos_exaurir_subregiao"
            remanescente_global_list.append(pool_subregiao)

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
                "manifestos_gerados_subregiao": int(total_manifestos_sub),
                "linhas_remanescentes_subregiao": int(len(pool_subregiao)),
                "tentativas_processadas_subregiao": int(total_tentativas_sub),
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
    df_tentativas_perfis_m5_4 = pd.DataFrame(tentativas_perfis_rows)
    df_tentativas_clientes_m5_4 = pd.DataFrame(tentativas_clientes_rows)

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
        "tentativas_perfis_m5_4": int(len(df_tentativas_perfis_m5_4)),
        "tentativas_clientes_m5_4": int(len(df_tentativas_clientes_m5_4)),
        "estrategia_m5_4": [
            "subregiao_por_subregiao",
            "cidade_ancora_por_maior_massa",
            "clientes_por_maior_massa",
            "perfil_do_maior_para_o_menor",
            "perfil_fixo_na_tentativa",
            "expansao_por_menor_distancia_entre_destinos",
            "cliente_ruim_nao_elimina_cidade",
            "saldo_vivo_ate_final_da_subregiao",
            "VERSAO_M5_4_2026_04_12",
        ],
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    outputs_m5_4 = {
        "df_premanifestos_m5_4": df_premanifestos_m5_4,
        "df_itens_premanifestados_m5_4": df_itens_premanifestados_m5_4,
        "df_remanescente_m5_4": df_remanescente_m5_4,
        "df_subregioes_processadas_m5_4": df_subregioes_processadas_m5_4,
        "df_tentativas_perfis_m5_4": df_tentativas_perfis_m5_4,
        "df_tentativas_clientes_m5_4": df_tentativas_clientes_m5_4,
    }

    meta_m5_4 = {
        "resumo_m5_4": resumo_m5_4,
    }

    return outputs_m5_4, meta_m5_4


# Aliases defensivos
def executar_m5_composicao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4_composicao_subregioes(*args, **kwargs)


def processar_m5_4_composicao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4_composicao_subregioes(*args, **kwargs)


def rodar_m5_4_composicao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4_composicao_subregioes(*args, **kwargs)
