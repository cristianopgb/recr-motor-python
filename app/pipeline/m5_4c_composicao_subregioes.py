from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

import pandas as pd


# =========================================================================================
# M5.4C - COMPOSIÇÃO POR SUBREGIÃO
# -----------------------------------------------------------------------------------------
# ENTRADAS
# - df_base_preparada_m5_4a
# - df_frota_aderente_m5_4b
#
# OBJETIVO
# - compor pré-manifestos reais por subregião
# - usando somente perfis aderentes do M5.4B
#
# REGRAS DE NEGÓCIO CONSOLIDADAS
# 1. não mistura subregiões
# 2. cidade âncora = maior massa do saldo atual
# 3. clientes por maior massa
# 4. perfil testado do maior aderente para o menor aderente
# 5. perfil fica fixo dentro da tentativa
# 6. se precisar, expande para outras cidades da mesma subregião
# 7. falha de cliente não elimina a cidade
# 8. fechou pré-manifesto, remove itens do pool
# 9. só o saldo final da subregião vira remanescente
# =========================================================================================


# -----------------------------------------------------------------------------------------
# Helpers
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


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "sim", "s", "yes", "y"}


def _ensure_column(df: pd.DataFrame, col: str, default: Any) -> None:
    if col not in df.columns:
        df[col] = default


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


def _km_ref(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(pd.to_numeric(df["distancia_rodoviaria_est_km"], errors="coerce").fillna(0).max())


def _qtd_paradas(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df["destinatario"].fillna("").astype(str).nunique())


def _ocupacao_minima_kg(vehicle_row: pd.Series) -> float:
    cap = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    perc = _safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0)
    return cap * (perc / 100.0)


def _ocupacao_maxima_kg(vehicle_row: pd.Series) -> float:
    cap = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    perc = _safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0)
    if perc <= 0:
        perc = 100.0
    return cap * (perc / 100.0)


# -----------------------------------------------------------------------------------------
# Geometria simples entre destinos
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
    df_frota_aderente_m5_4b: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = df_base_preparada_m5_4a.copy() if df_base_preparada_m5_4a is not None else pd.DataFrame()
    frota = df_frota_aderente_m5_4b.copy() if df_frota_aderente_m5_4b is not None else pd.DataFrame()

    if base.empty:
        return base, frota

    defaults_base = {
        "id_linha_pipeline": None,
        "subregiao": "",
        "uf": "",
        "cidade": "",
        "destinatario": "",
        "peso_calculado": 0.0,
        "vol_m3": 0.0,
        "distancia_rodoviaria_est_km": 0.0,
        "latitude_destinatario": pd.NA,
        "longitude_destinatario": pd.NA,
        "ordem_subregiao_m5_4a": 0,
        "ordem_cidade_na_subregiao_m5_4a": 0,
        "ordem_cliente_na_cidade_m5_4a": 0,
    }
    for col, default in defaults_base.items():
        _ensure_column(base, col, default)

    if base["id_linha_pipeline"].isna().any():
        raise ValueError("M5.4C exige id_linha_pipeline em todas as linhas da base preparada do M5.4A.")

    for col in [
        "peso_calculado",
        "vol_m3",
        "distancia_rodoviaria_est_km",
        "latitude_destinatario",
        "longitude_destinatario",
        "ordem_subregiao_m5_4a",
        "ordem_cidade_na_subregiao_m5_4a",
        "ordem_cliente_na_cidade_m5_4a",
    ]:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    for col in ["subregiao", "uf", "cidade", "destinatario"]:
        base[col] = base[col].fillna("").astype(str).str.strip()

    if not frota.empty:
        for col in [
            "subregiao",
            "uf",
            "perfil",
            "tipo",
            "ordem_tentativa_perfil_m5_4b",
            "capacidade_peso_kg",
            "capacidade_vol_m3",
            "max_entregas",
            "max_km_distancia",
            "ocupacao_minima_perc",
            "ocupacao_maxima_perc",
            "status_aderencia_m5_4b",
        ]:
            _ensure_column(frota, col, pd.NA if col not in ["subregiao", "uf", "perfil", "tipo"] else "")

        for col in [
            "ordem_tentativa_perfil_m5_4b",
            "capacidade_peso_kg",
            "capacidade_vol_m3",
            "max_entregas",
            "max_km_distancia",
            "ocupacao_minima_perc",
            "ocupacao_maxima_perc",
        ]:
            frota[col] = pd.to_numeric(frota[col], errors="coerce")

        for col in ["subregiao", "uf", "perfil", "tipo", "status_aderencia_m5_4b"]:
            frota[col] = frota[col].fillna("").astype(str).str.strip()

        frota = frota[frota["status_aderencia_m5_4b"].str.lower() == "aderente"].copy()

    return base, frota


# -----------------------------------------------------------------------------------------
# Agregações
# -----------------------------------------------------------------------------------------
def _centroide(df: pd.DataFrame) -> Tuple[float, float]:
    lat = pd.to_numeric(df["latitude_destinatario"], errors="coerce").dropna()
    lon = pd.to_numeric(df["longitude_destinatario"], errors="coerce").dropna()
    if lat.empty or lon.empty:
        return (math.nan, math.nan)
    return (float(lat.mean()), float(lon.mean()))


def _agrupar_cidades(df_sub: pd.DataFrame) -> pd.DataFrame:
    if df_sub.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for cidade, grupo in df_sub.groupby("cidade", dropna=False, sort=False):
        lat, lon = _centroide(grupo)
        rows.append(
            {
                "cidade": _safe_text(cidade),
                "peso_total_cidade": round(_peso_total(grupo), 3),
                "qtd_clientes_cidade": int(grupo["destinatario"].nunique()),
                "qtd_linhas_cidade": int(len(grupo)),
                "ordem_cidade_base": _safe_int(grupo["ordem_cidade_na_subregiao_m5_4a"].min(), 999999),
                "lat_centroide": lat,
                "lon_centroide": lon,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        by=["peso_total_cidade", "ordem_cidade_base", "cidade"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _agrupar_clientes_cidade(df_cidade: pd.DataFrame) -> pd.DataFrame:
    if df_cidade.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for destinatario, grupo in df_cidade.groupby("destinatario", dropna=False, sort=False):
        lat, lon = _centroide(grupo)
        rows.append(
            {
                "destinatario": _safe_text(destinatario),
                "peso_total_cliente": round(_peso_total(grupo), 3),
                "qtd_linhas_cliente": int(len(grupo)),
                "ordem_cliente_base": _safe_int(grupo["ordem_cliente_na_cidade_m5_4a"].min(), 999999),
                "lat_centroide": lat,
                "lon_centroide": lon,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        by=["peso_total_cliente", "ordem_cliente_base", "destinatario"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _itens_do_cliente(df: pd.DataFrame, cidade: str, destinatario: str) -> pd.DataFrame:
    return df[
        (df["cidade"].astype(str) == _safe_text(cidade))
        & (df["destinatario"].astype(str) == _safe_text(destinatario))
    ].copy()


# -----------------------------------------------------------------------------------------
# Validação do conjunto
# -----------------------------------------------------------------------------------------
def _validar_conjunto(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> Tuple[bool, str]:
    if df_itens.empty:
        return False, "conjunto_vazio"

    peso_total = _peso_total(df_itens)
    vol_total = _volume_total(df_itens)
    km_total = _km_ref(df_itens)
    qtd_paradas = _qtd_paradas(df_itens)

    cap_peso = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    cap_vol = _safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0)
    max_entregas = _safe_int(vehicle_row.get("max_entregas"), 0)
    max_km = _safe_float(vehicle_row.get("max_km_distancia"), 0.0)
    ocup_max = _ocupacao_maxima_kg(vehicle_row)

    if cap_peso > 0 and peso_total > cap_peso:
        return False, "excesso_capacidade_peso"
    if ocup_max > 0 and peso_total > ocup_max:
        return False, "ocupacao_acima_maxima"
    if cap_vol > 0 and vol_total > cap_vol:
        return False, "excesso_capacidade_volume"
    if max_entregas > 0 and qtd_paradas > max_entregas:
        return False, "excesso_paradas"
    if max_km > 0 and km_total > max_km:
        return False, "raio_excedido"

    return True, "ok"


def _bate_ocupacao_minima(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> bool:
    return _peso_total(df_itens) >= _ocupacao_minima_kg(vehicle_row)


# -----------------------------------------------------------------------------------------
# Expansão entre cidades
# -----------------------------------------------------------------------------------------
def _distancia_cidade_para_composicao(
    cidade_row: pd.Series,
    composicao_df: pd.DataFrame,
) -> float:
    if composicao_df.empty:
        return 999999.0

    lat_comp, lon_comp = _centroide(composicao_df)
    return _haversine_km(
        _safe_float(cidade_row.get("lat_centroide"), math.nan),
        _safe_float(cidade_row.get("lon_centroide"), math.nan),
        lat_comp,
        lon_comp,
    )


def _ordenar_cidades_candidatas(
    cidades_df: pd.DataFrame,
    composicao_df: pd.DataFrame,
    cidade_anchor: str,
) -> pd.DataFrame:
    if cidades_df.empty:
        return cidades_df.copy()

    temp = cidades_df.copy()
    temp["_mesma_ancora"] = temp["cidade"].astype(str) == _safe_text(cidade_anchor)
    temp["_dist_comp"] = temp.apply(
        lambda row: _distancia_cidade_para_composicao(row, composicao_df),
        axis=1,
    )

    return temp.sort_values(
        by=["_mesma_ancora", "_dist_comp", "peso_total_cidade", "cidade"],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------------------
# Tentativa real por perfil fixo
# -----------------------------------------------------------------------------------------
def _enriquecer_com_cidade(
    composicao_df: pd.DataFrame,
    df_pool_sub: pd.DataFrame,
    cidade: str,
    vehicle_row: pd.Series,
    clientes_testados: set[Tuple[str, str]],
    auditoria_clientes: List[Dict[str, Any]],
) -> pd.DataFrame:
    df_cidade = df_pool_sub[df_pool_sub["cidade"].astype(str) == _safe_text(cidade)].copy()
    if df_cidade.empty:
        return composicao_df

    clientes_df = _agrupar_clientes_cidade(df_cidade)

    for _, cli_row in clientes_df.iterrows():
        destinatario = _safe_text(cli_row.get("destinatario"))
        chave = (_safe_text(cidade), destinatario)
        if chave in clientes_testados:
            continue

        clientes_testados.add(chave)

        df_cliente = _itens_do_cliente(df_pool_sub, cidade, destinatario)
        if df_cliente.empty:
            auditoria_clientes.append(
                {
                    "cidade": cidade,
                    "destinatario": destinatario,
                    "status_cliente": "rejeitado",
                    "motivo_cliente": "cliente_sem_itens",
                }
            )
            continue

        tentativa_df = pd.concat([composicao_df, df_cliente], ignore_index=True)
        ok, motivo = _validar_conjunto(tentativa_df, vehicle_row)
        if not ok:
            auditoria_clientes.append(
                {
                    "cidade": cidade,
                    "destinatario": destinatario,
                    "status_cliente": "rejeitado",
                    "motivo_cliente": motivo,
                }
            )
            continue

        composicao_df = tentativa_df
        auditoria_clientes.append(
            {
                "cidade": cidade,
                "destinatario": destinatario,
                "status_cliente": "aceito",
                "motivo_cliente": "ok",
                "peso_pos_cliente": round(_peso_total(composicao_df), 3),
                "paradas_pos_cliente": int(_qtd_paradas(composicao_df)),
            }
        )

    return composicao_df


def _tentar_compor_com_perfil(
    df_pool_sub: pd.DataFrame,
    cidade_anchor: str,
    vehicle_row: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    auditoria_clientes: List[Dict[str, Any]] = []
    clientes_testados: set[Tuple[str, str]] = set()
    composicao_df = pd.DataFrame(columns=df_pool_sub.columns)

    # 1. ancora
    composicao_df = _enriquecer_com_cidade(
        composicao_df=composicao_df,
        df_pool_sub=df_pool_sub,
        cidade=cidade_anchor,
        vehicle_row=vehicle_row,
        clientes_testados=clientes_testados,
        auditoria_clientes=auditoria_clientes,
    )

    # 2. se ainda não bateu mínimo, expande
    if not composicao_df.empty and not _bate_ocupacao_minima(composicao_df, vehicle_row):
        cidades_df = _agrupar_cidades(df_pool_sub)
        cidades_ord = _ordenar_cidades_candidatas(
            cidades_df=cidades_df,
            composicao_df=composicao_df,
            cidade_anchor=cidade_anchor,
        )

        for _, cidade_row in cidades_ord.iterrows():
            cidade = _safe_text(cidade_row.get("cidade"))
            if cidade == _safe_text(cidade_anchor):
                continue

            composicao_df = _enriquecer_com_cidade(
                composicao_df=composicao_df,
                df_pool_sub=df_pool_sub,
                cidade=cidade,
                vehicle_row=vehicle_row,
                clientes_testados=clientes_testados,
                auditoria_clientes=auditoria_clientes,
            )

            if not composicao_df.empty and _bate_ocupacao_minima(composicao_df, vehicle_row):
                break

    if composicao_df.empty:
        return (
            pd.DataFrame(),
            {
                "status_tentativa": "falha",
                "motivo_tentativa": "nenhum_cliente_aceito",
            },
            auditoria_clientes,
        )

    ok_final, motivo_final = _validar_conjunto(composicao_df, vehicle_row)
    bate_min = _bate_ocupacao_minima(composicao_df, vehicle_row)

    if ok_final and bate_min:
        return (
            composicao_df.reset_index(drop=True).copy(),
            {
                "status_tentativa": "sucesso",
                "motivo_tentativa": "ok",
                "peso_total": round(_peso_total(composicao_df), 3),
                "qtd_paradas": int(_qtd_paradas(composicao_df)),
                "km_ref": round(_km_ref(composicao_df), 2),
            },
            auditoria_clientes,
        )

    return (
        pd.DataFrame(),
        {
            "status_tentativa": "falha",
            "motivo_tentativa": motivo_final if not ok_final else "abaixo_ocupacao_minima",
            "peso_total_tentado": round(_peso_total(composicao_df), 3),
            "qtd_paradas_tentado": int(_qtd_paradas(composicao_df)),
            "km_ref_tentado": round(_km_ref(composicao_df), 2),
        },
        auditoria_clientes,
    )


# -----------------------------------------------------------------------------------------
# Registro de manifesto
# -----------------------------------------------------------------------------------------
def _registro_manifesto(
    manifesto_id: str,
    subregiao: str,
    uf: str,
    vehicle_row: pd.Series,
    df_itens: pd.DataFrame,
    cidade_anchor: str,
) -> Dict[str, Any]:
    return {
        "premanifesto_id": manifesto_id,
        "subregiao": subregiao,
        "uf": uf,
        "perfil": _safe_text(vehicle_row.get("perfil")),
        "tipo": _safe_text(vehicle_row.get("tipo")),
        "cidade_anchor": cidade_anchor,
        "qtd_linhas": int(len(df_itens)),
        "qtd_paradas": int(_qtd_paradas(df_itens)),
        "peso_total_kg": round(_peso_total(df_itens), 3),
        "vol_total_m3": round(_volume_total(df_itens), 3),
        "km_referencia": round(_km_ref(df_itens), 2),
        "capacidade_peso_kg": round(_safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0), 3),
        "ocupacao_minima_kg": round(_ocupacao_minima_kg(vehicle_row), 3),
        "ocupacao_maxima_kg": round(_ocupacao_maxima_kg(vehicle_row), 3),
        "max_entregas": _safe_int(vehicle_row.get("max_entregas"), 0),
        "max_km_distancia": round(_safe_float(vehicle_row.get("max_km_distancia"), 0.0), 3),
        "origem_modulo": "M5.4C",
        "origem_etapa": "composicao_subregioes",
    }


# -----------------------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------------------
def executar_m5_4c_composicao_subregioes(
    df_base_preparada_m5_4a: pd.DataFrame,
    df_frota_aderente_m5_4b: pd.DataFrame,
    rodada_id: Optional[str] = None,
    data_base_roteirizacao: Optional[Any] = None,
    tipo_roteirizacao: str = "carteira",
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    del rodada_id, kwargs

    base, frota = _normalizar_inputs(
        df_base_preparada_m5_4a=df_base_preparada_m5_4a,
        df_frota_aderente_m5_4b=df_frota_aderente_m5_4b,
    )

    if base.empty:
        outputs_vazio = {
            "df_premanifestos_m5_4c": pd.DataFrame(),
            "df_itens_premanifestados_m5_4c": pd.DataFrame(),
            "df_remanescente_m5_4c": pd.DataFrame(),
            "df_tentativas_perfis_m5_4c": pd.DataFrame(),
            "df_tentativas_clientes_m5_4c": pd.DataFrame(),
        }
        meta_vazio = {
            "resumo_m5_4c": {
                "modulo": "M5.4C",
                "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
                "tipo_roteirizacao": tipo_roteirizacao,
                "linhas_entrada_m5_4c": 0,
                "premanifestos_gerados_m5_4c": 0,
                "itens_premanifestados_m5_4c": 0,
                "remanescente_m5_4c": 0,
                "estrategia_m5_4c": [
                    "subregiao_por_subregiao",
                    "cidade_ancora_por_maior_massa",
                    "perfil_aderente_do_maior_para_o_menor",
                    "expansao_multicidade",
                    "saldo_vivo_da_subregiao",
                    "VERSAO_M5_4C_2026_04_13",
                ],
                "caminhos_pipeline": caminhos_pipeline or {},
            }
        }
        return outputs_vazio, meta_vazio

    manifestos_rows: List[Dict[str, Any]] = []
    itens_manifestados_list: List[pd.DataFrame] = []
    remanescente_list: List[pd.DataFrame] = []
    tentativas_perfis_rows: List[Dict[str, Any]] = []
    tentativas_clientes_rows: List[Dict[str, Any]] = []

    manifesto_seq = 1

    subregioes = (
        base[["subregiao", "uf", "ordem_subregiao_m5_4a"]]
        .drop_duplicates()
        .sort_values(
            by=["ordem_subregiao_m5_4a", "subregiao", "uf"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    for _, sub_row in subregioes.iterrows():
        subregiao = _safe_text(sub_row.get("subregiao"))
        uf = _safe_text(sub_row.get("uf"))

        df_pool_sub = base[(base["subregiao"] == subregiao) & (base["uf"] == uf)].copy()
        if df_pool_sub.empty:
            continue

        perfis_sub = frota[(frota["subregiao"] == subregiao) & (frota["uf"] == uf)].copy()
        if perfis_sub.empty:
            df_pool_sub["status_m5_4c"] = "remanescente_m5_4c"
            df_pool_sub["motivo_m5_4c"] = "subregiao_sem_frota_aderente"
            remanescente_list.append(df_pool_sub)
            continue

        perfis_sub = perfis_sub.sort_values(
            by=["ordem_tentativa_perfil_m5_4b", "capacidade_peso_kg"],
            ascending=[True, False],
            kind="mergesort",
        ).reset_index(drop=True)

        while True:
            cidades_df = _agrupar_cidades(df_pool_sub)
            if cidades_df.empty:
                break

            houve_fechamento = False

            for _, cidade_row in cidades_df.iterrows():
                cidade_anchor = _safe_text(cidade_row.get("cidade"))

                for _, vehicle_row in perfis_sub.iterrows():
                    manifesto_df, info_tentativa, auditoria_clientes = _tentar_compor_com_perfil(
                        df_pool_sub=df_pool_sub,
                        cidade_anchor=cidade_anchor,
                        vehicle_row=vehicle_row,
                    )

                    tentativas_perfis_rows.append(
                        {
                            "subregiao": subregiao,
                            "uf": uf,
                            "cidade_anchor": cidade_anchor,
                            "perfil": _safe_text(vehicle_row.get("perfil")),
                            "tipo": _safe_text(vehicle_row.get("tipo")),
                            "status_tentativa": _safe_text(info_tentativa.get("status_tentativa")),
                            "motivo_tentativa": _safe_text(info_tentativa.get("motivo_tentativa")),
                            "peso_total_tentativa": round(_safe_float(info_tentativa.get("peso_total", info_tentativa.get("peso_total_tentado", 0.0)), 0.0), 3),
                            "qtd_paradas_tentativa": _safe_int(info_tentativa.get("qtd_paradas", info_tentativa.get("qtd_paradas_tentado", 0)), 0),
                            "km_ref_tentativa": round(_safe_float(info_tentativa.get("km_ref", info_tentativa.get("km_ref_tentado", 0.0)), 0.0), 2),
                        }
                    )

                    for reg in auditoria_clientes:
                        tentativas_clientes_rows.append(
                            {
                                "subregiao": subregiao,
                                "uf": uf,
                                "cidade_anchor": cidade_anchor,
                                "perfil": _safe_text(vehicle_row.get("perfil")),
                                "tipo": _safe_text(vehicle_row.get("tipo")),
                                **reg,
                            }
                        )

                    if manifesto_df.empty:
                        continue

                    manifesto_id = f"PM54C_{manifesto_seq:04d}"
                    manifesto_seq += 1

                    manifestos_rows.append(
                        _registro_manifesto(
                            manifesto_id=manifesto_id,
                            subregiao=subregiao,
                            uf=uf,
                            vehicle_row=vehicle_row,
                            df_itens=manifesto_df,
                            cidade_anchor=cidade_anchor,
                        )
                    )

                    manifesto_df = manifesto_df.copy()
                    manifesto_df["premanifesto_id"] = manifesto_id
                    manifesto_df["perfil_m5_4c"] = _safe_text(vehicle_row.get("perfil"))
                    manifesto_df["tipo_m5_4c"] = _safe_text(vehicle_row.get("tipo"))
                    manifesto_df["cidade_anchor_m5_4c"] = cidade_anchor
                    manifesto_df["origem_modulo"] = "M5.4C"
                    manifesto_df["origem_etapa"] = "composicao_subregioes"
                    itens_manifestados_list.append(manifesto_df)

                    used_ids = set(manifesto_df["id_linha_pipeline"].astype(str).tolist())
                    df_pool_sub = df_pool_sub[
                        ~df_pool_sub["id_linha_pipeline"].astype(str).isin(used_ids)
                    ].copy()

                    houve_fechamento = True
                    break

                if houve_fechamento:
                    break

            if not houve_fechamento:
                break

        if not df_pool_sub.empty:
            df_pool_sub["status_m5_4c"] = "remanescente_m5_4c"
            df_pool_sub["motivo_m5_4c"] = "nao_composto_apos_exaurir_subregiao"
            remanescente_list.append(df_pool_sub)

    df_premanifestos_m5_4c = pd.DataFrame(manifestos_rows)
    df_itens_premanifestados_m5_4c = (
        pd.concat(itens_manifestados_list, ignore_index=True) if itens_manifestados_list else pd.DataFrame()
    )
    df_remanescente_m5_4c = (
        pd.concat(remanescente_list, ignore_index=True) if remanescente_list else pd.DataFrame()
    )
    df_tentativas_perfis_m5_4c = pd.DataFrame(tentativas_perfis_rows)
    df_tentativas_clientes_m5_4c = pd.DataFrame(tentativas_clientes_rows)

    resumo_m5_4c = {
        "modulo": "M5.4C",
        "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
        "tipo_roteirizacao": tipo_roteirizacao,
        "linhas_entrada_m5_4c": int(len(base)),
        "subregioes_processadas_m5_4c": int(subregioes["subregiao"].nunique()),
        "premanifestos_gerados_m5_4c": int(len(df_premanifestos_m5_4c)),
        "itens_premanifestados_m5_4c": int(len(df_itens_premanifestados_m5_4c)),
        "remanescente_m5_4c": int(len(df_remanescente_m5_4c)),
        "tentativas_perfis_m5_4c": int(len(df_tentativas_perfis_m5_4c)),
        "tentativas_clientes_m5_4c": int(len(df_tentativas_clientes_m5_4c)),
        "estrategia_m5_4c": [
            "subregiao_por_subregiao",
            "cidade_ancora_por_maior_massa",
            "perfil_aderente_do_maior_para_o_menor",
            "expansao_multicidade",
            "saldo_vivo_da_subregiao",
            "VERSAO_M5_4C_2026_04_13",
        ],
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    outputs_m5_4c = {
        "df_premanifestos_m5_4c": df_premanifestos_m5_4c,
        "df_itens_premanifestados_m5_4c": df_itens_premanifestados_m5_4c,
        "df_remanescente_m5_4c": df_remanescente_m5_4c,
        "df_tentativas_perfis_m5_4c": df_tentativas_perfis_m5_4c,
        "df_tentativas_clientes_m5_4c": df_tentativas_clientes_m5_4c,
    }

    meta_m5_4c = {
        "resumo_m5_4c": resumo_m5_4c,
    }

    return outputs_m5_4c, meta_m5_4c


# Aliases defensivos
def executar_m5_composicao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4c_composicao_subregioes(*args, **kwargs)


def processar_m5_4c_composicao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4c_composicao_subregioes(*args, **kwargs)


def rodar_m5_4c_composicao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_4c_composicao_subregioes(*args, **kwargs)
