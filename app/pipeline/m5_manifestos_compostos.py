from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =========================================================================================
# M5.1A - TRIAGEM DE CIDADES E PERFIS VIÁVEIS
# -----------------------------------------------------------------------------------------
# OBJETIVO
# - receber SOMENTE o remanescente oficial do M4
# - agrupar por cidade
# - eliminar cidades inviáveis antes de qualquer composição
# - identificar perfis viáveis por cidade
# - devolver base auditável para a próxima etapa do M5
#
# ESTA ETAPA NÃO GERA PRÉ-MANIFESTO
# -----------------------------------------------------------------------------------------
# REGRAS DESTA ETAPA
# 1) cidade só segue se o peso total da cidade atingir a ocupação mínima do menor perfil
#    compatível por raio
# 2) perfil só fica viável na cidade se:
#    - o raio do perfil suportar a cidade
#    - o peso total da cidade atingir a ocupação mínima desse perfil
# 3) nada é composto aqui; só triagem
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
    df_remanescente_roteirizavel_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = (
        df_remanescente_roteirizavel_bloco_4.copy()
        if df_remanescente_roteirizavel_bloco_4 is not None
        else pd.DataFrame()
    )
    veic = df_veiculos_tratados.copy() if df_veiculos_tratados is not None else pd.DataFrame()

    if df.empty:
        return df, veic

    rename_map: Dict[str, str] = {}
    if "sub_regiao" in df.columns and "subregiao" not in df.columns:
        rename_map["sub_regiao"] = "subregiao"
    if "mesoregiao" in df.columns and "mesorregiao" not in df.columns:
        rename_map["mesoregiao"] = "mesorregiao"
    if rename_map:
        df = df.rename(columns=rename_map)

    defaults = {
        "id_linha_pipeline": None,
        "cidade": "",
        "uf": "",
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
    }
    for col, default in defaults.items():
        _ensure_column(df, col, default)

    if df["id_linha_pipeline"].isna().any():
        raise ValueError("M5.1A exige id_linha_pipeline em todas as linhas do remanescente do M4.")

    if "peso_calculado" not in df.columns and "peso_c" in df.columns:
        df["peso_calculado"] = df["peso_c"]

    if "peso_kg" not in df.columns:
        df["peso_kg"] = df["peso_calculado"]

    if "distancia_rodoviaria_est_km" not in df.columns and "distancia_km" in df.columns:
        df["distancia_rodoviaria_est_km"] = df["distancia_km"]

    numeric_cols = [
        "peso_calculado",
        "peso_kg",
        "vol_m3",
        "distancia_rodoviaria_est_km",
        "folga_dias",
        "prioridade_embarque_num",
        "prioridade_embarque",
        "ranking_prioridade_operacional",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = ["agendada", "veiculo_exclusivo", "veiculo_exclusivo_flag"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(_safe_bool)

    if veic.empty:
        raise ValueError("M5.1A exige df_veiculos_tratados.")

    if "tipo" not in veic.columns and "perfil" in veic.columns:
        veic["tipo"] = veic["perfil"]
    if "perfil" not in veic.columns and "tipo" in veic.columns:
        veic["perfil"] = veic["tipo"]

    for col in [
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ocupacao_minima_perc",
        "ocupacao_maxima_perc",
    ]:
        if col not in veic.columns:
            veic[col] = pd.NA
        veic[col] = pd.to_numeric(veic[col], errors="coerce")

    for col in ["tipo", "perfil"]:
        if col in veic.columns:
            veic[col] = veic[col].astype(str)

    return df, veic


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
    temp["_id_str_m5_1a"] = temp["id_linha_pipeline"].astype(str)
    temp["_cidade_key_m5_1a"] = temp["cidade"].fillna("").astype(str).str.strip()
    temp["_uf_key_m5_1a"] = temp["uf"].fillna("").astype(str).str.strip()
    temp["_cliente_key_m5_1a"] = temp["destinatario"].fillna("").astype(str).str.strip()

    prioridade = pd.to_numeric(
        temp["prioridade_embarque_num"].where(
            temp["prioridade_embarque_num"].notna(),
            temp["prioridade_embarque"],
        ),
        errors="coerce",
    )
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

    temp["_bucket_m5_1a"] = buckets
    temp["_prioridade_ord_m5_1a"] = prioridade_ord
    temp["_folga_ord_m5_1a"] = folga
    temp["_ranking_ord_m5_1a"] = ranking
    temp["_km_ord_m5_1a"] = km
    temp["_peso_ord_m5_1a"] = -peso

    return temp


def _ordenar_operacional(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "_bucket_m5_1a" not in df.columns:
        df = _precalcular_ordenacao(df)

    return (
        df.sort_values(
            by=[
                "_bucket_m5_1a",
                "_prioridade_ord_m5_1a",
                "_folga_ord_m5_1a",
                "_ranking_ord_m5_1a",
                "_km_ord_m5_1a",
                "_peso_ord_m5_1a",
                "_id_str_m5_1a",
            ],
            ascending=[True, True, True, True, True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )


# -----------------------------------------------------------------------------------------
# Veículos
# -----------------------------------------------------------------------------------------
def _veiculos_menor_para_maior(df_veiculos: pd.DataFrame) -> pd.DataFrame:
    temp = df_veiculos.copy()
    temp["_cap_peso_tmp"] = pd.to_numeric(temp["capacidade_peso_kg"], errors="coerce").fillna(0)
    temp["_cap_vol_tmp"] = pd.to_numeric(temp["capacidade_vol_m3"], errors="coerce").fillna(0)
    temp = (
        temp.sort_values(["_cap_peso_tmp", "_cap_vol_tmp"], ascending=[True, True], kind="mergesort")
        .drop(columns=["_cap_peso_tmp", "_cap_vol_tmp"])
        .reset_index(drop=True)
    )
    return temp


# -----------------------------------------------------------------------------------------
# Métricas do grupo
# -----------------------------------------------------------------------------------------
def _peso_total(df_itens: pd.DataFrame) -> float:
    if df_itens.empty:
        return 0.0
    return float(df_itens["peso_calculado"].fillna(0).sum())


def _volume_total(df_itens: pd.DataFrame) -> float:
    if df_itens.empty:
        return 0.0
    return float(df_itens["vol_m3"].fillna(0).sum())


def _km_referencia(df_itens: pd.DataFrame) -> float:
    if df_itens.empty:
        return 0.0
    return float(df_itens["distancia_rodoviaria_est_km"].fillna(0).max())


def _qtd_paradas(df_itens: pd.DataFrame) -> int:
    if df_itens.empty:
        return 0
    return int(df_itens["destinatario"].fillna("").astype(str).nunique())


# -----------------------------------------------------------------------------------------
# Regras de viabilidade
# -----------------------------------------------------------------------------------------
def _ocupacao_minima_kg(vehicle_row: pd.Series) -> float:
    cap_peso = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    ocup_min = _safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0)
    return cap_peso * (ocup_min / 100.0)


def _perfil_compativel_por_raio(city_df: pd.DataFrame, vehicle_row: pd.Series) -> bool:
    km_city = _km_referencia(city_df)
    max_km = _safe_float(vehicle_row.get("max_km_distancia"), 0.0)
    if max_km <= 0:
        return False
    return km_city <= max_km


def _perfil_viavel_na_cidade(city_df: pd.DataFrame, vehicle_row: pd.Series) -> Tuple[bool, str]:
    if not _perfil_compativel_por_raio(city_df, vehicle_row):
        return False, "raio_incompativel"

    peso_city = _peso_total(city_df)
    peso_minimo = _ocupacao_minima_kg(vehicle_row)

    if peso_city < peso_minimo:
        return False, "abaixo_ocupacao_minima"

    return True, "ok"


def _menor_perfil_compativel_por_raio(
    city_df: pd.DataFrame,
    veiculos_menor_maior: pd.DataFrame,
) -> Optional[pd.Series]:
    for _, vehicle_row in veiculos_menor_maior.iterrows():
        if _perfil_compativel_por_raio(city_df, vehicle_row):
            return vehicle_row.copy()
    return None


# -----------------------------------------------------------------------------------------
# Serialização interna de auditoria por cidade
# -----------------------------------------------------------------------------------------
def _registro_cidade(
    city_key: str,
    uf_key: str,
    city_df: pd.DataFrame,
    status_triagem: str,
    motivo: str,
    menor_perfil_compativel: Optional[pd.Series],
) -> Dict[str, Any]:
    peso_city = _peso_total(city_df)
    volume_city = _volume_total(city_df)
    km_city = _km_referencia(city_df)
    paradas_city = _qtd_paradas(city_df)

    return {
        "cidade": city_key,
        "uf": uf_key,
        "status_triagem_cidade": status_triagem,
        "motivo_triagem_cidade": motivo,
        "qtd_linhas": int(len(city_df)),
        "qtd_paradas": int(paradas_city),
        "peso_total_cidade": round(peso_city, 3),
        "volume_total_cidade": round(volume_city, 3),
        "km_referencia_cidade": round(km_city, 2),
        "menor_perfil_compativel_por_raio": None
        if menor_perfil_compativel is None
        else _safe_text(menor_perfil_compativel.get("tipo")),
        "ocupacao_minima_kg_menor_perfil": None
        if menor_perfil_compativel is None
        else round(_ocupacao_minima_kg(menor_perfil_compativel), 3),
        "capacidade_peso_kg_menor_perfil": None
        if menor_perfil_compativel is None
        else round(_safe_float(menor_perfil_compativel.get("capacidade_peso_kg"), 0.0), 3),
        "ocupacao_minima_perc_menor_perfil": None
        if menor_perfil_compativel is None
        else round(_safe_float(menor_perfil_compativel.get("ocupacao_minima_perc"), 70.0), 3),
    }


# -----------------------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------------------
def executar_m5_1_triagem_cidades(
    df_remanescente_roteirizavel_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
    rodada_id: Optional[str] = None,
    data_base_roteirizacao: Optional[Any] = None,
    tipo_roteirizacao: str = "carteira",
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    del rodada_id, kwargs

    df_input, df_veic = _normalizar_inputs(
        df_remanescente_roteirizavel_bloco_4=df_remanescente_roteirizavel_bloco_4,
        df_veiculos_tratados=df_veiculos_tratados,
    )

    if df_input.empty:
        outputs_vazio = {
            "df_saldo_elegivel_composicao_m5_1": pd.DataFrame(),
            "df_saldo_excluido_triagem_m5_1": pd.DataFrame(),
            "df_cidades_viaveis_m5_1": pd.DataFrame(),
            "df_cidades_inviaveis_m5_1": pd.DataFrame(),
            "df_perfis_viaveis_por_cidade_m5_1": pd.DataFrame(),
        }
        meta_vazio = {
            "resumo_m5_1_triagem": {
                "modulo": "M5.1A",
                "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
                "tipo_roteirizacao": tipo_roteirizacao,
                "remanescente_entrada_m5_1": 0,
                "cidades_total_analisadas": 0,
                "cidades_viaveis": 0,
                "cidades_inviaveis": 0,
                "linhas_elegiveis_composicao": 0,
                "linhas_excluidas_triagem": 0,
                "perfis_viaveis_total": 0,
                "estrategia_m5_1_triagem": [
                    "agrupamento_por_cidade",
                    "eliminacao_cidades_inviaveis",
                    "eliminacao_perfis_inviaveis",
                    "VERSAO_M5_1A_2026_04_10",
                ],
                "caminhos_pipeline": caminhos_pipeline or {},
            },
            "auditoria_m5_1_triagem": {
                "total_cidades": 0,
                "total_cidades_viaveis": 0,
                "total_cidades_inviaveis": 0,
                "total_linhas_elegiveis": 0,
                "total_linhas_excluidas": 0,
                "total_perfis_viaveis": 0,
            },
        }
        return outputs_vazio, meta_vazio

    saldo = _precalcular_ordenacao(df_input.copy())
    saldo = _ordenar_operacional(saldo)
    veiculos_menor_maior = _veiculos_menor_para_maior(df_veic)

    cidades_viaveis_rows: List[Dict[str, Any]] = []
    cidades_inviaveis_rows: List[Dict[str, Any]] = []
    perfis_viaveis_rows: List[Dict[str, Any]] = []

    df_elegiveis_list: List[pd.DataFrame] = []
    df_excluidas_list: List[pd.DataFrame] = []

    cidades_keys = (
        saldo[["_cidade_key_m5_1a", "_uf_key_m5_1a"]]
        .drop_duplicates()
        .sort_values(["_cidade_key_m5_1a", "_uf_key_m5_1a"], kind="mergesort")
        .values.tolist()
    )

    for cidade_key, uf_key in cidades_keys:
        city_df = saldo[
            (saldo["_cidade_key_m5_1a"] == cidade_key)
            & (saldo["_uf_key_m5_1a"] == uf_key)
        ].copy()

        if city_df.empty:
            continue

        menor_perfil = _menor_perfil_compativel_por_raio(
            city_df=city_df,
            veiculos_menor_maior=veiculos_menor_maior,
        )

        if menor_perfil is None:
            cidades_inviaveis_rows.append(
                _registro_cidade(
                    city_key=cidade_key,
                    uf_key=uf_key,
                    city_df=city_df,
                    status_triagem="inviavel",
                    motivo="sem_perfil_compativel_por_raio",
                    menor_perfil_compativel=None,
                )
            )
            city_df["status_triagem_m5_1"] = "cidade_inviavel"
            city_df["motivo_triagem_m5_1"] = "sem_perfil_compativel_por_raio"
            df_excluidas_list.append(city_df)
            continue

        peso_city = _peso_total(city_df)
        peso_minimo_menor = _ocupacao_minima_kg(menor_perfil)

        if peso_city < peso_minimo_menor:
            cidades_inviaveis_rows.append(
                _registro_cidade(
                    city_key=cidade_key,
                    uf_key=uf_key,
                    city_df=city_df,
                    status_triagem="inviavel",
                    motivo="peso_total_cidade_abaixo_da_ocupacao_minima_do_menor_perfil_compativel",
                    menor_perfil_compativel=menor_perfil,
                )
            )
            city_df["status_triagem_m5_1"] = "cidade_inviavel"
            city_df["motivo_triagem_m5_1"] = (
                "peso_total_cidade_abaixo_da_ocupacao_minima_do_menor_perfil_compativel"
            )
            df_excluidas_list.append(city_df)
            continue

        cidades_viaveis_rows.append(
            _registro_cidade(
                city_key=cidade_key,
                uf_key=uf_key,
                city_df=city_df,
                status_triagem="viavel",
                motivo="ok",
                menor_perfil_compativel=menor_perfil,
            )
        )

        city_df["status_triagem_m5_1"] = "cidade_viavel"
        city_df["motivo_triagem_m5_1"] = "ok"
        df_elegiveis_list.append(city_df)

        for _, vehicle_row in veiculos_menor_maior.iterrows():
            ok, motivo = _perfil_viavel_na_cidade(city_df, vehicle_row)

            if ok:
                perfis_viaveis_rows.append(
                    {
                        "cidade": cidade_key,
                        "uf": uf_key,
                        "perfil": _safe_text(vehicle_row.get("perfil")),
                        "tipo": _safe_text(vehicle_row.get("tipo")),
                        "capacidade_peso_kg": round(
                            _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0),
                            3,
                        ),
                        "capacidade_vol_m3": round(
                            _safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0),
                            3,
                        ),
                        "max_entregas": _safe_int(vehicle_row.get("max_entregas"), 0),
                        "max_km_distancia": round(
                            _safe_float(vehicle_row.get("max_km_distancia"), 0.0),
                            3,
                        ),
                        "ocupacao_minima_perc": round(
                            _safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0),
                            3,
                        ),
                        "ocupacao_minima_kg": round(_ocupacao_minima_kg(vehicle_row), 3),
                        "peso_total_cidade": round(peso_city, 3),
                        "km_referencia_cidade": round(_km_referencia(city_df), 2),
                        "status_perfil_cidade": "viavel",
                        "motivo_status_perfil_cidade": "ok",
                    }
                )

    df_saldo_elegivel_composicao_m5_1 = (
        pd.concat(df_elegiveis_list, ignore_index=True) if df_elegiveis_list else pd.DataFrame()
    )
    df_saldo_excluido_triagem_m5_1 = (
        pd.concat(df_excluidas_list, ignore_index=True) if df_excluidas_list else pd.DataFrame()
    )
    df_cidades_viaveis_m5_1 = pd.DataFrame(cidades_viaveis_rows)
    df_cidades_inviaveis_m5_1 = pd.DataFrame(cidades_inviaveis_rows)
    df_perfis_viaveis_por_cidade_m5_1 = pd.DataFrame(perfis_viaveis_rows)

    cols_drop = [
        "_id_str_m5_1a",
        "_cidade_key_m5_1a",
        "_uf_key_m5_1a",
        "_cliente_key_m5_1a",
        "_bucket_m5_1a",
        "_prioridade_ord_m5_1a",
        "_folga_ord_m5_1a",
        "_ranking_ord_m5_1a",
        "_km_ord_m5_1a",
        "_peso_ord_m5_1a",
    ]

    if not df_saldo_elegivel_composicao_m5_1.empty:
        df_saldo_elegivel_composicao_m5_1 = (
            df_saldo_elegivel_composicao_m5_1.drop(columns=cols_drop, errors="ignore")
            .reset_index(drop=True)
            .copy()
        )

    if not df_saldo_excluido_triagem_m5_1.empty:
        df_saldo_excluido_triagem_m5_1 = (
            df_saldo_excluido_triagem_m5_1.drop(columns=cols_drop, errors="ignore")
            .reset_index(drop=True)
            .copy()
        )

    resumo_m5_1_triagem = {
        "modulo": "M5.1A",
        "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
        "tipo_roteirizacao": tipo_roteirizacao,
        "remanescente_entrada_m5_1": int(len(df_input)),
        "cidades_total_analisadas": int(len(cidades_keys)),
        "cidades_viaveis": int(len(df_cidades_viaveis_m5_1)),
        "cidades_inviaveis": int(len(df_cidades_inviaveis_m5_1)),
        "linhas_elegiveis_composicao": int(len(df_saldo_elegivel_composicao_m5_1)),
        "linhas_excluidas_triagem": int(len(df_saldo_excluido_triagem_m5_1)),
        "perfis_viaveis_total": int(len(df_perfis_viaveis_por_cidade_m5_1)),
        "estrategia_m5_1_triagem": [
            "agrupamento_por_cidade",
            "eliminacao_cidades_inviaveis",
            "eliminacao_perfis_inviaveis",
            "VERSAO_M5_1A_2026_04_10",
        ],
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    auditoria_m5_1_triagem = {
        "total_cidades": int(len(cidades_keys)),
        "total_cidades_viaveis": int(len(df_cidades_viaveis_m5_1)),
        "total_cidades_inviaveis": int(len(df_cidades_inviaveis_m5_1)),
        "total_linhas_elegiveis": int(len(df_saldo_elegivel_composicao_m5_1)),
        "total_linhas_excluidas": int(len(df_saldo_excluido_triagem_m5_1)),
        "total_perfis_viaveis": int(len(df_perfis_viaveis_por_cidade_m5_1)),
    }

    outputs_m5_1_triagem = {
        "df_saldo_elegivel_composicao_m5_1": df_saldo_elegivel_composicao_m5_1,
        "df_saldo_excluido_triagem_m5_1": df_saldo_excluido_triagem_m5_1,
        "df_cidades_viaveis_m5_1": df_cidades_viaveis_m5_1,
        "df_cidades_inviaveis_m5_1": df_cidades_inviaveis_m5_1,
        "df_perfis_viaveis_por_cidade_m5_1": df_perfis_viaveis_por_cidade_m5_1,
    }

    meta_m5_1_triagem = {
        "resumo_m5_1_triagem": resumo_m5_1_triagem,
        "auditoria_m5_1_triagem": auditoria_m5_1_triagem,
    }

    return outputs_m5_1_triagem, meta_m5_1_triagem


# Aliases defensivos
def executar_m5_triagem_cidades(*args: Any, **kwargs: Any):
    return executar_m5_1_triagem_cidades(*args, **kwargs)


def processar_m5_1_triagem_cidades(*args: Any, **kwargs: Any):
    return executar_m5_1_triagem_cidades(*args, **kwargs)


def rodar_m5_1_triagem_cidades(*args: Any, **kwargs: Any):
    return executar_m5_1_triagem_cidades(*args, **kwargs)
