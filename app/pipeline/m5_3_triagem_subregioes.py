from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =========================================================================================
# M5.3 - TRIAGEM DE SUBREGIÕES
# -----------------------------------------------------------------------------------------
# OBJETIVO
# - receber o saldo global pós-etapa cidade
# - agrupar por subregião
# - eliminar subregiões inviáveis antes da composição
# - devolver base auditável para a próxima etapa (M5.4)
#
# REGRA ATUAL DE VALIDAÇÃO
# - nesta etapa NÃO olhar raio
# - nesta etapa NÃO subir perfil
# - nesta etapa NÃO compor
# - usar SOMENTE o menor perfil cadastrado como régua base
# - se o peso total da subregião atingir a ocupação mínima do menor perfil cadastrado:
#       => elegível para M5.4
# - se não atingir:
#       => remanescente / excluído da triagem M5.3
#
# ESTA ETAPA NÃO GERA PRÉ-MANIFESTO
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
    df_saldo_global_pos_cidade_m5: pd.DataFrame,
    df_perfis_base_m5: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    saldo = (
        df_saldo_global_pos_cidade_m5.copy()
        if df_saldo_global_pos_cidade_m5 is not None
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
    }
    for col, default in defaults.items():
        _ensure_column(saldo, col, default)

    if saldo["id_linha_pipeline"].isna().any():
        raise ValueError("M5.3 exige id_linha_pipeline em todas as linhas do saldo global pós-cidade.")

    if "peso_calculado" not in saldo.columns and "peso_c" in saldo.columns:
        saldo["peso_calculado"] = saldo["peso_c"]

    if "peso_kg" not in saldo.columns:
        saldo["peso_kg"] = saldo["peso_calculado"]

    if "distancia_rodoviaria_est_km" not in saldo.columns and "distancia_km" in saldo.columns:
        saldo["distancia_rodoviaria_est_km"] = saldo["distancia_km"]

    for col in [
        "peso_calculado",
        "peso_kg",
        "vol_m3",
        "distancia_rodoviaria_est_km",
        "folga_dias",
        "prioridade_embarque_num",
        "prioridade_embarque",
        "ranking_prioridade_operacional",
    ]:
        if col in saldo.columns:
            saldo[col] = pd.to_numeric(saldo[col], errors="coerce")

    for col in ["agendada", "veiculo_exclusivo", "veiculo_exclusivo_flag"]:
        if col in saldo.columns:
            saldo[col] = saldo[col].apply(_safe_bool)

    for col in ["cidade", "uf", "subregiao", "mesorregiao", "destinatario"]:
        if col in saldo.columns:
            saldo[col] = saldo[col].fillna("").astype(str).str.strip()

    if perfis.empty:
        raise ValueError("M5.3 exige df_perfis_base_m5.")

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
    temp["_id_str_m5_3"] = temp["id_linha_pipeline"].astype(str)
    temp["_subregiao_key_m5_3"] = temp["subregiao"].fillna("").astype(str).str.strip()
    temp["_uf_key_m5_3"] = temp["uf"].fillna("").astype(str).str.strip()
    temp["_cliente_key_m5_3"] = temp["destinatario"].fillna("").astype(str).str.strip()

    folga = pd.to_numeric(temp["folga_dias"], errors="coerce").fillna(999)
    ranking = pd.to_numeric(temp["ranking_prioridade_operacional"], errors="coerce").fillna(999)
    km = pd.to_numeric(temp["distancia_rodoviaria_est_km"], errors="coerce").fillna(999999)
    peso = pd.to_numeric(temp["peso_calculado"], errors="coerce").fillna(0.0)

    buckets: List[int] = []
    prioridade_ord: List[float] = []

    for _, row in temp.iterrows():
        buckets.append(_fase_bucket(row))
        prioridade = pd.to_numeric(
            row.get("prioridade_embarque_num", row.get("prioridade_embarque", pd.NA)),
            errors="coerce",
        )
        prioridade_ord.append(_safe_float(prioridade, 999.0) if not pd.isna(prioridade) else 999.0)

    temp["_bucket_m5_3"] = buckets
    temp["_prioridade_ord_m5_3"] = prioridade_ord
    temp["_folga_ord_m5_3"] = folga
    temp["_ranking_ord_m5_3"] = ranking
    temp["_km_ord_m5_3"] = km
    temp["_peso_ord_m5_3"] = -peso

    return temp


def _ordenar_operacional(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "_bucket_m5_3" not in df.columns:
        df = _precalcular_ordenacao(df)

    return (
        df.sort_values(
            by=[
                "_bucket_m5_3",
                "_prioridade_ord_m5_3",
                "_folga_ord_m5_3",
                "_ranking_ord_m5_3",
                "_km_ord_m5_3",
                "_peso_ord_m5_3",
                "_id_str_m5_3",
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
def _veiculos_menor_para_maior(df_perfis: pd.DataFrame) -> pd.DataFrame:
    temp = df_perfis.copy()
    temp["_cap_peso_tmp"] = pd.to_numeric(temp["capacidade_peso_kg"], errors="coerce").fillna(0)
    temp["_cap_vol_tmp"] = pd.to_numeric(temp["capacidade_vol_m3"], errors="coerce").fillna(0)

    return (
        temp.sort_values(
            ["_cap_peso_tmp", "_cap_vol_tmp", "tipo", "perfil"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )
        .drop(columns=["_cap_peso_tmp", "_cap_vol_tmp"], errors="ignore")
        .reset_index(drop=True)
        .copy()
    )


def _menor_perfil_cadastrado(veiculos_menor_maior: pd.DataFrame) -> pd.Series:
    if veiculos_menor_maior.empty:
        raise ValueError("M5.3 exige ao menos um perfil de veículo cadastrado.")
    return veiculos_menor_maior.iloc[0].copy()


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


# -----------------------------------------------------------------------------------------
# Auditoria por subregião
# -----------------------------------------------------------------------------------------
def _registro_subregiao(
    subregiao_key: str,
    uf_key: str,
    group_df: pd.DataFrame,
    status_triagem: str,
    motivo: str,
    menor_perfil_base: pd.Series,
) -> Dict[str, Any]:
    peso_group = _peso_total(group_df)
    volume_group = _volume_total(group_df)
    km_group = _km_referencia(group_df)
    paradas_group = _qtd_paradas(group_df)

    return {
        "subregiao": subregiao_key,
        "uf": uf_key,
        "status_triagem_subregiao": status_triagem,
        "motivo_triagem_subregiao": motivo,
        "qtd_linhas": int(len(group_df)),
        "qtd_paradas": int(paradas_group),
        "peso_total_subregiao": round(peso_group, 3),
        "volume_total_subregiao": round(volume_group, 3),
        "km_referencia_subregiao": round(km_group, 2),
        "perfil_base_triagem": _safe_text(menor_perfil_base.get("tipo")) or _safe_text(menor_perfil_base.get("perfil")),
        "ocupacao_minima_kg_perfil_base": round(_ocupacao_minima_kg(menor_perfil_base), 3),
        "capacidade_peso_kg_perfil_base": round(_safe_float(menor_perfil_base.get("capacidade_peso_kg"), 0.0), 3),
        "ocupacao_minima_perc_perfil_base": round(_safe_float(menor_perfil_base.get("ocupacao_minima_perc"), 70.0), 3),
    }


# -----------------------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------------------
def executar_m5_3_triagem_subregioes(
    df_saldo_global_pos_cidade_m5: pd.DataFrame,
    df_perfis_base_m5: pd.DataFrame,
    rodada_id: Optional[str] = None,
    data_base_roteirizacao: Optional[Any] = None,
    tipo_roteirizacao: str = "carteira",
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    del rodada_id, kwargs

    saldo, perfis = _normalizar_inputs(
        df_saldo_global_pos_cidade_m5=df_saldo_global_pos_cidade_m5,
        df_perfis_base_m5=df_perfis_base_m5,
    )

    if saldo.empty:
        outputs_vazio = {
            "df_saldo_elegivel_composicao_m5_3": pd.DataFrame(),
            "df_saldo_excluido_triagem_m5_3": pd.DataFrame(),
            "df_subregioes_viaveis_m5_3": pd.DataFrame(),
            "df_subregioes_inviaveis_m5_3": pd.DataFrame(),
            "df_perfis_viaveis_por_subregiao_m5_3": pd.DataFrame(),
        }
        meta_vazio = {
            "resumo_m5_3_triagem": {
                "modulo": "M5.3",
                "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
                "tipo_roteirizacao": tipo_roteirizacao,
                "linhas_entrada_m5_3": 0,
                "subregioes_total_analisadas": 0,
                "subregioes_viaveis": 0,
                "subregioes_inviaveis": 0,
                "linhas_elegiveis_composicao": 0,
                "linhas_excluidas_triagem": 0,
                "perfis_viaveis_total": 0,
                "estrategia_m5_3_triagem": [
                    "agrupamento_por_subregiao",
                    "triagem_somente_por_peso_agregado",
                    "uso_do_menor_perfil_cadastrado",
                    "sem_validacao_de_raio",
                    "sem_subida_de_perfil",
                    "VERSAO_M5_3_TRIAGEM_SIMPLIFICADA_2026_04_11",
                ],
                "caminhos_pipeline": caminhos_pipeline or {},
            },
            "auditoria_m5_3_triagem": {
                "total_subregioes": 0,
                "total_subregioes_viaveis": 0,
                "total_subregioes_inviaveis": 0,
                "total_linhas_elegiveis": 0,
                "total_linhas_excluidas": 0,
                "total_perfis_viaveis": 0,
            },
        }
        return outputs_vazio, meta_vazio

    saldo = _precalcular_ordenacao(saldo)
    saldo = _ordenar_operacional(saldo)

    veiculos_menor_maior = _veiculos_menor_para_maior(perfis)
    menor_perfil_base = _menor_perfil_cadastrado(veiculos_menor_maior)
    piso_minimo_menor_perfil = _ocupacao_minima_kg(menor_perfil_base)

    subregioes_viaveis_rows: List[Dict[str, Any]] = []
    subregioes_inviaveis_rows: List[Dict[str, Any]] = []
    perfis_viaveis_rows: List[Dict[str, Any]] = []

    df_elegiveis_list: List[pd.DataFrame] = []
    df_excluidas_list: List[pd.DataFrame] = []

    sub_keys = (
        saldo[["_subregiao_key_m5_3", "_uf_key_m5_3"]]
        .drop_duplicates()
        .sort_values(["_subregiao_key_m5_3", "_uf_key_m5_3"], kind="mergesort")
        .values.tolist()
    )

    for subregiao_key, uf_key in sub_keys:
        group_df = saldo[
            (saldo["_subregiao_key_m5_3"] == subregiao_key)
            & (saldo["_uf_key_m5_3"] == uf_key)
        ].copy()

        if group_df.empty:
            continue

        peso_group = _peso_total(group_df)

        if peso_group < piso_minimo_menor_perfil:
            subregioes_inviaveis_rows.append(
                _registro_subregiao(
                    subregiao_key=subregiao_key,
                    uf_key=uf_key,
                    group_df=group_df,
                    status_triagem="inviavel",
                    motivo="peso_total_subregiao_abaixo_do_piso_minimo_do_menor_perfil_cadastrado",
                    menor_perfil_base=menor_perfil_base,
                )
            )
            group_df["status_triagem_m5_3"] = "subregiao_inviavel"
            group_df["motivo_triagem_m5_3"] = (
                "peso_total_subregiao_abaixo_do_piso_minimo_do_menor_perfil_cadastrado"
            )
            group_df["perfil_base_triagem_m5_3"] = _safe_text(menor_perfil_base.get("tipo")) or _safe_text(
                menor_perfil_base.get("perfil")
            )
            group_df["capacidade_peso_kg_perfil_base_m5_3"] = round(
                _safe_float(menor_perfil_base.get("capacidade_peso_kg"), 0.0), 3
            )
            group_df["ocupacao_minima_perc_perfil_base_m5_3"] = round(
                _safe_float(menor_perfil_base.get("ocupacao_minima_perc"), 70.0), 3
            )
            group_df["piso_minimo_kg_perfil_base_m5_3"] = round(piso_minimo_menor_perfil, 3)
            group_df["peso_total_subregiao_m5_3"] = round(peso_group, 3)
            df_excluidas_list.append(group_df)
            continue

        subregioes_viaveis_rows.append(
            _registro_subregiao(
                subregiao_key=subregiao_key,
                uf_key=uf_key,
                group_df=group_df,
                status_triagem="viavel",
                motivo="ok",
                menor_perfil_base=menor_perfil_base,
            )
        )

        group_df["status_triagem_m5_3"] = "subregiao_viavel"
        group_df["motivo_triagem_m5_3"] = "ok"
        group_df["perfil_base_triagem_m5_3"] = _safe_text(menor_perfil_base.get("tipo")) or _safe_text(
            menor_perfil_base.get("perfil")
        )
        group_df["capacidade_peso_kg_perfil_base_m5_3"] = round(
            _safe_float(menor_perfil_base.get("capacidade_peso_kg"), 0.0), 3
        )
        group_df["ocupacao_minima_perc_perfil_base_m5_3"] = round(
            _safe_float(menor_perfil_base.get("ocupacao_minima_perc"), 70.0), 3
        )
        group_df["piso_minimo_kg_perfil_base_m5_3"] = round(piso_minimo_menor_perfil, 3)
        group_df["peso_total_subregiao_m5_3"] = round(peso_group, 3)
        df_elegiveis_list.append(group_df)

        perfis_viaveis_rows.append(
            {
                "subregiao": subregiao_key,
                "uf": uf_key,
                "perfil": _safe_text(menor_perfil_base.get("perfil")),
                "tipo": _safe_text(menor_perfil_base.get("tipo")),
                "capacidade_peso_kg": round(
                    _safe_float(menor_perfil_base.get("capacidade_peso_kg"), 0.0),
                    3,
                ),
                "capacidade_vol_m3": round(
                    _safe_float(menor_perfil_base.get("capacidade_vol_m3"), 0.0),
                    3,
                ),
                "max_entregas": _safe_int(menor_perfil_base.get("max_entregas"), 0),
                "max_km_distancia": round(
                    _safe_float(menor_perfil_base.get("max_km_distancia"), 0.0),
                    3,
                ),
                "ocupacao_minima_perc": round(
                    _safe_float(menor_perfil_base.get("ocupacao_minima_perc"), 70.0),
                    3,
                ),
                "ocupacao_minima_kg": round(piso_minimo_menor_perfil, 3),
                "peso_total_subregiao": round(peso_group, 3),
                "km_referencia_subregiao": round(_km_referencia(group_df), 2),
                "status_perfil_subregiao": "viavel",
                "motivo_status_perfil_subregiao": "ok",
                "regra_triagem_m5_3": "somente_menor_perfil_cadastrado_sem_raio",
            }
        )

    df_saldo_elegivel_composicao_m5_3 = (
        pd.concat(df_elegiveis_list, ignore_index=True) if df_elegiveis_list else pd.DataFrame()
    )
    df_saldo_excluido_triagem_m5_3 = (
        pd.concat(df_excluidas_list, ignore_index=True) if df_excluidas_list else pd.DataFrame()
    )
    df_subregioes_viaveis_m5_3 = pd.DataFrame(subregioes_viaveis_rows)
    df_subregioes_inviaveis_m5_3 = pd.DataFrame(subregioes_inviaveis_rows)
    df_perfis_viaveis_por_subregiao_m5_3 = pd.DataFrame(perfis_viaveis_rows)

    cols_drop = [
        "_id_str_m5_3",
        "_subregiao_key_m5_3",
        "_uf_key_m5_3",
        "_cliente_key_m5_3",
        "_bucket_m5_3",
        "_prioridade_ord_m5_3",
        "_folga_ord_m5_3",
        "_ranking_ord_m5_3",
        "_km_ord_m5_3",
        "_peso_ord_m5_3",
    ]

    if not df_saldo_elegivel_composicao_m5_3.empty:
        df_saldo_elegivel_composicao_m5_3 = (
            df_saldo_elegivel_composicao_m5_3.drop(columns=cols_drop, errors="ignore")
            .reset_index(drop=True)
            .copy()
        )

    if not df_saldo_excluido_triagem_m5_3.empty:
        df_saldo_excluido_triagem_m5_3 = (
            df_saldo_excluido_triagem_m5_3.drop(columns=cols_drop, errors="ignore")
            .reset_index(drop=True)
            .copy()
        )

    resumo_m5_3_triagem = {
        "modulo": "M5.3",
        "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
        "tipo_roteirizacao": tipo_roteirizacao,
        "linhas_entrada_m5_3": int(len(saldo)),
        "subregioes_total_analisadas": int(len(sub_keys)),
        "subregioes_viaveis": int(len(df_subregioes_viaveis_m5_3)),
        "subregioes_inviaveis": int(len(df_subregioes_inviaveis_m5_3)),
        "linhas_elegiveis_composicao": int(len(df_saldo_elegivel_composicao_m5_3)),
        "linhas_excluidas_triagem": int(len(df_saldo_excluido_triagem_m5_3)),
        "perfis_viaveis_total": int(len(df_perfis_viaveis_por_subregiao_m5_3)),
        "perfil_base_triagem": _safe_text(menor_perfil_base.get("tipo")) or _safe_text(menor_perfil_base.get("perfil")),
        "capacidade_peso_kg_perfil_base": round(_safe_float(menor_perfil_base.get("capacidade_peso_kg"), 0.0), 3),
        "ocupacao_minima_perc_perfil_base": round(_safe_float(menor_perfil_base.get("ocupacao_minima_perc"), 70.0), 3),
        "piso_minimo_kg_perfil_base": round(piso_minimo_menor_perfil, 3),
        "estrategia_m5_3_triagem": [
            "agrupamento_por_subregiao",
            "triagem_somente_por_peso_agregado",
            "uso_do_menor_perfil_cadastrado",
            "sem_validacao_de_raio",
            "sem_subida_de_perfil",
            "VERSAO_M5_3_TRIAGEM_SIMPLIFICADA_2026_04_11",
        ],
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    auditoria_m5_3_triagem = {
        "total_subregioes": int(len(sub_keys)),
        "total_subregioes_viaveis": int(len(df_subregioes_viaveis_m5_3)),
        "total_subregioes_inviaveis": int(len(df_subregioes_inviaveis_m5_3)),
        "total_linhas_elegiveis": int(len(df_saldo_elegivel_composicao_m5_3)),
        "total_linhas_excluidas": int(len(df_saldo_excluido_triagem_m5_3)),
        "total_perfis_viaveis": int(len(df_perfis_viaveis_por_subregiao_m5_3)),
    }

    outputs_m5_3 = {
        "df_saldo_elegivel_composicao_m5_3": df_saldo_elegivel_composicao_m5_3,
        "df_saldo_excluido_triagem_m5_3": df_saldo_excluido_triagem_m5_3,
        "df_subregioes_viaveis_m5_3": df_subregioes_viaveis_m5_3,
        "df_subregioes_inviaveis_m5_3": df_subregioes_inviaveis_m5_3,
        "df_perfis_viaveis_por_subregiao_m5_3": df_perfis_viaveis_por_subregiao_m5_3,
    }

    meta_m5_3 = {
        "resumo_m5_3_triagem": resumo_m5_3_triagem,
        "auditoria_m5_3_triagem": auditoria_m5_3_triagem,
    }

    return outputs_m5_3, meta_m5_3


# Aliases defensivos
def executar_m5_triagem_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_3_triagem_subregioes(*args, **kwargs)


def processar_m5_3_triagem_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_3_triagem_subregioes(*args, **kwargs)


def rodar_m5_3_triagem_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_3_triagem_subregioes(*args, **kwargs)
