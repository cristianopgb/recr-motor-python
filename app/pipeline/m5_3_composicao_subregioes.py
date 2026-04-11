from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =========================================================================================
# M5.3 - COMPOSIÇÃO POR SUBREGIÃO
# -----------------------------------------------------------------------------------------
# ENTRADA CORRETA:
# - df_saldo_global_pos_cidade_m5
#   = saldo global após a etapa cidade
#   = não elegíveis do M5.1 + remanescente do M5.2
#
# - df_perfis_base_m5
#   = base de perfis/veículos para reavaliar elegibilidade na subregião
#
# SAÍDA:
# - df_premanifestos_m5_3
# - df_itens_premanifestos_m5_3
# - df_tentativas_m5_3
# - df_remanescente_m5_3
#
# LÓGICA:
# 1) agrupar por subregião
# 2) tentar do maior perfil elegível para o menor
# 3) tentar subregião fechada primeiro
# 4) se não fechar, remover o menor bloco e tentar novamente
# 5) se fechar, remove do saldo da sub e tenta novamente na mesma sub
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
        if col in saldo.columns:
            saldo[col] = pd.to_numeric(saldo[col], errors="coerce")

    bool_cols = ["agendada", "veiculo_exclusivo", "veiculo_exclusivo_flag"]
    for col in bool_cols:
        if col in saldo.columns:
            saldo[col] = saldo[col].apply(_safe_bool)

    for col in ["cidade", "uf", "subregiao", "mesorregiao", "destinatario"]:
        if col in saldo.columns:
            saldo[col] = saldo[col].fillna("").astype(str).str.strip()

    if perfis.empty:
        raise ValueError("M5.3 exige df_perfis_base_m5.")

    # aceita perfis vindos do M5.1 ou outra base de perfis
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
        perfis[col] = perfis[col].fillna("").astype(str)

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
        p = pd.to_numeric(
            row.get("prioridade_embarque_num", row.get("prioridade_embarque", pd.NA)),
            errors="coerce",
        )
        prioridade_ord.append(_safe_float(p, 999.0) if not pd.isna(p) else 999.0)

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
# Métricas
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


def _ocupacao_perc(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> float:
    cap_peso = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    if cap_peso <= 0:
        return 0.0
    return (_peso_total(df_itens) / cap_peso) * 100.0


# -----------------------------------------------------------------------------------------
# Regras de veículo
# -----------------------------------------------------------------------------------------
def _veiculo_compatível_com_restricoes(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> bool:
    if "restricao_veiculo" not in df_itens.columns:
        return True

    tipo = _safe_text(vehicle_row.get("tipo")).upper()
    perfil = _safe_text(vehicle_row.get("perfil")).upper()

    for value in df_itens["restricao_veiculo"].tolist():
        txt = _safe_text(value).upper()
        if txt and txt not in {tipo, perfil}:
            return False
    return True


def _validar_hard_constraints(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> Tuple[bool, str]:
    if df_itens.empty:
        return False, "grupo_vazio"

    if not _veiculo_compatível_com_restricoes(df_itens, vehicle_row):
        return False, "restricao_veiculo_incompativel"

    peso_total = _peso_total(df_itens)
    volume_total = _volume_total(df_itens)
    paradas = _qtd_paradas(df_itens)
    km_ref = _km_referencia(df_itens)

    cap_peso = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    cap_vol = _safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0)
    max_entregas = _safe_int(vehicle_row.get("max_entregas"), 0)
    max_km = _safe_float(vehicle_row.get("max_km_distancia"), 0.0)
    ocup_max = _safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0)

    if cap_peso > 0 and peso_total > cap_peso:
        return False, "excede_capacidade_peso"
    if cap_vol > 0 and volume_total > cap_vol:
        return False, "excede_capacidade_volume"
    if max_entregas > 0 and paradas > max_entregas:
        return False, "excede_max_entregas"
    if max_km > 0 and km_ref > max_km:
        return False, "excede_max_km"
    if _ocupacao_perc(df_itens, vehicle_row) > ocup_max:
        return False, "excede_ocupacao_maxima"

    return True, "ok"


def _validar_fechamento(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> Tuple[bool, str]:
    ok_hard, motivo = _validar_hard_constraints(df_itens, vehicle_row)
    if not ok_hard:
        return False, motivo

    ocup_min = _safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0)
    ocup = _ocupacao_perc(df_itens, vehicle_row)

    if ocup < ocup_min:
        return False, "abaixo_ocupacao_minima"

    return True, "ok"


# -----------------------------------------------------------------------------------------
# Blocos por destinatário
# -----------------------------------------------------------------------------------------
def _build_blocks(sub_df: pd.DataFrame) -> pd.DataFrame:
    if sub_df.empty:
        return pd.DataFrame()

    temp = _ordenar_operacional(sub_df.copy())

    grouped = (
        temp.groupby(["_cliente_key_m5_3", "destinatario"], dropna=False)
        .agg(
            peso_total=("peso_calculado", "sum"),
            volume_total=("vol_m3", "sum"),
            km_referencia=("distancia_rodoviaria_est_km", "max"),
            qtd_linhas=("id_linha_pipeline", "count"),
            prioridade_min=("_bucket_m5_3", "min"),
            ranking_min=("_ranking_ord_m5_3", "min"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(
        by=["peso_total", "prioridade_min", "ranking_min", "_cliente_key_m5_3"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped["ordem_bloco_desc"] = range(1, len(grouped) + 1)
    return grouped


def _materializar_candidato_por_blocos(sub_df: pd.DataFrame, blocks_df: pd.DataFrame) -> pd.DataFrame:
    if sub_df.empty or blocks_df.empty:
        return pd.DataFrame(columns=sub_df.columns)

    keys = set(blocks_df["_cliente_key_m5_3"].tolist())
    candidato = sub_df[sub_df["_cliente_key_m5_3"].isin(keys)].copy()
    return _ordenar_operacional(candidato)


# -----------------------------------------------------------------------------------------
# Tentativas
# -----------------------------------------------------------------------------------------
def _tentativa_dict(
    subregiao: str,
    uf: str,
    vehicle_row: Optional[pd.Series],
    resultado: str,
    motivo: str,
    df_candidato: Optional[pd.DataFrame],
    tentativa_idx: int,
    blocos_considerados: int,
) -> Dict[str, Any]:
    candidato = df_candidato if df_candidato is not None else pd.DataFrame()
    return {
        "subregiao": subregiao,
        "uf": uf,
        "tentativa_idx": tentativa_idx,
        "blocos_considerados": blocos_considerados,
        "veiculo_tipo_tentado": None if vehicle_row is None else _safe_text(vehicle_row.get("tipo")),
        "veiculo_perfil_tentado": None if vehicle_row is None else _safe_text(vehicle_row.get("perfil")),
        "resultado": resultado,
        "motivo": motivo,
        "qtd_itens_candidato": int(len(candidato)),
        "qtd_paradas_candidato": _qtd_paradas(candidato),
        "peso_total_candidato": round(_peso_total(candidato), 3),
        "volume_total_candidato": round(_volume_total(candidato), 3),
        "km_referencia_candidato": round(_km_referencia(candidato), 2),
        "ocupacao_perc_candidato": round(_ocupacao_perc(candidato, vehicle_row), 2)
        if vehicle_row is not None and not candidato.empty
        else 0.0,
    }


# -----------------------------------------------------------------------------------------
# Manifesto
# -----------------------------------------------------------------------------------------
def _build_manifesto_id(seq: int) -> str:
    return f"PM53_{seq:04d}"


def _drop_internal_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    cols_internal = [
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
    return df.drop(columns=cols_internal, errors="ignore").copy()


def _build_manifesto(
    df_itens: pd.DataFrame,
    vehicle_row: pd.Series,
    manifesto_id: str,
    subregiao: str,
    uf: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_itens_limpo = _drop_internal_cols(df_itens)

    manifesto = {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": "pre_manifesto_bloco_5_3_subregiao",
        "subregiao": subregiao,
        "uf": uf,
        "veiculo_tipo": _safe_text(vehicle_row.get("tipo")),
        "veiculo_perfil": _safe_text(vehicle_row.get("perfil")),
        "qtd_itens": int(len(df_itens_limpo)),
        "qtd_ctes": int(df_itens_limpo["cte"].nunique(dropna=True)) if "cte" in df_itens_limpo.columns else int(len(df_itens_limpo)),
        "qtd_paradas": _qtd_paradas(df_itens_limpo),
        "base_carga_oficial": round(_peso_total(df_itens_limpo), 3),
        "peso_total_kg": round(_peso_total(df_itens_limpo), 3),
        "vol_total_m3": round(_volume_total(df_itens_limpo), 3),
        "km_referencia": round(_km_referencia(df_itens_limpo), 2),
        "ocupacao_oficial_perc": round(_ocupacao_perc(df_itens_limpo, vehicle_row), 2),
        "capacidade_peso_kg_veiculo": _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0),
        "capacidade_vol_m3_veiculo": _safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0),
        "max_entregas_veiculo": _safe_int(vehicle_row.get("max_entregas"), 0),
        "max_km_distancia_veiculo": _safe_float(vehicle_row.get("max_km_distancia"), 0.0),
        "ocupacao_minima_perc_veiculo": _safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0),
        "ignorar_ocupacao_minima": False,
        "origem_modulo": 5,
        "origem_etapa": "m5_3_composicao_subregiao",
    }

    df_manifesto = pd.DataFrame([manifesto])
    df_itens_saida = df_itens_limpo.copy()
    for k, v in manifesto.items():
        df_itens_saida[k] = v

    return df_manifesto, df_itens_saida


# -----------------------------------------------------------------------------------------
# Veículos elegíveis para a subregião
# -----------------------------------------------------------------------------------------
def _get_eligible_vehicles_for_sub(
    perfis_base_df: pd.DataFrame,
    sub_df_atual: pd.DataFrame,
) -> pd.DataFrame:
    base = perfis_base_df.copy()
    if base.empty:
        return base

    peso_sub = _peso_total(sub_df_atual)
    km_sub = _km_referencia(sub_df_atual)

    base["capacidade_peso_kg"] = pd.to_numeric(base["capacidade_peso_kg"], errors="coerce")
    base["capacidade_vol_m3"] = pd.to_numeric(base["capacidade_vol_m3"], errors="coerce")
    base["max_km_distancia"] = pd.to_numeric(base["max_km_distancia"], errors="coerce")
    base["ocupacao_minima_perc"] = pd.to_numeric(base["ocupacao_minima_perc"], errors="coerce").fillna(70.0)

    base["peso_minimo_kg_dinamico"] = base["capacidade_peso_kg"] * (base["ocupacao_minima_perc"] / 100.0)

    base = base[
        (base["max_km_distancia"].fillna(0) >= km_sub)
        & (peso_sub >= base["peso_minimo_kg_dinamico"].fillna(float("inf")))
    ].copy()

    if base.empty:
        return base

    return (
        base.sort_values(
            by=["capacidade_peso_kg", "capacidade_vol_m3", "tipo", "perfil"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------------------
# Fecha um manifesto na subregião
# -----------------------------------------------------------------------------------------
def _tentar_fechar_um_manifesto_na_sub(
    sub_df: pd.DataFrame,
    perfis_base_df: pd.DataFrame,
    subregiao: str,
    uf: str,
    tentativas: List[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], str]:
    if sub_df.empty:
        return None, None, "subregiao_vazia"

    vehicles_sub = _get_eligible_vehicles_for_sub(
        perfis_base_df=perfis_base_df,
        sub_df_atual=sub_df,
    )
    if vehicles_sub.empty:
        tentativas.append(
            _tentativa_dict(
                subregiao=subregiao,
                uf=uf,
                vehicle_row=None,
                resultado="falhou",
                motivo="sem_perfil_elegivel_restante_na_subregiao",
                df_candidato=sub_df,
                tentativa_idx=1,
                blocos_considerados=0,
            )
        )
        return None, None, "sem_perfil_elegivel_restante_na_subregiao"

    blocks_df = _build_blocks(sub_df)
    if blocks_df.empty:
        return None, None, "sem_blocos_na_subregiao"

    melhor_motivo = "nenhum_fechamento"
    tentativa_idx = 1

    for _, vehicle_row in vehicles_sub.iterrows():
        blocks_atual = blocks_df.copy()

        while len(blocks_atual) > 0:
            candidato = _materializar_candidato_por_blocos(sub_df, blocks_atual)
            ok, motivo = _validar_fechamento(candidato, vehicle_row)

            tentativas.append(
                _tentativa_dict(
                    subregiao=subregiao,
                    uf=uf,
                    vehicle_row=vehicle_row,
                    resultado="fechado" if ok else "falhou",
                    motivo=motivo,
                    df_candidato=candidato,
                    tentativa_idx=tentativa_idx,
                    blocos_considerados=int(len(blocks_atual)),
                )
            )
            tentativa_idx += 1
            melhor_motivo = motivo

            if ok:
                return candidato.copy(), vehicle_row.copy(), "ok"

            if len(blocks_atual) == 1:
                break

            # remove o menor bloco e tenta novamente
            blocks_atual = blocks_atual.iloc[:-1].copy()

    return None, None, melhor_motivo


# -----------------------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------------------
def executar_m5_3_composicao_subregioes(
    df_saldo_global_pos_cidade_m5: pd.DataFrame,
    df_perfis_base_m5: pd.DataFrame,
    rodada_id: Optional[str] = None,
    data_base_roteirizacao: Optional[Any] = None,
    tipo_roteirizacao: str = "carteira",
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    del rodada_id, kwargs

    saldo, perfis_base = _normalizar_inputs(
        df_saldo_global_pos_cidade_m5=df_saldo_global_pos_cidade_m5,
        df_perfis_base_m5=df_perfis_base_m5,
    )

    if saldo.empty:
        outputs_vazio = {
            "df_premanifestos_m5_3": pd.DataFrame(),
            "df_itens_premanifestos_m5_3": pd.DataFrame(),
            "df_tentativas_m5_3": pd.DataFrame(),
            "df_remanescente_m5_3": pd.DataFrame(),
        }
        meta_vazio = {
            "resumo_m5_3": {
                "modulo": "M5.3",
                "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
                "tipo_roteirizacao": tipo_roteirizacao,
                "linhas_entrada_m5_3": 0,
                "pre_manifestos_gerados_m5_3": 0,
                "itens_pre_manifestados_m5_3": 0,
                "remanescente_saida_m5_3": 0,
                "subregioes_processadas_m5_3": 0,
                "estrategia_m5_3": [
                    "subregiao_por_subregiao",
                    "maior_perfil_elegivel_para_menor",
                    "subregiao_fechada_primeiro",
                    "remove_menor_bloco_se_nao_fechar",
                    "multiplos_fechamentos_na_mesma_subregiao",
                    "VERSAO_M5_3_2026_04_10_FIX_01",
                ],
                "caminhos_pipeline": caminhos_pipeline or {},
            },
            "auditoria_m5_3": {
                "total_tentativas": 0,
                "total_pre_manifestos": 0,
                "total_itens_pre_manifestados": 0,
                "total_remanescentes": 0,
                "total_subregioes_processadas": 0,
            },
        }
        return outputs_vazio, meta_vazio

    saldo = _precalcular_ordenacao(saldo)
    saldo = _ordenar_operacional(saldo)

    manifestos_list: List[pd.DataFrame] = []
    itens_manifestados_list: List[pd.DataFrame] = []
    tentativas: List[Dict[str, Any]] = []

    manifesto_seq = 1
    subregioes_processadas = 0

    sub_keys = (
        saldo[["_subregiao_key_m5_3", "_uf_key_m5_3"]]
        .drop_duplicates()
        .sort_values(["_subregiao_key_m5_3", "_uf_key_m5_3"], kind="mergesort")
        .values.tolist()
    )

    for subregiao_key, uf_key in sub_keys:
        subregioes_processadas += 1

        while True:
            sub_df = saldo[
                (saldo["_subregiao_key_m5_3"] == subregiao_key)
                & (saldo["_uf_key_m5_3"] == uf_key)
            ].copy()

            if sub_df.empty:
                break

            candidato, vehicle_row, motivo = _tentar_fechar_um_manifesto_na_sub(
                sub_df=sub_df,
                perfis_base_df=perfis_base,
                subregiao=subregiao_key,
                uf=uf_key,
                tentativas=tentativas,
            )

            if candidato is None or vehicle_row is None:
                tentativas.append(
                    {
                        "subregiao": subregiao_key,
                        "uf": uf_key,
                        "tentativa_idx": None,
                        "blocos_considerados": 0,
                        "veiculo_tipo_tentado": None,
                        "veiculo_perfil_tentado": None,
                        "resultado": "saldo",
                        "motivo": motivo,
                        "qtd_itens_candidato": int(len(sub_df)),
                        "qtd_paradas_candidato": _qtd_paradas(sub_df),
                        "peso_total_candidato": round(_peso_total(sub_df), 3),
                        "volume_total_candidato": round(_volume_total(sub_df), 3),
                        "km_referencia_candidato": round(_km_referencia(sub_df), 2),
                        "ocupacao_perc_candidato": 0.0,
                    }
                )
                break

            manifesto_id = _build_manifesto_id(manifesto_seq)
            manifesto_seq += 1

            df_manifesto, df_itens = _build_manifesto(
                df_itens=candidato,
                vehicle_row=vehicle_row,
                manifesto_id=manifesto_id,
                subregiao=subregiao_key,
                uf=uf_key,
            )

            manifestos_list.append(df_manifesto)
            itens_manifestados_list.append(df_itens)

            ids_consumidos = set(candidato["_id_str_m5_3"].tolist())
            saldo = saldo[~saldo["_id_str_m5_3"].isin(ids_consumidos)].copy()
            saldo = _ordenar_operacional(saldo)

    df_premanifestos_m5_3 = (
        pd.concat(manifestos_list, ignore_index=True) if manifestos_list else pd.DataFrame()
    )
    df_itens_premanifestos_m5_3 = (
        pd.concat(itens_manifestados_list, ignore_index=True) if itens_manifestados_list else pd.DataFrame()
    )
    df_tentativas_m5_3 = pd.DataFrame(tentativas)
    df_remanescente_m5_3 = _drop_internal_cols(saldo.reset_index(drop=True))

    resumo_m5_3 = {
        "modulo": "M5.3",
        "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
        "tipo_roteirizacao": tipo_roteirizacao,
        "linhas_entrada_m5_3": int(len(df_saldo_global_pos_cidade_m5)),
        "pre_manifestos_gerados_m5_3": int(df_premanifestos_m5_3["manifesto_id"].nunique())
        if not df_premanifestos_m5_3.empty
        else 0,
        "itens_pre_manifestados_m5_3": int(len(df_itens_premanifestos_m5_3)),
        "remanescente_saida_m5_3": int(len(df_remanescente_m5_3)),
        "subregioes_processadas_m5_3": int(subregioes_processadas),
        "estrategia_m5_3": [
            "subregiao_por_subregiao",
            "maior_perfil_elegivel_para_menor",
            "subregiao_fechada_primeiro",
            "remove_menor_bloco_se_nao_fechar",
            "multiplos_fechamentos_na_mesma_subregiao",
            "VERSAO_M5_3_2026_04_10_FIX_01",
        ],
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    auditoria_m5_3 = {
        "total_tentativas": int(len(df_tentativas_m5_3)),
        "total_pre_manifestos": int(df_premanifestos_m5_3["manifesto_id"].nunique())
        if not df_premanifestos_m5_3.empty
        else 0,
        "total_itens_pre_manifestados": int(len(df_itens_premanifestos_m5_3)),
        "total_remanescentes": int(len(df_remanescente_m5_3)),
        "total_subregioes_processadas": int(subregioes_processadas),
    }

    outputs_m5_3 = {
        "df_premanifestos_m5_3": df_premanifestos_m5_3,
        "df_itens_premanifestos_m5_3": df_itens_premanifestos_m5_3,
        "df_tentativas_m5_3": df_tentativas_m5_3,
        "df_remanescente_m5_3": df_remanescente_m5_3,
    }

    meta_m5_3 = {
        "resumo_m5_3": resumo_m5_3,
        "auditoria_m5_3": auditoria_m5_3,
    }

    return outputs_m5_3, meta_m5_3


# aliases defensivos
def executar_m5_composicao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_3_composicao_subregioes(*args, **kwargs)


def processar_m5_3_composicao_subregioes(*args: Any, **kwargs: Any):
    return executar_m5_3_composicao_subregioes(*args, **kwargs)


def rodar_m5_3(*args: Any, **kwargs: Any):
    return executar_m5_3_composicao_subregioes(*args, **kwargs)
