from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =========================================================================================
# M5.2 - COMPOSIÇÃO POR CIDADE
# -----------------------------------------------------------------------------------------
# ENTRADA
# - df_saldo_elegivel_composicao_m5_1: saldo elegível vindo do M5.1A
# - df_perfis_viaveis_por_cidade_m5_1: perfis viáveis por cidade do M5.1A
#
# SAÍDA
# - pré-manifestos da composição por cidade
# - itens dos pré-manifestos
# - remanescente total após tentar composição por cidade
# - tentativas auditáveis
#
# LÓGICA
# 1) processa cidade por cidade
# 2) para cada cidade, tenta perfis elegíveis do MAIOR para o MENOR
# 3) tenta primeiro a cidade "cheia" por blocos de destinatário
# 4) se não fechar, remove o menor bloco e tenta novamente
# 5) se fechar, remove itens usados e tenta novo pré-manifesto na mesma cidade
# 6) para quando a cidade não consegue mais fechar nenhum pré-manifesto
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
    df_saldo_elegivel_composicao_m5_1: pd.DataFrame,
    df_perfis_viaveis_por_cidade_m5_1: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    saldo = (
        df_saldo_elegivel_composicao_m5_1.copy()
        if df_saldo_elegivel_composicao_m5_1 is not None
        else pd.DataFrame()
    )
    perfis = (
        df_perfis_viaveis_por_cidade_m5_1.copy()
        if df_perfis_viaveis_por_cidade_m5_1 is not None
        else pd.DataFrame()
    )

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
        raise ValueError("M5.2 exige id_linha_pipeline em todas as linhas elegíveis para composição.")

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

    if perfis.empty:
        raise ValueError("M5.2 exige df_perfis_viaveis_por_cidade_m5_1.")

    for col in [
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ocupacao_minima_perc",
        "ocupacao_minima_kg",
    ]:
        if col not in perfis.columns:
            perfis[col] = pd.NA
        perfis[col] = pd.to_numeric(perfis[col], errors="coerce")

    for col in ["cidade", "uf", "perfil", "tipo", "status_perfil_cidade", "motivo_status_perfil_cidade"]:
        if col not in perfis.columns:
            perfis[col] = ""
        perfis[col] = perfis[col].astype(str)

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
    temp["_id_str_m5_2"] = temp["id_linha_pipeline"].astype(str)
    temp["_cidade_key_m5_2"] = temp["cidade"].fillna("").astype(str).str.strip()
    temp["_uf_key_m5_2"] = temp["uf"].fillna("").astype(str).str.strip()
    temp["_cliente_key_m5_2"] = temp["destinatario"].fillna("").astype(str).str.strip()

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

    temp["_bucket_m5_2"] = buckets
    temp["_prioridade_ord_m5_2"] = prioridade_ord
    temp["_folga_ord_m5_2"] = folga
    temp["_ranking_ord_m5_2"] = ranking
    temp["_km_ord_m5_2"] = km
    temp["_peso_ord_m5_2"] = -peso

    return temp


def _ordenar_operacional(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "_bucket_m5_2" not in df.columns:
        df = _precalcular_ordenacao(df)

    return (
        df.sort_values(
            by=[
                "_bucket_m5_2",
                "_prioridade_ord_m5_2",
                "_folga_ord_m5_2",
                "_ranking_ord_m5_2",
                "_km_ord_m5_2",
                "_peso_ord_m5_2",
                "_id_str_m5_2",
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
# Validação de restrições
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

    ocup = _ocupacao_perc(df_itens, vehicle_row)
    if ocup > ocup_max:
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
# Blocos por destinatário dentro da cidade
# -----------------------------------------------------------------------------------------
def _build_blocks(city_df: pd.DataFrame) -> pd.DataFrame:
    """
    Um bloco = um destinatário dentro da cidade.
    É o menor bloco removível para reduzir a cidade.
    """
    if city_df.empty:
        return pd.DataFrame()

    temp = _ordenar_operacional(city_df.copy())

    grouped = (
        temp.groupby(["_cliente_key_m5_2", "destinatario"], dropna=False)
        .agg(
            peso_total=("peso_calculado", "sum"),
            volume_total=("vol_m3", "sum"),
            km_referencia=("distancia_rodoviaria_est_km", "max"),
            qtd_linhas=("id_linha_pipeline", "count"),
            prioridade_min=("_bucket_m5_2", "min"),
            ranking_min=("_ranking_ord_m5_2", "min"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(
        by=["peso_total", "prioridade_min", "ranking_min", "_cliente_key_m5_2"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped["ordem_bloco_desc"] = range(1, len(grouped) + 1)
    return grouped


def _materializar_candidato_por_blocos(city_df: pd.DataFrame, blocks_df: pd.DataFrame) -> pd.DataFrame:
    if city_df.empty or blocks_df.empty:
        return pd.DataFrame(columns=city_df.columns)

    keys = set(blocks_df["_cliente_key_m5_2"].tolist())
    candidato = city_df[city_df["_cliente_key_m5_2"].isin(keys)].copy()
    return _ordenar_operacional(candidato)


# -----------------------------------------------------------------------------------------
# Tentativas
# -----------------------------------------------------------------------------------------
def _tentativa_dict(
    cidade: str,
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
        "cidade": cidade,
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
    return f"PM52_{seq:04d}"


def _drop_internal_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    cols_internal = [
        "_id_str_m5_2",
        "_cidade_key_m5_2",
        "_uf_key_m5_2",
        "_cliente_key_m5_2",
        "_bucket_m5_2",
        "_prioridade_ord_m5_2",
        "_folga_ord_m5_2",
        "_ranking_ord_m5_2",
        "_km_ord_m5_2",
        "_peso_ord_m5_2",
    ]
    existentes = [c for c in cols_internal if c in df.columns]
    if not existentes:
        return df.copy()
    return df.drop(columns=existentes, errors="ignore").copy()


def _build_manifesto(
    df_itens: pd.DataFrame,
    vehicle_row: pd.Series,
    manifesto_id: str,
    cidade: str,
    uf: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_itens_limpo = _drop_internal_cols(df_itens)

    qtd_itens = int(len(df_itens_limpo))
    qtd_ctes = int(df_itens_limpo["cte"].nunique(dropna=True)) if "cte" in df_itens_limpo.columns else qtd_itens

    manifesto = {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": "pre_manifesto_bloco_5_2_cidade",
        "cidade": cidade,
        "uf": uf,
        "veiculo_tipo": _safe_text(vehicle_row.get("tipo")),
        "veiculo_perfil": _safe_text(vehicle_row.get("perfil")),
        "qtd_itens": qtd_itens,
        "qtd_ctes": qtd_ctes,
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
        "origem_etapa": "m5_2_composicao_cidade",
    }

    df_manifesto = pd.DataFrame([manifesto])
    df_itens_saida = df_itens_limpo.copy()
    for k, v in manifesto.items():
        df_itens_saida[k] = v

    return df_manifesto, df_itens_saida


# -----------------------------------------------------------------------------------------
# Perfis elegíveis por cidade
# -----------------------------------------------------------------------------------------
def _get_eligible_vehicles_for_city(
    city: str,
    uf: str,
    perfis_viaveis_df: pd.DataFrame,
    city_df_atual: pd.DataFrame,
) -> pd.DataFrame:
    base = perfis_viaveis_df[
        (perfis_viaveis_df["cidade"].fillna("").astype(str).str.strip() == city)
        & (perfis_viaveis_df["uf"].fillna("").astype(str).str.strip() == uf)
    ].copy()

    if base.empty:
        return pd.DataFrame()

    # Refiltra dinamicamente com o saldo atual da cidade
    peso_city = _peso_total(city_df_atual)
    km_city = _km_referencia(city_df_atual)

    base["capacidade_peso_kg"] = pd.to_numeric(base["capacidade_peso_kg"], errors="coerce")
    base["max_km_distancia"] = pd.to_numeric(base["max_km_distancia"], errors="coerce")
    base["ocupacao_minima_perc"] = pd.to_numeric(base["ocupacao_minima_perc"], errors="coerce").fillna(70.0)

    base["peso_minimo_kg_dinamico"] = base["capacidade_peso_kg"] * (base["ocupacao_minima_perc"] / 100.0)
    base = base[
        (base["max_km_distancia"].fillna(0) >= km_city)
        & (peso_city >= base["peso_minimo_kg_dinamico"].fillna(float("inf")))
    ].copy()

    if base.empty:
        return base

    # maior para menor
    base = base.sort_values(
        by=["capacidade_peso_kg", "capacidade_vol_m3", "tipo", "perfil"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return base


# -----------------------------------------------------------------------------------------
# Tenta fechar um manifesto dentro de uma cidade
# -----------------------------------------------------------------------------------------
def _tentar_fechar_um_manifesto_na_cidade(
    city_df: pd.DataFrame,
    perfis_viaveis_df: pd.DataFrame,
    cidade: str,
    uf: str,
    tentativas: List[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], str]:
    if city_df.empty:
        return None, None, "cidade_vazia"

    vehicles_city = _get_eligible_vehicles_for_city(
        city=cidade,
        uf=uf,
        perfis_viaveis_df=perfis_viaveis_df,
        city_df_atual=city_df,
    )
    if vehicles_city.empty:
        tentativas.append(
            _tentativa_dict(
                cidade=cidade,
                uf=uf,
                vehicle_row=None,
                resultado="falhou",
                motivo="sem_perfil_elegivel_restante_na_cidade",
                df_candidato=city_df,
                tentativa_idx=1,
                blocos_considerados=0,
            )
        )
        return None, None, "sem_perfil_elegivel_restante_na_cidade"

    blocks_df = _build_blocks(city_df)
    if blocks_df.empty:
        return None, None, "sem_blocos_na_cidade"

    melhor_motivo = "nenhum_fechamento"

    tentativa_idx = 1
    for _, vehicle_row in vehicles_city.iterrows():
        # tenta cidade "cheia" por blocos
        blocks_atual = blocks_df.copy()

        while len(blocks_atual) > 0:
            candidato = _materializar_candidato_por_blocos(city_df, blocks_atual)
            ok, motivo = _validar_fechamento(candidato, vehicle_row)

            tentativas.append(
                _tentativa_dict(
                    cidade=cidade,
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

            # não desiste na cidade fechada: remove o menor bloco e tenta de novo
            if len(blocks_atual) == 1:
                break

            blocks_atual = blocks_atual.iloc[:-1].copy()

    return None, None, melhor_motivo


# -----------------------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------------------
def executar_m5_2_composicao_cidades(
    df_saldo_elegivel_composicao_m5_1: pd.DataFrame,
    df_perfis_viaveis_por_cidade_m5_1: pd.DataFrame,
    rodada_id: Optional[str] = None,
    data_base_roteirizacao: Optional[Any] = None,
    tipo_roteirizacao: str = "carteira",
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    del rodada_id, kwargs

    saldo, perfis_viaveis = _normalizar_inputs(
        df_saldo_elegivel_composicao_m5_1=df_saldo_elegivel_composicao_m5_1,
        df_perfis_viaveis_por_cidade_m5_1=df_perfis_viaveis_por_cidade_m5_1,
    )

    if saldo.empty:
        outputs_vazio = {
            "df_premanifestos_m5_2": pd.DataFrame(),
            "df_itens_premanifestos_m5_2": pd.DataFrame(),
            "df_tentativas_m5_2": pd.DataFrame(),
            "df_remanescente_m5_2": pd.DataFrame(),
        }
        meta_vazio = {
            "resumo_m5_2": {
                "modulo": "M5.2",
                "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
                "tipo_roteirizacao": tipo_roteirizacao,
                "linhas_entrada_m5_2": 0,
                "pre_manifestos_gerados_m5_2": 0,
                "itens_pre_manifestados_m5_2": 0,
                "remanescente_saida_m5_2": 0,
                "cidades_processadas_m5_2": 0,
                "estrategia_m5_2": [
                    "cidade_por_cidade",
                    "maior_perfil_elegivel_para_menor",
                    "cidade_fechada_primeiro",
                    "remove_menor_bloco_se_nao_fechar",
                    "multiplos_fechamentos_na_mesma_cidade",
                    "VERSAO_M5_2_2026_04_10",
                ],
                "caminhos_pipeline": caminhos_pipeline or {},
            },
            "auditoria_m5_2": {
                "total_tentativas": 0,
                "total_pre_manifestos": 0,
                "total_itens_pre_manifestados": 0,
                "total_remanescentes": 0,
                "total_cidades_processadas": 0,
            },
        }
        return outputs_vazio, meta_vazio

    saldo = _precalcular_ordenacao(saldo)
    saldo = _ordenar_operacional(saldo)

    manifestos_list: List[pd.DataFrame] = []
    itens_manifestados_list: List[pd.DataFrame] = []
    tentativas: List[Dict[str, Any]] = []

    manifesto_seq = 1
    cidades_processadas = 0

    cidades_keys = (
        saldo[["_cidade_key_m5_2", "_uf_key_m5_2"]]
        .drop_duplicates()
        .sort_values(["_cidade_key_m5_2", "_uf_key_m5_2"], kind="mergesort")
        .values.tolist()
    )

    for cidade_key, uf_key in cidades_keys:
        cidades_processadas += 1

        while True:
            city_df = saldo[
                (saldo["_cidade_key_m5_2"] == cidade_key)
                & (saldo["_uf_key_m5_2"] == uf_key)
            ].copy()

            if city_df.empty:
                break

            candidato, vehicle_row, motivo = _tentar_fechar_um_manifesto_na_cidade(
                city_df=city_df,
                perfis_viaveis_df=perfis_viaveis,
                cidade=cidade_key,
                uf=uf_key,
                tentativas=tentativas,
            )

            if candidato is None or vehicle_row is None:
                tentativas.append(
                    {
                        "cidade": cidade_key,
                        "uf": uf_key,
                        "tentativa_idx": None,
                        "blocos_considerados": 0,
                        "veiculo_tipo_tentado": None,
                        "veiculo_perfil_tentado": None,
                        "resultado": "saldo",
                        "motivo": motivo,
                        "qtd_itens_candidato": int(len(city_df)),
                        "qtd_paradas_candidato": _qtd_paradas(city_df),
                        "peso_total_candidato": round(_peso_total(city_df), 3),
                        "volume_total_candidato": round(_volume_total(city_df), 3),
                        "km_referencia_candidato": round(_km_referencia(city_df), 2),
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
                cidade=cidade_key,
                uf=uf_key,
            )

            manifestos_list.append(df_manifesto)
            itens_manifestados_list.append(df_itens)

            ids_consumidos = set(candidato["_id_str_m5_2"].tolist())
            saldo = saldo[~saldo["_id_str_m5_2"].isin(ids_consumidos)].copy()
            saldo = _ordenar_operacional(saldo)

    df_premanifestos_m5_2 = pd.concat(manifestos_list, ignore_index=True) if manifestos_list else pd.DataFrame()
    df_itens_premanifestos_m5_2 = (
        pd.concat(itens_manifestados_list, ignore_index=True) if itens_manifestados_list else pd.DataFrame()
    )
    df_tentativas_m5_2 = pd.DataFrame(tentativas)

    df_remanescente_m5_2 = _drop_internal_cols(saldo.reset_index(drop=True))

    resumo_m5_2 = {
        "modulo": "M5.2",
        "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
        "tipo_roteirizacao": tipo_roteirizacao,
        "linhas_entrada_m5_2": int(len(df_saldo_elegivel_composicao_m5_1)),
        "pre_manifestos_gerados_m5_2": int(df_premanifestos_m5_2["manifesto_id"].nunique()) if not df_premanifestos_m5_2.empty else 0,
        "itens_pre_manifestados_m5_2": int(len(df_itens_premanifestos_m5_2)),
        "remanescente_saida_m5_2": int(len(df_remanescente_m5_2)),
        "cidades_processadas_m5_2": int(cidades_processadas),
        "estrategia_m5_2": [
            "cidade_por_cidade",
            "maior_perfil_elegivel_para_menor",
            "cidade_fechada_primeiro",
            "remove_menor_bloco_se_nao_fechar",
            "multiplos_fechamentos_na_mesma_cidade",
            "VERSAO_M5_2_2026_04_10",
        ],
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    auditoria_m5_2 = {
        "total_tentativas": int(len(df_tentativas_m5_2)),
        "total_pre_manifestos": int(df_premanifestos_m5_2["manifesto_id"].nunique()) if not df_premanifestos_m5_2.empty else 0,
        "total_itens_pre_manifestados": int(len(df_itens_premanifestos_m5_2)),
        "total_remanescentes": int(len(df_remanescente_m5_2)),
        "total_cidades_processadas": int(cidades_processadas),
    }

    outputs_m5_2 = {
        "df_premanifestos_m5_2": df_premanifestos_m5_2,
        "df_itens_premanifestos_m5_2": df_itens_premanifestos_m5_2,
        "df_tentativas_m5_2": df_tentativas_m5_2,
        "df_remanescente_m5_2": df_remanescente_m5_2,
    }

    meta_m5_2 = {
        "resumo_m5_2": resumo_m5_2,
        "auditoria_m5_2": auditoria_m5_2,
    }

    return outputs_m5_2, meta_m5_2


# Aliases defensivos
def executar_m5_composicao_cidades(*args: Any, **kwargs: Any):
    return executar_m5_2_composicao_cidades(*args, **kwargs)


def processar_m5_2_composicao_cidades(*args: Any, **kwargs: Any):
    return executar_m5_2_composicao_cidades(*args, **kwargs)


def rodar_m5_2_composicao_cidades(*args: Any, **kwargs: Any):
    return executar_m5_2_composicao_cidades(*args, **kwargs)
