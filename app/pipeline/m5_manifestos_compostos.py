from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import pandas as pd


# =========================================================================================
# M5.1 - MANIFESTOS COMPOSTOS
# -----------------------------------------------------------------------------------------
# CONTRATO:
# - compatível com pipeline_service.py
# - função principal: executar_m5_manifestos_compostos(...)
# - retorno: outputs_m5_1, meta_m5_1
#
# LÓGICA:
# 1) entrada dura: só remanescente do M4
# 2) fase 1: consolidação por mesmo cliente
# 3) fase 2: saldo -> cidade -> subregião -> mesorregião
# 4) regionalidade é UNIVERSO ELEGÍVEL, não bloco indivisível
# 5) quando grupo cheio não fecha, tenta subconjunto viável
#
# OTIMIZAÇÕES DESTA VERSÃO:
# - pré-cálculo de colunas de prioridade
# - menos apply(axis=1) em loops
# - menos reordenações caras
# - subconjunto viável sem recriar DataFrame a cada linha
# - sem uso de atributos frágeis em linhas iteradas
# =========================================================================================


INTERNAL_COLS_M5_1 = [
    "id_str_m5_1",
    "cliente_chave_m5_1",
    "cidade_key_m5_1",
    "subregiao_key_m5_1",
    "mesorregiao_key_m5_1",
    "bucket_fase2_m5_1",
    "prioridade_ord_m5_1",
    "folga_ord_m5_1",
    "ranking_oper_ord_m5_1",
    "km_ord_m5_1",
    "peso_ord_m5_1",
    "anchor_attempted_m5_1",
]


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


def _first_existing(row: pd.Series, candidates: List[str], default: Any = None) -> Any:
    for col in candidates:
        if col in row.index:
            value = row[col]
            if not pd.isna(value):
                return value
    return default


def _clean_key_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _drop_internal_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    cols_to_drop = [c for c in INTERNAL_COLS_M5_1 if c in df.columns]
    if not cols_to_drop:
        return df.copy()
    return df.drop(columns=cols_to_drop, errors="ignore").copy()


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
        "destinatario": "",
        "cidade": "",
        "subregiao": "",
        "mesorregiao": "",
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
        raise ValueError("M5.1 exige id_linha_pipeline em todas as linhas do remanescente do M4.")

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
        raise ValueError("M5.1 exige df_veiculos_tratados.")

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
# Prioridade operacional
# -----------------------------------------------------------------------------------------
def _fase2_bucket_from_values(
    prioridade_embarque: float,
    agendada: bool,
    folga: float,
) -> int:
    if not math.isnan(prioridade_embarque) and prioridade_embarque > 0:
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


def _precalcular_priorizacao(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    temp = df.copy()

    temp["id_str_m5_1"] = temp["id_linha_pipeline"].astype(str)
    temp["cliente_chave_m5_1"] = _clean_key_series(temp["destinatario"])
    temp["cidade_key_m5_1"] = _clean_key_series(temp["cidade"])
    temp["subregiao_key_m5_1"] = _clean_key_series(temp["subregiao"])
    temp["mesorregiao_key_m5_1"] = _clean_key_series(temp["mesorregiao"])

    prioridade_base = pd.to_numeric(
        temp["prioridade_embarque_num"].where(temp["prioridade_embarque_num"].notna(), temp["prioridade_embarque"]),
        errors="coerce",
    )

    agendada = temp["agendada"].fillna(False).astype(bool)
    folga = pd.to_numeric(temp["folga_dias"], errors="coerce").fillna(999)
    ranking_oper = pd.to_numeric(temp["ranking_prioridade_operacional"], errors="coerce").fillna(999)
    km = pd.to_numeric(temp["distancia_rodoviaria_est_km"], errors="coerce").fillna(999999)
    peso = pd.to_numeric(temp["peso_calculado"], errors="coerce").fillna(0.0)

    buckets: List[int] = []
    prioridade_ord: List[float] = []
    for prio, ag, fg in zip(prioridade_base.tolist(), agendada.tolist(), folga.tolist()):
        prio_float = _safe_float(prio, math.nan)
        buckets.append(_fase2_bucket_from_values(prio_float, bool(ag), _safe_float(fg, 999)))
        prioridade_ord.append(prio_float if not math.isnan(prio_float) else 999.0)

    temp["bucket_fase2_m5_1"] = buckets
    temp["prioridade_ord_m5_1"] = prioridade_ord
    temp["folga_ord_m5_1"] = folga
    temp["ranking_oper_ord_m5_1"] = ranking_oper
    temp["km_ord_m5_1"] = km
    temp["peso_ord_m5_1"] = -peso
    temp["anchor_attempted_m5_1"] = False

    return temp


def _elegivel_fase2_df(df: pd.DataFrame) -> pd.Series:
    return df["bucket_fase2_m5_1"] < 99


def _ordenar_operacional(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    cols_ord = [
        "bucket_fase2_m5_1",
        "prioridade_ord_m5_1",
        "folga_ord_m5_1",
        "ranking_oper_ord_m5_1",
        "km_ord_m5_1",
        "peso_ord_m5_1",
        "id_str_m5_1",
    ]
    existentes = [c for c in cols_ord if c in df.columns]
    if len(existentes) != len(cols_ord):
        df = _precalcular_priorizacao(df)

    return (
        df.sort_values(
            cols_ord,
            ascending=[True, True, True, True, True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )


# -----------------------------------------------------------------------------------------
# Veículos
# -----------------------------------------------------------------------------------------
def _veiculos_maior_para_menor(df_veiculos: pd.DataFrame) -> pd.DataFrame:
    temp = df_veiculos.copy()
    temp["_cap_peso_tmp"] = temp["capacidade_peso_kg"].fillna(0)
    temp["_cap_vol_tmp"] = temp["capacidade_vol_m3"].fillna(0)
    temp = temp.sort_values(
        ["_cap_peso_tmp", "_cap_vol_tmp"],
        ascending=[False, False],
        kind="mergesort",
    ).drop(columns=["_cap_peso_tmp", "_cap_vol_tmp"])
    return temp.reset_index(drop=True)


def _restricao_compativel_linha(restricao_value: Any, vehicle_row: pd.Series) -> bool:
    txt = _safe_text(restricao_value).upper()
    if not txt:
        return True

    tipo = _safe_text(vehicle_row.get("tipo")).upper()
    perfil = _safe_text(vehicle_row.get("perfil")).upper()
    return txt in {tipo, perfil}


def _restricao_compatível(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> bool:
    if "restricao_veiculo" not in df_itens.columns:
        return True

    for value in df_itens["restricao_veiculo"].tolist():
        if not _restricao_compativel_linha(value, vehicle_row):
            return False
    return True


# -----------------------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------------------
def _qtd_paradas(df_itens: pd.DataFrame) -> int:
    if df_itens.empty:
        return 0
    return int(df_itens["destinatario"].fillna("").astype(str).nunique())


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


def _ocupacao_perc(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> float:
    capacidade = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    if capacidade <= 0:
        return 0.0
    return (_peso_total(df_itens) / capacidade) * 100.0


# -----------------------------------------------------------------------------------------
# Validação
# -----------------------------------------------------------------------------------------
def _validar_hard_constraints(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> Tuple[bool, str]:
    if df_itens.empty:
        return False, "grupo_vazio"

    if not _restricao_compatível(df_itens, vehicle_row):
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
# Auditoria
# -----------------------------------------------------------------------------------------
def _tentativa_dict(
    fase: str,
    camada: str,
    anchor_id: Optional[str],
    vehicle_row: Optional[pd.Series],
    resultado: str,
    motivo: str,
    df_candidato: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    candidato = df_candidato if df_candidato is not None else pd.DataFrame()
    return {
        "fase": fase,
        "camada": camada,
        "anchor_id_linha_pipeline": anchor_id,
        "veiculo_tipo_tentado": None if vehicle_row is None else _safe_text(vehicle_row.get("tipo")),
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
    return f"PM51_{seq:04d}"


def _build_manifesto(
    df_itens: pd.DataFrame,
    vehicle_row: pd.Series,
    manifesto_id: str,
    fase: str,
    camada: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_itens_limpo = _drop_internal_cols(df_itens)

    qtd_itens = int(len(df_itens_limpo))
    qtd_ctes = (
        int(df_itens_limpo["cte"].nunique(dropna=True))
        if "cte" in df_itens_limpo.columns
        else qtd_itens
    )

    manifesto = {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": "pre_manifesto_bloco_5_1",
        "veiculo_tipo": _safe_text(vehicle_row.get("tipo")),
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
        "ignorar_ocupacao_minima": False,
        "origem_modulo": 5,
        "origem_etapa": f"{fase}_{camada}",
    }

    df_manifesto = pd.DataFrame([manifesto])
    df_itens_saida = df_itens_limpo.copy()
    for k, v in manifesto.items():
        df_itens_saida[k] = v

    return df_manifesto, df_itens_saida


# -----------------------------------------------------------------------------------------
# Fase 1 - mesmo cliente
# -----------------------------------------------------------------------------------------
def _tentar_fechar_grupo_inteiro(
    df_grupo: pd.DataFrame,
    veiculos_ordenados: pd.DataFrame,
    tentativas: List[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], str]:
    melhor_motivo = "nenhum_veiculo_compativel"

    for _, vehicle_row in veiculos_ordenados.iterrows():
        ok, motivo = _validar_fechamento(df_grupo, vehicle_row)
        tentativas.append(
            _tentativa_dict(
                fase="fase_1_mesmo_cliente",
                camada="mesmo_cliente",
                anchor_id=None,
                vehicle_row=vehicle_row,
                resultado="fechado" if ok else "falhou",
                motivo=motivo,
                df_candidato=df_grupo,
            )
        )
        melhor_motivo = motivo
        if ok:
            return df_grupo.copy(), vehicle_row.copy(), "ok"

    return None, None, melhor_motivo


# -----------------------------------------------------------------------------------------
# Fase 2 - composição regional com subconjunto viável
# -----------------------------------------------------------------------------------------
def _montar_subconjunto_regional_para_veiculo(
    pool_regional: pd.DataFrame,
    anchor_row: pd.Series,
    vehicle_row: pd.Series,
) -> pd.DataFrame:
    """
    Regionalidade = pool elegível.
    O algoritmo monta subconjunto viável dentro do pool.
    Esta versão evita recriar DataFrame a cada item.
    """
    if pool_regional.empty:
        return pd.DataFrame(columns=pool_regional.columns)

    pool = _ordenar_operacional(pool_regional.copy())

    anchor_id = str(anchor_row["id_linha_pipeline"])
    if "id_str_m5_1" not in pool.columns:
        pool["id_str_m5_1"] = pool["id_linha_pipeline"].astype(str)

    if anchor_id not in set(pool["id_str_m5_1"].tolist()):
        return pd.DataFrame(columns=pool.columns)

    # Garante âncora primeiro
    pool["anchor_first_tmp_m5_1"] = pool["id_str_m5_1"] == anchor_id
    pool = pool.sort_values(
        ["anchor_first_tmp_m5_1", "bucket_fase2_m5_1", "prioridade_ord_m5_1", "folga_ord_m5_1", "ranking_oper_ord_m5_1", "km_ord_m5_1", "peso_ord_m5_1", "id_str_m5_1"],
        ascending=[False, True, True, True, True, True, True, True],
        kind="mergesort",
    ).drop(columns=["anchor_first_tmp_m5_1"]).reset_index(drop=True)

    cap_peso = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    cap_vol = _safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0)
    max_entregas = _safe_int(vehicle_row.get("max_entregas"), 0)
    max_km = _safe_float(vehicle_row.get("max_km_distancia"), 0.0)
    ocup_max = _safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0)

    selected_ids: List[str] = []
    selected_destinatarios: set[str] = set()
    peso_total = 0.0
    volume_total = 0.0
    km_ref = 0.0

    for row in pool.itertuples(index=False, name="RowM51"):
        row_id = str(getattr(row, "id_str_m5_1"))
        restricao_value = getattr(row, "restricao_veiculo", None)

        if row_id in selected_ids:
            continue

        if not _restricao_compativel_linha(restricao_value, vehicle_row):
            continue

        row_peso = _safe_float(getattr(row, "peso_calculado", 0.0), 0.0)
        row_vol = _safe_float(getattr(row, "vol_m3", 0.0), 0.0)
        row_km = _safe_float(getattr(row, "distancia_rodoviaria_est_km", 0.0), 0.0)
        row_dest = _safe_text(getattr(row, "destinatario", ""))

        novo_peso = peso_total + row_peso
        novo_volume = volume_total + row_vol
        novo_km = max(km_ref, row_km)

        novos_destinatarios = selected_destinatarios.copy()
        if row_dest:
            novos_destinatarios.add(row_dest)
        novas_paradas = len(novos_destinatarios)

        if cap_peso > 0 and novo_peso > cap_peso:
            continue
        if cap_vol > 0 and novo_volume > cap_vol:
            continue
        if max_entregas > 0 and novas_paradas > max_entregas:
            continue
        if max_km > 0 and novo_km > max_km:
            continue
        if cap_peso > 0:
            ocup = (novo_peso / cap_peso) * 100.0
            if ocup > ocup_max:
                continue

        selected_ids.append(row_id)
        selected_destinatarios = novos_destinatarios
        peso_total = novo_peso
        volume_total = novo_volume
        km_ref = novo_km

    if not selected_ids:
        return pd.DataFrame(columns=pool.columns)

    df_sel = pool[pool["id_str_m5_1"].isin(selected_ids)].copy()
    df_sel = _ordenar_operacional(df_sel)

    # Se por algum motivo a âncora não entrou, não aceita
    if anchor_id not in set(df_sel["id_str_m5_1"].tolist()):
        return pd.DataFrame(columns=pool.columns)

    return df_sel.reset_index(drop=True)


def _tentar_pool_regional(
    pool_regional: pd.DataFrame,
    anchor_row: pd.Series,
    veiculos_ordenados: pd.DataFrame,
    camada: str,
    tentativas: List[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], str]:
    melhor_motivo = "nenhum_veiculo_compativel"
    anchor_id = str(anchor_row["id_linha_pipeline"])

    # Primeiro tenta o grupo cheio por veículo
    for _, vehicle_row in veiculos_ordenados.iterrows():
        ok_cheio, motivo_cheio = _validar_fechamento(pool_regional, vehicle_row)
        tentativas.append(
            _tentativa_dict(
                fase="fase_2_regional",
                camada=f"{camada}_grupo_cheio",
                anchor_id=anchor_id,
                vehicle_row=vehicle_row,
                resultado="fechado" if ok_cheio else "falhou",
                motivo=motivo_cheio,
                df_candidato=pool_regional,
            )
        )
        if ok_cheio:
            return pool_regional.copy(), vehicle_row.copy(), "ok"

    # Se grupo cheio não fechou, tenta subconjunto viável
    for _, vehicle_row in veiculos_ordenados.iterrows():
        candidato = _montar_subconjunto_regional_para_veiculo(
            pool_regional=pool_regional,
            anchor_row=anchor_row,
            vehicle_row=vehicle_row,
        )

        if candidato.empty:
            tentativas.append(
                _tentativa_dict(
                    fase="fase_2_regional",
                    camada=f"{camada}_subconjunto",
                    anchor_id=anchor_id,
                    vehicle_row=vehicle_row,
                    resultado="falhou",
                    motivo="sem_subconjunto_viavel_hard_constraints",
                    df_candidato=candidato,
                )
            )
            melhor_motivo = "sem_subconjunto_viavel_hard_constraints"
            continue

        ok, motivo = _validar_fechamento(candidato, vehicle_row)
        tentativas.append(
            _tentativa_dict(
                fase="fase_2_regional",
                camada=f"{camada}_subconjunto",
                anchor_id=anchor_id,
                vehicle_row=vehicle_row,
                resultado="fechado" if ok else "falhou",
                motivo=motivo,
                df_candidato=candidato,
            )
        )
        melhor_motivo = motivo

        if ok:
            return candidato.copy(), vehicle_row.copy(), "ok"

    return None, None, melhor_motivo


# -----------------------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------------------
def executar_m5_manifestos_compostos(
    df_remanescente_roteirizavel_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
    rodada_id: Optional[str] = None,
    data_base_roteirizacao: Optional[Any] = None,
    tipo_roteirizacao: str = "carteira",
    configuracao_frota: Optional[Any] = None,
    df_uso_frota_m4: Optional[pd.DataFrame] = None,
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    del configuracao_frota, df_uso_frota_m4, kwargs, rodada_id

    df_input, df_veic = _normalizar_inputs(
        df_remanescente_roteirizavel_bloco_4=df_remanescente_roteirizavel_bloco_4,
        df_veiculos_tratados=df_veiculos_tratados,
    )

    if df_input.empty:
        outputs_vazio = {
            "df_premanifestos_m5_1": pd.DataFrame(),
            "df_itens_premanifestos_m5_1": pd.DataFrame(),
            "df_tentativas_m5_1": pd.DataFrame(),
            "df_remanescente_m5_1": pd.DataFrame(),
            "df_nao_roteirizados_bloco_5_1": pd.DataFrame(),
            "df_uso_frota_m5_1": pd.DataFrame(),
        }
        meta_vazio = {
            "resumo_m5_1": {
                "modulo": "M5.1",
                "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
                "tipo_roteirizacao": tipo_roteirizacao,
                "remanescente_entrada_m5_1": 0,
                "pre_manifestos_gerados_m5_1": 0,
                "itens_pre_manifestados_m5_1": 0,
                "remanescente_saida_m5_1": 0,
                "nao_roteirizados_bloco_5_1": 0,
                "coluna_tipo_veiculo_utilizada": "tipo",
                "estrategia_m5_1": [
                    "fase_1_mesmo_cliente",
                    "saldo_para_cidade_subregiao_mesorregiao",
                    "regionalidade_como_pool_elegivel",
                    "subconjunto_viavel_ativo",
                    "VERSAO_M5_1_2026_04_10_FIX_01",
                ],
                "ocupacao_minima_padrao_perc": 70,
                "ocupacao_maxima_padrao_perc": 100,
                "anchors_geradas_m5_1": 0,
                "persistiu_artefatos": False,
                "caminhos_pipeline": caminhos_pipeline or {},
            },
            "auditoria_m5_1": {
                "total_tentativas": 0,
                "total_pre_manifestos": 0,
                "total_itens_pre_manifestados": 0,
                "total_remanescentes": 0,
                "anchors_geradas_m5_1": 0,
                "regionalidade_tratada_como_pool": True,
                "subconjunto_viavel_ativo": True,
            },
        }
        return outputs_vazio, meta_vazio

    veiculos_ordenados = _veiculos_maior_para_menor(df_veic)
    saldo = _precalcular_priorizacao(df_input.copy())
    saldo = _ordenar_operacional(saldo)

    tentativas: List[Dict[str, Any]] = []
    manifestos_list: List[pd.DataFrame] = []
    itens_manifestados_list: List[pd.DataFrame] = []
    manifest_seq = 1
    anchors_geradas = 0

    # =====================================================================================
    # FASE 1 - MESMO CLIENTE
    # =====================================================================================
    ids_consumidos_fase1: set[str] = set()

    grupos_cliente = []
    for cliente_chave, grp in saldo.groupby("cliente_chave_m5_1", sort=False):
        grupos_cliente.append((cliente_chave, _ordenar_operacional(grp.copy())))

    for cliente_chave, df_grupo in grupos_cliente:
        ids_grupo = set(df_grupo["id_str_m5_1"].tolist())
        if ids_consumidos_fase1 & ids_grupo:
            continue

        candidato, vehicle_row, motivo = _tentar_fechar_grupo_inteiro(
            df_grupo=df_grupo,
            veiculos_ordenados=veiculos_ordenados,
            tentativas=tentativas,
        )

        if candidato is not None and vehicle_row is not None:
            manifesto_id = _build_manifesto_id(manifest_seq)
            manifest_seq += 1
            df_manifesto, df_itens = _build_manifesto(
                df_itens=candidato,
                vehicle_row=vehicle_row,
                manifesto_id=manifesto_id,
                fase="fase_1_mesmo_cliente",
                camada="mesmo_cliente",
            )
            manifestos_list.append(df_manifesto)
            itens_manifestados_list.append(df_itens)
            ids_consumidos_fase1.update(ids_grupo)
        else:
            tentativas.append(
                {
                    "fase": "fase_1_mesmo_cliente",
                    "camada": "mesmo_cliente",
                    "anchor_id_linha_pipeline": None,
                    "veiculo_tipo_tentado": None,
                    "resultado": "saldo",
                    "motivo": motivo,
                    "qtd_itens_candidato": int(len(df_grupo)),
                    "qtd_paradas_candidato": _qtd_paradas(df_grupo),
                    "peso_total_candidato": round(_peso_total(df_grupo), 3),
                    "volume_total_candidato": round(_volume_total(df_grupo), 3),
                    "km_referencia_candidato": round(_km_referencia(df_grupo), 2),
                    "ocupacao_perc_candidato": 0.0,
                    "cliente_chave": cliente_chave,
                }
            )

    if ids_consumidos_fase1:
        saldo = saldo[~saldo["id_str_m5_1"].isin(ids_consumidos_fase1)].copy()
        saldo = _ordenar_operacional(saldo)

    # =====================================================================================
    # FASE 2 - SALDO PARA CIDADE / SUB / MESO
    # =====================================================================================
    while True:
        elegiveis = saldo[
            (saldo["anchor_attempted_m5_1"] == False)  # noqa: E712
            & _elegivel_fase2_df(saldo)
        ].copy()

        if elegiveis.empty:
            break

        elegiveis = _ordenar_operacional(elegiveis)
        anchor_row = elegiveis.iloc[0].copy()
        anchor_id = str(anchor_row["id_linha_pipeline"])
        anchors_geradas += 1

        saldo.loc[saldo["id_str_m5_1"] == anchor_id, "anchor_attempted_m5_1"] = True

        fechou_anchor = False
        melhor_motivo_anchor = "sem_pool_regional"

        for camada, col_key in [
            ("cidade", "cidade_key_m5_1"),
            ("subregiao", "subregiao_key_m5_1"),
            ("mesorregiao", "mesorregiao_key_m5_1"),
        ]:
            valor = _safe_text(anchor_row.get(col_key))
            if not valor:
                tentativas.append(
                    _tentativa_dict(
                        fase="fase_2_regional",
                        camada=camada,
                        anchor_id=anchor_id,
                        vehicle_row=None,
                        resultado="falhou",
                        motivo=f"{camada}_vazia",
                        df_candidato=None,
                    )
                )
                melhor_motivo_anchor = f"{camada}_vazia"
                continue

            pool = saldo[saldo[col_key] == valor].copy()
            pool = _ordenar_operacional(pool)

            if pool.empty:
                tentativas.append(
                    _tentativa_dict(
                        fase="fase_2_regional",
                        camada=camada,
                        anchor_id=anchor_id,
                        vehicle_row=None,
                        resultado="falhou",
                        motivo="pool_regional_vazio",
                        df_candidato=None,
                    )
                )
                melhor_motivo_anchor = "pool_regional_vazio"
                continue

            candidato, vehicle_row, motivo = _tentar_pool_regional(
                pool_regional=pool,
                anchor_row=anchor_row,
                veiculos_ordenados=veiculos_ordenados,
                camada=camada,
                tentativas=tentativas,
            )

            melhor_motivo_anchor = motivo

            if candidato is not None and vehicle_row is not None:
                manifesto_id = _build_manifesto_id(manifest_seq)
                manifest_seq += 1
                df_manifesto, df_itens = _build_manifesto(
                    df_itens=candidato,
                    vehicle_row=vehicle_row,
                    manifesto_id=manifesto_id,
                    fase="fase_2_regional",
                    camada=camada,
                )
                manifestos_list.append(df_manifesto)
                itens_manifestados_list.append(df_itens)

                consumed_ids = set(candidato["id_str_m5_1"].tolist())
                saldo = saldo[~saldo["id_str_m5_1"].isin(consumed_ids)].copy()
                saldo = _ordenar_operacional(saldo)
                fechou_anchor = True
                break

        if not fechou_anchor:
            tentativas.append(
                {
                    "fase": "fase_2_regional",
                    "camada": "fim_anchor",
                    "anchor_id_linha_pipeline": anchor_id,
                    "veiculo_tipo_tentado": None,
                    "resultado": "saldo",
                    "motivo": melhor_motivo_anchor,
                    "qtd_itens_candidato": 1,
                    "qtd_paradas_candidato": 1,
                    "peso_total_candidato": round(_safe_float(anchor_row.get("peso_calculado"), 0.0), 3),
                    "volume_total_candidato": round(_safe_float(anchor_row.get("vol_m3"), 0.0), 3),
                    "km_referencia_candidato": round(_safe_float(anchor_row.get("distancia_rodoviaria_est_km"), 0.0), 2),
                    "ocupacao_perc_candidato": 0.0,
                }
            )

    # =====================================================================================
    # SAÍDAS
    # =====================================================================================
    df_premanifestos_m5_1 = (
        pd.concat(manifestos_list, ignore_index=True) if manifestos_list else pd.DataFrame()
    )
    df_itens_premanifestos_m5_1 = (
        pd.concat(itens_manifestados_list, ignore_index=True) if itens_manifestados_list else pd.DataFrame()
    )
    df_tentativas_m5_1 = pd.DataFrame(tentativas)

    saldo = _drop_internal_cols(saldo.reset_index(drop=True))
    df_remanescente_m5_1 = saldo.copy()
    df_nao_roteirizados_bloco_5_1 = saldo.copy()

    if not df_itens_premanifestos_m5_1.empty:
        df_uso_frota_m5_1 = (
            df_itens_premanifestos_m5_1.groupby("veiculo_tipo", dropna=False)
            .agg(
                pre_manifestos=("manifesto_id", "nunique"),
                itens=("id_linha_pipeline", "count"),
                peso_total_kg=("peso_calculado", "sum"),
                paradas=("destinatario", "nunique"),
            )
            .reset_index()
        )
    else:
        df_uso_frota_m5_1 = pd.DataFrame(
            columns=["veiculo_tipo", "pre_manifestos", "itens", "peso_total_kg", "paradas"]
        )

    resumo_m5_1 = {
        "modulo": "M5.1",
        "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
        "tipo_roteirizacao": tipo_roteirizacao,
        "remanescente_entrada_m5_1": int(len(df_input)),
        "pre_manifestos_gerados_m5_1": int(df_premanifestos_m5_1["manifesto_id"].nunique())
        if not df_premanifestos_m5_1.empty
        else 0,
        "itens_pre_manifestados_m5_1": int(len(df_itens_premanifestos_m5_1)),
        "remanescente_saida_m5_1": int(len(df_remanescente_m5_1)),
        "nao_roteirizados_bloco_5_1": int(len(df_nao_roteirizados_bloco_5_1)),
        "coluna_tipo_veiculo_utilizada": "tipo",
        "estrategia_m5_1": [
            "fase_1_mesmo_cliente",
            "saldo_para_cidade_subregiao_mesorregiao",
            "regionalidade_como_pool_elegivel",
            "subconjunto_viavel_ativo",
            "VERSAO_M5_1_2026_04_10_FIX_01",
        ],
        "ocupacao_minima_padrao_perc": 70,
        "ocupacao_maxima_padrao_perc": 100,
        "anchors_geradas_m5_1": int(anchors_geradas),
        "persistiu_artefatos": False,
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    auditoria_m5_1 = {
        "total_tentativas": int(len(df_tentativas_m5_1)),
        "total_pre_manifestos": int(df_premanifestos_m5_1["manifesto_id"].nunique())
        if not df_premanifestos_m5_1.empty
        else 0,
        "total_itens_pre_manifestados": int(len(df_itens_premanifestos_m5_1)),
        "total_remanescentes": int(len(df_remanescente_m5_1)),
        "anchors_geradas_m5_1": int(anchors_geradas),
        "regionalidade_tratada_como_pool": True,
        "subconjunto_viavel_ativo": True,
    }

    outputs_m5_1 = {
        "df_premanifestos_m5_1": df_premanifestos_m5_1,
        "df_itens_premanifestos_m5_1": df_itens_premanifestos_m5_1,
        "df_tentativas_m5_1": df_tentativas_m5_1,
        "df_remanescente_m5_1": df_remanescente_m5_1,
        "df_nao_roteirizados_bloco_5_1": df_nao_roteirizados_bloco_5_1,
        "df_uso_frota_m5_1": df_uso_frota_m5_1,
    }

    meta_m5_1 = {
        "resumo_m5_1": resumo_m5_1,
        "auditoria_m5_1": auditoria_m5_1,
    }

    return outputs_m5_1, meta_m5_1


# Aliases defensivos
def executar_m5_1(*args: Any, **kwargs: Any):
    return executar_m5_manifestos_compostos(*args, **kwargs)


def processar_m5_1(*args: Any, **kwargs: Any):
    return executar_m5_manifestos_compostos(*args, **kwargs)


def rodar_m5_1(*args: Any, **kwargs: Any):
    return executar_m5_manifestos_compostos(*args, **kwargs)
