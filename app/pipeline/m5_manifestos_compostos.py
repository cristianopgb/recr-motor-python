from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =========================================================================================
# M5 - BLOCO DE COMPOSIÇÃO (VERSÃO ENXUTA PARA VALIDAÇÃO)
# -----------------------------------------------------------------------------------------
# OBJETIVO DESTA VERSÃO
# - manter contrato com pipeline_service.py
# - receber SOMENTE o remanescente do M4
# - executar composição por MESMA CIDADE
# - eliminar cedo combinações inviáveis
# - reduzir drasticamente a quantidade de cálculo
#
# LÓGICA DESTA VERSÃO
# 1) entrada dura = apenas remanescente do M4
# 2) agrupa por cidade
# 3) tenta primeiro o cluster completo
# 4) testa veículos do MAIOR para o MENOR
# 5) se cluster completo não fecha por capacidade / entregas / ocupação:
#    vai retirando a última entrega do cluster e testa novamente
# 6) se fechar, gera pré-manifesto
# 7) se não fechar, volta para o saldo
#
# OBSERVAÇÃO
# - esta versão não faz cidade -> subregião -> mesorregião no mesmo bloco
# - ela é propositalmente mais simples para validar performance com dataset real
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


def _drop_internal_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    cols_internal = [
        "_id_str_m5",
        "_cidade_key_m5",
        "_cliente_key_m5",
        "_bucket_m5",
        "_prioridade_ord_m5",
        "_folga_ord_m5",
        "_ranking_ord_m5",
        "_km_ord_m5",
        "_peso_ord_m5",
    ]
    existentes = [c for c in cols_internal if c in df.columns]
    if not existentes:
        return df.copy()
    return df.drop(columns=existentes, errors="ignore").copy()


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
        raise ValueError("M5 exige id_linha_pipeline em todas as linhas do remanescente do M4.")

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
        raise ValueError("M5 exige df_veiculos_tratados.")

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
    temp["_id_str_m5"] = temp["id_linha_pipeline"].astype(str)
    temp["_cidade_key_m5"] = temp["cidade"].fillna("").astype(str).str.strip()
    temp["_cliente_key_m5"] = temp["destinatario"].fillna("").astype(str).str.strip()

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

    temp["_bucket_m5"] = buckets
    temp["_prioridade_ord_m5"] = prioridade_ord
    temp["_folga_ord_m5"] = folga
    temp["_ranking_ord_m5"] = ranking
    temp["_km_ord_m5"] = km
    temp["_peso_ord_m5"] = -peso

    return temp


def _ordenar_operacional(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "_bucket_m5" not in df.columns:
        df = _precalcular_ordenacao(df)

    return (
        df.sort_values(
            by=[
                "_bucket_m5",
                "_prioridade_ord_m5",
                "_folga_ord_m5",
                "_ranking_ord_m5",
                "_km_ord_m5",
                "_peso_ord_m5",
                "_id_str_m5",
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
def _veiculos_maior_para_menor(df_veiculos: pd.DataFrame) -> pd.DataFrame:
    temp = df_veiculos.copy()
    temp["_cap_peso"] = temp["capacidade_peso_kg"].fillna(0)
    temp["_cap_vol"] = temp["capacidade_vol_m3"].fillna(0)
    temp = (
        temp.sort_values(["_cap_peso", "_cap_vol"], ascending=[False, False], kind="mergesort")
        .drop(columns=["_cap_peso", "_cap_vol"])
        .reset_index(drop=True)
    )
    return temp


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
# Auditoria
# -----------------------------------------------------------------------------------------
def _tentativa_dict(
    fase: str,
    camada: str,
    cluster_chave: Optional[str],
    vehicle_row: Optional[pd.Series],
    resultado: str,
    motivo: str,
    df_candidato: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    candidato = df_candidato if df_candidato is not None else pd.DataFrame()
    return {
        "fase": fase,
        "camada": camada,
        "cluster_chave": cluster_chave,
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
    qtd_ctes = int(df_itens_limpo["cte"].nunique(dropna=True)) if "cte" in df_itens_limpo.columns else qtd_itens

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
# Batida mínima de viabilidade do cluster
# -----------------------------------------------------------------------------------------
def _motivo_cluster_impossivel_por_raio(cluster_df: pd.DataFrame, veiculos_ordenados: pd.DataFrame) -> Optional[str]:
    km_cluster = _km_referencia(cluster_df)
    if veiculos_ordenados.empty:
        return "sem_veiculos"

    compativeis = veiculos_ordenados[
        pd.to_numeric(veiculos_ordenados["max_km_distancia"], errors="coerce").fillna(0) >= km_cluster
    ].copy()

    if compativeis.empty:
        return "sem_veiculo_compativel_por_raio"

    return None


def _menor_veiculo_compativel_por_raio(cluster_df: pd.DataFrame, veiculos_ordenados: pd.DataFrame) -> Optional[pd.Series]:
    km_cluster = _km_referencia(cluster_df)
    compativeis = veiculos_ordenados[
        pd.to_numeric(veiculos_ordenados["max_km_distancia"], errors="coerce").fillna(0) >= km_cluster
    ].copy()

    if compativeis.empty:
        return None

    compativeis["_cap_tmp"] = pd.to_numeric(compativeis["capacidade_peso_kg"], errors="coerce").fillna(0)
    compativeis = compativeis.sort_values("_cap_tmp", ascending=True, kind="mergesort").drop(columns=["_cap_tmp"])
    return compativeis.iloc[0].copy()


def _cluster_cheio_nem_atinge_minimo_do_menor(cluster_df: pd.DataFrame, veiculos_ordenados: pd.DataFrame) -> bool:
    menor = _menor_veiculo_compativel_por_raio(cluster_df, veiculos_ordenados)
    if menor is None:
        return False

    cap_peso = _safe_float(menor.get("capacidade_peso_kg"), 0.0)
    ocup_min = _safe_float(menor.get("ocupacao_minima_perc"), 70.0)
    peso_total = _peso_total(cluster_df)

    if cap_peso <= 0:
        return False

    peso_minimo = cap_peso * (ocup_min / 100.0)
    return peso_total < peso_minimo


# -----------------------------------------------------------------------------------------
# Tenta cluster da cidade
# -----------------------------------------------------------------------------------------
def _tentar_cluster_cidade(
    cluster_df: pd.DataFrame,
    veiculos_ordenados: pd.DataFrame,
    tentativas: List[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], str]:
    cidade_chave = _safe_text(cluster_df["_cidade_key_m5"].iloc[0]) if not cluster_df.empty else ""

    if cluster_df.empty:
        return None, None, "cluster_vazio"

    motivo_raio = _motivo_cluster_impossivel_por_raio(cluster_df, veiculos_ordenados)
    if motivo_raio is not None:
        tentativas.append(
            _tentativa_dict(
                fase="fase_2_mesma_cidade",
                camada="cidade_cluster_inteiro",
                cluster_chave=cidade_chave,
                vehicle_row=None,
                resultado="falhou",
                motivo=motivo_raio,
                df_candidato=cluster_df,
            )
        )
        return None, None, motivo_raio

    if _cluster_cheio_nem_atinge_minimo_do_menor(cluster_df, veiculos_ordenados):
        tentativas.append(
            _tentativa_dict(
                fase="fase_2_mesma_cidade",
                camada="cidade_cluster_inteiro",
                cluster_chave=cidade_chave,
                vehicle_row=None,
                resultado="falhou",
                motivo="cluster_cheio_nem_atinge_ocupacao_minima_do_menor_veiculo",
                df_candidato=cluster_df,
            )
        )
        return None, None, "cluster_cheio_nem_atinge_ocupacao_minima_do_menor_veiculo"

    cluster_ordenado = _ordenar_operacional(cluster_df.copy())
    melhor_motivo = "nenhum_veiculo_compativel"

    # primeiro tenta o cluster cheio
    for _, vehicle_row in veiculos_ordenados.iterrows():
        ok, motivo = _validar_fechamento(cluster_ordenado, vehicle_row)
        tentativas.append(
            _tentativa_dict(
                fase="fase_2_mesma_cidade",
                camada="cidade_cluster_inteiro",
                cluster_chave=cidade_chave,
                vehicle_row=vehicle_row,
                resultado="fechado" if ok else "falhou",
                motivo=motivo,
                df_candidato=cluster_ordenado,
            )
        )
        melhor_motivo = motivo
        if ok:
            return cluster_ordenado.copy(), vehicle_row.copy(), "ok"

    # se não fechou, vai reduzindo o cluster
    n = len(cluster_ordenado)
    if n <= 1:
        return None, None, melhor_motivo

    for k in range(n - 1, 0, -1):
        candidato = cluster_ordenado.head(k).copy()

        # eliminação rápida por raio
        motivo_raio_candidato = _motivo_cluster_impossivel_por_raio(candidato, veiculos_ordenados)
        if motivo_raio_candidato is not None:
            tentativas.append(
                _tentativa_dict(
                    fase="fase_2_mesma_cidade",
                    camada="cidade_subcluster",
                    cluster_chave=cidade_chave,
                    vehicle_row=None,
                    resultado="falhou",
                    motivo=motivo_raio_candidato,
                    df_candidato=candidato,
                )
            )
            melhor_motivo = motivo_raio_candidato
            continue

        # se nem o menor veículo por raio bate ocupação mínima com o cluster cheio do candidato, pula
        if _cluster_cheio_nem_atinge_minimo_do_menor(candidato, veiculos_ordenados):
            tentativas.append(
                _tentativa_dict(
                    fase="fase_2_mesma_cidade",
                    camada="cidade_subcluster",
                    cluster_chave=cidade_chave,
                    vehicle_row=None,
                    resultado="falhou",
                    motivo="subcluster_nem_atinge_ocupacao_minima_do_menor_veiculo",
                    df_candidato=candidato,
                )
            )
            melhor_motivo = "subcluster_nem_atinge_ocupacao_minima_do_menor_veiculo"
            continue

        for _, vehicle_row in veiculos_ordenados.iterrows():
            ok, motivo = _validar_fechamento(candidato, vehicle_row)
            tentativas.append(
                _tentativa_dict(
                    fase="fase_2_mesma_cidade",
                    camada="cidade_subcluster",
                    cluster_chave=cidade_chave,
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
                "modulo": "M5.2",
                "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
                "tipo_roteirizacao": tipo_roteirizacao,
                "remanescente_entrada_m5_1": 0,
                "pre_manifestos_gerados_m5_1": 0,
                "itens_pre_manifestados_m5_1": 0,
                "remanescente_saida_m5_1": 0,
                "nao_roteirizados_bloco_5_1": 0,
                "coluna_tipo_veiculo_utilizada": "tipo",
                "estrategia_m5_1": [
                    "entrada_dura_remanescente_m4",
                    "cluster_mesma_cidade",
                    "cluster_inteiro_primeiro",
                    "reducao_progressiva_do_cluster",
                    "veiculo_maior_para_menor",
                    "VERSAO_M5_CIDADE_2026_04_10",
                ],
                "ocupacao_minima_padrao_perc": 70,
                "ocupacao_maxima_padrao_perc": 100,
                "clusters_cidade_processados": 0,
                "persistiu_artefatos": False,
                "caminhos_pipeline": caminhos_pipeline or {},
            },
            "auditoria_m5_1": {
                "total_tentativas": 0,
                "total_pre_manifestos": 0,
                "total_itens_pre_manifestados": 0,
                "total_remanescentes": 0,
                "clusters_cidade_processados": 0,
            },
        }
        return outputs_vazio, meta_vazio

    veiculos_ordenados = _veiculos_maior_para_menor(df_veic)
    saldo = _precalcular_ordenacao(df_input.copy())
    saldo = _ordenar_operacional(saldo)

    tentativas: List[Dict[str, Any]] = []
    manifestos_list: List[pd.DataFrame] = []
    itens_manifestados_list: List[pd.DataFrame] = []

    manifest_seq = 1
    clusters_cidade_processados = 0

    # agrupa por cidade uma única vez, do jeito mais barato possível
    cidades = [
        cidade
        for cidade in saldo["_cidade_key_m5"].dropna().astype(str).tolist()
        if cidade.strip()
    ]
    cidades_ordenadas = list(dict.fromkeys(cidades))  # preserva ordem

    for cidade_key in cidades_ordenadas:
        cluster = saldo[saldo["_cidade_key_m5"] == cidade_key].copy()
        if cluster.empty:
            continue

        cluster = _ordenar_operacional(cluster)
        clusters_cidade_processados += 1

        candidato, vehicle_row, motivo = _tentar_cluster_cidade(
            cluster_df=cluster,
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
                fase="fase_2_mesma_cidade",
                camada="cidade",
            )

            manifestos_list.append(df_manifesto)
            itens_manifestados_list.append(df_itens)

            ids_consumidos = set(candidato["_id_str_m5"].tolist())
            saldo = saldo[~saldo["_id_str_m5"].isin(ids_consumidos)].copy()
            saldo = _ordenar_operacional(saldo)
        else:
            tentativas.append(
                {
                    "fase": "fase_2_mesma_cidade",
                    "camada": "cidade_fim_cluster",
                    "cluster_chave": cidade_key,
                    "veiculo_tipo_tentado": None,
                    "resultado": "saldo",
                    "motivo": motivo,
                    "qtd_itens_candidato": int(len(cluster)),
                    "qtd_paradas_candidato": _qtd_paradas(cluster),
                    "peso_total_candidato": round(_peso_total(cluster), 3),
                    "volume_total_candidato": round(_volume_total(cluster), 3),
                    "km_referencia_candidato": round(_km_referencia(cluster), 2),
                    "ocupacao_perc_candidato": 0.0,
                }
            )

    df_premanifestos_m5_1 = pd.concat(manifestos_list, ignore_index=True) if manifestos_list else pd.DataFrame()
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
        "modulo": "M5.2",
        "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
        "tipo_roteirizacao": tipo_roteirizacao,
        "remanescente_entrada_m5_1": int(len(df_input)),
        "pre_manifestos_gerados_m5_1": int(df_premanifestos_m5_1["manifesto_id"].nunique()) if not df_premanifestos_m5_1.empty else 0,
        "itens_pre_manifestados_m5_1": int(len(df_itens_premanifestos_m5_1)),
        "remanescente_saida_m5_1": int(len(df_remanescente_m5_1)),
        "nao_roteirizados_bloco_5_1": int(len(df_nao_roteirizados_bloco_5_1)),
        "coluna_tipo_veiculo_utilizada": "tipo",
        "estrategia_m5_1": [
            "entrada_dura_remanescente_m4",
            "cluster_mesma_cidade",
            "cluster_inteiro_primeiro",
            "reducao_progressiva_do_cluster",
            "veiculo_maior_para_menor",
            "eliminacao_precoce_por_raio_e_ocupacao_minima",
            "VERSAO_M5_CIDADE_2026_04_10",
        ],
        "ocupacao_minima_padrao_perc": 70,
        "ocupacao_maxima_padrao_perc": 100,
        "clusters_cidade_processados": int(clusters_cidade_processados),
        "persistiu_artefatos": False,
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    auditoria_m5_1 = {
        "total_tentativas": int(len(df_tentativas_m5_1)),
        "total_pre_manifestos": int(df_premanifestos_m5_1["manifesto_id"].nunique()) if not df_premanifestos_m5_1.empty else 0,
        "total_itens_pre_manifestados": int(len(df_itens_premanifestos_m5_1)),
        "total_remanescentes": int(len(df_remanescente_m5_1)),
        "clusters_cidade_processados": int(clusters_cidade_processados),
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
