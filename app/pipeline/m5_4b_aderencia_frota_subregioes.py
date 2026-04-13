from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =========================================================================================
# M5.4B - ADERÊNCIA DE FROTA POR SUBREGIÃO
# -----------------------------------------------------------------------------------------
# OBJETIVO
# - receber a base preparada do M5.4A
# - analisar perfil por subregião
# - eliminar perfis claramente inviáveis
# - manter só os perfis que valem tentativa no M5.4C
#
# ESTA ETAPA NÃO:
# - não compõe
# - não fecha pré-manifesto
# - não remove itens do pool
#
# REGRA CONSOLIDADA DESTA VERSÃO
# - o raio fino NÃO derruba perfil no M5.4B
# - o raio fino fica para o M5.4C
# - no M5.4B a análise é:
#   1. massa total da subregião
#   2. combinação simples pelo limite de entregas
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


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


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


def _km_max(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(pd.to_numeric(df["distancia_rodoviaria_est_km"], errors="coerce").fillna(0).max())


def _qtd_clientes(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df["destinatario"].fillna("").astype(str).nunique())


def _qtd_cidades(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df["cidade"].fillna("").astype(str).nunique())


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
        "distancia_rodoviaria_est_km": 0.0,
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
        "distancia_rodoviaria_est_km",
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
        _ensure_column(veiculos, col, pd.NA if col not in ["perfil", "tipo"] else "")

    for col in [
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ocupacao_minima_perc",
        "ocupacao_maxima_perc",
    ]:
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
# Veículos do maior para o menor
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
# Blocos de cliente da subregião
# -----------------------------------------------------------------------------------------
def _blocos_clientes_subregiao(df_sub: pd.DataFrame) -> pd.DataFrame:
    if df_sub.empty:
        return pd.DataFrame()

    blocos = (
        df_sub.groupby(["cidade", "destinatario"], dropna=False, sort=False)
        .agg(
            peso_total_cliente=("peso_calculado", "sum"),
            km_cliente=("distancia_rodoviaria_est_km", "max"),
            ordem_cidade=("ordem_cidade_na_subregiao_m5_4a", "min"),
            ordem_cliente=("ordem_cliente_na_cidade_m5_4a", "min"),
        )
        .reset_index()
    )

    blocos = blocos.sort_values(
        by=["peso_total_cliente", "ordem_cidade", "ordem_cliente", "cidade", "destinatario"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return blocos


# -----------------------------------------------------------------------------------------
# Teste simples de entregas
# -----------------------------------------------------------------------------------------
def _existe_combinacao_simples_viavel(
    blocos: pd.DataFrame,
    vehicle_row: pd.Series,
    max_trocas: int = 20,
) -> Tuple[bool, Dict[str, Any]]:
    if blocos.empty:
        return False, {
            "motivo": "sem_blocos_na_subregiao",
            "peso_melhor_combinacao": 0.0,
            "qtd_entregas_melhor_combinacao": 0,
        }

    max_entregas = _safe_int(vehicle_row.get("max_entregas"), 0)
    if max_entregas <= 0:
        return False, {
            "motivo": "perfil_sem_max_entregas",
            "peso_melhor_combinacao": 0.0,
            "qtd_entregas_melhor_combinacao": 0,
        }

    ocup_min_kg = _ocupacao_minima_kg(vehicle_row)
    ocup_max_kg = _ocupacao_maxima_kg(vehicle_row)

    pesos = blocos["peso_total_cliente"].tolist()
    n = len(pesos)
    k = min(max_entregas, n)

    if k <= 0:
        return False, {
            "motivo": "sem_entregas_possiveis",
            "peso_melhor_combinacao": 0.0,
            "qtd_entregas_melhor_combinacao": 0,
        }

    melhor_soma_abaixo = 0.0
    melhor_qtd_abaixo = 0

    # tentativa 1: top k maiores
    escolhidos = list(range(k))
    soma = sum(pesos[i] for i in escolhidos)

    if ocup_min_kg <= soma <= ocup_max_kg:
        return True, {
            "motivo": "ok_com_maiores_blocos",
            "peso_melhor_combinacao": round(soma, 3),
            "qtd_entregas_melhor_combinacao": k,
        }

    if soma < ocup_min_kg:
        melhor_soma_abaixo = soma
        melhor_qtd_abaixo = k

    # tentativa 2: menores quantidades até k
    for qtd in range(1, k + 1):
        soma_qtd = sum(pesos[:qtd])

        if ocup_min_kg <= soma_qtd <= ocup_max_kg:
            return True, {
                "motivo": "ok_com_quantidade_menor_entregas",
                "peso_melhor_combinacao": round(soma_qtd, 3),
                "qtd_entregas_melhor_combinacao": qtd,
            }

        if soma_qtd < ocup_min_kg and soma_qtd > melhor_soma_abaixo:
            melhor_soma_abaixo = soma_qtd
            melhor_qtd_abaixo = qtd

    # tentativa 3: se top k passou do máximo, troca o último por blocos menores
    if soma > ocup_max_kg and n > k:
        base_idxs = list(range(k))
        reserva_idxs = list(range(k, n))
        tentativas = 0

        for pos_sub in range(k - 1, -1, -1):
            for idx_reserva in reserva_idxs:
                tentativas += 1
                if tentativas > max_trocas:
                    break

                tentativa_idxs = base_idxs.copy()
                tentativa_idxs[pos_sub] = idx_reserva
                tentativa_idxs = sorted(set(tentativa_idxs))

                if len(tentativa_idxs) == 0 or len(tentativa_idxs) > k:
                    continue

                soma_tent = sum(pesos[i] for i in tentativa_idxs)

                if ocup_min_kg <= soma_tent <= ocup_max_kg:
                    return True, {
                        "motivo": "ok_com_substituicao_bloco_menor",
                        "peso_melhor_combinacao": round(soma_tent, 3),
                        "qtd_entregas_melhor_combinacao": len(tentativa_idxs),
                    }

                if soma_tent < ocup_min_kg and soma_tent > melhor_soma_abaixo:
                    melhor_soma_abaixo = soma_tent
                    melhor_qtd_abaixo = len(tentativa_idxs)

            if tentativas > max_trocas:
                break

    return False, {
        "motivo": "nao_ha_combinacao_simples_dentro_limite_entregas",
        "peso_melhor_combinacao": round(melhor_soma_abaixo, 3),
        "qtd_entregas_melhor_combinacao": int(melhor_qtd_abaixo),
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
        }
        meta_vazio = {
            "resumo_m5_4b": {
                "modulo": "M5.4B",
                "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
                "tipo_roteirizacao": tipo_roteirizacao,
                "linhas_entrada_m5_4b": 0,
                "total_registros_frota_aderente_m5_4b": 0,
                "total_registros_frota_nao_aderente_m5_4b": 0,
                "estrategia_m5_4b": [
                    "analise_por_subregiao",
                    "perfis_do_maior_para_o_menor",
                    "sem_filtro_fino_de_raio_no_m5_4b",
                    "teste_simples_por_limite_de_entregas",
                    "VERSAO_M5_4B_2026_04_12_C",
                ],
                "caminhos_pipeline": caminhos_pipeline or {},
            }
        }
        return outputs_vazio, meta_vazio

    veiculos_ord = _veiculos_maior_para_menor(veiculos)

    aderentes_rows: List[Dict[str, Any]] = []
    nao_aderentes_rows: List[Dict[str, Any]] = []

    subregioes = (
        base[
            [
                "subregiao",
                "uf",
                "ordem_subregiao_m5_4a",
                "peso_total_subregiao_m5_4a",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            by=["ordem_subregiao_m5_4a", "peso_total_subregiao_m5_4a", "subregiao", "uf"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    for _, sub_row in subregioes.iterrows():
        subregiao = _safe_text(sub_row.get("subregiao"))
        uf = _safe_text(sub_row.get("uf"))
        ordem_sub = _safe_int(sub_row.get("ordem_subregiao_m5_4a"), 0)

        df_sub = base[(base["subregiao"] == subregiao) & (base["uf"] == uf)].copy()
        if df_sub.empty:
            continue

        peso_total_sub = _peso_total(df_sub)
        qtd_clientes_sub = _qtd_clientes(df_sub)
        qtd_cidades_sub = _qtd_cidades(df_sub)
        km_max_sub = _km_max(df_sub)

        blocos_sub = _blocos_clientes_subregiao(df_sub)

        for ordem_perfil, (_, vehicle_row) in enumerate(veiculos_ord.iterrows(), start=1):
            perfil = _safe_text(vehicle_row.get("perfil"))
            tipo = _safe_text(vehicle_row.get("tipo"))

            cap_peso = _safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
            max_entregas = _safe_int(vehicle_row.get("max_entregas"), 0)
            max_km = _safe_float(vehicle_row.get("max_km_distancia"), 0.0)
            ocup_min_kg = _ocupacao_minima_kg(vehicle_row)
            ocup_max_kg = _ocupacao_maxima_kg(vehicle_row)

            registro_base = {
                "subregiao": subregiao,
                "uf": uf,
                "ordem_subregiao_m5_4b": ordem_sub,
                "perfil": perfil,
                "tipo": tipo,
                "ordem_tentativa_perfil_m5_4b": ordem_perfil,
                "capacidade_peso_kg": round(cap_peso, 3),
                "max_entregas": max_entregas,
                "max_km_distancia": round(max_km, 3),
                "ocupacao_minima_perc": round(_safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0), 3),
                "ocupacao_maxima_perc": round(_safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0), 3),
                "ocupacao_minima_kg": round(ocup_min_kg, 3),
                "ocupacao_maxima_kg": round(ocup_max_kg, 3),
                "peso_total_subregiao": round(peso_total_sub, 3),
                "qtd_clientes_subregiao": qtd_clientes_sub,
                "qtd_cidades_subregiao": qtd_cidades_sub,
                "km_max_subregiao": round(km_max_sub, 2),
                # mantidos só para auditoria visual, sem corte de lógica no 5.4B
                "peso_total_util_raio": round(peso_total_sub, 3),
                "qtd_clientes_util_raio": qtd_clientes_sub,
                "km_max_util_raio": round(km_max_sub, 2),
                "qtd_cidades_fora_do_raio": 0,
            }

            # 1) elimina por ocupação mínima da subregião inteira
            if peso_total_sub < ocup_min_kg:
                nao_aderentes_rows.append(
                    {
                        **registro_base,
                        "status_aderencia_m5_4b": "nao_aderente",
                        "motivo_aderencia_m5_4b": "peso_total_subregiao_abaixo_ocupacao_minima",
                    }
                )
                continue

            # 2) elimina por combinação simples de entregas
            ok_entregas, info_combo = _existe_combinacao_simples_viavel(
                blocos=blocos_sub,
                vehicle_row=vehicle_row,
            )

            if not ok_entregas:
                nao_aderentes_rows.append(
                    {
                        **registro_base,
                        "status_aderencia_m5_4b": "nao_aderente",
                        "motivo_aderencia_m5_4b": _safe_text(info_combo.get("motivo")),
                        "peso_melhor_combinacao": round(_safe_float(info_combo.get("peso_melhor_combinacao"), 0.0), 3),
                        "qtd_entregas_melhor_combinacao": _safe_int(info_combo.get("qtd_entregas_melhor_combinacao"), 0),
                    }
                )
                continue

            aderentes_rows.append(
                {
                    **registro_base,
                    "status_aderencia_m5_4b": "aderente",
                    "motivo_aderencia_m5_4b": "perfil_aderente",
                    "peso_melhor_combinacao": round(_safe_float(info_combo.get("peso_melhor_combinacao"), 0.0), 3),
                    "qtd_entregas_melhor_combinacao": _safe_int(info_combo.get("qtd_entregas_melhor_combinacao"), 0),
                    "motivo_combinacao_simples": _safe_text(info_combo.get("motivo")),
                }
            )

    df_frota_aderente_m5_4b = pd.DataFrame(aderentes_rows)
    df_frota_nao_aderente_m5_4b = pd.DataFrame(nao_aderentes_rows)

    resumo_m5_4b = {
        "modulo": "M5.4B",
        "data_base_roteirizacao": str(data_base_roteirizacao) if data_base_roteirizacao is not None else None,
        "tipo_roteirizacao": tipo_roteirizacao,
        "linhas_entrada_m5_4b": int(len(base)),
        "subregioes_analisadas_m5_4b": int(subregioes["subregiao"].nunique()),
        "total_registros_frota_aderente_m5_4b": int(len(df_frota_aderente_m5_4b)),
        "total_registros_frota_nao_aderente_m5_4b": int(len(df_frota_nao_aderente_m5_4b)),
        "estrategia_m5_4b": [
            "analise_por_subregiao",
            "perfis_do_maior_para_o_menor",
            "sem_filtro_fino_de_raio_no_m5_4b",
            "teste_simples_por_limite_de_entregas",
            "VERSAO_M5_4B_2026_04_12_C",
        ],
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    outputs_m5_4b = {
        "df_frota_aderente_m5_4b": df_frota_aderente_m5_4b,
        "df_frota_nao_aderente_m5_4b": df_frota_nao_aderente_m5_4b,
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
