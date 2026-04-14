from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.pipeline.m5_common import (
    normalize_saldo_m5,
    safe_float,
    safe_int,
    safe_text,
    precalcular_ordenacao_m5,
    ordenar_operacional_m5,
    peso_total,
    peso_auditoria_total,
    volume_total,
    km_referencia,
    qtd_paradas,
    ocupacao_perc,
    grupo_respeita_restricao_veiculo,
)


# =========================================================================================
# M5.2 - COMPOSIÇÃO POR CIDADE
# -----------------------------------------------------------------------------------------
# REGRA OPERACIONAL
# - processa cidade por cidade
# - dentro da cidade, trabalha por blocos de cliente
# - busca a MELHOR combinação de clientes da rodada
# - objetivo: tirar o máximo de cargas possível com a melhor ocupação válida
# - não prioriza simplesmente o maior cliente
# - pode gerar múltiplos pré-manifestos na mesma cidade
# - perfis elegíveis são orientativos; não desistir no primeiro perfil que falhar
#
# REGRAS DE PESO
# - base oficial = peso_calculado
# - peso_kg = auditoria
# - vol_m3 = volume
# - M5.2 não recria peso_calculado
# =========================================================================================


MAX_BLOCOS_PARA_BUSCA_EXATA = 12
MAX_COMBINACOES_POR_PERFIL = 6000


# -----------------------------------------------------------------------------------------
# Helpers locais
# -----------------------------------------------------------------------------------------
def _drop_internal_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    cols_internal = [
        f"_id_str_{suffix}",
        f"_cidade_key_{suffix}",
        f"_uf_key_{suffix}",
        f"_cliente_key_{suffix}",
        f"_bucket_{suffix}",
        f"_prioridade_ord_{suffix}",
        f"_folga_ord_{suffix}",
        f"_ranking_ord_{suffix}",
        f"_km_ord_{suffix}",
        f"_peso_ord_{suffix}",
    ]
    existentes = [c for c in cols_internal if c in df.columns]
    if not existentes:
        return df.copy()
    return df.drop(columns=existentes, errors="ignore").copy()


def _ordenar_cidades_por_massa(df_saldo: pd.DataFrame) -> List[Tuple[str, str]]:
    if df_saldo.empty:
        return []

    agrupado = (
        df_saldo.groupby(["cidade", "uf"], dropna=False, sort=False)
        .agg(
            peso_total_cidade=("peso_calculado", "sum"),
            qtd_linhas_cidade=("id_linha_pipeline", "count"),
        )
        .reset_index()
        .sort_values(
            by=["peso_total_cidade", "cidade", "uf"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return [(safe_text(r["cidade"]), safe_text(r["uf"])) for _, r in agrupado.iterrows()]


def _agrupar_blocos_cliente_na_cidade(city_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if city_df.empty:
        return pd.DataFrame()

    temp = city_df.copy()

    cliente_key_col = f"_cliente_key_{suffix}"
    bucket_col = f"_bucket_{suffix}"
    ranking_col = f"_ranking_ord_{suffix}"

    grouped = (
        temp.groupby([cliente_key_col, "destinatario"], dropna=False)
        .agg(
            peso_total_bloco=("peso_calculado", "sum"),
            peso_kg_total_bloco=("peso_kg", "sum"),
            volume_total_bloco=("vol_m3", "sum"),
            km_referencia_bloco=("distancia_rodoviaria_est_km", "max"),
            qtd_linhas_bloco=("id_linha_pipeline", "count"),
            prioridade_min=(bucket_col, "min"),
            ranking_min=(ranking_col, "min"),
        )
        .reset_index()
        .sort_values(
            by=["peso_total_bloco", "prioridade_min", "ranking_min", cliente_key_col],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    grouped["ordem_bloco_desc"] = range(1, len(grouped) + 1)
    return grouped


def _materializar_candidato_por_blocos(
    city_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    suffix: str,
) -> pd.DataFrame:
    if city_df.empty or blocks_df.empty:
        return pd.DataFrame(columns=city_df.columns)

    cliente_key_col = f"_cliente_key_{suffix}"
    keys = set(blocks_df[cliente_key_col].tolist())

    candidato = city_df[city_df[cliente_key_col].isin(keys)].copy()
    candidato = ordenar_operacional_m5(candidato, suffix=suffix)
    return candidato.reset_index(drop=True)


def _validar_hard_constraints(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> Tuple[bool, str]:
    if df_itens.empty:
        return False, "grupo_vazio"

    if not grupo_respeita_restricao_veiculo(df_itens, vehicle_row):
        return False, "restricao_veiculo_incompativel"

    peso_oficial = peso_total(df_itens)
    volume = volume_total(df_itens)
    paradas = qtd_paradas(df_itens)
    km_ref = km_referencia(df_itens)

    cap_peso = safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    cap_vol = safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0)
    max_entregas = safe_int(vehicle_row.get("max_entregas"), 0)
    max_km = safe_float(vehicle_row.get("max_km_distancia"), 0.0)
    ocup_max = safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0)

    if cap_peso > 0 and peso_oficial > cap_peso:
        return False, "excede_capacidade_peso"
    if cap_vol > 0 and volume > cap_vol:
        return False, "excede_capacidade_volume"
    if max_entregas > 0 and paradas > max_entregas:
        return False, "excede_max_entregas"
    if max_km > 0 and km_ref > max_km:
        return False, "excede_max_km"

    ocup = ocupacao_perc(df_itens, vehicle_row)
    if ocup > ocup_max:
        return False, "excede_ocupacao_maxima"

    return True, "ok"


def _validar_fechamento(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> Tuple[bool, str]:
    ok_hard, motivo_hard = _validar_hard_constraints(df_itens, vehicle_row)
    if not ok_hard:
        return False, motivo_hard

    ocup_min = safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0)
    ocup = ocupacao_perc(df_itens, vehicle_row)

    if ocup < ocup_min:
        return False, "abaixo_ocupacao_minima"

    return True, "ok"


def _score_candidato(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> Tuple[float, float, int, float]:
    """
    Quanto maior, melhor.
    Ordem:
    1. ocupação válida mais alta
    2. maior peso expedido
    3. mais clientes
    4. menor capacidade do veículo como desempate implícito via negativo
    """
    ocup = ocupacao_perc(df_itens, vehicle_row)
    peso = peso_total(df_itens)
    clientes = qtd_paradas(df_itens)
    cap = safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)

    return (
        round(ocup, 6),
        round(peso, 6),
        int(clientes),
        -cap,
    )


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
        "veiculo_tipo_tentado": None if vehicle_row is None else safe_text(vehicle_row.get("tipo")),
        "veiculo_perfil_tentado": None if vehicle_row is None else safe_text(vehicle_row.get("perfil")),
        "resultado": resultado,
        "motivo": motivo,
        "qtd_itens_candidato": int(len(candidato)),
        "qtd_paradas_candidato": qtd_paradas(candidato),
        "peso_total_candidato": round(peso_total(candidato), 3),
        "peso_kg_total_candidato": round(peso_auditoria_total(candidato), 3),
        "volume_total_candidato": round(volume_total(candidato), 3),
        "km_referencia_candidato": round(km_referencia(candidato), 2),
        "ocupacao_perc_candidato": round(ocupacao_perc(candidato, vehicle_row), 2)
        if vehicle_row is not None and not candidato.empty
        else 0.0,
    }


def _build_manifesto_id(seq: int) -> str:
    return f"PM52_{seq:04d}"


def _build_manifesto(
    df_itens: pd.DataFrame,
    vehicle_row: pd.Series,
    manifesto_id: str,
    cidade: str,
    uf: str,
    suffix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_itens_limpo = _drop_internal_cols(df_itens, suffix=suffix)

    qtd_itens = int(len(df_itens_limpo))
    qtd_ctes = int(df_itens_limpo["cte"].nunique(dropna=True)) if "cte" in df_itens_limpo.columns else qtd_itens

    manifesto = {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": "pre_manifesto_bloco_5_2_cidade",
        "cidade": cidade,
        "uf": uf,
        "veiculo_tipo": safe_text(vehicle_row.get("tipo")),
        "veiculo_perfil": safe_text(vehicle_row.get("perfil")),
        "qtd_itens": qtd_itens,
        "qtd_ctes": qtd_ctes,
        "qtd_paradas": qtd_paradas(df_itens_limpo),
        "base_carga_oficial": round(peso_total(df_itens_limpo), 3),
        "peso_total_kg": round(peso_auditoria_total(df_itens_limpo), 3),
        "vol_total_m3": round(volume_total(df_itens_limpo), 3),
        "km_referencia": round(km_referencia(df_itens_limpo), 2),
        "ocupacao_oficial_perc": round(ocupacao_perc(df_itens_limpo, vehicle_row), 2),
        "capacidade_peso_kg_veiculo": safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0),
        "capacidade_vol_m3_veiculo": safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0),
        "max_entregas_veiculo": safe_int(vehicle_row.get("max_entregas"), 0),
        "max_km_distancia_veiculo": safe_float(vehicle_row.get("max_km_distancia"), 0.0),
        "ocupacao_minima_perc_veiculo": safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0),
        "ocupacao_maxima_perc_veiculo": safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0),
        "ignorar_ocupacao_minima": False,
        "origem_modulo": 5,
        "origem_etapa": "m5_2_composicao_cidade",
    }

    df_manifesto = pd.DataFrame([manifesto])

    df_itens_saida = df_itens_limpo.copy()
    for k, v in manifesto.items():
        df_itens_saida[k] = v

    return df_manifesto, df_itens_saida


def _get_eligible_vehicles_for_city(
    city: str,
    uf: str,
    perfis_elegiveis_df: pd.DataFrame,
) -> pd.DataFrame:
    base = perfis_elegiveis_df[
        (perfis_elegiveis_df["cidade"].fillna("").astype(str).str.strip() == city)
        & (perfis_elegiveis_df["uf"].fillna("").astype(str).str.strip() == uf)
    ].copy()

    if base.empty:
        return pd.DataFrame()

    for col in [
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ocupacao_minima_perc",
        "ocupacao_maxima_perc",
    ]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    # Maior para menor: perfis são orientativos, não impeditivos no primeiro teste.
    base = base.sort_values(
        by=["capacidade_peso_kg", "capacidade_vol_m3", "tipo", "perfil"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return base


def _gerar_combinacoes_blocos(
    blocks_df: pd.DataFrame,
    max_blocos_busca_exata: int = MAX_BLOCOS_PARA_BUSCA_EXATA,
    max_combinacoes: int = MAX_COMBINACOES_POR_PERFIL,
) -> List[pd.DataFrame]:
    """
    Gera combinações candidatas de blocos.
    Busca exata limitada quando o número de blocos é pequeno.
    Para volumes maiores, usa uma busca reduzida guiada.
    """
    if blocks_df.empty:
        return []

    n = len(blocks_df)
    resultados: List[pd.DataFrame] = []

    if n <= max_blocos_busca_exata:
        contador = 0
        # Preferir combinações maiores primeiro ajuda a tirar mais carga
        for r in range(n, 0, -1):
            for idxs in combinations(range(n), r):
                resultados.append(blocks_df.iloc[list(idxs)].copy())
                contador += 1
                if contador >= max_combinacoes:
                    return resultados
        return resultados

    # Busca reduzida para cidades com muitos blocos:
    # usa prefixos e janelas sobre blocos já ordenados por peso/prioridade.
    limites = [min(n, x) for x in [3, 4, 5, 6, 7, 8, 10, 12]]
    vistos: set[tuple[str, ...]] = set()

    for lim in limites:
        candidato = blocks_df.head(lim).copy()
        chave = tuple(candidato.iloc[:, 0].astype(str).tolist())
        if chave not in vistos:
            vistos.add(chave)
            resultados.append(candidato)

    # Também combinações retirando 1 e 2 blocos do topo mais relevante
    base = blocks_df.head(min(n, 12)).copy()
    idxs_base = list(range(len(base)))

    for remover_qtd in [1, 2]:
        for remover in combinations(idxs_base, remover_qtd):
            manter = [i for i in idxs_base if i not in remover]
            if not manter:
                continue
            candidato = base.iloc[manter].copy()
            chave = tuple(candidato.iloc[:, 0].astype(str).tolist())
            if chave not in vistos:
                vistos.add(chave)
                resultados.append(candidato)
            if len(resultados) >= max_combinacoes:
                return resultados

    return resultados[:max_combinacoes]


def _buscar_melhor_fechamento_na_cidade(
    city_df: pd.DataFrame,
    perfis_elegiveis_df: pd.DataFrame,
    cidade: str,
    uf: str,
    tentativas: List[Dict[str, Any]],
    suffix: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], str]:
    if city_df.empty:
        return None, None, "cidade_vazia"

    vehicles_city = _get_eligible_vehicles_for_city(
        city=cidade,
        uf=uf,
        perfis_elegiveis_df=perfis_elegiveis_df,
    )
    if vehicles_city.empty:
        tentativas.append(
            _tentativa_dict(
                cidade=cidade,
                uf=uf,
                vehicle_row=None,
                resultado="falhou",
                motivo="sem_perfil_elegivel_na_cidade",
                df_candidato=city_df,
                tentativa_idx=1,
                blocos_considerados=0,
            )
        )
        return None, None, "sem_perfil_elegivel_na_cidade"

    blocks_df = _agrupar_blocos_cliente_na_cidade(city_df, suffix=suffix)
    if blocks_df.empty:
        return None, None, "sem_blocos_na_cidade"

    combinacoes_blocos = _gerar_combinacoes_blocos(blocks_df)

    melhor_df: Optional[pd.DataFrame] = None
    melhor_vehicle: Optional[pd.Series] = None
    melhor_score: Optional[Tuple[float, float, int, float]] = None
    melhor_motivo = "nenhum_fechamento"

    tentativa_idx = 1

    for _, vehicle_row in vehicles_city.iterrows():
        for blocks_candidato in combinacoes_blocos:
            candidato = _materializar_candidato_por_blocos(city_df, blocks_candidato, suffix=suffix)
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
                    blocos_considerados=int(len(blocks_candidato)),
                )
            )
            tentativa_idx += 1
            melhor_motivo = motivo

            if not ok:
                continue

            score = _score_candidato(candidato, vehicle_row)

            if melhor_score is None or score > melhor_score:
                melhor_score = score
                melhor_df = candidato.copy()
                melhor_vehicle = vehicle_row.copy()

    if melhor_df is None or melhor_vehicle is None:
        return None, None, melhor_motivo

    return melhor_df, melhor_vehicle, "ok"


# -----------------------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------------------
def executar_m5_2_composicao_cidades(
    df_saldo_elegivel_composicao_m5_1: pd.DataFrame,
    df_perfis_elegiveis_por_cidade_m5_1: pd.DataFrame,
    rodada_id: Optional[str] = None,
    data_base_roteirizacao: Optional[Any] = None,
    tipo_roteirizacao: str = "carteira",
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    del rodada_id, kwargs

    suffix = "m5_2"

    saldo = normalize_saldo_m5(
        df_input=df_saldo_elegivel_composicao_m5_1,
        etapa="M5.2",
        require_geo=True,
        require_subregiao=False,
        require_mesorregiao=False,
    )

    perfis_elegiveis = (
        df_perfis_elegiveis_por_cidade_m5_1.copy()
        if df_perfis_elegiveis_por_cidade_m5_1 is not None
        else pd.DataFrame()
    )

    if perfis_elegiveis.empty:
        raise ValueError("M5.2 exige df_perfis_elegiveis_por_cidade_m5_1.")

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
                    "solver_melhor_combinacao",
                    "maximiza_ocupacao_e_aproveitamento",
                    "multiplos_fechamentos_na_mesma_cidade",
                    "VERSAO_M5_2_2026_04_14",
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

    saldo = precalcular_ordenacao_m5(saldo, suffix=suffix)
    saldo = ordenar_operacional_m5(saldo, suffix=suffix)

    manifestos_list: List[pd.DataFrame] = []
    itens_manifestados_list: List[pd.DataFrame] = []
    tentativas: List[Dict[str, Any]] = []

    manifesto_seq = 1
    cidades_processadas = 0

    cidades_keys = _ordenar_cidades_por_massa(saldo)

    for cidade_key, uf_key in cidades_keys:
        cidades_processadas += 1

        while True:
            city_df = saldo[
                (saldo[f"_cidade_key_{suffix}"] == cidade_key)
                & (saldo[f"_uf_key_{suffix}"] == uf_key)
            ].copy()

            if city_df.empty:
                break

            candidato, vehicle_row, motivo = _buscar_melhor_fechamento_na_cidade(
                city_df=city_df,
                perfis_elegiveis_df=perfis_elegiveis,
                cidade=cidade_key,
                uf=uf_key,
                tentativas=tentativas,
                suffix=suffix,
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
                        "qtd_paradas_candidato": qtd_paradas(city_df),
                        "peso_total_candidato": round(peso_total(city_df), 3),
                        "peso_kg_total_candidato": round(peso_auditoria_total(city_df), 3),
                        "volume_total_candidato": round(volume_total(city_df), 3),
                        "km_referencia_candidato": round(km_referencia(city_df), 2),
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
                suffix=suffix,
            )

            manifestos_list.append(df_manifesto)
            itens_manifestados_list.append(df_itens)

            ids_consumidos = set(candidato[f"_id_str_{suffix}"].tolist())
            saldo = saldo[~saldo[f"_id_str_{suffix}"].isin(ids_consumidos)].copy()

            if saldo.empty:
                break

            saldo = ordenar_operacional_m5(saldo, suffix=suffix)

    df_premanifestos_m5_2 = (
        pd.concat(manifestos_list, ignore_index=True)
        if manifestos_list
        else pd.DataFrame()
    )

    df_itens_premanifestos_m5_2 = (
        pd.concat(itens_manifestados_list, ignore_index=True)
        if itens_manifestados_list
        else pd.DataFrame()
    )

    df_tentativas_m5_2 = pd.DataFrame(tentativas)
    df_remanescente_m5_2 = _drop_internal_cols(saldo.reset_index(drop=True), suffix=suffix)

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
            "solver_melhor_combinacao",
            "maximiza_ocupacao_e_aproveitamento",
            "multiplos_fechamentos_na_mesma_cidade",
            "VERSAO_M5_2_2026_04_14",
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
