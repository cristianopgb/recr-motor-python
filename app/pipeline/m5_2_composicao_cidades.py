from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

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
# VERSÃO PERFORMANCE
# - sem busca combinatória
# - sem materialização de DataFrame em loop interno
# - composição gulosa por blocos de cliente
# - múltiplos fechamentos por cidade
#
# REGRAS DE PESO
# - base oficial = peso_calculado
# - peso_kg = auditoria
# - vol_m3 = volume
# - M5.2 não recria peso_calculado
# =========================================================================================


MAX_SEEDS_POR_VEICULO = 4
MAX_BLOCOS_PRETRIAGEM = 80
ALVO_OCUPACAO_FRAC = 0.92


# -----------------------------------------------------------------------------------------
# Helpers gerais
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


def _build_manifesto_id(seq: int) -> str:
    return f"PM52_{seq:04d}"


def _tentativa_dict(
    cidade: str,
    uf: str,
    vehicle_row: Optional[pd.Series],
    resultado: str,
    motivo: str,
    tentativa_idx: int,
    blocos_considerados: int,
    estrategia: str,
    peso_total_candidato: float = 0.0,
    peso_kg_total_candidato: float = 0.0,
    volume_total_candidato: float = 0.0,
    km_referencia_candidato: float = 0.0,
    ocupacao_perc_candidato: float = 0.0,
    qtd_itens_candidato: int = 0,
    qtd_paradas_candidato: int = 0,
) -> Dict[str, Any]:
    return {
        "cidade": cidade,
        "uf": uf,
        "tentativa_idx": tentativa_idx,
        "blocos_considerados": blocos_considerados,
        "estrategia_tentativa": estrategia,
        "veiculo_tipo_tentado": None if vehicle_row is None else safe_text(vehicle_row.get("tipo")),
        "veiculo_perfil_tentado": None if vehicle_row is None else safe_text(vehicle_row.get("perfil")),
        "resultado": resultado,
        "motivo": motivo,
        "qtd_itens_candidato": int(qtd_itens_candidato),
        "qtd_paradas_candidato": int(qtd_paradas_candidato),
        "peso_total_candidato": round(safe_float(peso_total_candidato, 0.0), 3),
        "peso_kg_total_candidato": round(safe_float(peso_kg_total_candidato, 0.0), 3),
        "volume_total_candidato": round(safe_float(volume_total_candidato, 0.0), 3),
        "km_referencia_candidato": round(safe_float(km_referencia_candidato, 0.0), 2),
        "ocupacao_perc_candidato": round(safe_float(ocupacao_perc_candidato, 0.0), 2),
    }


# -----------------------------------------------------------------------------------------
# Veículos elegíveis
# -----------------------------------------------------------------------------------------
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

    base = base.sort_values(
        by=["capacidade_peso_kg", "capacidade_vol_m3", "tipo", "perfil"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return base


# -----------------------------------------------------------------------------------------
# Estruturas leves por bloco
# -----------------------------------------------------------------------------------------
def _agrupar_blocos_cliente_na_cidade(city_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if city_df.empty:
        return pd.DataFrame()

    temp = city_df.copy()

    cliente_key_col = f"_cliente_key_{suffix}"
    bucket_col = f"_bucket_{suffix}"
    ranking_col = f"_ranking_ord_{suffix}"
    folga_col = f"_folga_ord_{suffix}"
    km_ord_col = f"_km_ord_{suffix}"
    peso_ord_col = f"_peso_ord_{suffix}"

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
            folga_min=(folga_col, "min"),
            km_ord_min=(km_ord_col, "min"),
            peso_ord_min=(peso_ord_col, "min"),
        )
        .reset_index()
        .sort_values(
            by=[
                "prioridade_min",
                "ranking_min",
                "folga_min",
                "km_referencia_bloco",
                "peso_total_bloco",
                cliente_key_col,
            ],
            ascending=[True, True, True, True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    grouped["ordem_bloco"] = range(1, len(grouped) + 1)
    return grouped


def _build_block_structures(
    city_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    suffix: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """
    Retorna:
    - blocks_map: métricas leves por bloco
    - block_to_ids: ids de linhas da cidade por bloco
    """
    cliente_key_col = f"_cliente_key_{suffix}"
    id_col = f"_id_str_{suffix}"

    block_to_ids = (
        city_df.groupby(cliente_key_col, dropna=False)[id_col]
        .apply(lambda s: [safe_text(x) for x in s.tolist()])
        .to_dict()
    )

    blocks_map: Dict[str, Dict[str, Any]] = {}
    for _, row in blocks_df.iterrows():
        key = safe_text(row.get(cliente_key_col))
        if not key:
            continue

        blocks_map[key] = {
            "key": key,
            "destinatario": safe_text(row.get("destinatario")),
            "peso": safe_float(row.get("peso_total_bloco"), 0.0),
            "peso_kg": safe_float(row.get("peso_kg_total_bloco"), 0.0),
            "volume": safe_float(row.get("volume_total_bloco"), 0.0),
            "km": safe_float(row.get("km_referencia_bloco"), 0.0),
            "paradas": 1,
            "qtd_linhas": safe_int(row.get("qtd_linhas_bloco"), 0),
            "prioridade": safe_float(row.get("prioridade_min"), 999999.0),
            "ranking": safe_float(row.get("ranking_min"), 999999.0),
            "folga": safe_float(row.get("folga_min"), 999999.0),
            "ordem_bloco": safe_int(row.get("ordem_bloco"), 999999),
            "ids": block_to_ids.get(key, []),
        }

    return blocks_map, block_to_ids


# -----------------------------------------------------------------------------------------
# Estado incremental do grupo
# -----------------------------------------------------------------------------------------
def _empty_group_state() -> Dict[str, Any]:
    return {
        "keys": set(),
        "peso": 0.0,
        "peso_kg": 0.0,
        "volume": 0.0,
        "km": 0.0,
        "paradas": 0,
        "qtd_linhas": 0,
    }


def _clone_group_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "keys": set(state["keys"]),
        "peso": float(state["peso"]),
        "peso_kg": float(state["peso_kg"]),
        "volume": float(state["volume"]),
        "km": float(state["km"]),
        "paradas": int(state["paradas"]),
        "qtd_linhas": int(state["qtd_linhas"]),
    }


def _group_add_block(state: Dict[str, Any], block: Dict[str, Any]) -> Dict[str, Any]:
    new_state = _clone_group_state(state)
    if block["key"] in new_state["keys"]:
        return new_state

    new_state["keys"].add(block["key"])
    new_state["peso"] += block["peso"]
    new_state["peso_kg"] += block["peso_kg"]
    new_state["volume"] += block["volume"]
    new_state["km"] = max(new_state["km"], block["km"])
    new_state["paradas"] += block["paradas"]
    new_state["qtd_linhas"] += block["qtd_linhas"]
    return new_state


def _group_ocupacao_perc(state: Dict[str, Any], vehicle_row: pd.Series) -> float:
    cap_peso = safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    if cap_peso <= 0:
        return 0.0
    return (safe_float(state["peso"], 0.0) / cap_peso) * 100.0


def _group_validate_hard(state: Dict[str, Any], vehicle_row: pd.Series) -> Tuple[bool, str]:
    if not state["keys"]:
        return False, "grupo_vazio"

    cap_peso = safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    cap_vol = safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0)
    max_entregas = safe_int(vehicle_row.get("max_entregas"), 0)
    max_km = safe_float(vehicle_row.get("max_km_distancia"), 0.0)
    ocup_max = safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0)

    if cap_peso > 0 and state["peso"] > cap_peso:
        return False, "excede_capacidade_peso"
    if cap_vol > 0 and state["volume"] > cap_vol:
        return False, "excede_capacidade_volume"
    if max_entregas > 0 and state["paradas"] > max_entregas:
        return False, "excede_max_entregas"
    if max_km > 0 and state["km"] > max_km:
        return False, "excede_max_km"

    ocup = _group_ocupacao_perc(state, vehicle_row)
    if ocup > ocup_max:
        return False, "excede_ocupacao_maxima"

    return True, "ok"


def _group_validate_close(state: Dict[str, Any], vehicle_row: pd.Series) -> Tuple[bool, str]:
    ok_hard, motivo = _group_validate_hard(state, vehicle_row)
    if not ok_hard:
        return False, motivo

    ocup_min = safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0)
    ocup = _group_ocupacao_perc(state, vehicle_row)

    if ocup < ocup_min:
        return False, "abaixo_ocupacao_minima"

    return True, "ok"


def _group_score(state: Dict[str, Any], vehicle_row: pd.Series) -> Tuple[float, float, int, float]:
    ocup = _group_ocupacao_perc(state, vehicle_row)
    peso = safe_float(state["peso"], 0.0)
    clientes = safe_int(state["paradas"], 0)
    cap = safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)

    return (
        round(ocup, 6),
        round(peso, 6),
        int(clientes),
        -cap,
    )


# -----------------------------------------------------------------------------------------
# Materialização final
# -----------------------------------------------------------------------------------------
def _materializar_candidato_por_keys(
    city_df: pd.DataFrame,
    selected_keys: Set[str],
    suffix: str,
) -> pd.DataFrame:
    if city_df.empty or not selected_keys:
        return pd.DataFrame(columns=city_df.columns)

    cliente_key_col = f"_cliente_key_{suffix}"
    candidato = city_df[city_df[cliente_key_col].isin(selected_keys)].copy()
    candidato = ordenar_operacional_m5(candidato, suffix=suffix)
    return candidato.reset_index(drop=True)


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


# -----------------------------------------------------------------------------------------
# Compatibilidade de restrição de veículo por bloco
# -----------------------------------------------------------------------------------------
def _validar_bloco_com_dataframe_unico(
    city_df: pd.DataFrame,
    key: str,
    vehicle_row: pd.Series,
    suffix: str,
) -> bool:
    """
    Valida compatibilidade de restrição de veículo uma única vez por bloco.
    """
    bloco_df = _materializar_candidato_por_keys(city_df, {key}, suffix=suffix)
    if bloco_df.empty:
        return False
    return bool(grupo_respeita_restricao_veiculo(bloco_df, vehicle_row))


def _filtrar_blocos_elegiveis_para_veiculo(
    city_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    blocks_map: Dict[str, Dict[str, Any]],
    vehicle_row: pd.Series,
    suffix: str,
) -> List[Dict[str, Any]]:
    if city_df.empty or blocks_df.empty:
        return []

    cap_peso = safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    cap_vol = safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0)
    max_entregas = safe_int(vehicle_row.get("max_entregas"), 0)
    max_km = safe_float(vehicle_row.get("max_km_distancia"), 0.0)
    ocup_max = safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0)
    ocup_min = safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0)

    elegiveis: List[Dict[str, Any]] = []
    cliente_key_col = f"_cliente_key_{suffix}"

    for _, row in blocks_df.iterrows():
        key = safe_text(row.get(cliente_key_col))
        if not key or key not in blocks_map:
            continue

        bloco = blocks_map[key]
        peso_bloco = bloco["peso"]
        vol_bloco = bloco["volume"]
        km_bloco = bloco["km"]

        if cap_peso > 0 and peso_bloco > cap_peso:
            continue
        if cap_vol > 0 and vol_bloco > cap_vol:
            continue
        if max_entregas > 0 and 1 > max_entregas:
            continue
        if max_km > 0 and km_bloco > max_km:
            continue

        ocup_bloco = (peso_bloco / cap_peso) * 100.0 if cap_peso > 0 else 0.0
        if ocup_bloco > ocup_max:
            continue

        if not _validar_bloco_com_dataframe_unico(city_df, key, vehicle_row, suffix=suffix):
            continue

        gap_ocup_min = abs(ocup_min - ocup_bloco)

        bloco_out = dict(bloco)
        bloco_out["fit_ocupacao_bloco"] = ocup_bloco
        bloco_out["gap_ocup_min"] = gap_ocup_min
        elegiveis.append(bloco_out)

    if not elegiveis:
        return []

    elegiveis.sort(
        key=lambda b: (
            b["prioridade"],
            b["ranking"],
            b["gap_ocup_min"],
            b["km"],
            -b["peso"],
            b["ordem_bloco"],
        )
    )

    return elegiveis[:MAX_BLOCOS_PRETRIAGEM]


# -----------------------------------------------------------------------------------------
# Sementes e composição gulosa
# -----------------------------------------------------------------------------------------
def _gerar_sementes_blocos(
    blocks_vehicle: List[Dict[str, Any]],
    vehicle_row: pd.Series,
) -> List[List[str]]:
    if not blocks_vehicle:
        return []

    seeds: List[List[str]] = []

    top_peso = sorted(
        blocks_vehicle,
        key=lambda b: (-b["peso"], b["prioridade"], b["ranking"])
    )[:2]

    top_prioridade = sorted(
        blocks_vehicle,
        key=lambda b: (b["prioridade"], b["ranking"], -b["peso"])
    )[:2]

    cap_peso = safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    ocup_min = safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0)

    top_fit = sorted(
        blocks_vehicle,
        key=lambda b: (
            abs((((b["peso"] / cap_peso) * 100.0) if cap_peso > 0 else 0.0) - ocup_min),
            b["prioridade"],
            b["ranking"],
            -b["peso"],
        )
    )[:2]

    for grupo in [top_peso, top_prioridade, top_fit]:
        for bloco in grupo:
            seed = [bloco["key"]]
            if seed not in seeds:
                seeds.append(seed)

    if not seeds and blocks_vehicle:
        seeds.append([blocks_vehicle[0]["key"]])

    return seeds[:MAX_SEEDS_POR_VEICULO]


def _score_add_candidate(
    current_state: Dict[str, Any],
    block: Dict[str, Any],
    vehicle_row: pd.Series,
) -> Tuple[float, float, float, float]:
    novo_peso = current_state["peso"] + block["peso"]
    cap_peso = safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    ocup = (novo_peso / cap_peso) * 100.0 if cap_peso > 0 else 0.0
    return (
        round(ocup, 6),
        round(novo_peso, 6),
        -block["prioridade"],
        -block["km"],
    )


def _construir_guloso_por_blocos(
    blocks_vehicle: List[Dict[str, Any]],
    vehicle_row: pd.Series,
    seed_keys: List[str],
) -> Dict[str, Any]:
    blocks_by_key = {b["key"]: b for b in blocks_vehicle}

    state = _empty_group_state()

    for key in seed_keys:
        bloco = blocks_by_key.get(key)
        if bloco is None:
            continue
        new_state = _group_add_block(state, bloco)
        ok_hard, _ = _group_validate_hard(new_state, vehicle_row)
        if ok_hard:
            state = new_state

    if not state["keys"]:
        return state

    alvo_ocup = safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0) * ALVO_OCUPACAO_FRAC

    restantes = [b for b in blocks_vehicle if b["key"] not in state["keys"]]

    while restantes:
        melhor_bloco: Optional[Dict[str, Any]] = None
        melhor_score: Optional[Tuple[float, float, float, float]] = None

        for bloco in restantes:
            candidato = _group_add_block(state, bloco)
            ok_hard, _ = _group_validate_hard(candidato, vehicle_row)
            if not ok_hard:
                continue

            score = _score_add_candidate(state, bloco, vehicle_row)
            if melhor_score is None or score > melhor_score:
                melhor_score = score
                melhor_bloco = bloco

        if melhor_bloco is None:
            break

        state = _group_add_block(state, melhor_bloco)
        if _group_ocupacao_perc(state, vehicle_row) >= alvo_ocup:
            break

        restantes = [b for b in restantes if b["key"] != melhor_bloco["key"]]

    return state


def _complementar_com_blocos_pequenos(
    state: Dict[str, Any],
    blocks_vehicle: List[Dict[str, Any]],
    vehicle_row: pd.Series,
) -> Dict[str, Any]:
    """
    Tenta completar a ocupação mínima com blocos menores.
    """
    if not state["keys"]:
        return state

    ocup_min = safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0)
    ocup_atual = _group_ocupacao_perc(state, vehicle_row)

    if ocup_atual >= ocup_min:
        return state

    restantes = [b for b in blocks_vehicle if b["key"] not in state["keys"]]
    restantes.sort(key=lambda b: (b["peso"], b["km"], b["prioridade"]))

    best_state = _clone_group_state(state)
    best_gap = max(0.0, ocup_min - ocup_atual)

    for bloco in restantes:
        candidato = _group_add_block(best_state, bloco)
        ok_hard, _ = _group_validate_hard(candidato, vehicle_row)
        if not ok_hard:
            continue

        ocup = _group_ocupacao_perc(candidato, vehicle_row)
        gap = max(0.0, ocup_min - ocup)

        if gap < best_gap:
            best_state = candidato
            best_gap = gap

        if ocup >= ocup_min:
            break

    return best_state


# -----------------------------------------------------------------------------------------
# Busca do melhor fechamento por cidade
# -----------------------------------------------------------------------------------------
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
                tentativa_idx=1,
                blocos_considerados=0,
                estrategia="sem_perfil",
            )
        )
        return None, None, "sem_perfil_elegivel_na_cidade"

    blocks_df = _agrupar_blocos_cliente_na_cidade(city_df, suffix=suffix)
    if blocks_df.empty:
        return None, None, "sem_blocos_na_cidade"

    blocks_map, _ = _build_block_structures(city_df, blocks_df, suffix=suffix)

    melhor_state: Optional[Dict[str, Any]] = None
    melhor_vehicle: Optional[pd.Series] = None
    melhor_score: Optional[Tuple[float, float, int, float]] = None
    melhor_motivo = "nenhum_fechamento"

    tentativa_idx = 1

    for _, vehicle_row in vehicles_city.iterrows():
        blocks_vehicle = _filtrar_blocos_elegiveis_para_veiculo(
            city_df=city_df,
            blocks_df=blocks_df,
            blocks_map=blocks_map,
            vehicle_row=vehicle_row,
            suffix=suffix,
        )

        if not blocks_vehicle:
            tentativas.append(
                _tentativa_dict(
                    cidade=cidade,
                    uf=uf,
                    vehicle_row=vehicle_row,
                    resultado="falhou",
                    motivo="sem_blocos_elegiveis_para_veiculo",
                    tentativa_idx=tentativa_idx,
                    blocos_considerados=0,
                    estrategia="pretriagem",
                )
            )
            tentativa_idx += 1
            melhor_motivo = "sem_blocos_elegiveis_para_veiculo"
            continue

        sementes = _gerar_sementes_blocos(blocks_vehicle, vehicle_row)
        if not sementes:
            tentativas.append(
                _tentativa_dict(
                    cidade=cidade,
                    uf=uf,
                    vehicle_row=vehicle_row,
                    resultado="falhou",
                    motivo="sem_sementes",
                    tentativa_idx=tentativa_idx,
                    blocos_considerados=0,
                    estrategia="semente",
                )
            )
            tentativa_idx += 1
            melhor_motivo = "sem_sementes"
            continue

        melhor_state_veiculo: Optional[Dict[str, Any]] = None
        melhor_score_veiculo: Optional[Tuple[float, float, int, float]] = None

        for seed_keys in sementes:
            candidato = _construir_guloso_por_blocos(
                blocks_vehicle=blocks_vehicle,
                vehicle_row=vehicle_row,
                seed_keys=seed_keys,
            )

            candidato = _complementar_com_blocos_pequenos(
                state=candidato,
                blocks_vehicle=blocks_vehicle,
                vehicle_row=vehicle_row,
            )

            ok, motivo = _group_validate_close(candidato, vehicle_row)

            tentativas.append(
                _tentativa_dict(
                    cidade=cidade,
                    uf=uf,
                    vehicle_row=vehicle_row,
                    resultado="fechado" if ok else "falhou",
                    motivo=motivo,
                    tentativa_idx=tentativa_idx,
                    blocos_considerados=len(candidato["keys"]),
                    estrategia="guloso_incremental",
                    peso_total_candidato=candidato["peso"],
                    peso_kg_total_candidato=candidato["peso_kg"],
                    volume_total_candidato=candidato["volume"],
                    km_referencia_candidato=candidato["km"],
                    ocupacao_perc_candidato=_group_ocupacao_perc(candidato, vehicle_row),
                    qtd_itens_candidato=candidato["qtd_linhas"],
                    qtd_paradas_candidato=candidato["paradas"],
                )
            )
            tentativa_idx += 1
            melhor_motivo = motivo

            if not ok:
                continue

            score = _group_score(candidato, vehicle_row)

            if melhor_score_veiculo is None or score > melhor_score_veiculo:
                melhor_score_veiculo = score
                melhor_state_veiculo = candidato

        if melhor_state_veiculo is None:
            continue

        if melhor_score is None or melhor_score_veiculo > melhor_score:
            melhor_score = melhor_score_veiculo
            melhor_state = melhor_state_veiculo
            melhor_vehicle = vehicle_row.copy()

    if melhor_state is None or melhor_vehicle is None:
        return None, None, melhor_motivo

    melhor_df = _materializar_candidato_por_keys(
        city_df=city_df,
        selected_keys=melhor_state["keys"],
        suffix=suffix,
    )

    if melhor_df.empty:
        return None, None, "materializacao_vazia"

    ok_restr, motivo_restr = _validar_fechamento_final_df(melhor_df, melhor_vehicle)
    if not ok_restr:
        return None, None, motivo_restr

    return melhor_df, melhor_vehicle, "ok"


def _validar_fechamento_final_df(df_itens: pd.DataFrame, vehicle_row: pd.Series) -> Tuple[bool, str]:
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
    ocup_min = safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0)

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
    if ocup < ocup_min:
        return False, "abaixo_ocupacao_minima"

    return True, "ok"


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
                    "guloso_incremental_por_blocos",
                    "sem_combinatoria_exaustiva",
                    "sem_dataframe_em_loop_interno",
                    "VERSAO_M5_2_FAST_2026_04_14",
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
        while True:
            if saldo.empty:
                break

            city_df = saldo[
                (saldo["cidade"].fillna("").astype(str).str.strip() == cidade_key)
                & (saldo["uf"].fillna("").astype(str).str.strip() == uf_key)
            ].copy()

            if city_df.empty:
                break

            cidades_processadas += 1

            candidato, vehicle_row, motivo = _buscar_melhor_fechamento_na_cidade(
                city_df=city_df,
                perfis_elegiveis_df=perfis_elegiveis,
                cidade=cidade_key,
                uf=uf_key,
                tentativas=tentativas,
                suffix=suffix,
            )

            if candidato is None or vehicle_row is None or candidato.empty:
                tentativas.append(
                    {
                        "cidade": cidade_key,
                        "uf": uf_key,
                        "tentativa_idx": None,
                        "blocos_considerados": 0,
                        "estrategia_tentativa": "encerramento_cidade",
                        "veiculo_tipo_tentado": None,
                        "veiculo_perfil_tentado": None,
                        "resultado": "sem_fechamento",
                        "motivo": motivo,
                        "qtd_itens_candidato": 0,
                        "qtd_paradas_candidato": 0,
                        "peso_total_candidato": 0.0,
                        "peso_kg_total_candidato": 0.0,
                        "volume_total_candidato": 0.0,
                        "km_referencia_candidato": 0.0,
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
            "guloso_incremental_por_blocos",
            "sem_combinatoria_exaustiva",
            "sem_dataframe_em_loop_interno",
            "VERSAO_M5_2_FAST_2026_04_14",
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


# -----------------------------------------------------------------------------------------
# Aliases defensivos
# -----------------------------------------------------------------------------------------
def executar_m5_composicao_cidades(*args: Any, **kwargs: Any):
    return executar_m5_2_composicao_cidades(*args, **kwargs)


def processar_m5_2_composicao_cidades(*args: Any, **kwargs: Any):
    return executar_m5_2_composicao_cidades(*args, **kwargs)


def rodar_m5_2_composicao_cidades(*args: Any, **kwargs: Any):
    return executar_m5_2_composicao_cidades(*args, **kwargs)
