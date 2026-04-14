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
# ESTRATÉGIA NOVA
# - processa cidade por cidade
# - trabalha por blocos de cliente dentro da cidade
# - substitui busca combinatória por:
#     1) construção gulosa
#     2) melhoria local leve (add/remove/swap)
# - gera múltiplos pré-manifestos na mesma cidade
# - mantém hard constraints de veículo
#
# REGRAS DE PESO
# - base oficial = peso_calculado
# - peso_kg = auditoria
# - vol_m3 = volume
# - M5.2 não recria peso_calculado
# =========================================================================================


MAX_BLOCOS_BASE_GULOSO = 60
MAX_BLOCOS_COMPLEMENTARES = 120
MAX_ITERACOES_LOCAL_SEARCH = 3
MAX_CANDIDATOS_POR_VEICULO = 5


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
    estrategia: str,
) -> Dict[str, Any]:
    candidato = df_candidato if df_candidato is not None else pd.DataFrame()

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

    base = base.sort_values(
        by=["capacidade_peso_kg", "capacidade_vol_m3", "tipo", "perfil"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return base


def _avaliar_bloco_vs_veiculo(block_row: pd.Series, vehicle_row: pd.Series) -> Tuple[bool, str]:
    cap_peso = safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    cap_vol = safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0)
    max_entregas = safe_int(vehicle_row.get("max_entregas"), 0)
    max_km = safe_float(vehicle_row.get("max_km_distancia"), 0.0)
    ocup_max = safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0)

    peso_bloco = safe_float(block_row.get("peso_total_bloco"), 0.0)
    vol_bloco = safe_float(block_row.get("volume_total_bloco"), 0.0)
    km_bloco = safe_float(block_row.get("km_referencia_bloco"), 0.0)
    qtd_paradas_bloco = 1

    if cap_peso > 0 and peso_bloco > cap_peso:
        return False, "bloco_excede_peso"
    if cap_vol > 0 and vol_bloco > cap_vol:
        return False, "bloco_excede_volume"
    if max_entregas > 0 and qtd_paradas_bloco > max_entregas:
        return False, "bloco_excede_paradas"
    if max_km > 0 and km_bloco > max_km:
        return False, "bloco_excede_km"

    if cap_peso > 0:
        ocup_bloco = (peso_bloco / cap_peso) * 100.0
        if ocup_bloco > ocup_max:
            return False, "bloco_excede_ocupacao_maxima"

    return True, "ok"


def _filtrar_blocos_elegiveis_para_veiculo(
    city_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    vehicle_row: pd.Series,
    suffix: str,
) -> pd.DataFrame:
    if city_df.empty or blocks_df.empty:
        return pd.DataFrame()

    cliente_key_col = f"_cliente_key_{suffix}"
    elegiveis: List[Dict[str, Any]] = []

    for _, block_row in blocks_df.iterrows():
        ok_bloco, motivo = _avaliar_bloco_vs_veiculo(block_row, vehicle_row)
        if not ok_bloco:
            continue

        key = safe_text(block_row.get(cliente_key_col))
        bloco_df = _materializar_candidato_por_keys(city_df, {key}, suffix=suffix)

        ok_hard, _ = _validar_hard_constraints(bloco_df, vehicle_row)
        if not ok_hard:
            continue

        cap_peso = safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
        peso_bloco = safe_float(block_row.get("peso_total_bloco"), 0.0)
        ocup_est = (peso_bloco / cap_peso) * 100.0 if cap_peso > 0 else 0.0

        row = block_row.to_dict()
        row["fit_ocupacao_bloco"] = ocup_est
        row["motivo_pretriagem"] = motivo
        elegiveis.append(row)

    if not elegiveis:
        return pd.DataFrame()

    base = pd.DataFrame(elegiveis)

    base["gap_ocupacao_min"] = (
        safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0) - pd.to_numeric(base["fit_ocupacao_bloco"], errors="coerce").fillna(0.0)
    ).abs()

    base = base.sort_values(
        by=[
            "prioridade_min",
            "ranking_min",
            "gap_ocupacao_min",
            "km_referencia_bloco",
            "peso_total_bloco",
            "ordem_bloco",
        ],
        ascending=[True, True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    limite = max(MAX_BLOCOS_BASE_GULOSO, safe_int(vehicle_row.get("max_entregas"), 0) * 10)
    limite = max(20, min(limite, MAX_BLOCOS_COMPLEMENTARES))
    return base.head(limite).copy().reset_index(drop=True)


def _score_adicao_bloco(
    candidato_df: pd.DataFrame,
    vehicle_row: pd.Series,
    block_row: pd.Series,
) -> Tuple[float, float, float, float]:
    ocup = ocupacao_perc(candidato_df, vehicle_row)
    peso = peso_total(candidato_df)
    prioridade = -safe_float(block_row.get("prioridade_min"), 999999.0)
    km = -safe_float(block_row.get("km_referencia_bloco"), 999999.0)
    return (round(ocup, 6), round(peso, 6), prioridade, km)


def _gerar_sementes_blocos(
    blocks_df: pd.DataFrame,
    vehicle_row: pd.Series,
    suffix: str,
) -> List[List[str]]:
    if blocks_df.empty:
        return []

    cliente_key_col = f"_cliente_key_{suffix}"
    seeds: List[List[str]] = []

    top_peso = (
        blocks_df.sort_values(
            by=["peso_total_bloco", "prioridade_min", "ranking_min"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .head(2)
    )

    top_prioridade = (
        blocks_df.sort_values(
            by=["prioridade_min", "ranking_min", "peso_total_bloco"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        .head(2)
    )

    cap_peso = safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    ocup_min = safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0)

    by_fit = blocks_df.copy()
    if cap_peso > 0:
        by_fit["fit_gap"] = (
            ((pd.to_numeric(by_fit["peso_total_bloco"], errors="coerce").fillna(0.0) / cap_peso) * 100.0) - ocup_min
        ).abs()
    else:
        by_fit["fit_gap"] = 999999.0

    top_fit = (
        by_fit.sort_values(
            by=["fit_gap", "prioridade_min", "ranking_min", "peso_total_bloco"],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
        .head(2)
    )

    for df_seed in [top_peso, top_prioridade, top_fit]:
        for _, row in df_seed.iterrows():
            key = safe_text(row.get(cliente_key_col))
            if not key:
                continue
            seed = [key]
            if seed not in seeds:
                seeds.append(seed)

    if not seeds:
        first_key = safe_text(blocks_df.iloc[0][cliente_key_col])
        if first_key:
            seeds.append([first_key])

    return seeds[:MAX_CANDIDATOS_POR_VEICULO]


def _construir_guloso_por_blocos(
    city_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    vehicle_row: pd.Series,
    suffix: str,
    seed_keys: List[str],
) -> pd.DataFrame:
    if city_df.empty or blocks_df.empty:
        return pd.DataFrame()

    cliente_key_col = f"_cliente_key_{suffix}"
    blocks_map = {
        safe_text(r[cliente_key_col]): r
        for _, r in blocks_df.iterrows()
    }

    selecionados: Set[str] = set()

    for key in seed_keys:
        if key not in blocks_map:
            continue
        candidato = _materializar_candidato_por_keys(city_df, selecionados | {key}, suffix=suffix)
        ok_hard, _ = _validar_hard_constraints(candidato, vehicle_row)
        if ok_hard:
            selecionados.add(key)

    if not selecionados:
        return pd.DataFrame(columns=city_df.columns)

    restantes = [k for k in blocks_map.keys() if k not in selecionados]

    while restantes:
        melhor_key: Optional[str] = None
        melhor_score: Optional[Tuple[float, float, float, float]] = None

        for key in restantes:
            candidato = _materializar_candidato_por_keys(city_df, selecionados | {key}, suffix=suffix)
            ok_hard, _ = _validar_hard_constraints(candidato, vehicle_row)
            if not ok_hard:
                continue

            score_add = _score_adicao_bloco(candidato, vehicle_row, blocks_map[key])

            if melhor_score is None or score_add > melhor_score:
                melhor_score = score_add
                melhor_key = key

        if melhor_key is None:
            break

        selecionados.add(melhor_key)
        restantes = [k for k in restantes if k != melhor_key]

        candidato_atual = _materializar_candidato_por_keys(city_df, selecionados, suffix=suffix)
        ocup_atual = ocupacao_perc(candidato_atual, vehicle_row)
        ocup_max = safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0)

        if ocup_atual >= ocup_max * 0.97:
            break

    return _materializar_candidato_por_keys(city_df, selecionados, suffix=suffix)


def _melhorar_por_adicao(
    city_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    vehicle_row: pd.Series,
    suffix: str,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    if current_df.empty:
        return current_df

    cliente_key_col = f"_cliente_key_{suffix}"
    selecionados = set(current_df[cliente_key_col].astype(str).unique().tolist())
    blocks_map = {
        safe_text(r[cliente_key_col]): r
        for _, r in blocks_df.iterrows()
    }

    melhor_df = current_df.copy()
    melhor_score = _score_candidato(melhor_df, vehicle_row)

    for key, block_row in blocks_map.items():
        if key in selecionados:
            continue
        candidato = _materializar_candidato_por_keys(city_df, selecionados | {key}, suffix=suffix)
        ok_hard, _ = _validar_hard_constraints(candidato, vehicle_row)
        if not ok_hard:
            continue
        score = _score_candidato(candidato, vehicle_row)
        if score > melhor_score:
            melhor_df = candidato
            melhor_score = score

    return melhor_df


def _melhorar_por_remocao(
    city_df: pd.DataFrame,
    vehicle_row: pd.Series,
    suffix: str,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    if current_df.empty:
        return current_df

    cliente_key_col = f"_cliente_key_{suffix}"
    selecionados = current_df[cliente_key_col].astype(str).unique().tolist()
    if len(selecionados) <= 1:
        return current_df

    melhor_df = current_df.copy()
    melhor_score = _score_candidato(melhor_df, vehicle_row)

    for key in selecionados:
        novos = set(selecionados) - {key}
        candidato = _materializar_candidato_por_keys(city_df, novos, suffix=suffix)
        ok, _ = _validar_fechamento(candidato, vehicle_row)
        if not ok:
            continue
        score = _score_candidato(candidato, vehicle_row)
        if score > melhor_score:
            melhor_df = candidato
            melhor_score = score

    return melhor_df


def _melhorar_por_troca(
    city_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    vehicle_row: pd.Series,
    suffix: str,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    if current_df.empty:
        return current_df

    cliente_key_col = f"_cliente_key_{suffix}"
    selecionados = set(current_df[cliente_key_col].astype(str).unique().tolist())
    nao_selecionados = [
        safe_text(r[cliente_key_col])
        for _, r in blocks_df.iterrows()
        if safe_text(r[cliente_key_col]) not in selecionados
    ]

    melhor_df = current_df.copy()
    melhor_score = _score_candidato(melhor_df, vehicle_row)

    removiveis = list(selecionados)
    for key_out in removiveis:
        base = set(selecionados) - {key_out}
        for key_in in nao_selecionados[:40]:
            candidato = _materializar_candidato_por_keys(city_df, base | {key_in}, suffix=suffix)
            ok, _ = _validar_fechamento(candidato, vehicle_row)
            if not ok:
                continue
            score = _score_candidato(candidato, vehicle_row)
            if score > melhor_score:
                melhor_df = candidato
                melhor_score = score

    return melhor_df


def _melhoria_local(
    city_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    vehicle_row: pd.Series,
    suffix: str,
    candidato_inicial: pd.DataFrame,
) -> pd.DataFrame:
    if candidato_inicial.empty:
        return candidato_inicial

    melhor_df = candidato_inicial.copy()

    for _ in range(MAX_ITERACOES_LOCAL_SEARCH):
        mudou = False

        candidato_add = _melhorar_por_adicao(city_df, blocks_df, vehicle_row, suffix, melhor_df)
        if not candidato_add.empty and _score_candidato(candidato_add, vehicle_row) > _score_candidato(melhor_df, vehicle_row):
            melhor_df = candidato_add
            mudou = True

        candidato_swap = _melhorar_por_troca(city_df, blocks_df, vehicle_row, suffix, melhor_df)
        if not candidato_swap.empty and _score_candidato(candidato_swap, vehicle_row) > _score_candidato(melhor_df, vehicle_row):
            melhor_df = candidato_swap
            mudou = True

        candidato_rem = _melhorar_por_remocao(city_df, vehicle_row, suffix, melhor_df)
        if not candidato_rem.empty and _score_candidato(candidato_rem, vehicle_row) > _score_candidato(melhor_df, vehicle_row):
            melhor_df = candidato_rem
            mudou = True

        if not mudou:
            break

    return melhor_df


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
                estrategia="sem_perfil",
            )
        )
        return None, None, "sem_perfil_elegivel_na_cidade"

    blocks_df = _agrupar_blocos_cliente_na_cidade(city_df, suffix=suffix)
    if blocks_df.empty:
        return None, None, "sem_blocos_na_cidade"

    melhor_df: Optional[pd.DataFrame] = None
    melhor_vehicle: Optional[pd.Series] = None
    melhor_score: Optional[Tuple[float, float, int, float]] = None
    melhor_motivo = "nenhum_fechamento"

    tentativa_idx = 1

    for _, vehicle_row in vehicles_city.iterrows():
        blocks_vehicle = _filtrar_blocos_elegiveis_para_veiculo(
            city_df=city_df,
            blocks_df=blocks_df,
            vehicle_row=vehicle_row,
            suffix=suffix,
        )

        if blocks_vehicle.empty:
            tentativas.append(
                _tentativa_dict(
                    cidade=cidade,
                    uf=uf,
                    vehicle_row=vehicle_row,
                    resultado="falhou",
                    motivo="sem_blocos_elegiveis_para_veiculo",
                    df_candidato=pd.DataFrame(),
                    tentativa_idx=tentativa_idx,
                    blocos_considerados=0,
                    estrategia="pretriagem",
                )
            )
            tentativa_idx += 1
            melhor_motivo = "sem_blocos_elegiveis_para_veiculo"
            continue

        sementes = _gerar_sementes_blocos(blocks_vehicle, vehicle_row, suffix=suffix)
        if not sementes:
            tentativas.append(
                _tentativa_dict(
                    cidade=cidade,
                    uf=uf,
                    vehicle_row=vehicle_row,
                    resultado="falhou",
                    motivo="sem_sementes",
                    df_candidato=pd.DataFrame(),
                    tentativa_idx=tentativa_idx,
                    blocos_considerados=0,
                    estrategia="semente",
                )
            )
            tentativa_idx += 1
            melhor_motivo = "sem_sementes"
            continue

        melhor_por_veiculo_df: Optional[pd.DataFrame] = None
        melhor_por_veiculo_score: Optional[Tuple[float, float, int, float]] = None

        for seed_keys in sementes:
            candidato_guloso = _construir_guloso_por_blocos(
                city_df=city_df,
                blocks_df=blocks_vehicle,
                vehicle_row=vehicle_row,
                suffix=suffix,
                seed_keys=seed_keys,
            )

            if candidato_guloso.empty:
                tentativas.append(
                    _tentativa_dict(
                        cidade=cidade,
                        uf=uf,
                        vehicle_row=vehicle_row,
                        resultado="falhou",
                        motivo="guloso_vazio",
                        df_candidato=candidato_guloso,
                        tentativa_idx=tentativa_idx,
                        blocos_considerados=0,
                        estrategia="guloso",
                    )
                )
                tentativa_idx += 1
                melhor_motivo = "guloso_vazio"
                continue

            candidato_melhorado = _melhoria_local(
                city_df=city_df,
                blocks_df=blocks_vehicle,
                vehicle_row=vehicle_row,
                suffix=suffix,
                candidato_inicial=candidato_guloso,
            )

            ok, motivo = _validar_fechamento(candidato_melhorado, vehicle_row)

            tentativas.append(
                _tentativa_dict(
                    cidade=cidade,
                    uf=uf,
                    vehicle_row=vehicle_row,
                    resultado="fechado" if ok else "falhou",
                    motivo=motivo,
                    df_candidato=candidato_melhorado,
                    tentativa_idx=tentativa_idx,
                    blocos_considerados=int(
                        len(candidato_melhorado[f"_cliente_key_{suffix}"].astype(str).unique())
                    )
                    if not candidato_melhorado.empty and f"_cliente_key_{suffix}" in candidato_melhorado.columns
                    else 0,
                    estrategia="guloso_local_search",
                )
            )
            tentativa_idx += 1
            melhor_motivo = motivo

            if not ok:
                continue

            score = _score_candidato(candidato_melhorado, vehicle_row)

            if melhor_por_veiculo_score is None or score > melhor_por_veiculo_score:
                melhor_por_veiculo_score = score
                melhor_por_veiculo_df = candidato_melhorado.copy()

        if melhor_por_veiculo_df is None:
            continue

        if melhor_score is None or melhor_por_veiculo_score > melhor_score:
            melhor_score = melhor_por_veiculo_score
            melhor_df = melhor_por_veiculo_df.copy()
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
                    "guloso_por_blocos",
                    "melhoria_local_add_remove_swap",
                    "multiplos_fechamentos_na_mesma_cidade",
                    "VERSAO_M5_2_GREEDY_2026_04_14",
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
            "guloso_por_blocos",
            "melhoria_local_add_remove_swap",
            "multiplos_fechamentos_na_mesma_cidade",
            "VERSAO_M5_2_GREEDY_2026_04_14",
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
