from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# MÓDULO 4 - GERAÇÃO DE MANIFESTOS FECHADOS
# ADAPTAÇÃO FIEL DO NOTEBOOK PARA O SISTEMA 2
# ============================================================

OCUPACAO_DOMINANTE_MIN = 0.70
OCUPACAO_SECUNDARIA_MIN = 0.20

CHAVES_PARADA = ["destinatario", "cidade", "uf"]
CHAVE_CLIENTE = "destinatario"
MAX_ITERACOES_VARREDURA_4C = 10


def _to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or len(df) == 0:
        return []
    df2 = df.copy()
    for col in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[col]):
            df2[col] = df2[col].astype(str)
    df2 = df2.where(pd.notnull(df2), None)
    return df2.to_dict(orient="records")


def _resolver_coluna_tipo_veiculo(df_veiculos: pd.DataFrame) -> str:
    if "tipo" in df_veiculos.columns:
        return "tipo"
    if "perfil" in df_veiculos.columns:
        return "perfil"
    raise Exception("Faltam colunas mínimas na base de veículos:\n- tipo ou perfil")


def _txt_norm(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def _chave_parada_df(df_: pd.DataFrame) -> pd.Series:
    return (
        df_["destinatario"].astype(str).fillna("").str.strip().str.upper()
        + "|"
        + df_["cidade"].astype(str).fillna("").str.strip().str.upper()
        + "|"
        + df_["uf"].astype(str).fillna("").str.strip().str.upper()
    )


def _calcular_ocupacoes(
    peso_total: float,
    vol_total: float,
    cap_peso: float,
    cap_vol: float,
) -> Tuple[float, float, float, float]:
    ocup_peso = peso_total / cap_peso if pd.notna(cap_peso) and cap_peso > 0 else np.nan
    ocup_vol = vol_total / cap_vol if pd.notna(cap_vol) and cap_vol > 0 else np.nan
    dominante = np.nanmax([ocup_peso, ocup_vol])
    secundaria = np.nanmin([ocup_peso, ocup_vol])
    return ocup_peso, ocup_vol, dominante, secundaria


def _bucket_temporal(row: pd.Series) -> str:
    ag = bool(row.get("agendada", False))
    dt_ag = row.get("data_agenda", pd.NaT)
    if ag and pd.notna(dt_ag):
        return f"AGENDA_{pd.Timestamp(dt_ag).strftime('%Y-%m-%d')}"
    return "LEADTIME"


def _score_linha(row: pd.Series) -> Tuple[Any, ...]:
    ranking = row.get("ranking_prioridade", 999999)
    score = row.get("score_prioridade_preliminar", 0)
    peso = row.get("peso_kg", 0)
    vol = row.get("vol_m3", 0)
    return (
        -1 if bool(row.get("agendada", False)) else 0,
        ranking if pd.notna(ranking) else 999999,
        -(score if pd.notna(score) else 0),
        -(vol if pd.notna(vol) else 0),
        -(peso if pd.notna(peso) else 0),
    )


def _ordenar_df_prioridade(df_: pd.DataFrame) -> pd.DataFrame:
    if len(df_) == 0:
        return df_.copy()
    return (
        df_.assign(__ord__=df_.apply(_score_linha, axis=1))
        .sort_values("__ord__")
        .drop(columns="__ord__")
        .reset_index(drop=True)
    )


def _score_macro_cliente(df_cliente: pd.DataFrame) -> Tuple[Any, ...]:
    ranking_min = pd.to_numeric(
        df_cliente.get("ranking_prioridade", pd.Series([999999] * len(df_cliente))),
        errors="coerce",
    ).min()
    score_max = pd.to_numeric(
        df_cliente.get("score_prioridade_preliminar", pd.Series([0] * len(df_cliente))),
        errors="coerce",
    ).max()
    tem_agendada = bool((df_cliente["agendada"] == True).any()) if "agendada" in df_cliente.columns else False
    peso_total = pd.to_numeric(df_cliente["peso_kg"], errors="coerce").sum()
    vol_total = pd.to_numeric(df_cliente["vol_m3"], errors="coerce").sum()

    return (
        1 if tem_agendada else 0,
        -(ranking_min if pd.notna(ranking_min) else 999999),
        score_max if pd.notna(score_max) else 0,
        peso_total,
        vol_total,
    )


def _score_grupo_parada(df_grupo: pd.DataFrame) -> Tuple[Any, ...]:
    ranking_min = pd.to_numeric(
        df_grupo.get("ranking_prioridade", pd.Series([999999] * len(df_grupo))),
        errors="coerce",
    ).min()
    score_max = pd.to_numeric(
        df_grupo.get("score_prioridade_preliminar", pd.Series([0] * len(df_grupo))),
        errors="coerce",
    ).max()
    tem_agendada = bool((df_grupo["agendada"] == True).any()) if "agendada" in df_grupo.columns else False
    km_max = pd.to_numeric(df_grupo["distancia_rodoviaria_est_km"], errors="coerce").max()
    peso_total = pd.to_numeric(df_grupo["peso_kg"], errors="coerce").sum()
    vol_total = pd.to_numeric(df_grupo["vol_m3"], errors="coerce").sum()

    return (
        1 if tem_agendada else 0,
        -(ranking_min if pd.notna(ranking_min) else 999999),
        score_max if pd.notna(score_max) else 0,
        peso_total,
        vol_total,
        -(km_max if pd.notna(km_max) else 0),
    )


def _preparar_catalogo_veiculos(df_veic: pd.DataFrame, coluna_tipo_veiculo: str) -> pd.DataFrame:
    cat = df_veic.copy()

    cat = cat.loc[
        cat[coluna_tipo_veiculo].notna()
        & cat["capacidade_peso_kg"].notna()
        & cat["capacidade_vol_m3"].notna()
        & cat["max_entregas"].notna()
        & cat["max_km_distancia"].notna()
    ].copy()

    cat["tipo"] = cat[coluna_tipo_veiculo].astype(str).str.strip()

    cat = (
        cat.groupby("tipo", as_index=False)
        .agg(
            {
                "capacidade_peso_kg": "max",
                "capacidade_vol_m3": "max",
                "max_entregas": "max",
                "max_km_distancia": "max",
            }
        )
        .sort_values(
            by=["capacidade_vol_m3", "capacidade_peso_kg", "max_entregas", "max_km_distancia"],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )

    cat["ordem_porte"] = np.arange(1, len(cat) + 1)
    return cat


def _avaliar_combo_no_veiculo(df_combo: pd.DataFrame, veic: pd.Series) -> Dict[str, Any]:
    peso_total = float(pd.to_numeric(df_combo["peso_kg"], errors="coerce").sum())
    vol_total = float(pd.to_numeric(df_combo["vol_m3"], errors="coerce").sum())
    km_combo = float(pd.to_numeric(df_combo["distancia_rodoviaria_est_km"], errors="coerce").max())
    qtd_ctes = int(df_combo["cte"].astype(str).nunique())
    qtd_itens = int(len(df_combo))
    qtd_paradas = int(_chave_parada_df(df_combo).nunique())

    cap_peso = float(veic["capacidade_peso_kg"])
    cap_vol = float(veic["capacidade_vol_m3"])
    max_entregas = int(veic["max_entregas"])
    max_km = float(veic["max_km_distancia"])

    cabe_peso = peso_total <= cap_peso
    cabe_vol = vol_total <= cap_vol
    cabe_paradas = qtd_paradas <= max_entregas
    cabe_km = km_combo <= max_km

    ocup_peso, ocup_vol, ocup_dom, ocup_sec = _calcular_ocupacoes(
        peso_total, vol_total, cap_peso, cap_vol
    )

    passa_ocupacao = (
        pd.notna(ocup_dom)
        and pd.notna(ocup_sec)
        and ocup_dom >= OCUPACAO_DOMINANTE_MIN
        and ocup_sec >= OCUPACAO_SECUNDARIA_MIN
    )

    return {
        "veiculo_tipo": veic["tipo"],
        "capacidade_peso_kg": cap_peso,
        "capacidade_vol_m3": cap_vol,
        "max_entregas": max_entregas,
        "max_km_distancia": max_km,
        "peso_total_kg": round(peso_total, 3),
        "vol_total_m3": round(vol_total, 3),
        "km_referencia": round(km_combo, 2),
        "qtd_itens": qtd_itens,
        "qtd_ctes": qtd_ctes,
        "qtd_paradas": qtd_paradas,
        "cabe_peso": cabe_peso,
        "cabe_vol": cabe_vol,
        "cabe_paradas": cabe_paradas,
        "cabe_km": cabe_km,
        "ocupacao_peso_perc": round(float(ocup_peso * 100), 2) if pd.notna(ocup_peso) else np.nan,
        "ocupacao_vol_perc": round(float(ocup_vol * 100), 2) if pd.notna(ocup_vol) else np.nan,
        "ocupacao_dominante_perc": round(float(ocup_dom * 100), 2) if pd.notna(ocup_dom) else np.nan,
        "ocupacao_secundaria_perc": round(float(ocup_sec * 100), 2) if pd.notna(ocup_sec) else np.nan,
        "passa_ocupacao": passa_ocupacao,
        "aceito": bool(cabe_peso and cabe_vol and cabe_paradas and cabe_km and passa_ocupacao),
    }


def _avaliar_combo(df_combo: pd.DataFrame, catalogo_veiculos: pd.DataFrame) -> Dict[str, Any]:
    tentativas = []
    for _, veic in catalogo_veiculos.sort_values("ordem_porte").iterrows():
        r = _avaliar_combo_no_veiculo(df_combo, veic)
        tentativas.append({**r, "resultado_teste": "aceito" if r["aceito"] else "rejeitado"})
        if r["aceito"]:
            return {**r, "tentativas": tentativas}
    return {
        "aceito": False,
        "motivo_reprovacao": "nao_cabe_ou_nao_atinge_ocupacao_minima",
        "tentativas": tentativas,
    }


def _construir_subconjunto_guloso(df_base: pd.DataFrame, veic: pd.Series) -> pd.DataFrame:
    if len(df_base) == 0:
        return pd.DataFrame(columns=df_base.columns)

    df_ord = _ordenar_df_prioridade(df_base.copy())
    selecionados = []

    for _, row in df_ord.iterrows():
        if len(selecionados) == 0:
            teste = pd.DataFrame([row])
        else:
            teste = pd.concat([pd.DataFrame(selecionados), pd.DataFrame([row])], ignore_index=True)

        aval = _avaliar_combo_no_veiculo(teste, veic)
        if aval["cabe_peso"] and aval["cabe_vol"] and aval["cabe_paradas"] and aval["cabe_km"]:
            selecionados.append(row)

    if len(selecionados) == 0:
        return pd.DataFrame(columns=df_base.columns)

    return pd.DataFrame(selecionados).reset_index(drop=True)


def _encontrar_melhor_subconjunto_fechavel(
    df_base: pd.DataFrame, catalogo_veiculos: pd.DataFrame
) -> Dict[str, Any] | None:
    melhor = None

    for _, veic in catalogo_veiculos.sort_values("ordem_porte").iterrows():
        subset = _construir_subconjunto_guloso(df_base, veic)
        if len(subset) == 0:
            continue

        aval = _avaliar_combo_no_veiculo(subset, veic)
        if not aval["aceito"]:
            continue

        registro = {
            "df_combo": subset.copy(),
            "avaliacao": aval,
            "ordem_porte": int(veic["ordem_porte"]),
        }

        if melhor is None:
            melhor = registro
        else:
            chave_nova = (
                len(registro["df_combo"]),
                registro["avaliacao"]["vol_total_m3"],
                registro["avaliacao"]["peso_total_kg"],
                -registro["ordem_porte"],
            )
            chave_atual = (
                len(melhor["df_combo"]),
                melhor["avaliacao"]["vol_total_m3"],
                melhor["avaliacao"]["peso_total_kg"],
                -melhor["ordem_porte"],
            )
            if chave_nova > chave_atual:
                melhor = registro

    return melhor


def _gerar_resumo_manifesto(
    df_combo: pd.DataFrame,
    avaliacao: Dict[str, Any],
    manifesto_id: str,
    tipo_manifesto: str,
    origem_etapa: str,
) -> Dict[str, Any]:
    linha = {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": tipo_manifesto,
        "veiculo_tipo": avaliacao["veiculo_tipo"],
        "qtd_itens": avaliacao["qtd_itens"],
        "qtd_ctes": avaliacao["qtd_ctes"],
        "qtd_paradas": avaliacao["qtd_paradas"],
        "peso_total_kg": avaliacao["peso_total_kg"],
        "vol_total_m3": avaliacao["vol_total_m3"],
        "km_referencia": avaliacao["km_referencia"],
        "ocupacao_peso_perc": avaliacao["ocupacao_peso_perc"],
        "ocupacao_vol_perc": avaliacao["ocupacao_vol_perc"],
        "ocupacao_dominante_perc": avaliacao["ocupacao_dominante_perc"],
        "ocupacao_secundaria_perc": avaliacao["ocupacao_secundaria_perc"],
        "capacidade_peso_kg_veiculo": avaliacao["capacidade_peso_kg"],
        "capacidade_vol_m3_veiculo": avaliacao["capacidade_vol_m3"],
        "max_entregas_veiculo": avaliacao["max_entregas"],
        "max_km_distancia_veiculo": avaliacao["max_km_distancia"],
        "ranking_prioridade_min": pd.to_numeric(
            df_combo.get("ranking_prioridade", pd.Series([np.nan] * len(df_combo))),
            errors="coerce",
        ).min(),
        "score_prioridade_max": pd.to_numeric(
            df_combo.get("score_prioridade_preliminar", pd.Series([np.nan] * len(df_combo))),
            errors="coerce",
        ).max(),
        "origem_modulo": 4,
        "origem_etapa": origem_etapa,
    }

    if avaliacao["qtd_paradas"] == 1:
        linha["destinatario"] = df_combo["destinatario"].iloc[0]
        linha["cidade"] = df_combo["cidade"].iloc[0]
        linha["uf"] = df_combo["uf"].iloc[0]
        linha["mesorregiao"] = df_combo["mesorregiao"].iloc[0] if "mesorregiao" in df_combo.columns else np.nan
        linha["microrregiao"] = df_combo["microrregiao"].iloc[0] if "microrregiao" in df_combo.columns else np.nan
    else:
        linha["destinatario"] = df_combo["destinatario"].iloc[0]
        linha["cidade"] = "MULTICIDADE"
        linha["uf"] = (
            df_combo["uf"].mode().iloc[0]
            if "uf" in df_combo.columns and df_combo["uf"].notna().any()
            else np.nan
        )
        linha["mesorregiao"] = "MULTI"
        linha["microrregiao"] = "MULTI"

    return linha


def executar_m4_manifestos_fechados(
    df_input_oficial_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
    rodada_id: str,
    data_base_roteirizacao: pd.Timestamp,
    caminhos_pipeline: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    # ------------------------------------------------------------
    # 1) CÓPIAS
    # ------------------------------------------------------------
    fila = df_input_oficial_bloco_4.copy().reset_index(drop=True)
    veiculos = df_veiculos_tratados.copy().reset_index(drop=True)
    caminhos_pipeline = caminhos_pipeline or {}

    # ------------------------------------------------------------
    # 2) VALIDAR ESTRUTURA
    # ------------------------------------------------------------
    coluna_tipo_veiculo = _resolver_coluna_tipo_veiculo(veiculos)

    colunas_minimas_fila = [
        "id_linha_pipeline",
        "destinatario",
        "cidade",
        "uf",
        "peso_kg",
        "vol_m3",
        "distancia_rodoviaria_est_km",
        "status_triagem",
        "grupo_saida",
    ]

    colunas_minimas_veiculos = [
        coluna_tipo_veiculo,
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
    ]

    faltam_fila = [c for c in colunas_minimas_fila if c not in fila.columns]
    faltam_veiculos = [c for c in colunas_minimas_veiculos if c not in veiculos.columns]

    if faltam_fila:
        raise Exception("Faltam colunas mínimas na fila oficial do Bloco 4:\n- " + "\n- ".join(faltam_fila))

    if faltam_veiculos:
        raise Exception("Faltam colunas mínimas na base de veículos:\n- " + "\n- ".join(faltam_veiculos))

    # ------------------------------------------------------------
    # 3) GARANTIA DURA DE INPUT CORRETO
    # ------------------------------------------------------------
    linhas_input_invalido = fila.loc[
        (fila["status_triagem"].astype(str) != "roteirizavel")
        | (fila["grupo_saida"].astype(str) != "df_carteira_roteirizavel")
    ].copy()

    if len(linhas_input_invalido) > 0:
        raise Exception(
            "O BLOCO 4 recebeu linhas incompatíveis com o estágio. "
            "Há registros com status_triagem != 'roteirizavel' ou grupo_saida inválido."
        )

    if fila["id_linha_pipeline"].astype(str).duplicated().any():
        qtd_dup = int(fila["id_linha_pipeline"].astype(str).duplicated().sum())
        raise Exception(f"O input oficial do Bloco 4 possui id_linha_pipeline duplicado: {qtd_dup}")

    # ------------------------------------------------------------
    # 4) PADRONIZAÇÕES
    # ------------------------------------------------------------
    for col in [
        "peso_kg",
        "vol_m3",
        "distancia_rodoviaria_est_km",
        "ranking_prioridade",
        "ranking_prioridade_operacional",
        "score_prioridade_preliminar",
        "folga_dias",
    ]:
        if col in fila.columns:
            fila[col] = pd.to_numeric(fila[col], errors="coerce")

    for col in ["capacidade_peso_kg", "capacidade_vol_m3", "max_entregas", "max_km_distancia"]:
        veiculos[col] = pd.to_numeric(veiculos[col], errors="coerce")

    for col in ["data_agenda", "data_leadtime", "data_limite_considerada"]:
        if col in fila.columns:
            fila[col] = pd.to_datetime(fila[col], errors="coerce")

    if "agendada" in fila.columns:
        fila["agendada"] = fila["agendada"].fillna(False).astype(bool)
    else:
        fila["agendada"] = False

    if "cte" not in fila.columns:
        fila["cte"] = fila["id_linha_pipeline"].astype(str)

    if "ranking_prioridade" not in fila.columns:
        if "ranking_prioridade_operacional" in fila.columns:
            fila["ranking_prioridade"] = fila["ranking_prioridade_operacional"]
        else:
            fila["ranking_prioridade"] = 999999

    if "score_prioridade_preliminar" not in fila.columns:
        fila["score_prioridade_preliminar"] = 0.0

    # ------------------------------------------------------------
    # 5) PASTA DE SAÍDA INTERNA
    # ------------------------------------------------------------
    pasta_saida_base_str = caminhos_pipeline.get("pasta_saida_base")
    if pasta_saida_base_str:
        pasta_saida_base = Path(pasta_saida_base_str)
    else:
        pasta_saida_base = Path("/tmp/rec_roteirizador") / str(rodada_id)

    pasta_modulo_4 = pasta_saida_base / "bloco_4_manifestos_fechados"
    pasta_modulo_4.mkdir(parents=True, exist_ok=True)

    arq_manifestos_xlsx = pasta_modulo_4 / "df_manifestos_fechados_bloco_4.xlsx"
    arq_itens_csv = pasta_modulo_4 / "df_itens_manifestos_fechados_bloco_4.csv"
    arq_tentativas_csv = pasta_modulo_4 / "df_tentativas_fechamento_bloco_4.csv"
    arq_remanescente_csv = pasta_modulo_4 / "df_remanescente_roteirizavel_bloco_4.csv"
    arq_resumo_xlsx = pasta_modulo_4 / "resumo_modulo_4.xlsx"
    arq_metadata_json = pasta_modulo_4 / "metadata_modulo_4.json"

    # ------------------------------------------------------------
    # 6) CATÁLOGO DE VEÍCULOS
    # ------------------------------------------------------------
    catalogo_veiculos = _preparar_catalogo_veiculos(veiculos, coluna_tipo_veiculo)

    # ------------------------------------------------------------
    # 7) ESTRUTURAS DE SAÍDA
    # ------------------------------------------------------------
    manifestos_fechados: List[Dict[str, Any]] = []
    itens_manifestos_fechados: List[pd.DataFrame] = []
    tentativas_fechamento: List[Dict[str, Any]] = []

    ids_alocados: set[str] = set()
    contador_manifesto = 1

    def registrar_manifesto(
        df_combo: pd.DataFrame,
        avaliacao: Dict[str, Any],
        origem_etapa: str,
        bucket_temporal_valor: str,
    ) -> None:
        nonlocal contador_manifesto, manifestos_fechados, itens_manifestos_fechados, ids_alocados

        manifesto_id = f"MF4_{contador_manifesto:04d}"
        contador_manifesto += 1

        resumo = _gerar_resumo_manifesto(
            df_combo=df_combo,
            avaliacao=avaliacao,
            manifesto_id=manifesto_id,
            tipo_manifesto="fechado_bloco_4",
            origem_etapa=origem_etapa,
        )
        resumo["bucket_temporal_4a"] = bucket_temporal_valor
        manifestos_fechados.append(resumo)

        itens = df_combo.copy()
        itens["manifesto_id"] = manifesto_id
        itens["tipo_manifesto"] = "fechado_bloco_4"
        itens["veiculo_tipo"] = avaliacao["veiculo_tipo"]
        itens["capacidade_peso_kg_veiculo"] = avaliacao["capacidade_peso_kg"]
        itens["capacidade_vol_m3_veiculo"] = avaliacao["capacidade_vol_m3"]
        itens["max_entregas_veiculo"] = avaliacao["max_entregas"]
        itens["max_km_distancia_veiculo"] = avaliacao["max_km_distancia"]
        itens["ocupacao_peso_perc_manifesto"] = avaliacao["ocupacao_peso_perc"]
        itens["ocupacao_vol_perc_manifesto"] = avaliacao["ocupacao_vol_perc"]
        itens["ocupacao_dominante_perc_manifesto"] = avaliacao["ocupacao_dominante_perc"]
        itens["ocupacao_secundaria_perc_manifesto"] = avaliacao["ocupacao_secundaria_perc"]
        itens["origem_modulo"] = 4
        itens["origem_etapa"] = origem_etapa
        itens["bucket_temporal_4a"] = bucket_temporal_valor
        itens_manifestos_fechados.append(itens)

        ids_alocados.update(df_combo["id_linha_pipeline"].astype(str).tolist())

    # ------------------------------------------------------------
    # 8) 4A - FECHAMENTO MACRO POR MESMO CLIENTE + BUCKET TEMPORAL
    # ------------------------------------------------------------
    fila["bucket_temporal_4a"] = fila.apply(_bucket_temporal, axis=1)

    blocos_cliente = []
    for (cliente, bucket), df_bucket in fila.groupby([CHAVE_CLIENTE, "bucket_temporal_4a"], dropna=False):
        blocos_cliente.append(
            {
                "cliente": cliente,
                "bucket_temporal": bucket,
                "df_bucket": df_bucket.copy().reset_index(drop=True),
            }
        )

    blocos_cliente = sorted(blocos_cliente, key=lambda x: _score_macro_cliente(x["df_bucket"]), reverse=True)

    for bloco in blocos_cliente:
        cliente = bloco["cliente"]
        bucket = bloco["bucket_temporal"]

        df_bucket = bloco["df_bucket"].copy()
        df_bucket = df_bucket.loc[
            ~df_bucket["id_linha_pipeline"].astype(str).isin(ids_alocados)
        ].copy().reset_index(drop=True)

        if len(df_bucket) == 0:
            continue

        while len(df_bucket) > 0:
            df_bucket = _ordenar_df_prioridade(df_bucket)

            avaliacao_full = _avaliar_combo(df_bucket, catalogo_veiculos)

            for tent in avaliacao_full["tentativas"]:
                tent["etapa_fechamento"] = "4A_mesmo_cliente_bucket"
                tent["chave_analise"] = f"{_txt_norm(cliente)}|{_txt_norm(bucket)}"
                tent["cliente_referencia"] = cliente
                tent["bucket_temporal"] = bucket
                tent["qtd_linhas_grupo"] = len(df_bucket)
                tent["qtd_paradas_grupo"] = int(_chave_parada_df(df_bucket).nunique())
                tent["tipo_tentativa"] = "grupo_inteiro"
                tentativas_fechamento.append(tent)

            if avaliacao_full["aceito"]:
                registrar_manifesto(df_bucket, avaliacao_full, "4A_mesmo_cliente_bucket", bucket)
                df_bucket = df_bucket.iloc[0:0].copy()
                break

            melhor_subset = _encontrar_melhor_subconjunto_fechavel(df_bucket, catalogo_veiculos)

            if melhor_subset is None:
                break

            df_subset = melhor_subset["df_combo"].copy().reset_index(drop=True)
            avaliacao_subset = melhor_subset["avaliacao"]

            for _, veic in catalogo_veiculos.sort_values("ordem_porte").iterrows():
                tent_sub = _avaliar_combo_no_veiculo(df_subset, veic)
                tent_sub["resultado_teste"] = "aceito" if tent_sub["aceito"] else "rejeitado"
                tent_sub["etapa_fechamento"] = "4A_mesmo_cliente_bucket"
                tent_sub["chave_analise"] = f"{_txt_norm(cliente)}|{_txt_norm(bucket)}"
                tent_sub["cliente_referencia"] = cliente
                tent_sub["bucket_temporal"] = bucket
                tent_sub["qtd_linhas_grupo"] = len(df_bucket)
                tent_sub["qtd_paradas_grupo"] = int(_chave_parada_df(df_bucket).nunique())
                tent_sub["tipo_tentativa"] = "subconjunto_forte"
                tent_sub["qtd_linhas_subset"] = len(df_subset)
                tentativas_fechamento.append(tent_sub)

            registrar_manifesto(df_subset, avaliacao_subset, "4A_subconjunto_mesmo_cliente", bucket)

            df_bucket = df_bucket.loc[
                ~df_bucket["id_linha_pipeline"].astype(str).isin(
                    set(df_subset["id_linha_pipeline"].astype(str))
                )
            ].copy().reset_index(drop=True)

    # ------------------------------------------------------------
    # 9) 4B - FECHAMENTO POR PARADA NATURAL NO SALDO
    # ------------------------------------------------------------
    fila_saldo_4b = fila.loc[
        ~fila["id_linha_pipeline"].astype(str).isin(ids_alocados)
    ].copy().reset_index(drop=True)

    grupos_parada = []
    for chaves, df_grupo in fila_saldo_4b.groupby(CHAVES_PARADA, dropna=False):
        grupos_parada.append(
            {
                "destinatario": chaves[0],
                "cidade": chaves[1],
                "uf": chaves[2],
                "df_grupo": df_grupo.copy().reset_index(drop=True),
            }
        )

    grupos_parada = sorted(grupos_parada, key=lambda x: _score_grupo_parada(x["df_grupo"]), reverse=True)

    for bloco_parada in grupos_parada:
        df_grupo = bloco_parada["df_grupo"].copy()
        df_grupo = df_grupo.loc[
            ~df_grupo["id_linha_pipeline"].astype(str).isin(ids_alocados)
        ].copy().reset_index(drop=True)

        if len(df_grupo) == 0:
            continue

        avaliacao = _avaliar_combo(df_grupo, catalogo_veiculos)

        for tent in avaliacao["tentativas"]:
            tent["etapa_fechamento"] = "4B_parada_natural"
            tent["chave_analise"] = (
                f"{_txt_norm(bloco_parada['destinatario'])}|"
                f"{_txt_norm(bloco_parada['cidade'])}|"
                f"{_txt_norm(bloco_parada['uf'])}"
            )
            tent["cliente_referencia"] = bloco_parada["destinatario"]
            tent["bucket_temporal"] = "NA"
            tent["qtd_linhas_grupo"] = len(df_grupo)
            tent["qtd_paradas_grupo"] = int(_chave_parada_df(df_grupo).nunique())
            tent["tipo_tentativa"] = "grupo_inteiro"
            tentativas_fechamento.append(tent)

        if avaliacao["aceito"]:
            registrar_manifesto(df_grupo, avaliacao, "4B_parada_natural", "NA")

    # ------------------------------------------------------------
    # 10) 4C - VARREDURA FINAL DO REMANESCENTE
    # ------------------------------------------------------------
    iteracao_4c = 0
    novos_manifestos_4c = 0

    while iteracao_4c < MAX_ITERACOES_VARREDURA_4C:
        iteracao_4c += 1
        houve_novo_fechamento = False

        fila_saldo_4c = fila.loc[
            ~fila["id_linha_pipeline"].astype(str).isin(ids_alocados)
        ].copy().reset_index(drop=True)

        if len(fila_saldo_4c) == 0:
            break

        fila_saldo_4c["bucket_temporal_4a"] = fila_saldo_4c.apply(_bucket_temporal, axis=1)

        candidatos_4c = []

        for (cliente, bucket), df_bucket in fila_saldo_4c.groupby([CHAVE_CLIENTE, "bucket_temporal_4a"], dropna=False):
            candidatos_4c.append(
                {
                    "tipo_candidato": "cliente_bucket",
                    "cliente": cliente,
                    "bucket_temporal": bucket,
                    "chave_analise": f"{_txt_norm(cliente)}|{_txt_norm(bucket)}",
                    "df_base": df_bucket.copy().reset_index(drop=True),
                    "score": _score_macro_cliente(df_bucket),
                }
            )

        for chaves, df_grupo in fila_saldo_4c.groupby(CHAVES_PARADA, dropna=False):
            candidatos_4c.append(
                {
                    "tipo_candidato": "parada_natural",
                    "cliente": chaves[0],
                    "bucket_temporal": "NA",
                    "chave_analise": f"{_txt_norm(chaves[0])}|{_txt_norm(chaves[1])}|{_txt_norm(chaves[2])}",
                    "df_base": df_grupo.copy().reset_index(drop=True),
                    "score": _score_grupo_parada(df_grupo),
                }
            )

        candidatos_4c = sorted(candidatos_4c, key=lambda x: x["score"], reverse=True)

        for cand in candidatos_4c:
            df_base = cand["df_base"].copy()

            df_base = df_base.loc[
                ~df_base["id_linha_pipeline"].astype(str).isin(ids_alocados)
            ].copy().reset_index(drop=True)

            if len(df_base) == 0:
                continue

            avaliacao_full = _avaliar_combo(df_base, catalogo_veiculos)

            for tent in avaliacao_full["tentativas"]:
                tent["etapa_fechamento"] = "4C_varredura_final"
                tent["chave_analise"] = cand["chave_analise"]
                tent["cliente_referencia"] = cand["cliente"]
                tent["bucket_temporal"] = cand["bucket_temporal"]
                tent["qtd_linhas_grupo"] = len(df_base)
                tent["qtd_paradas_grupo"] = int(_chave_parada_df(df_base).nunique())
                tent["tipo_tentativa"] = f"{cand['tipo_candidato']}_grupo_inteiro"
                tent["iteracao_4c"] = iteracao_4c
                tentativas_fechamento.append(tent)

            if avaliacao_full["aceito"]:
                registrar_manifesto(
                    df_base,
                    avaliacao_full,
                    f"4C_varredura_final_{cand['tipo_candidato']}",
                    cand["bucket_temporal"],
                )
                novos_manifestos_4c += 1
                houve_novo_fechamento = True
                continue

            melhor_subset = _encontrar_melhor_subconjunto_fechavel(df_base, catalogo_veiculos)

            if melhor_subset is not None:
                df_subset = melhor_subset["df_combo"].copy().reset_index(drop=True)
                avaliacao_subset = melhor_subset["avaliacao"]

                for _, veic in catalogo_veiculos.sort_values("ordem_porte").iterrows():
                    tent_sub = _avaliar_combo_no_veiculo(df_subset, veic)
                    tent_sub["resultado_teste"] = "aceito" if tent_sub["aceito"] else "rejeitado"
                    tent_sub["etapa_fechamento"] = "4C_varredura_final"
                    tent_sub["chave_analise"] = cand["chave_analise"]
                    tent_sub["cliente_referencia"] = cand["cliente"]
                    tent_sub["bucket_temporal"] = cand["bucket_temporal"]
                    tent_sub["qtd_linhas_grupo"] = len(df_base)
                    tent_sub["qtd_paradas_grupo"] = int(_chave_parada_df(df_base).nunique())
                    tent_sub["tipo_tentativa"] = f"{cand['tipo_candidato']}_subconjunto_forte"
                    tent_sub["qtd_linhas_subset"] = len(df_subset)
                    tent_sub["iteracao_4c"] = iteracao_4c
                    tentativas_fechamento.append(tent_sub)

                registrar_manifesto(
                    df_subset,
                    avaliacao_subset,
                    f"4C_varredura_final_subset_{cand['tipo_candidato']}",
                    cand["bucket_temporal"],
                )
                novos_manifestos_4c += 1
                houve_novo_fechamento = True

        if not houve_novo_fechamento:
            break

    # ------------------------------------------------------------
    # 11) DATAFRAMES FINAIS
    # ------------------------------------------------------------
    df_manifestos_fechados_bloco_4 = pd.DataFrame(manifestos_fechados)

    if len(itens_manifestos_fechados) > 0:
        df_itens_manifestos_fechados_bloco_4 = pd.concat(itens_manifestos_fechados, ignore_index=True)

        if df_itens_manifestos_fechados_bloco_4["id_linha_pipeline"].astype(str).duplicated().any():
            qtd_dup = int(
                df_itens_manifestos_fechados_bloco_4["id_linha_pipeline"].astype(str).duplicated().sum()
            )
            raise Exception(
                f"Falha estrutural: id_linha_pipeline duplicado dentro dos itens manifestados do M4: {qtd_dup}"
            )
    else:
        df_itens_manifestos_fechados_bloco_4 = pd.DataFrame(
            columns=list(fila.columns)
            + [
                "manifesto_id",
                "tipo_manifesto",
                "veiculo_tipo",
                "capacidade_peso_kg_veiculo",
                "capacidade_vol_m3_veiculo",
                "max_entregas_veiculo",
                "max_km_distancia_veiculo",
                "ocupacao_peso_perc_manifesto",
                "ocupacao_vol_perc_manifesto",
                "ocupacao_dominante_perc_manifesto",
                "ocupacao_secundaria_perc_manifesto",
                "origem_modulo",
                "origem_etapa",
                "bucket_temporal_4a",
            ]
        )

    df_tentativas_fechamento_bloco_4 = pd.DataFrame(tentativas_fechamento)

    df_remanescente_roteirizavel_bloco_4 = (
        fila.loc[~fila["id_linha_pipeline"].astype(str).isin(ids_alocados)].copy().reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # 12) FECHAMENTO CONTÁBIL OBRIGATÓRIO
    # ------------------------------------------------------------
    roteirizavel_entrada_m4 = len(fila)
    itens_manifestados_m4 = (
        int(df_itens_manifestos_fechados_bloco_4["id_linha_pipeline"].astype(str).nunique())
        if len(df_itens_manifestos_fechados_bloco_4) > 0
        else 0
    )
    remanescente_roteirizavel_m4 = len(df_remanescente_roteirizavel_bloco_4)

    if roteirizavel_entrada_m4 != (itens_manifestados_m4 + remanescente_roteirizavel_m4):
        raise Exception(
            "Falha de fechamento contábil do Módulo 4:\n"
            f"- entrada = {roteirizavel_entrada_m4}\n"
            f"- manifestados_unicos = {itens_manifestados_m4}\n"
            f"- remanescente = {remanescente_roteirizavel_m4}\n"
            f"- soma = {itens_manifestados_m4 + remanescente_roteirizavel_m4}"
        )

    intersecao_ids = set(df_remanescente_roteirizavel_bloco_4["id_linha_pipeline"].astype(str)).intersection(
        set(df_itens_manifestos_fechados_bloco_4["id_linha_pipeline"].astype(str))
    )

    if len(intersecao_ids) > 0:
        raise Exception(
            "Falha estrutural: há IDs ao mesmo tempo em manifestados e remanescente. "
            f"Exemplo: {sorted(list(intersecao_ids))[:10]}"
        )

    # ------------------------------------------------------------
    # 13) RESUMOS
    # ------------------------------------------------------------
    grupos_naturais_testados_4b = int(
        fila_saldo_4b.groupby(CHAVES_PARADA, dropna=False).ngroups
    ) if len(fila_saldo_4b) > 0 else 0

    resumo_execucao = pd.DataFrame(
        [
            {"indicador": "roteirizavel_entrada_m4", "valor": int(roteirizavel_entrada_m4)},
            {"indicador": "clientes_macro_testados_4a", "valor": int(fila["destinatario"].nunique(dropna=True))},
            {
                "indicador": "buckets_temporais_testados_4a",
                "valor": int(fila.groupby(["destinatario", "bucket_temporal_4a"], dropna=False).ngroups),
            },
            {"indicador": "grupos_naturais_testados_4b", "valor": grupos_naturais_testados_4b},
            {"indicador": "iteracoes_varredura_4c_executadas", "valor": int(iteracao_4c)},
            {"indicador": "novos_manifestos_gerados_4c", "valor": int(novos_manifestos_4c)},
            {"indicador": "manifestos_fechados_gerados_m4", "valor": int(len(df_manifestos_fechados_bloco_4))},
            {"indicador": "itens_manifestados_m4", "valor": int(itens_manifestados_m4)},
            {"indicador": "remanescente_roteirizavel_m4", "valor": int(remanescente_roteirizavel_m4)},
        ]
    )

    if len(df_manifestos_fechados_bloco_4) > 0:
        resumo_por_veiculo = (
            df_manifestos_fechados_bloco_4.groupby("veiculo_tipo", as_index=False)
            .agg(
                {
                    "manifesto_id": "count",
                    "peso_total_kg": "sum",
                    "vol_total_m3": "sum",
                    "ocupacao_dominante_perc": "mean",
                    "ocupacao_secundaria_perc": "mean",
                }
            )
            .rename(columns={"manifesto_id": "qtd_manifestos"})
            .sort_values(by=["qtd_manifestos", "peso_total_kg"], ascending=[False, False])
            .reset_index(drop=True)
        )
    else:
        resumo_por_veiculo = pd.DataFrame(
            columns=[
                "veiculo_tipo",
                "qtd_manifestos",
                "peso_total_kg",
                "vol_total_m3",
                "ocupacao_dominante_perc",
                "ocupacao_secundaria_perc",
            ]
        )

    if len(df_manifestos_fechados_bloco_4) > 0:
        resumo_por_etapa = (
            df_manifestos_fechados_bloco_4.groupby("origem_etapa", as_index=False)
            .agg(
                {
                    "manifesto_id": "count",
                    "peso_total_kg": "sum",
                    "vol_total_m3": "sum",
                    "qtd_paradas": "sum",
                }
            )
            .rename(columns={"manifesto_id": "qtd_manifestos"})
            .sort_values(by=["qtd_manifestos"], ascending=False)
            .reset_index(drop=True)
        )
    else:
        resumo_por_etapa = pd.DataFrame(
            columns=["origem_etapa", "qtd_manifestos", "peso_total_kg", "vol_total_m3", "qtd_paradas"]
        )

    # ------------------------------------------------------------
    # 14) SALVAR OUTPUTS INTERNOS
    # ------------------------------------------------------------
    try:
        with pd.ExcelWriter(arq_manifestos_xlsx, engine="openpyxl") as writer:
            df_manifestos_fechados_bloco_4.to_excel(writer, sheet_name="manifestos_fechados", index=False)

        df_itens_manifestos_fechados_bloco_4.to_csv(
            arq_itens_csv, index=False, encoding="utf-8-sig", sep=";"
        )

        df_tentativas_fechamento_bloco_4.to_csv(
            arq_tentativas_csv, index=False, encoding="utf-8-sig", sep=";"
        )

        df_remanescente_roteirizavel_bloco_4.to_csv(
            arq_remanescente_csv, index=False, encoding="utf-8-sig", sep=";"
        )

        with pd.ExcelWriter(arq_resumo_xlsx, engine="openpyxl") as writer:
            resumo_execucao.to_excel(writer, sheet_name="resumo_execucao", index=False)
            resumo_por_veiculo.to_excel(writer, sheet_name="resumo_veiculo", index=False)
            resumo_por_etapa.to_excel(writer, sheet_name="resumo_etapa", index=False)
            df_manifestos_fechados_bloco_4.to_excel(writer, sheet_name="manifestos_fechados", index=False)
            df_tentativas_fechamento_bloco_4.to_excel(writer, sheet_name="tentativas", index=False)
            df_remanescente_roteirizavel_bloco_4.to_excel(writer, sheet_name="remanescente", index=False)

        metadata = {
            "modulo": "4_manifestos_fechados_reforcado_final",
            "data_execucao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_base_projeto": pd.Timestamp(data_base_roteirizacao).strftime("%Y-%m-%d"),
            "regras": {
                "etapa_4a_mesmo_cliente": True,
                "etapa_4a_bucket_temporal": True,
                "etapa_4a_subconjunto_forte": True,
                "etapa_4b_parada_natural": True,
                "etapa_4c_varredura_final_remanescente": True,
                "max_entregas_como_paradas": True,
                "respeita_max_km_distancia": True,
                "ocupacao_dominante_min": OCUPACAO_DOMINANTE_MIN,
                "ocupacao_secundaria_min": OCUPACAO_SECUNDARIA_MIN,
            },
            "totais": {
                "roteirizavel_entrada_m4": int(roteirizavel_entrada_m4),
                "manifestos_fechados_gerados_m4": int(len(df_manifestos_fechados_bloco_4)),
                "itens_manifestados_m4": int(itens_manifestados_m4),
                "remanescente_roteirizavel_m4": int(remanescente_roteirizavel_m4),
                "novos_manifestos_gerados_4c": int(novos_manifestos_4c),
                "iteracoes_varredura_4c_executadas": int(iteracao_4c),
            },
            "outputs": {
                "df_manifestos_fechados_bloco_4_xlsx": str(arq_manifestos_xlsx),
                "df_itens_manifestos_fechados_bloco_4_csv": str(arq_itens_csv),
                "df_tentativas_fechamento_bloco_4_csv": str(arq_tentativas_csv),
                "df_remanescente_roteirizavel_bloco_4_csv": str(arq_remanescente_csv),
                "resumo_modulo_4_xlsx": str(arq_resumo_xlsx),
            },
        }

        with open(arq_metadata_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        caminhos_pipeline["df_manifestos_fechados_bloco_4_xlsx"] = str(arq_manifestos_xlsx)
        caminhos_pipeline["df_itens_manifestos_fechados_bloco_4_csv"] = str(arq_itens_csv)
        caminhos_pipeline["df_tentativas_fechamento_bloco_4_csv"] = str(arq_tentativas_csv)
        caminhos_pipeline["df_remanescente_roteirizavel_bloco_4_csv"] = str(arq_remanescente_csv)
        caminhos_pipeline["resumo_modulo_4_xlsx"] = str(arq_resumo_xlsx)
        caminhos_pipeline["metadata_modulo_4_json"] = str(arq_metadata_json)
    except Exception:
        # Não quebra a API se falhar apenas a persistência dos artefatos.
        pass

    # ------------------------------------------------------------
    # 15) META E OUTPUTS
    # ------------------------------------------------------------
    resumo_m4 = {
        "modulo": "M4",
        "data_base_roteirizacao": pd.Timestamp(data_base_roteirizacao).isoformat(),
        "coluna_tipo_veiculo_utilizada": coluna_tipo_veiculo,
        "roteirizavel_entrada_m4": int(roteirizavel_entrada_m4),
        "manifestos_fechados_gerados_m4": int(len(df_manifestos_fechados_bloco_4)),
        "itens_manifestados_m4": int(itens_manifestados_m4),
        "remanescente_roteirizavel_m4": int(remanescente_roteirizavel_m4),
        "clientes_macro_testados_4a": int(fila["destinatario"].nunique(dropna=True)),
        "buckets_temporais_testados_4a": int(fila.groupby(["destinatario", "bucket_temporal_4a"], dropna=False).ngroups),
        "grupos_naturais_testados_4b": int(grupos_naturais_testados_4b),
        "iteracoes_varredura_4c_executadas": int(iteracao_4c),
        "novos_manifestos_gerados_4c": int(novos_manifestos_4c),
        "ocupacao_dominante_min_perc": round(OCUPACAO_DOMINANTE_MIN * 100, 2),
        "ocupacao_secundaria_min_perc": round(OCUPACAO_SECUNDARIA_MIN * 100, 2),
        "caminhos_pipeline": caminhos_pipeline,
    }

    outputs = {
        "df_manifestos_fechados_bloco_4": df_manifestos_fechados_bloco_4,
        "df_itens_manifestos_fechados_bloco_4": df_itens_manifestos_fechados_bloco_4,
        "df_tentativas_fechamento_bloco_4": df_tentativas_fechamento_bloco_4,
        "df_remanescente_roteirizavel_bloco_4": df_remanescente_roteirizavel_bloco_4,
    }

    meta_m4 = {
        "resumo_m4": resumo_m4,
        "resumo_execucao": _to_records(resumo_execucao),
        "resumo_por_veiculo": _to_records(resumo_por_veiculo),
        "resumo_por_etapa": _to_records(resumo_por_etapa),
        "outputs_m4": outputs,
    }

    return outputs, meta_m4
