from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from app.pipeline.m5_common import (
    normalize_saldo_m5,
    normalize_veiculos_m5,
    ordenar_operacional_m5,
    agrupar_saldo_por_cidade,
    peso_total,
    km_referencia,
    ocupacao_perc,
    safe_float,
    safe_int,
    safe_text,
)


# =========================================================================================
# M5.1 - TRIAGEM DE CIDADES
# -----------------------------------------------------------------------------------------
# OBJETIVO
# - receber o remanescente global oficial do M4
# - agrupar por cidade
# - ordenar da maior massa para a menor
# - testar todos os perfis por cidade
# - nesta etapa olhar SOMENTE ocupação mínima >= 70%
# - nesta etapa NÃO olhar raio
# - nesta etapa NÃO olhar ocupação máxima
#
# SAÍDA
# - cidades consolidadas
# - perfis elegíveis por cidade
# - saldo elegível para composição por cidade (M5.2)
# - cidades remanescentes para seguir ao agrupamento por subregião (M5.3)
# - tentativas auditáveis cidade x perfil
# =========================================================================================


def _veiculos_menor_para_maior(df_veiculos: pd.DataFrame) -> pd.DataFrame:
    temp = df_veiculos.copy()
    temp["_cap_peso_tmp"] = pd.to_numeric(temp["capacidade_peso_kg"], errors="coerce").fillna(0)
    temp["_cap_vol_tmp"] = pd.to_numeric(temp["capacidade_vol_m3"], errors="coerce").fillna(0)

    return (
        temp.sort_values(
            by=["_cap_peso_tmp", "_cap_vol_tmp", "tipo", "perfil"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )
        .drop(columns=["_cap_peso_tmp", "_cap_vol_tmp"], errors="ignore")
        .reset_index(drop=True)
        .copy()
    )


def _ordenar_cidades_por_massa(df_cidades: pd.DataFrame) -> pd.DataFrame:
    if df_cidades.empty:
        return df_cidades.copy()

    return (
        df_cidades.sort_values(
            by=["peso_total_cidade", "cidade", "uf"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )


def _filtrar_itens_cidade(df_saldo: pd.DataFrame, cidade: str, uf: str) -> pd.DataFrame:
    return (
        df_saldo.loc[
            (df_saldo["cidade"].astype(str) == safe_text(cidade))
            & (df_saldo["uf"].astype(str) == safe_text(uf))
        ]
        .copy()
        .reset_index(drop=True)
    )


def _avaliar_perfil_na_cidade(
    df_cidade: pd.DataFrame,
    vehicle_row: pd.Series,
) -> Dict[str, Any]:
    peso_cidade = peso_total(df_cidade)
    km_cidade = km_referencia(df_cidade)
    ocupacao = ocupacao_perc(df_cidade, vehicle_row)

    return {
        "perfil": safe_text(vehicle_row.get("perfil")),
        "tipo": safe_text(vehicle_row.get("tipo")),
        "capacidade_peso_kg": safe_float(vehicle_row.get("capacidade_peso_kg"), 0.0),
        "capacidade_vol_m3": safe_float(vehicle_row.get("capacidade_vol_m3"), 0.0),
        "max_entregas": safe_int(vehicle_row.get("max_entregas"), 0),
        "max_km_distancia": safe_float(vehicle_row.get("max_km_distancia"), 0.0),
        "ocupacao_minima_perc": safe_float(vehicle_row.get("ocupacao_minima_perc"), 70.0),
        "ocupacao_maxima_perc": safe_float(vehicle_row.get("ocupacao_maxima_perc"), 100.0),
        "peso_total_cidade": round(peso_cidade, 3),
        "km_referencia_cidade": round(km_cidade, 2),
        "ocupacao_calculada_perc": round(ocupacao, 2),
        "status_perfil_cidade": "elegivel" if ocupacao >= 70.0 else "nao_elegivel",
        "motivo_status_perfil_cidade": (
            "atinge_ocupacao_minima_70"
            if ocupacao >= 70.0
            else "abaixo_ocupacao_minima_70"
        ),
        "regra_aplicada": "somente_ocupacao_minima_sem_raio_sem_ocupacao_maxima",
    }


def _montar_cidades_consolidadas(
    df_cidades_agg: pd.DataFrame,
    df_perfis_viaveis: pd.DataFrame,
) -> pd.DataFrame:
    if df_cidades_agg.empty:
        return pd.DataFrame()

    base = df_cidades_agg.copy()

    if df_perfis_viaveis.empty:
        base["qtd_perfis_elegiveis"] = 0
        base["cidade_elegivel_m5_1"] = False
        base["motivo_status_cidade_m5_1"] = "nenhum_perfil_atinge_ocupacao_minima_70"
        base["ordem_cidade_m5_1"] = range(1, len(base) + 1)
        return base

    elegiveis = (
        df_perfis_viaveis.loc[df_perfis_viaveis["status_perfil_cidade"] == "elegivel"]
        .groupby(["cidade", "uf"], as_index=False)
        .agg(qtd_perfis_elegiveis=("perfil", "count"))
    )

    base = base.merge(elegiveis, how="left", on=["cidade", "uf"])
    base["qtd_perfis_elegiveis"] = pd.to_numeric(base["qtd_perfis_elegiveis"], errors="coerce").fillna(0).astype(int)
    base["cidade_elegivel_m5_1"] = base["qtd_perfis_elegiveis"] > 0
    base["motivo_status_cidade_m5_1"] = base["cidade_elegivel_m5_1"].map(
        {
            True: "cidade_tem_ao_menos_um_perfil_com_ocupacao_minima",
            False: "nenhum_perfil_atinge_ocupacao_minima_70",
        }
    )
    base["ordem_cidade_m5_1"] = range(1, len(base) + 1)

    return base.reset_index(drop=True).copy()


def executar_m5_1_triagem_cidades(
    df_remanescente_roteirizavel_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    M5.1 novo:
    - entrada = remanescente global oficial do M4
    - agrupa por cidade
    - testa todos os perfis por cidade
    - critério único desta etapa = ocupação mínima >= 70%
    - não olha raio
    - não olha ocupação máxima
    """

    # -------------------------------------------------------------------------
    # 1) NORMALIZAÇÃO DO CONTRATO INTERNO
    # -------------------------------------------------------------------------
    saldo = normalize_saldo_m5(
        df_input=df_remanescente_roteirizavel_bloco_4,
        etapa="M5.1",
        require_geo=True,
        require_subregiao=False,
        require_mesorregiao=False,
    )
    veiculos = normalize_veiculos_m5(
        df_veiculos=df_veiculos_tratados,
        etapa="M5.1",
    )

    if saldo.empty:
        outputs_vazios = {
            "df_cidades_consolidadas_m5_1": pd.DataFrame(),
            "df_perfis_viaveis_por_cidade_m5_1": pd.DataFrame(),
            "df_saldo_elegivel_composicao_m5_1": pd.DataFrame(),
            "df_cidades_remanescentes_m5_1": pd.DataFrame(),
            "df_tentativas_triagem_cidades_m5_1": pd.DataFrame(),
        }
        meta = {
            "resumo_m5_1": {
                "modulo": "M5.1",
                "linhas_entrada": 0,
                "cidades_total": 0,
                "cidades_elegiveis": 0,
                "cidades_remanescentes": 0,
                "perfis_testados_total": 0,
                "perfis_elegiveis_total": 0,
                "regra_m5_1": "ocupacao_minima_sem_raio_sem_ocupacao_maxima",
            }
        }
        return outputs_vazios, meta

    # -------------------------------------------------------------------------
    # 2) ORDENAÇÃO OPERACIONAL DO SALDO
    # -------------------------------------------------------------------------
    saldo = ordenar_operacional_m5(saldo, suffix="m5_1")

    # -------------------------------------------------------------------------
    # 3) AGREGAÇÃO POR CIDADE
    # -------------------------------------------------------------------------
    df_cidades_agg = agrupar_saldo_por_cidade(saldo)
    df_cidades_agg = _ordenar_cidades_por_massa(df_cidades_agg)

    veiculos_ord = _veiculos_menor_para_maior(veiculos)

    # -------------------------------------------------------------------------
    # 4) TESTE CIDADE X PERFIL
    # -------------------------------------------------------------------------
    tentativas: List[Dict[str, Any]] = []

    for _, row_cidade in df_cidades_agg.iterrows():
        cidade = safe_text(row_cidade.get("cidade"))
        uf = safe_text(row_cidade.get("uf"))

        df_cidade = _filtrar_itens_cidade(saldo, cidade=cidade, uf=uf)

        for _, row_veic in veiculos_ord.iterrows():
            avaliacao = _avaliar_perfil_na_cidade(
                df_cidade=df_cidade,
                vehicle_row=row_veic,
            )

            tentativas.append(
                {
                    "cidade": cidade,
                    "uf": uf,
                    "perfil": avaliacao["perfil"],
                    "tipo": avaliacao["tipo"],
                    "peso_total_cidade": avaliacao["peso_total_cidade"],
                    "km_referencia_cidade": avaliacao["km_referencia_cidade"],
                    "ocupacao_calculada_perc": avaliacao["ocupacao_calculada_perc"],
                    "capacidade_peso_kg": avaliacao["capacidade_peso_kg"],
                    "ocupacao_minima_perc": avaliacao["ocupacao_minima_perc"],
                    "status_perfil_cidade": avaliacao["status_perfil_cidade"],
                    "motivo_status_perfil_cidade": avaliacao["motivo_status_perfil_cidade"],
                    "regra_aplicada": avaliacao["regra_aplicada"],
                }
            )

    df_tentativas_triagem_cidades_m5_1 = pd.DataFrame(tentativas)

    # -------------------------------------------------------------------------
    # 5) PERFIS VIÁVEIS POR CIDADE
    # -------------------------------------------------------------------------
    df_perfis_viaveis_por_cidade_m5_1 = df_tentativas_triagem_cidades_m5_1.copy()

    # -------------------------------------------------------------------------
    # 6) CIDADES CONSOLIDADAS
    # -------------------------------------------------------------------------
    df_cidades_consolidadas_m5_1 = _montar_cidades_consolidadas(
        df_cidades_agg=df_cidades_agg,
        df_perfis_viaveis=df_perfis_viaveis_por_cidade_m5_1,
    )

    cidades_elegiveis = set(
        df_cidades_consolidadas_m5_1.loc[
            df_cidades_consolidadas_m5_1["cidade_elegivel_m5_1"] == True,
            ["cidade", "uf"],
        ].apply(lambda row: (safe_text(row["cidade"]), safe_text(row["uf"])), axis=1).tolist()
    )

    cidades_remanescentes = set(
        df_cidades_consolidadas_m5_1.loc[
            df_cidades_consolidadas_m5_1["cidade_elegivel_m5_1"] == False,
            ["cidade", "uf"],
        ].apply(lambda row: (safe_text(row["cidade"]), safe_text(row["uf"])), axis=1).tolist()
    )

    # -------------------------------------------------------------------------
    # 7) SALDO ELEGÍVEL PARA M5.2
    # -------------------------------------------------------------------------
    if cidades_elegiveis:
        df_saldo_elegivel_composicao_m5_1 = (
            saldo.loc[
                saldo.apply(
                    lambda row: (safe_text(row["cidade"]), safe_text(row["uf"])) in cidades_elegiveis,
                    axis=1,
                )
            ]
            .copy()
            .reset_index(drop=True)
        )
    else:
        df_saldo_elegivel_composicao_m5_1 = pd.DataFrame(columns=saldo.columns)

    # -------------------------------------------------------------------------
    # 8) CIDADES REMANESCENTES PARA M5.3
    # -------------------------------------------------------------------------
    if cidades_remanescentes:
        df_cidades_remanescentes_m5_1 = (
            saldo.loc[
                saldo.apply(
                    lambda row: (safe_text(row["cidade"]), safe_text(row["uf"])) in cidades_remanescentes,
                    axis=1,
                )
            ]
            .copy()
            .reset_index(drop=True)
        )
    else:
        df_cidades_remanescentes_m5_1 = pd.DataFrame(columns=saldo.columns)

    # -------------------------------------------------------------------------
    # 9) RESUMO
    # -------------------------------------------------------------------------
    perfis_elegiveis_total = int(
        (df_perfis_viaveis_por_cidade_m5_1["status_perfil_cidade"] == "elegivel").sum()
    ) if not df_perfis_viaveis_por_cidade_m5_1.empty else 0

    resumo_m5_1 = {
        "modulo": "M5.1",
        "linhas_entrada": int(len(saldo)),
        "cidades_total": int(len(df_cidades_consolidadas_m5_1)),
        "cidades_elegiveis": int(df_cidades_consolidadas_m5_1["cidade_elegivel_m5_1"].sum()) if not df_cidades_consolidadas_m5_1.empty else 0,
        "cidades_remanescentes": int((df_cidades_consolidadas_m5_1["cidade_elegivel_m5_1"] == False).sum()) if not df_cidades_consolidadas_m5_1.empty else 0,
        "perfis_testados_total": int(len(df_perfis_viaveis_por_cidade_m5_1)),
        "perfis_elegiveis_total": perfis_elegiveis_total,
        "linhas_saldo_elegivel_composicao_m5_1": int(len(df_saldo_elegivel_composicao_m5_1)),
        "linhas_cidades_remanescentes_m5_1": int(len(df_cidades_remanescentes_m5_1)),
        "regra_m5_1": "ocupacao_minima_sem_raio_sem_ocupacao_maxima",
    }

    outputs = {
        "df_cidades_consolidadas_m5_1": df_cidades_consolidadas_m5_1,
        "df_perfis_viaveis_por_cidade_m5_1": df_perfis_viaveis_por_cidade_m5_1,
        "df_saldo_elegivel_composicao_m5_1": df_saldo_elegivel_composicao_m5_1,
        "df_cidades_remanescentes_m5_1": df_cidades_remanescentes_m5_1,
        "df_tentativas_triagem_cidades_m5_1": df_tentativas_triagem_cidades_m5_1,
    }

    meta = {
        "resumo_m5_1": resumo_m5_1
    }

    return outputs, meta
