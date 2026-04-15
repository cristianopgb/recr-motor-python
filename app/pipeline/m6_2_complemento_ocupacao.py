from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# M6.2 - COMPLEMENTO DE OCUPAÇÃO
# CONTRATO FECHADO
#
# Entradas oficiais:
# - df_manifestos_base_m6
# - df_estatisticas_manifestos_antes_m6
# - df_itens_manifestos_base_m6
# - df_remanescente_m5_4
#
# Regras:
# - usa ocupacao_base_antes_m6 do M6.1
# - usa capacidade_peso_kg_veiculo do M6.1
# - usa max_km_distancia_veiculo do M6.1
# - usa mesorregiao_manifesto_m6 da base de estatísticas do M6.1
# - elegíveis: ocupação < 85%
# - busca remanescente nesta ordem:
#     1. mesmo cliente
#     2. mesma cidade
#     3. mesma sub-região
#     4. mesma mesorregião
# - prioridade:
#     1. agendado primeiro
#     2. folga positiva menor -> maior
#     3. folga negativa por último
# - ocupação calculada por peso_calculado / capacidade_peso_kg_veiculo
# ============================================================


COLS_MANIFESTOS_OBRIGATORIAS = [
    "manifesto_id",
    "origem_manifesto_modulo",
    "origem_manifesto_tipo",
    "veiculo_tipo",
    "veiculo_perfil",
    "peso_base_antes_m6",
    "km_base_antes_m6",
    "ocupacao_base_antes_m6",
    "capacidade_peso_kg_veiculo",
    "max_km_distancia_veiculo",
    "qtd_itens_base_antes_m6",
    "qtd_ctes_base_antes_m6",
    "qtd_paradas_base_antes_m6",
]

COLS_ESTATS_OBRIGATORIAS = [
    "manifesto_id",
    "mesorregiao_manifesto_m6",
]

COLS_ITENS_OBRIGATORIAS = [
    "manifesto_id",
    "id_linha_pipeline",
    "destinatario",
    "cidade",
    "uf",
    "subregiao",
    "mesorregiao",
    "distancia_rodoviaria_est_km",
    "peso_calculado",
    "vol_m3",
]

COLS_REMANESCENTE_OBRIGATORIAS = [
    "id_linha_pipeline",
    "destinatario",
    "cidade",
    "uf",
    "subregiao",
    "mesorregiao",
    "distancia_rodoviaria_est_km",
    "peso_calculado",
    "vol_m3",
]

CHAVES_PARADA = ["destinatario", "cidade", "uf"]


def executar_m6_2_complemento_ocupacao(
    df_manifestos_base_m6: pd.DataFrame,
    df_estatisticas_manifestos_antes_m6: pd.DataFrame,
    df_itens_manifestos_base_m6: pd.DataFrame,
    df_remanescente_m5_4: pd.DataFrame,
    data_base_roteirizacao: datetime,
    tipo_roteirizacao: str,
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    ocupacao_alvo_perc: float = 85.0,
) -> Dict[str, Any]:
    df_manifestos = _normalizar_manifestos(df_manifestos_base_m6)
    df_estats = _normalizar_estatisticas_m6(df_estatisticas_manifestos_antes_m6)
    df_itens = _normalizar_itens_manifestos(df_itens_manifestos_base_m6)
    df_remanescente = _normalizar_remanescente(df_remanescente_m5_4)

    _validar_entrada(df_manifestos, df_estats, df_itens, df_remanescente)

    df_manifestos = _enriquecer_manifestos_com_estatisticas(df_manifestos, df_estats)

    df_remanescente_original = df_remanescente.copy()

    if "flag_otimizado_m6_2" not in df_itens.columns:
        df_itens["flag_otimizado_m6_2"] = False
    if "origem_item_m6_2" not in df_itens.columns:
        df_itens["origem_item_m6_2"] = "original_m6_1"

    manifestos_alvo = _selecionar_manifestos_alvo(df_manifestos, ocupacao_alvo_perc)

    tentativas: List[Dict[str, Any]] = []
    movimentos_aceitos: List[Dict[str, Any]] = []

    for manifesto_id in manifestos_alvo:
        row_manifesto = df_manifestos.loc[df_manifestos["manifesto_id"] == manifesto_id].head(1)
        if row_manifesto.empty:
            continue

        manifesto = row_manifesto.iloc[0].to_dict()
        itens_manifesto_atual = df_itens.loc[df_itens["manifesto_id"] == manifesto_id].copy()
        if itens_manifesto_atual.empty:
            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "tipo_tentativa": "manifesto_sem_itens",
                    "nivel_hierarquia": None,
                    "aceito": False,
                    "motivo": "Manifesto elegível sem itens associados.",
                }
            )
            continue

        meso_manifesto = _txt_norm(manifesto.get("mesorregiao_manifesto", ""))
        if meso_manifesto == "":
            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "tipo_tentativa": "manifesto_sem_mesorregiao_oficial",
                    "nivel_hierarquia": None,
                    "aceito": False,
                    "motivo": "Manifesto sem mesorregião oficial vinda do M6.1.",
                }
            )
            continue

        rem_mesmo_meso = df_remanescente.loc[
            df_remanescente["mesorregiao"].astype(str).str.upper() == meso_manifesto
        ].copy()

        if rem_mesmo_meso.empty:
            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "tipo_tentativa": "sem_remanescente_mesma_mesorregiao",
                    "nivel_hierarquia": None,
                    "aceito": False,
                    "motivo": "Não há remanescente do M5 na mesma mesorregião do manifesto.",
                }
            )
            continue

        houve_movimento_neste_manifesto = False

        contexto_manifesto = {
            "cliente_dominante": _cliente_dominante(itens_manifesto_atual),
            "cidade_dominante": _cidade_dominante(itens_manifesto_atual),
            "subregiao_dominante": _subregiao_dominante(itens_manifesto_atual),
            "mesorregiao": meso_manifesto,
        }

        niveis_hierarquia = [
            ("mesmo_cliente", "destinatario"),
            ("mesma_cidade", "cidade"),
            ("mesma_subregiao", "subregiao"),
            ("mesma_mesorregiao", "mesorregiao"),
        ]

        for nome_nivel, coluna_nivel in niveis_hierarquia:
            candidatos = _selecionar_candidatos_por_hierarquia(
                rem_mesmo_meso=rem_mesmo_meso,
                nome_nivel=nome_nivel,
                coluna_nivel=coluna_nivel,
                contexto_manifesto=contexto_manifesto,
            )

            if candidatos.empty:
                tentativas.append(
                    {
                        "manifesto_id": manifesto_id,
                        "tipo_tentativa": "sem_candidatos_nivel",
                        "nivel_hierarquia": nome_nivel,
                        "aceito": False,
                        "motivo": f"Sem candidatos no nível {nome_nivel}.",
                    }
                )
                continue

            candidatos_ordenados = _ordenar_candidatos_por_prioridade_operacional(candidatos)

            for _, item_row in candidatos_ordenados.iterrows():
                item_df = pd.DataFrame([item_row.to_dict()])

                valido, motivo, comparativo = _simular_adicao_item(
                    manifesto=manifesto,
                    itens_manifesto=itens_manifesto_atual,
                    item_candidato=item_df,
                )

                tentativa = {
                    "manifesto_id": manifesto_id,
                    "tipo_tentativa": "adicao_item_remanescente_m5",
                    "nivel_hierarquia": nome_nivel,
                    "id_linha_pipeline": str(item_row["id_linha_pipeline"]),
                    "destinatario": str(item_row.get("destinatario", "")),
                    "cidade": str(item_row.get("cidade", "")),
                    "subregiao": str(item_row.get("subregiao", "")),
                    "mesorregiao": str(item_row.get("mesorregiao", "")),
                    "agendada": bool(item_row.get("agendada", False)),
                    "folga_dias": _to_float(item_row.get("folga_dias")),
                    "aceito": bool(valido),
                    "motivo": motivo,
                    **comparativo,
                }
                tentativas.append(tentativa)

                if not valido:
                    continue

                houve_movimento_neste_manifesto = True

                item_aplicar = item_df.copy()
                item_aplicar["manifesto_id"] = manifesto_id
                item_aplicar["flag_otimizado_m6_2"] = True
                item_aplicar["origem_item_m6_2"] = "adicionado_do_remanescente_m5"

                df_itens = pd.concat([df_itens, item_aplicar], ignore_index=True)

                df_remanescente = df_remanescente.loc[
                    df_remanescente["id_linha_pipeline"].astype(str) != str(item_row["id_linha_pipeline"])
                ].copy()

                df_manifestos = _recalcular_manifesto_unico(df_manifestos, df_itens, manifesto_id)
                manifesto = df_manifestos.loc[df_manifestos["manifesto_id"] == manifesto_id].head(1).iloc[0].to_dict()
                itens_manifesto_atual = df_itens.loc[df_itens["manifesto_id"] == manifesto_id].copy()

                movimentos_aceitos.append(
                    {
                        "manifesto_id": manifesto_id,
                        "nivel_hierarquia": nome_nivel,
                        "id_linha_pipeline": str(item_row["id_linha_pipeline"]),
                        "destinatario": str(item_row.get("destinatario", "")),
                        "cidade": str(item_row.get("cidade", "")),
                        "subregiao": str(item_row.get("subregiao", "")),
                        "mesorregiao": str(item_row.get("mesorregiao", "")),
                        "agendada": bool(item_row.get("agendada", False)),
                        "folga_dias": _to_float(item_row.get("folga_dias")),
                        **comparativo,
                    }
                )

                ocupacao_depois = float(manifesto.get("ocupacao_final_m6_2", 0) or 0)
                if ocupacao_depois >= ocupacao_alvo_perc:
                    break

            manifesto_atualizado = df_manifestos.loc[df_manifestos["manifesto_id"] == manifesto_id].head(1)
            if not manifesto_atualizado.empty:
                ocupacao_depois = float(manifesto_atualizado.iloc[0].get("ocupacao_final_m6_2", 0) or 0)
                if ocupacao_depois >= ocupacao_alvo_perc:
                    break

        if not houve_movimento_neste_manifesto:
            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "tipo_tentativa": "sem_movimento_aceito",
                    "nivel_hierarquia": None,
                    "aceito": False,
                    "motivo": "Nenhum item do remanescente do M5 pôde ser adicionado respeitando as restrições.",
                }
            )

    df_manifestos = _recalcular_todos_manifestos(df_manifestos, df_itens)

    _validar_integridade_final(
        df_itens_manifestos_final=df_itens,
        df_remanescente_final=df_remanescente,
        df_itens_manifestos_base_m6=df_itens_manifestos_base_m6,
        df_remanescente_original=df_remanescente_original,
    )

    df_tentativas = pd.DataFrame(tentativas)
    df_movimentos_aceitos = pd.DataFrame(movimentos_aceitos)

    resumo_m6_2 = {
        "modulo": "M6.2",
        "data_base_roteirizacao": data_base_roteirizacao.isoformat(),
        "tipo_roteirizacao": tipo_roteirizacao,
        "ocupacao_alvo_perc": float(ocupacao_alvo_perc),
        "manifestos_base_total_m6_1": int(len(df_manifestos_base_m6)),
        "itens_manifestos_base_total_m6_1": int(len(df_itens_manifestos_base_m6)),
        "remanescente_m5_original_total": int(len(df_remanescente_original)),
        "manifestos_alvo_abaixo_ocupacao_alvo": int(len(manifestos_alvo)),
        "movimentos_aceitos_m6_2": int(len(df_movimentos_aceitos)),
        "tentativas_total_m6_2": int(len(df_tentativas)),
        "itens_manifestos_total_m6_2": int(len(df_itens)),
        "itens_remanescente_m6_2": int(len(df_remanescente)),
        "itens_adicionados_a_manifestos_m6_2": int(
            len(df_itens.loc[df_itens["flag_otimizado_m6_2"] == True])
        ),
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    return {
        "outputs_m6_2": {
            "df_manifestos_m6_2": df_manifestos.reset_index(drop=True),
            "df_itens_manifestos_m6_2": df_itens.reset_index(drop=True),
            "df_remanescente_m6_2": df_remanescente.reset_index(drop=True),
            "df_remanescente_m5_original_m6_2": df_remanescente_original.reset_index(drop=True),
            "df_tentativas_m6_2": df_tentativas.reset_index(drop=True),
            "df_movimentos_aceitos_m6_2": df_movimentos_aceitos.reset_index(drop=True),
        },
        "resumo_m6_2": resumo_m6_2,
    }


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def _normalizar_manifestos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _validar_colunas_minimas(out, COLS_MANIFESTOS_OBRIGATORIAS, "df_manifestos_base_m6")

    out["manifesto_id"] = out["manifesto_id"].astype(str)
    out["origem_manifesto_modulo"] = out["origem_manifesto_modulo"].astype(str)
    out["origem_manifesto_tipo"] = out["origem_manifesto_tipo"].astype(str)
    out["veiculo_tipo"] = out["veiculo_tipo"].astype(str)
    out["veiculo_perfil"] = out["veiculo_perfil"].astype(str)

    cols_num = [
        "peso_base_antes_m6",
        "km_base_antes_m6",
        "ocupacao_base_antes_m6",
        "capacidade_peso_kg_veiculo",
        "max_km_distancia_veiculo",
        "qtd_itens_base_antes_m6",
        "qtd_ctes_base_antes_m6",
        "qtd_paradas_base_antes_m6",
    ]
    for col in cols_num:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["perfil_final_m6_2"] = out["veiculo_perfil"].replace("", np.nan).fillna(out["veiculo_tipo"])
    out["capacidade_peso_kg"] = out["capacidade_peso_kg_veiculo"]
    out["max_km_distancia"] = out["max_km_distancia_veiculo"]

    # ocupação oficial de entrada
    out["ocupacao_entrada_perc"] = out["ocupacao_base_antes_m6"]
    out["ocupacao_final_m6_2"] = out["ocupacao_base_antes_m6"]

    # placeholders de saída
    out["peso_final_m6_2"] = out["peso_base_antes_m6"]
    out["km_final_m6_2"] = out["km_base_antes_m6"]
    out["qtd_itens_final_m6_2"] = out["qtd_itens_base_antes_m6"]
    out["qtd_paradas_final_m6_2"] = out["qtd_paradas_base_antes_m6"]
    out["flag_otimizado_m6_2"] = False
    out["mesorregiao_manifesto"] = ""

    return out.reset_index(drop=True)


def _normalizar_estatisticas_m6(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _validar_colunas_minimas(out, COLS_ESTATS_OBRIGATORIAS, "df_estatisticas_manifestos_antes_m6")

    out["manifesto_id"] = out["manifesto_id"].astype(str)
    out["mesorregiao_manifesto_m6"] = out["mesorregiao_manifesto_m6"].astype(str)

    if "ocupacao_recalculada_antes_m6" in out.columns:
        out["ocupacao_recalculada_antes_m6"] = pd.to_numeric(out["ocupacao_recalculada_antes_m6"], errors="coerce")

    return out.reset_index(drop=True)


def _normalizar_itens_manifestos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _validar_colunas_minimas(out, COLS_ITENS_OBRIGATORIAS, "df_itens_manifestos_base_m6")

    out["manifesto_id"] = out["manifesto_id"].astype(str)
    out["id_linha_pipeline"] = out["id_linha_pipeline"].astype(str)
    out["destinatario"] = out["destinatario"].astype(str)
    out["cidade"] = out["cidade"].astype(str)
    out["uf"] = out["uf"].astype(str)
    out["subregiao"] = out["subregiao"].astype(str)
    out["mesorregiao"] = out["mesorregiao"].astype(str)
    out["restricao_veiculo"] = out["restricao_veiculo"].astype(str) if "restricao_veiculo" in out.columns else ""

    cols_num = ["distancia_rodoviaria_est_km", "peso_calculado", "peso_kg", "vol_m3"]
    for col in cols_num:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        else:
            out[col] = 0.0

    return out.reset_index(drop=True)


def _normalizar_remanescente(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _validar_colunas_minimas(out, COLS_REMANESCENTE_OBRIGATORIAS, "df_remanescente_m5_4")

    out["id_linha_pipeline"] = out["id_linha_pipeline"].astype(str)
    out["destinatario"] = out["destinatario"].astype(str)
    out["cidade"] = out["cidade"].astype(str)
    out["uf"] = out["uf"].astype(str)
    out["subregiao"] = out["subregiao"].astype(str)
    out["mesorregiao"] = out["mesorregiao"].astype(str)

    cols_num = ["distancia_rodoviaria_est_km", "peso_calculado", "vol_m3"]
    for col in cols_num:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    if "peso_kg" in out.columns:
        out["peso_kg"] = pd.to_numeric(out["peso_kg"], errors="coerce").fillna(0)
    else:
        out["peso_kg"] = 0.0

    if "restricao_veiculo" not in out.columns:
        out["restricao_veiculo"] = ""
    out["restricao_veiculo"] = out["restricao_veiculo"].astype(str)

    if "agendada" not in out.columns:
        out["agendada"] = False
    out["agendada"] = out["agendada"].apply(_to_bool)

    if "folga_dias" not in out.columns:
        out["folga_dias"] = np.nan
    out["folga_dias"] = pd.to_numeric(out["folga_dias"], errors="coerce")

    return out.reset_index(drop=True)


# ============================================================
# VALIDAÇÃO DE CONTRATO
# ============================================================

def _validar_colunas_minimas(df: pd.DataFrame, cols: List[str], nome_df: str) -> None:
    faltando = [c for c in cols if c not in df.columns]
    if faltando:
        raise Exception(f"{nome_df} sem colunas obrigatórias: {faltando}")


def _validar_entrada(
    df_manifestos: pd.DataFrame,
    df_estats: pd.DataFrame,
    df_itens: pd.DataFrame,
    df_remanescente: pd.DataFrame,
) -> None:
    if df_manifestos.empty:
        raise Exception("M6.2 recebeu df_manifestos_base_m6 vazio.")
    if df_estats.empty:
        raise Exception("M6.2 recebeu df_estatisticas_manifestos_antes_m6 vazio.")
    if df_itens.empty:
        raise Exception("M6.2 recebeu df_itens_manifestos_base_m6 vazio.")

    if df_itens["id_linha_pipeline"].duplicated().any():
        raise Exception("M6.2 recebeu duplicidade de id_linha_pipeline nos itens dos manifestos.")

    if df_remanescente["id_linha_pipeline"].duplicated().any():
        raise Exception("M6.2 recebeu duplicidade de id_linha_pipeline no remanescente do M5.")

    if df_estats["manifesto_id"].duplicated().any():
        # Se houver mais de uma linha por manifesto, a base estatística está ambígua para este uso
        raise Exception("M6.2 recebeu df_estatisticas_manifestos_antes_m6 com manifesto_id duplicado.")


def _enriquecer_manifestos_com_estatisticas(
    df_manifestos: pd.DataFrame,
    df_estats: pd.DataFrame,
) -> pd.DataFrame:
    out = df_manifestos.merge(
        df_estats[["manifesto_id", "mesorregiao_manifesto_m6"]],
        on="manifesto_id",
        how="left",
    )
    out["mesorregiao_manifesto"] = out["mesorregiao_manifesto_m6"].fillna("").astype(str)
    out.drop(columns=["mesorregiao_manifesto_m6"], inplace=True, errors="ignore")
    return out


# ============================================================
# LÓGICA DE NEGÓCIO
# ============================================================

def _selecionar_manifestos_alvo(df_manifestos: pd.DataFrame, ocupacao_alvo_perc: float) -> List[str]:
    base = df_manifestos.copy()
    base["ocupacao_entrada_perc"] = pd.to_numeric(base["ocupacao_entrada_perc"], errors="coerce")
    base = base.loc[base["ocupacao_entrada_perc"].notna()].copy()
    base = base.loc[base["ocupacao_entrada_perc"] < ocupacao_alvo_perc].copy()
    base = base.sort_values(by=["ocupacao_entrada_perc", "qtd_paradas_base_antes_m6"], ascending=[True, True])
    return base["manifesto_id"].astype(str).tolist()


def _cliente_dominante(df_itens_manifesto: pd.DataFrame) -> str:
    if df_itens_manifesto.empty:
        return ""
    vc = df_itens_manifesto["destinatario"].astype(str).str.strip().value_counts()
    return str(vc.index[0]).strip() if len(vc) > 0 else ""


def _cidade_dominante(df_itens_manifesto: pd.DataFrame) -> str:
    if df_itens_manifesto.empty:
        return ""
    vc = df_itens_manifesto["cidade"].astype(str).str.strip().value_counts()
    return str(vc.index[0]).strip() if len(vc) > 0 else ""


def _subregiao_dominante(df_itens_manifesto: pd.DataFrame) -> str:
    if df_itens_manifesto.empty:
        return ""
    vc = df_itens_manifesto["subregiao"].astype(str).str.strip().value_counts()
    return str(vc.index[0]).strip() if len(vc) > 0 else ""


def _selecionar_candidatos_por_hierarquia(
    rem_mesmo_meso: pd.DataFrame,
    nome_nivel: str,
    coluna_nivel: str,
    contexto_manifesto: Dict[str, Any],
) -> pd.DataFrame:
    base = rem_mesmo_meso.copy()
    if base.empty:
        return base

    if nome_nivel == "mesmo_cliente":
        alvo = _txt_norm(contexto_manifesto.get("cliente_dominante", ""))
        if alvo == "":
            return pd.DataFrame()
        return base.loc[base["destinatario"].astype(str).str.upper() == alvo].copy()

    if nome_nivel == "mesma_cidade":
        alvo = _txt_norm(contexto_manifesto.get("cidade_dominante", ""))
        if alvo == "":
            return pd.DataFrame()
        return base.loc[base["cidade"].astype(str).str.upper() == alvo].copy()

    if nome_nivel == "mesma_subregiao":
        alvo = _txt_norm(contexto_manifesto.get("subregiao_dominante", ""))
        if alvo == "":
            return pd.DataFrame()
        return base.loc[base["subregiao"].astype(str).str.upper() == alvo].copy()

    if nome_nivel == "mesma_mesorregiao":
        return base.copy()

    return pd.DataFrame()


def _ordenar_candidatos_por_prioridade_operacional(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["agendada"] = out["agendada"].fillna(False).astype(bool)
    out["folga_dias"] = pd.to_numeric(out["folga_dias"], errors="coerce")

    def _grupo_folga(valor: Any) -> int:
        if pd.isna(valor):
            return 2
        valor = float(valor)
        if valor >= 0:
            return 0
        return 1

    out["ord_agendada"] = np.where(out["agendada"] == True, 0, 1)
    out["ord_grupo_folga"] = out["folga_dias"].apply(_grupo_folga)
    out["ord_folga"] = out["folga_dias"].fillna(999999)

    out = out.sort_values(
        by=[
            "ord_agendada",
            "ord_grupo_folga",
            "ord_folga",
            "peso_calculado",
            "distancia_rodoviaria_est_km",
        ],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    return out


def _simular_adicao_item(
    manifesto: Dict[str, Any],
    itens_manifesto: pd.DataFrame,
    item_candidato: pd.DataFrame,
) -> Tuple[bool, str, Dict[str, Any]]:
    antes = _calcular_metricas_manifesto(manifesto, itens_manifesto)
    depois_df = pd.concat([itens_manifesto, item_candidato], ignore_index=True)
    depois = _calcular_metricas_manifesto(manifesto, depois_df)

    comparativo = {
        "ocupacao_antes_perc": round(float(antes["ocupacao_final_m6_2"]), 4),
        "ocupacao_depois_perc": round(float(depois["ocupacao_final_m6_2"]), 4),
        "ganho_ocupacao_perc": round(float(depois["ocupacao_final_m6_2"] - antes["ocupacao_final_m6_2"]), 4),
        "distancia_antes_km": round(float(antes["km_final_m6_2"]), 4),
        "distancia_depois_km": round(float(depois["km_final_m6_2"]), 4),
        "delta_distancia_km": round(float(depois["km_final_m6_2"] - antes["km_final_m6_2"]), 4),
        "qtd_paradas_antes": int(antes["qtd_paradas_final_m6_2"]),
        "qtd_paradas_depois": int(depois["qtd_paradas_final_m6_2"]),
    }

    restricoes = _validar_restricoes_manifesto(manifesto, depois_df)
    if restricoes is not True:
        return False, str(restricoes), comparativo

    if depois["ocupacao_final_m6_2"] <= antes["ocupacao_final_m6_2"]:
        return False, "A adição do item não melhora a ocupação do manifesto.", comparativo

    if depois["km_final_m6_2"] > antes["km_final_m6_2"]:
        return False, "A adição do item piora a distância máxima do manifesto.", comparativo

    ocup_max = _to_float(manifesto.get("ocupacao_maxima_perc"))
    if ocup_max is not None and depois["ocupacao_final_m6_2"] > ocup_max:
        return False, "A adição do item ultrapassa a ocupação máxima do manifesto.", comparativo

    return True, "Item aceito no complemento de ocupação.", comparativo


def _calcular_metricas_manifesto(manifesto: Dict[str, Any], df_itens: pd.DataFrame) -> Dict[str, Any]:
    capacidade_peso = _to_float(manifesto.get("capacidade_peso_kg"))
    if capacidade_peso is None or capacidade_peso <= 0:
        raise Exception(f"M6.2 encontrou manifesto sem capacidade_peso_kg válida: {manifesto.get('manifesto_id')}")

    peso_total = float(pd.to_numeric(df_itens["peso_calculado"], errors="coerce").fillna(0).sum())
    vol_total = float(pd.to_numeric(df_itens["vol_m3"], errors="coerce").fillna(0).sum())
    distancia_total = float(pd.to_numeric(df_itens["distancia_rodoviaria_est_km"], errors="coerce").fillna(0).max())
    qtd_itens = int(len(df_itens))
    qtd_paradas = int(_contar_entregas(df_itens))

    ocupacao = (peso_total / capacidade_peso) * 100.0

    return {
        "peso_final_m6_2": peso_total,
        "vol_total_m3": vol_total,
        "km_final_m6_2": distancia_total,
        "qtd_itens_final_m6_2": qtd_itens,
        "qtd_paradas_final_m6_2": qtd_paradas,
        "ocupacao_final_m6_2": ocupacao,
    }


def _validar_restricoes_manifesto(manifesto: Dict[str, Any], df_itens: pd.DataFrame) -> bool | str:
    metricas = _calcular_metricas_manifesto(manifesto, df_itens)

    capacidade_peso = _to_float(manifesto.get("capacidade_peso_kg"))
    max_km = _to_float(manifesto.get("max_km_distancia"))
    max_entregas = _to_int(manifesto.get("qtd_paradas_base_antes_m6"))  # fallback auditável; não trava se não existir
    meso_manifesto = _txt_norm(manifesto.get("mesorregiao_manifesto", ""))
    perfil_manifesto = _txt_norm(manifesto.get("perfil_final_m6_2", ""))

    if capacidade_peso is not None and metricas["peso_final_m6_2"] > capacidade_peso:
        return "Excede capacidade de peso."

    if max_km is not None and metricas["km_final_m6_2"] > max_km:
        return "Excede raio/distância máxima."

    if max_entregas is not None and metricas["qtd_paradas_final_m6_2"] > max_entregas:
        return "Excede máximo de paradas do manifesto."

    mesorregioes = set(df_itens["mesorregiao"].astype(str).str.strip().str.upper().tolist())
    mesorregioes.discard("")
    if len(mesorregioes) > 1:
        return "Mistura mais de uma mesorregião no manifesto."
    if len(mesorregioes) == 1 and list(mesorregioes)[0] != meso_manifesto:
        return "Item do remanescente está em mesorregião diferente do manifesto."

    restricoes = set(df_itens["restricao_veiculo"].astype(str).str.strip().str.upper().tolist())
    restricoes.discard("")
    if len(restricoes) > 1:
        return "Itens com restrições de veículo conflitantes."
    if len(restricoes) == 1 and perfil_manifesto != "" and list(restricoes)[0] != perfil_manifesto:
        return "Restrição de veículo do item é incompatível com o perfil do manifesto."

    return True


def _recalcular_manifesto_unico(
    df_manifestos: pd.DataFrame,
    df_itens: pd.DataFrame,
    manifesto_id: str,
) -> pd.DataFrame:
    out = df_manifestos.copy()
    idx = out.index[out["manifesto_id"].astype(str) == str(manifesto_id)]
    if len(idx) == 0:
        return out

    i = idx[0]
    manifesto = out.loc[i].to_dict()
    itens = df_itens.loc[df_itens["manifesto_id"].astype(str) == str(manifesto_id)].copy()
    metricas = _calcular_metricas_manifesto(manifesto, itens)

    out.loc[i, "peso_final_m6_2"] = metricas["peso_final_m6_2"]
    out.loc[i, "km_final_m6_2"] = metricas["km_final_m6_2"]
    out.loc[i, "qtd_itens_final_m6_2"] = metricas["qtd_itens_final_m6_2"]
    out.loc[i, "qtd_paradas_final_m6_2"] = metricas["qtd_paradas_final_m6_2"]
    out.loc[i, "ocupacao_final_m6_2"] = metricas["ocupacao_final_m6_2"]
    out.loc[i, "flag_otimizado_m6_2"] = bool(
        metricas["qtd_itens_final_m6_2"] > _to_int(out.loc[i, "qtd_itens_base_antes_m6"], default=0)
    )
    return out


def _recalcular_todos_manifestos(df_manifestos: pd.DataFrame, df_itens: pd.DataFrame) -> pd.DataFrame:
    out = df_manifestos.copy()
    for manifesto_id in out["manifesto_id"].astype(str).tolist():
        out = _recalcular_manifesto_unico(out, df_itens, manifesto_id)
    return out.reset_index(drop=True)


# ============================================================
# INTEGRIDADE
# ============================================================

def _validar_integridade_final(
    df_itens_manifestos_final: pd.DataFrame,
    df_remanescente_final: pd.DataFrame,
    df_itens_manifestos_base_m6: pd.DataFrame,
    df_remanescente_original: pd.DataFrame,
) -> None:
    ids_base_manifestos = set(df_itens_manifestos_base_m6["id_linha_pipeline"].astype(str))
    ids_base_remanescente = set(df_remanescente_original["id_linha_pipeline"].astype(str))
    universo_base = ids_base_manifestos.union(ids_base_remanescente)

    ids_final_manifestos = set(df_itens_manifestos_final["id_linha_pipeline"].astype(str))
    ids_final_remanescente = set(df_remanescente_final["id_linha_pipeline"].astype(str))
    universo_final = ids_final_manifestos.union(ids_final_remanescente)

    if universo_base != universo_final:
        faltando = list(universo_base - universo_final)[:20]
        sobrando = list(universo_final - universo_base)[:20]
        raise Exception(
            f"M6.2 violou integridade do universo de itens. Faltando={faltando} | Sobrando={sobrando}"
        )

    if df_itens_manifestos_final["id_linha_pipeline"].duplicated().any():
        raise Exception("M6.2 gerou itens duplicados nos manifestos finais.")

    if df_remanescente_final["id_linha_pipeline"].duplicated().any():
        raise Exception("M6.2 gerou itens duplicados no remanescente final.")

    intersec = ids_final_manifestos.intersection(ids_final_remanescente)
    if intersec:
        raise Exception(
            f"M6.2 deixou itens ao mesmo tempo no manifesto e no remanescente: {list(intersec)[:20]}"
        )


# ============================================================
# HELPERS
# ============================================================

def _to_bool(valor: Any) -> bool:
    if isinstance(valor, bool):
        return valor
    if pd.isna(valor):
        return False
    txt = str(valor).strip().lower()
    return txt in {"1", "true", "sim", "s", "yes", "y", "agendada"}


def _txt_norm(valor: Any) -> str:
    if pd.isna(valor):
        return ""
    return str(valor).strip().upper()


def _to_float(valor: Any, default: Optional[float] = None) -> Optional[float]:
    x = pd.to_numeric(valor, errors="coerce")
    if pd.isna(x):
        return default
    return float(x)


def _to_int(valor: Any, default: Optional[int] = None) -> Optional[int]:
    x = pd.to_numeric(valor, errors="coerce")
    if pd.isna(x):
        return default
    return int(x)


def _contar_entregas(df: pd.DataFrame) -> int:
    cols = [c for c in CHAVES_PARADA if c in df.columns]
    if len(cols) < 3:
        return int(len(df))

    chave = (
        df["destinatario"].astype(str).str.strip().str.upper()
        + "|"
        + df["cidade"].astype(str).str.strip().str.upper()
        + "|"
        + df["uf"].astype(str).str.strip().str.upper()
    )
    return int(chave.nunique())
