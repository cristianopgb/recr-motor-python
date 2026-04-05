from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def executar_m3_triagem(
    df_carteira_enriquecida: pd.DataFrame,
    data_base_roteirizacao: datetime,
    caminhos_pipeline: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    M3 real adaptado ao Sistema 2 (API).

    Regras de triagem:
    - agendada sem data_agenda -> aguardando_agendamento
    - nao agendada -> roteirizavel
    - agendada com folga entre 0 e 1 -> roteirizavel
    - agendada com folga >= 2 -> entrega_futura
    - demais cenários -> excecao_triagem

    Regra crítica ajustada:
    - folga_dias == 2 entra em entrega_futura
    """

    carteira = df_carteira_enriquecida.copy()

    _validar_colunas_minimas(carteira)

    carteira["agendada"] = carteira["agendada"].fillna(False).astype(bool)
    carteira["data_agenda"] = pd.to_datetime(carteira["data_agenda"], errors="coerce")
    carteira["data_leadtime"] = pd.to_datetime(carteira["data_leadtime"], errors="coerce")
    carteira["data_limite_considerada"] = pd.to_datetime(carteira["data_limite_considerada"], errors="coerce")
    carteira["folga_dias"] = pd.to_numeric(carteira["folga_dias"], errors="coerce")
    carteira["transit_time_dias"] = pd.to_numeric(carteira["transit_time_dias"], errors="coerce")
    carteira["dias_ate_data_alvo"] = pd.to_numeric(carteira["dias_ate_data_alvo"], errors="coerce")

    if "score_prioridade_preliminar" not in carteira.columns:
        carteira["score_prioridade_preliminar"] = 0

    if "ranking_preliminar" not in carteira.columns:
        carteira["ranking_preliminar"] = pd.Series(range(1, len(carteira) + 1), index=carteira.index)

    if "distancia_rodoviaria_est_km" not in carteira.columns:
        carteira["distancia_rodoviaria_est_km"] = np.nan

    carteira["score_prioridade_preliminar"] = pd.to_numeric(
        carteira["score_prioridade_preliminar"], errors="coerce"
    ).fillna(0)

    carteira["ranking_preliminar"] = pd.to_numeric(
        carteira["ranking_preliminar"], errors="coerce"
    )

    carteira["status_triagem"] = carteira.apply(_classificar_status_triagem, axis=1)
    carteira["motivo_triagem"] = carteira.apply(_definir_motivo_triagem, axis=1)
    carteira["grupo_saida"] = carteira["status_triagem"].apply(_definir_grupo_saida)
    carteira["prioridade_label"] = carteira.apply(_definir_prioridade_label, axis=1)
    carteira["ranking_prioridade_operacional"] = carteira.apply(_definir_ranking_operacional, axis=1)

    carteira["flag_roteirizavel"] = carteira["status_triagem"].eq("roteirizavel")
    carteira["flag_entrega_futura"] = carteira["status_triagem"].eq("entrega_futura")
    carteira["flag_aguardando_agendamento"] = carteira["status_triagem"].eq("aguardando_agendamento")
    carteira["flag_excecao_triagem"] = carteira["status_triagem"].eq("excecao_triagem")

    df_carteira_triagem = carteira.copy()

    df_carteira_roteirizavel = (
        carteira.loc[carteira["status_triagem"] == "roteirizavel"]
        .sort_values(
            by=[
                "ranking_prioridade_operacional",
                "score_prioridade_preliminar",
                "distancia_rodoviaria_est_km",
            ],
            ascending=[True, False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    df_carteira_entrega_futura = (
        carteira.loc[carteira["status_triagem"] == "entrega_futura"]
        .sort_values(
            by=["data_limite_considerada", "score_prioridade_preliminar"],
            ascending=[True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    df_carteira_aguardando_agendamento = (
        carteira.loc[carteira["status_triagem"] == "aguardando_agendamento"]
        .sort_values(
            by=["score_prioridade_preliminar", "distancia_rodoviaria_est_km"],
            ascending=[False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    df_carteira_excecoes_triagem = (
        carteira.loc[carteira["status_triagem"] == "excecao_triagem"]
        .sort_values(
            by=["score_prioridade_preliminar", "distancia_rodoviaria_est_km"],
            ascending=[False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    _validar_integridade_fechamento(
        df_entrada=carteira,
        df_carteira_roteirizavel=df_carteira_roteirizavel,
        df_carteira_entrega_futura=df_carteira_entrega_futura,
        df_carteira_aguardando_agendamento=df_carteira_aguardando_agendamento,
        df_carteira_excecoes_triagem=df_carteira_excecoes_triagem,
    )

    resumo = _montar_resumo_m3(
        df_carteira_triagem=df_carteira_triagem,
        df_carteira_roteirizavel=df_carteira_roteirizavel,
        df_carteira_entrega_futura=df_carteira_entrega_futura,
        df_carteira_aguardando_agendamento=df_carteira_aguardando_agendamento,
        df_carteira_excecoes_triagem=df_carteira_excecoes_triagem,
        data_base_roteirizacao=data_base_roteirizacao,
        caminhos_pipeline=caminhos_pipeline or {},
    )

    resultado = {
        "df_carteira_triagem": df_carteira_triagem,
        "df_carteira_roteirizavel": df_carteira_roteirizavel,
        "df_carteira_entrega_futura": df_carteira_entrega_futura,
        "df_carteira_aguardando_agendamento": df_carteira_aguardando_agendamento,
        "df_carteira_excecoes_triagem": df_carteira_excecoes_triagem,
    }

    return df_carteira_triagem, {
        "resumo_m3": resumo,
        "outputs_m3": resultado,
    }


def _validar_colunas_minimas(df: pd.DataFrame) -> None:
    colunas_minimas = [
        "agendada",
        "data_agenda",
        "data_leadtime",
        "data_limite_considerada",
        "tipo_data_limite",
        "dias_ate_data_alvo",
        "transit_time_dias",
        "folga_dias",
        "status_folga",
    ]

    faltam = [c for c in colunas_minimas if c not in df.columns]
    if faltam:
        raise Exception(
            "Faltam colunas mínimas na carteira enriquecida para executar o M3:\n- "
            + "\n- ".join(faltam)
        )


def _classificar_status_triagem(row: pd.Series) -> str:
    agendada = bool(row["agendada"])
    data_agenda = row["data_agenda"]
    folga = row["folga_dias"]

    if agendada and pd.isna(data_agenda):
        return "aguardando_agendamento"

    if not agendada:
        return "roteirizavel"

    if agendada and pd.notna(folga) and 0 <= folga <= 1:
        return "roteirizavel"

    if agendada and pd.notna(folga) and folga >= 2:
        return "entrega_futura"

    return "excecao_triagem"


def _definir_motivo_triagem(row: pd.Series) -> str:
    status = row["status_triagem"]
    agendada = bool(row["agendada"])
    folga = row["folga_dias"]

    if status == "roteirizavel" and not agendada:
        return "nao_agendada_entra_na_carteira_roteirizavel"

    if status == "roteirizavel" and agendada:
        return "agendada_com_folga_entre_0_e_1"

    if status == "entrega_futura":
        if pd.notna(folga) and folga == 2:
            return "agendada_com_folga_igual_a_2"
        return "agendada_com_folga_maior_ou_igual_a_2"

    if status == "aguardando_agendamento":
        return "marcada_como_agendada_sem_data_agenda"

    if status == "excecao_triagem":
        if agendada and pd.isna(folga):
            return "agendada_sem_folga_calculada"
        if agendada and pd.notna(folga) and folga < 0:
            return "agendada_inviavel_para_agenda_atual"
        return "cenario_nao_mapeado_pela_regra_atual"

    return "sem_motivo"


def _definir_grupo_saida(status: str) -> str:
    if status == "roteirizavel":
        return "df_carteira_roteirizavel"
    if status == "entrega_futura":
        return "df_carteira_entrega_futura"
    if status == "aguardando_agendamento":
        return "df_carteira_aguardando_agendamento"
    return "df_carteira_excecoes_triagem"


def _definir_prioridade_label(row: pd.Series) -> str:
    status = row["status_triagem"]
    agendada = bool(row["agendada"])
    folga = row["folga_dias"]

    if status != "roteirizavel":
        return "fora_da_carteira_roteirizavel"

    if agendada:
        return "prioridade_1_agendada"

    if pd.notna(folga) and folga <= 0:
        return "prioridade_2_leadtime_critico"

    if pd.notna(folga) and folga > 0:
        return "prioridade_3_leadtime_com_folga"

    return "prioridade_sem_classificacao"


def _definir_ranking_operacional(row: pd.Series) -> int:
    status = row["status_triagem"]
    agendada = bool(row["agendada"])
    folga = row["folga_dias"]

    if status != "roteirizavel":
        return 9
    if agendada:
        return 1
    if pd.notna(folga) and folga <= 0:
        return 2
    if pd.notna(folga) and folga > 0:
        return 3
    return 9


def _validar_integridade_fechamento(
    df_entrada: pd.DataFrame,
    df_carteira_roteirizavel: pd.DataFrame,
    df_carteira_entrega_futura: pd.DataFrame,
    df_carteira_aguardando_agendamento: pd.DataFrame,
    df_carteira_excecoes_triagem: pd.DataFrame,
) -> None:
    qtd_entrada = len(df_entrada)
    qtd_saida = (
        len(df_carteira_roteirizavel)
        + len(df_carteira_entrega_futura)
        + len(df_carteira_aguardando_agendamento)
        + len(df_carteira_excecoes_triagem)
    )

    if qtd_entrada != qtd_saida:
        raise Exception(
            f"Falha de integridade do M3: entrada={qtd_entrada} e saída={qtd_saida}."
        )

    violacoes_roteirizavel = df_carteira_roteirizavel.loc[
        (df_carteira_roteirizavel["agendada"] == True)
        & (
            df_carteira_roteirizavel["folga_dias"].isna()
            | (df_carteira_roteirizavel["folga_dias"] < 0)
            | (df_carteira_roteirizavel["folga_dias"] > 1)
        )
    ]

    if len(violacoes_roteirizavel) > 0:
        raise Exception(
            "A carteira roteirizável ficou contaminada com linhas agendadas fora da faixa permitida (0 a 1)."
        )

    violacoes_entrega_futura = df_carteira_entrega_futura.loc[
        ~(
            (df_carteira_entrega_futura["agendada"] == True)
            & (df_carteira_entrega_futura["folga_dias"] >= 2)
        )
    ]

    if len(violacoes_entrega_futura) > 0:
        raise Exception(
            "A carteira de entrega futura ficou com linhas incompatíveis com a regra (agendada com folga >= 2)."
        )


def _montar_resumo_m3(
    df_carteira_triagem: pd.DataFrame,
    df_carteira_roteirizavel: pd.DataFrame,
    df_carteira_entrega_futura: pd.DataFrame,
    df_carteira_aguardando_agendamento: pd.DataFrame,
    df_carteira_excecoes_triagem: pd.DataFrame,
    data_base_roteirizacao: datetime,
    caminhos_pipeline: Dict[str, Any],
) -> Dict[str, Any]:
    status_counts = (
        df_carteira_triagem["status_triagem"]
        .fillna("sem_classificacao")
        .value_counts(dropna=False)
        .to_dict()
    )

    prioridade_counts = (
        df_carteira_roteirizavel["prioridade_label"]
        .fillna("sem_classificacao")
        .value_counts(dropna=False)
        .to_dict()
    )

    qtd_folga_2_futura = int(
        (
            (df_carteira_entrega_futura["agendada"] == True)
            & (pd.to_numeric(df_carteira_entrega_futura["folga_dias"], errors="coerce") == 2)
        ).sum()
    )

    return {
        "modulo": "M3",
        "data_base_roteirizacao": pd.to_datetime(data_base_roteirizacao).isoformat(),
        "linhas_entrada": int(len(df_carteira_triagem)),
        "linhas_saida_total": int(len(df_carteira_triagem)),
        "carteira_roteirizavel": int(len(df_carteira_roteirizavel)),
        "carteira_entrega_futura": int(len(df_carteira_entrega_futura)),
        "carteira_aguardando_agendamento": int(len(df_carteira_aguardando_agendamento)),
        "carteira_excecoes_triagem": int(len(df_carteira_excecoes_triagem)),
        "agendadas_na_roteirizavel": int((df_carteira_roteirizavel["agendada"] == True).sum()),
        "nao_agendadas_na_roteirizavel": int((df_carteira_roteirizavel["agendada"] == False).sum()),
        "agendadas_folga_igual_2_em_entrega_futura": qtd_folga_2_futura,
        "status_triagem_counts": status_counts,
        "prioridade_roteirizavel_counts": prioridade_counts,
        "regra_folga_agendada_entrega_futura": "folga_dias >= 2",
        "caminhos_pipeline": caminhos_pipeline,
    }
