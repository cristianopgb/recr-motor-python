from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, Tuple

import pandas as pd


COLUNAS_MINIMAS_M31 = [
    "status_triagem",
    "grupo_saida",
    "agendada",
    "folga_dias",
    "peso_kg",
    "vol_m3",
    "distancia_rodoviaria_est_km",
    "destinatario",
    "cidade",
    "uf",
]


COLUNAS_BASE_HASH_PREFERENCIA = [
    "nro_documento",
    "romaneio",
    "serie_romaneio",
    "serie",
    "filial_roteirizacao",
    "filial_origem",
    "destinatario",
    "cidade",
    "uf",
    "peso_kg",
    "vol_m3",
    "data_agenda",
    "data_leadtime",
]


def executar_m3_1_validacao_fronteira(
    df_carteira_roteirizavel: pd.DataFrame,
    data_base_roteirizacao: datetime,
    caminhos_pipeline: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    M3.1 real adaptado ao Sistema 2.

    Objetivo:
    - Receber somente a carteira roteirizável do M3
    - Validar ausência de contaminação na fronteira do bloco 4
    - Criar a chave técnica oficial id_linha_pipeline
    - Devolver o dataframe oficial de entrada do bloco 4
    """

    df_input = df_carteira_roteirizavel.copy().reset_index(drop=True)
    _validar_colunas_minimas(df_input)
    _tipagem_defensiva(df_input)
    _validacoes_duras(df_input)

    colunas_base_hash = [c for c in COLUNAS_BASE_HASH_PREFERENCIA if c in df_input.columns]
    if len(colunas_base_hash) < 6:
        raise Exception(
            "Não há colunas suficientes para criar a chave técnica oficial do pipeline na fronteira do bloco 4."
        )

    df_input["id_linha_pipeline"] = df_input.apply(
        lambda row: _gerar_id_linha_pipeline(row, colunas_base_hash), axis=1
    )

    if df_input["id_linha_pipeline"].duplicated().any():
        duplicados = int(df_input["id_linha_pipeline"].duplicated().sum())
        raise Exception(
            f"A chave técnica id_linha_pipeline ficou duplicada em {duplicados} linhas. "
            "O contrato do pipeline precisa ser único nesta fronteira."
        )

    colunas_finais = ["id_linha_pipeline"] + [c for c in df_input.columns if c != "id_linha_pipeline"]
    df_input_oficial_bloco_4 = df_input[colunas_finais].copy()

    resumo_m31 = {
        "modulo": "m3_1_validacao_fronteira",
        "data_execucao": datetime.utcnow().isoformat(),
        "data_base_roteirizacao": data_base_roteirizacao.isoformat(),
        "linhas_input": int(len(df_input_oficial_bloco_4)),
        "colunas_input": int(len(df_input_oficial_bloco_4.columns)),
        "agendadas_validas": int((df_input_oficial_bloco_4["agendada"] == True).sum()),
        "nao_agendadas": int((df_input_oficial_bloco_4["agendada"] == False).sum()),
        "peso_nulo": int(df_input_oficial_bloco_4["peso_kg"].isna().sum()),
        "volume_nulo": int(df_input_oficial_bloco_4["vol_m3"].isna().sum()),
        "km_nulo": int(df_input_oficial_bloco_4["distancia_rodoviaria_est_km"].isna().sum()),
        "ids_tecnicos_unicos": int(df_input_oficial_bloco_4["id_linha_pipeline"].nunique()),
        "colunas_base_hash": colunas_base_hash,
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    return df_input_oficial_bloco_4, {
        "resumo_m31": resumo_m31,
        "outputs_m31": {
            "df_input_oficial_bloco_4": df_input_oficial_bloco_4,
        },
    }


def _validar_colunas_minimas(df: pd.DataFrame) -> None:
    faltam = [c for c in COLUNAS_MINIMAS_M31 if c not in df.columns]
    if faltam:
        raise Exception(
            "Faltam colunas mínimas no input do Bloco 4:\n- " + "\n- ".join(faltam)
        )


def _tipagem_defensiva(df: pd.DataFrame) -> None:
    df["status_triagem"] = df["status_triagem"].astype(str).str.strip()
    df["grupo_saida"] = df["grupo_saida"].astype(str).str.strip()
    df["agendada"] = df["agendada"].fillna(False).astype(bool)
    df["folga_dias"] = pd.to_numeric(df["folga_dias"], errors="coerce")
    df["peso_kg"] = pd.to_numeric(df["peso_kg"], errors="coerce")
    df["vol_m3"] = pd.to_numeric(df["vol_m3"], errors="coerce")
    df["distancia_rodoviaria_est_km"] = pd.to_numeric(df["distancia_rodoviaria_est_km"], errors="coerce")


def _validacoes_duras(df: pd.DataFrame) -> None:
    problemas: list[str] = []

    invalidas_status = df.loc[df["status_triagem"] != "roteirizavel"]
    if len(invalidas_status) > 0:
        problemas.append(
            f"Linhas com status_triagem diferente de 'roteirizavel': {len(invalidas_status)}"
        )

    invalidas_grupo = df.loc[df["grupo_saida"] != "df_carteira_roteirizavel"]
    if len(invalidas_grupo) > 0:
        problemas.append(
            f"Linhas com grupo_saida diferente de 'df_carteira_roteirizavel': {len(invalidas_grupo)}"
        )

    agendadas_invalidas = df.loc[
        (df["agendada"] == True)
        & (
            df["folga_dias"].isna()
            | (df["folga_dias"] < 0)
            | (df["folga_dias"] > 1)
        )
    ]
    if len(agendadas_invalidas) > 0:
        problemas.append(
            f"Linhas agendadas fora da faixa permitida (0 a 1): {len(agendadas_invalidas)}"
        )

    linhas_sem_peso = int(df["peso_kg"].isna().sum())
    linhas_sem_vol = int(df["vol_m3"].isna().sum())
    linhas_sem_km = int(df["distancia_rodoviaria_est_km"].isna().sum())

    if linhas_sem_peso > 0:
        problemas.append(f"Linhas sem peso_kg: {linhas_sem_peso}")
    if linhas_sem_vol > 0:
        problemas.append(f"Linhas sem vol_m3: {linhas_sem_vol}")
    if linhas_sem_km > 0:
        problemas.append(f"Linhas sem distancia_rodoviaria_est_km: {linhas_sem_km}")

    if problemas:
        raise Exception("A fronteira de input do Bloco 4 falhou:\n- " + "\n- ".join(problemas))


def _gerar_id_linha_pipeline(row: pd.Series, colunas_base_hash: list[str]) -> str:
    partes: list[str] = []
    for coluna in colunas_base_hash:
        valor = row[coluna]
        if pd.isna(valor):
            partes.append("")
        elif isinstance(valor, pd.Timestamp):
            partes.append(valor.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            partes.append(str(valor).strip())

    payload = "||".join(partes)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
