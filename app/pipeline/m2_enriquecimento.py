from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


FATOR_KM_RODOVIARIO_PADRAO = 1.20
KM_MINIMO_OPERACIONAL = 5.0
VELOCIDADE_MEDIA_KM_H_PADRAO = 50.0
HORAS_DIRECAO_DIA_PADRAO = 8.0
KM_DIA_OPERACIONAL_PADRAO = VELOCIDADE_MEDIA_KM_H_PADRAO * HORAS_DIRECAO_DIA_PADRAO


def executar_m2_enriquecimento(
    df_carteira_tratada: pd.DataFrame,
    df_geo_tratado: pd.DataFrame,
    df_parametros_tratados: pd.DataFrame,
    data_base_roteirizacao: datetime,
    caminhos_pipeline: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    M2 real adaptado ao Sistema 2 (API).

    Regras:
    - origem = filial da rodada
    - data base = data_base_roteirizacao
    - transit time = ceil(km_rodoviario / 400)
    - regionalidades NÃO precisam ter latitude/longitude
    - latitude/longitude de destino vêm da carteira
    """

    carteira = df_carteira_tratada.copy()
    geo = df_geo_tratado.copy()
    parametros = df_parametros_tratados.copy()

    _validar_colunas_minimas(carteira, geo)

    for c in [
        "latitude_filial",
        "longitude_filial",
        "latitude_destinatario",
        "longitude_destinatario",
    ]:
        carteira[c] = pd.to_numeric(carteira[c], errors="coerce")

    for c in ["data_agenda", "data_leadtime"]:
        if c in carteira.columns:
            carteira[c] = pd.to_datetime(carteira[c], errors="coerce", dayfirst=True)

    data_base = pd.to_datetime(data_base_roteirizacao, errors="coerce")
    if pd.isna(data_base):
        raise Exception("data_base_roteirizacao inválida no M2.")

    fator_km_rodoviario = _obter_valor_parametro(
        parametros,
        ["fator_km_rodoviario", "fator_rodoviario", "fator_km_rodoviario_estimado"],
        FATOR_KM_RODOVIARIO_PADRAO,
    )

    velocidade_media_km_h = _obter_valor_parametro(
        parametros,
        ["velocidade_media_km_h"],
        VELOCIDADE_MEDIA_KM_H_PADRAO,
    )

    horas_direcao_dia = _obter_valor_parametro(
        parametros,
        ["horas_direcao_dia"],
        HORAS_DIRECAO_DIA_PADRAO,
    )

    km_dia_operacional = _obter_valor_parametro(
        parametros,
        ["km_dia_operacional", "km_dia_max", "km_max_dia", "km_dia"],
        KM_DIA_OPERACIONAL_PADRAO,
    )

    if pd.isna(km_dia_operacional) or float(km_dia_operacional) <= 0:
        km_dia_operacional = float(velocidade_media_km_h) * float(horas_direcao_dia)

    if pd.isna(km_dia_operacional) or float(km_dia_operacional) <= 0:
        km_dia_operacional = KM_DIA_OPERACIONAL_PADRAO

    fator_km_rodoviario = float(fator_km_rodoviario)
    velocidade_media_km_h = float(velocidade_media_km_h)
    horas_direcao_dia = float(horas_direcao_dia)
    km_dia_operacional = float(km_dia_operacional)

    carteira["origem_latitude"] = carteira["latitude_filial"]
    carteira["origem_longitude"] = carteira["longitude_filial"]

    carteira["distancia_km"] = carteira.apply(
        lambda row: _haversine_km(
            row["origem_latitude"],
            row["origem_longitude"],
            row["latitude_destinatario"],
            row["longitude_destinatario"],
        ),
        axis=1,
    )

    carteira["distancia_km"] = carteira["distancia_km"].clip(lower=KM_MINIMO_OPERACIONAL)
    carteira["distancia_rodoviaria_est_km"] = (
        carteira["distancia_km"] * fator_km_rodoviario
    ).round(2)

    carteira["horas_viagem_estimadas"] = (
        carteira["distancia_rodoviaria_est_km"] / velocidade_media_km_h
    ).round(2)

    carteira["transit_time_dias"] = carteira["distancia_rodoviaria_est_km"].apply(
        lambda x: _calcular_transit_time_dias(x, km_dia_operacional)
    )

    carteira["faixa_km_cd"] = carteira["distancia_rodoviaria_est_km"].apply(_classificar_faixa_km)

    carteira["quadrante"] = carteira.apply(
        lambda row: _classificar_quadrante(
            row["origem_latitude"],
            row["origem_longitude"],
            row["latitude_destinatario"],
            row["longitude_destinatario"],
        ),
        axis=1,
    )

    carteira["data_limite_considerada"] = np.where(
        (carteira["agendada"] == True) & (carteira["data_agenda"].notna()),
        carteira["data_agenda"],
        carteira["data_leadtime"],
    )
    carteira["data_limite_considerada"] = pd.to_datetime(
        carteira["data_limite_considerada"], errors="coerce"
    )

    carteira["tipo_data_limite"] = np.where(
        (carteira["agendada"] == True) & (carteira["data_agenda"].notna()),
        "agenda",
        np.where(carteira["data_leadtime"].notna(), "leadtime", "sem_data"),
    )

    carteira["data_base_roteirizacao"] = data_base
    carteira["dias_ate_data_alvo"] = (
        carteira["data_limite_considerada"].dt.normalize() - data_base.normalize()
    ).dt.days

    carteira["folga_dias"] = carteira["dias_ate_data_alvo"] - carteira["transit_time_dias"]
    carteira["status_folga"] = carteira["folga_dias"].apply(_classificar_status_folga)

    # regionalidades: só cidade/uf/meso/micro
    geo["_cidade_norm"] = geo["nome"].apply(_normalizar_texto)
    geo["_uf_norm"] = geo["uf"].apply(_normalizar_texto)

    carteira["_cidade_norm"] = carteira["cidade"].apply(_normalizar_texto)
    carteira["_uf_norm"] = carteira["uf"].apply(_normalizar_texto)

    geo_chaves = geo[
        ["_cidade_norm", "_uf_norm", "mesorregiao", "microrregiao"]
    ].drop_duplicates()

    if "mesorregiao" not in carteira.columns:
        carteira["mesorregiao"] = np.nan
    if "microrregiao" not in carteira.columns:
        carteira["microrregiao"] = np.nan

    carteira = carteira.merge(
        geo_chaves,
        how="left",
        on=["_cidade_norm", "_uf_norm"],
        suffixes=("_carteira", "_geo"),
    )

    col_meso_carteira = "mesorregiao_carteira" if "mesorregiao_carteira" in carteira.columns else None
    col_meso_geo = "mesorregiao_geo" if "mesorregiao_geo" in carteira.columns else None
    col_micro_carteira = "microrregiao_carteira" if "microrregiao_carteira" in carteira.columns else None
    col_micro_geo = "microrregiao_geo" if "microrregiao_geo" in carteira.columns else None

    if col_meso_carteira is not None or col_meso_geo is not None:
        carteira["mesorregiao"] = carteira.apply(
            lambda row: _primeira_nao_nula(
                row[col_meso_carteira] if col_meso_carteira else np.nan,
                row[col_meso_geo] if col_meso_geo else np.nan,
            ),
            axis=1,
        )

    if col_micro_carteira is not None or col_micro_geo is not None:
        carteira["microrregiao"] = carteira.apply(
            lambda row: _primeira_nao_nula(
                row[col_micro_carteira] if col_micro_carteira else np.nan,
                row[col_micro_geo] if col_micro_geo else np.nan,
            ),
            axis=1,
        )

    carteira["status_geo"] = np.where(
        carteira["mesorregiao"].notna() & carteira["microrregiao"].notna(),
        "ok",
        "pendencia_geo",
    )

    carteira["perfil_veiculo_referencia"] = carteira["distancia_rodoviaria_est_km"].apply(
        _classificar_perfil_veiculo_referencia
    )

    carteira["score_prioridade_preliminar"] = carteira.apply(_calcular_score, axis=1)
    carteira["ranking_preliminar"] = (
        carteira["score_prioridade_preliminar"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )

    colunas_descartar = [
        "_cidade_norm",
        "_uf_norm",
        "mesorregiao_carteira",
        "mesorregiao_geo",
        "microrregiao_carteira",
        "microrregiao_geo",
    ]
    carteira.drop(columns=[c for c in colunas_descartar if c in carteira.columns], inplace=True, errors="ignore")

    colunas_ordem_preferencial = [
        "ranking_preliminar",
        "score_prioridade_preliminar",
        "filial_roteirizacao",
        "romaneio",
        "filial_origem",
        "serie_romaneio",
        "nro_documento",
        "embarcador",
        "destinatario",
        "cidade",
        "uf",
        "mesorregiao",
        "microrregiao",
        "latitude_filial",
        "longitude_filial",
        "latitude_destinatario",
        "longitude_destinatario",
        "peso_kg",
        "vol_m3",
        "agendada",
        "data_agenda",
        "data_leadtime",
        "data_limite_considerada",
        "tipo_data_limite",
        "data_base_roteirizacao",
        "dias_ate_data_alvo",
        "horas_viagem_estimadas",
        "transit_time_dias",
        "folga_dias",
        "status_folga",
        "distancia_km",
        "distancia_rodoviaria_est_km",
        "faixa_km_cd",
        "quadrante",
        "perfil_veiculo_referencia",
        "status_geo",
    ]

    colunas_existentes = [c for c in colunas_ordem_preferencial if c in carteira.columns]
    colunas_restantes = [c for c in carteira.columns if c not in colunas_existentes]

    df_carteira_enriquecida = carteira[colunas_existentes + colunas_restantes].copy()

    _validar_saida_m2(df_carteira_enriquecida)

    resumo = {
        "linhas_carteira": int(len(df_carteira_enriquecida)),
        "colunas_carteira": int(len(df_carteira_enriquecida.columns)),
        "fator_km_rodoviario": fator_km_rodoviario,
        "velocidade_media_km_h": velocidade_media_km_h,
        "horas_direcao_dia": horas_direcao_dia,
        "km_dia_operacional": km_dia_operacional,
        "distancia_km_nulos": int(df_carteira_enriquecida["distancia_km"].isna().sum()),
        "transit_time_nulos": int(df_carteira_enriquecida["transit_time_dias"].isna().sum()),
        "folga_nulos": int(df_carteira_enriquecida["folga_dias"].isna().sum()),
        "mesorregiao_nulos": int(df_carteira_enriquecida["mesorregiao"].isna().sum()),
        "microrregiao_nulos": int(df_carteira_enriquecida["microrregiao"].isna().sum()),
        "status_geo_ok": int((df_carteira_enriquecida["status_geo"] == "ok").sum()),
        "status_geo_pendencia": int((df_carteira_enriquecida["status_geo"] == "pendencia_geo").sum()),
    }

    return df_carteira_enriquecida, resumo


def _validar_colunas_minimas(carteira: pd.DataFrame, geo: pd.DataFrame) -> None:
    colunas_minimas_carteira = [
        "cidade",
        "uf",
        "latitude_filial",
        "longitude_filial",
        "latitude_destinatario",
        "longitude_destinatario",
        "agendada",
        "data_agenda",
        "data_leadtime",
    ]
    faltam_carteira = [c for c in colunas_minimas_carteira if c not in carteira.columns]
    if faltam_carteira:
        raise Exception(
            "Faltam colunas mínimas na carteira tratada para o M2:\n- " +
            "\n- ".join(faltam_carteira)
        )

    # regionalidades no contrato atual NÃO têm latitude/longitude
    colunas_minimas_geo = ["nome", "uf", "mesorregiao", "microrregiao"]
    faltam_geo = [c for c in colunas_minimas_geo if c not in geo.columns]
    if faltam_geo:
        raise Exception(
            "Faltam colunas mínimas na base geográfica tratada para o M2:\n- " +
            "\n- ".join(faltam_geo)
        )


def _validar_saida_m2(df: pd.DataFrame) -> None:
    colunas_obrigatorias_saida = [
        "mesorregiao",
        "microrregiao",
        "distancia_km",
        "distancia_rodoviaria_est_km",
        "transit_time_dias",
        "folga_dias",
        "faixa_km_cd",
        "quadrante",
    ]
    faltam = [c for c in colunas_obrigatorias_saida if c not in df.columns]
    if faltam:
        raise Exception(
            "A saída do M2 ficou incompleta. Faltam colunas obrigatórias:\n- " +
            "\n- ".join(faltam)
        )


def _obter_valor_parametro(
    df_parametros: pd.DataFrame,
    chaves_possiveis: list[str],
    default: Any = None,
) -> Any:
    if df_parametros.empty or "parametro" not in df_parametros.columns:
        return default

    dfp = df_parametros.copy()
    dfp["_parametro_norm"] = dfp["parametro"].astype(str).str.strip().str.lower()

    if "valor" not in dfp.columns:
        return default

    for chave in chaves_possiveis:
        linha = dfp.loc[dfp["_parametro_norm"] == str(chave).strip().lower()]
        if len(linha) > 0:
            valor = linha.iloc[0]["valor"]
            if pd.notna(valor):
                try:
                    return float(valor)
                except Exception:
                    return valor

    return default


def _haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan

    raio_terra = 6371.0

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return raio_terra * c


def _calcular_transit_time_dias(km_rodoviario: Any, km_dia_operacional: float) -> Any:
    if pd.isna(km_rodoviario) or pd.isna(km_dia_operacional) or km_dia_operacional <= 0:
        return np.nan
    return int(np.ceil(float(km_rodoviario) / float(km_dia_operacional)))


def _classificar_faixa_km(km: Any) -> str:
    if pd.isna(km):
        return "sem_km"
    km = float(km)
    if km <= 50:
        return "ate_50"
    elif km <= 100:
        return "51_100"
    elif km <= 150:
        return "101_150"
    elif km <= 200:
        return "151_200"
    elif km <= 300:
        return "201_300"
    return "acima_300"


def _classificar_quadrante(lat_origem: Any, lon_origem: Any, lat_destino: Any, lon_destino: Any) -> str:
    if any(pd.isna(v) for v in [lat_origem, lon_origem, lat_destino, lon_destino]):
        return "sem_coord"

    norte = float(lat_destino) >= float(lat_origem)
    leste = float(lon_destino) >= float(lon_origem)

    if norte and leste:
        return "NE"
    elif norte and not leste:
        return "NO"
    elif not norte and leste:
        return "SE"
    return "SO"


def _classificar_status_folga(folga: Any) -> str:
    if pd.isna(folga):
        return "sem_folga"
    folga = float(folga)
    if folga < 0:
        return "negativa"
    if folga == 0:
        return "zero"
    if folga == 1:
        return "um_dia"
    if folga == 2:
        return "dois_dias"
    return "maior_que_2"


def _classificar_perfil_veiculo_referencia(km: Any) -> str:
    if pd.isna(km):
        return "indefinido"
    km = float(km)
    if km <= 50:
        return "VUC"
    elif km <= 180:
        return "3/4"
    elif km <= 500:
        return "TOCO"
    elif km <= 2000:
        return "TRUCK"
    return "CARRETA"


def _calcular_score(row: pd.Series) -> int:
    score = 0

    if row.get("agendada") is True:
        score += 100

    folga = row.get("folga_dias")
    if pd.notna(folga):
        folga = float(folga)
        if folga < 0:
            score += 80
        elif folga == 0:
            score += 60
        elif folga == 1:
            score += 40
        elif folga == 2:
            score += 10

    km = row.get("distancia_rodoviaria_est_km")
    if pd.notna(km):
        km = float(km)
        if km > 300:
            score += 10
        elif km > 150:
            score += 5

    return score


def _remover_acentos(texto: Any) -> Any:
    if pd.isna(texto):
        return np.nan
    texto = str(texto)
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )


def _normalizar_texto(valor: Any) -> Any:
    if pd.isna(valor):
        return np.nan
    valor = str(valor).strip()
    valor = _remover_acentos(valor)
    valor = re.sub(r"\s+", " ", valor)
    return valor.upper()


def _primeira_nao_nula(a: Any, b: Any) -> Any:
    if pd.notna(a):
        return a
    return b
