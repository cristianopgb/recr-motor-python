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
    - transit time = ceil(km_rodoviario / km_dia_operacional)
    - regionalidades são FALLBACK
    - latitude/longitude de destino vêm da carteira
    - não pode multiplicar linhas da carteira
    """

    carteira = df_carteira_tratada.copy()
    geo = df_geo_tratado.copy()
    parametros = df_parametros_tratados.copy()

    _validar_colunas_minimas(carteira, geo)

    qtd_linhas_entrada = len(carteira)

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

    data_base = _normalizar_datetime_sem_timezone(data_base_roteirizacao)
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

    carteira["data_limite_considerada"] = carteira["data_limite_considerada"].apply(
        _normalizar_datetime_sem_timezone
    )

    if "data_agenda" in carteira.columns:
        carteira["data_agenda"] = carteira["data_agenda"].apply(_normalizar_datetime_sem_timezone)

    if "data_leadtime" in carteira.columns:
        carteira["data_leadtime"] = carteira["data_leadtime"].apply(_normalizar_datetime_sem_timezone)

    carteira["tipo_data_limite"] = np.where(
        (carteira["agendada"] == True) & (carteira["data_agenda"].notna()),
        "agenda",
        np.where(carteira["data_leadtime"].notna(), "leadtime", "sem_data"),
    )

    carteira["data_base_roteirizacao"] = data_base

    carteira["dias_ate_data_alvo"] = (
        pd.to_datetime(carteira["data_limite_considerada"]).dt.normalize()
        - pd.to_datetime(data_base).normalize()
    ).dt.days

    carteira["folga_dias"] = carteira["dias_ate_data_alvo"] - carteira["transit_time_dias"]
    carteira["status_folga"] = carteira["folga_dias"].apply(_classificar_status_folga)

    # ============================================================
    # FALLBACK DE REGIONALIDADES SEM DUPLICAR LINHAS
    # de/para:
    # cidade      -> cidade
    # regiao      -> microrregiao
    # subregiao   -> microrregiao
    # mesorregiao -> mesorregiao
    # ============================================================

    # garante colunas-base na carteira
    if "regiao" not in carteira.columns:
        carteira["regiao"] = np.nan
    if "subregiao" not in carteira.columns:
        carteira["subregiao"] = np.nan
    if "mesorregiao" not in carteira.columns:
        carteira["mesorregiao"] = np.nan

    # normaliza carteira
    carteira["_cidade_norm"] = carteira["cidade"].apply(_normalizar_texto)
    carteira["_uf_norm"] = carteira["uf"].apply(_normalizar_texto)

    # normaliza geo
    geo = geo.copy()

    if "nome" in geo.columns:
        geo["cidade_fallback"] = geo["nome"]
    elif "cidade" in geo.columns:
        geo["cidade_fallback"] = geo["cidade"]
    else:
        raise Exception("A base de regionalidades não possui coluna 'nome' nem 'cidade'.")

    geo["_cidade_norm"] = geo["cidade_fallback"].apply(_normalizar_texto)
    geo["_uf_norm"] = geo["uf"].apply(_normalizar_texto)

    # saneia colunas fallback
    for c in ["mesorregiao", "microrregiao"]:
        if c not in geo.columns:
            geo[c] = np.nan

    geo["mesorregiao"] = geo["mesorregiao"].apply(_limpar_vazio)
    geo["microrregiao"] = geo["microrregiao"].apply(_limpar_vazio)

    # score para escolher a melhor linha por cidade+uf
    # prioridade:
    # 1) tem meso e micro
    # 2) tem meso
    # 3) tem micro
    # 4) resto
    geo["_score_geo"] = (
        geo["mesorregiao"].notna().astype(int) * 2
        + geo["microrregiao"].notna().astype(int) * 1
    )

    # deduplicação 1:1 pela chave do merge
    geo_chaves = (
        geo.sort_values(
            by=["_cidade_norm", "_uf_norm", "_score_geo"],
            ascending=[True, True, False]
        )
        .drop_duplicates(subset=["_cidade_norm", "_uf_norm"], keep="first")
        [[" _cidade_norm".strip(), "_uf_norm", "mesorregiao", "microrregiao"]]
        .rename(
            columns={
                "mesorregiao": "mesorregiao_fallback",
                "microrregiao": "microrregiao_fallback",
            }
        )
        .copy()
    )

    carteira = carteira.merge(
        geo_chaves,
        how="left",
        on=["_cidade_norm", "_uf_norm"],
        validate="m:1",
    )

    # fallback controlado: só preenche se estiver vazio na carteira
    carteira["mesorregiao"] = carteira.apply(
        lambda row: _preencher_somente_se_vazio(
            row.get("mesorregiao"),
            row.get("mesorregiao_fallback"),
        ),
        axis=1,
    )

    # subregiao fallback = microrregiao
    carteira["subregiao"] = carteira.apply(
        lambda row: _preencher_somente_se_vazio(
            row.get("subregiao"),
            row.get("microrregiao_fallback"),
        ),
        axis=1,
    )

    # regiao fallback = microrregiao
    carteira["regiao"] = carteira.apply(
        lambda row: _preencher_somente_se_vazio(
            row.get("regiao"),
            row.get("microrregiao_fallback"),
        ),
        axis=1,
    )

    carteira["status_geo"] = np.where(
        carteira["mesorregiao"].notna() & carteira["subregiao"].notna(),
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
        "mesorregiao_fallback",
        "microrregiao_fallback",
    ]
    carteira.drop(
        columns=[c for c in colunas_descartar if c in carteira.columns],
        inplace=True,
        errors="ignore",
    )

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
        "regiao",
        "subregiao",
        "mesorregiao",
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

    # blindagem cardinalidade
    if len(df_carteira_enriquecida) != qtd_linhas_entrada:
        raise Exception(
            "O M2 alterou a cardinalidade da carteira, o que não é permitido.\n"
            f"Linhas entrada: {qtd_linhas_entrada}\n"
            f"Linhas saída: {len(df_carteira_enriquecida)}"
        )

    _validar_saida_m2(df_carteira_enriquecida)

    resumo = {
        "linhas_carteira_entrada": int(qtd_linhas_entrada),
        "linhas_carteira_saida": int(len(df_carteira_enriquecida)),
        "colunas_carteira": int(len(df_carteira_enriquecida.columns)),
        "fator_km_rodoviario": fator_km_rodoviario,
        "velocidade_media_km_h": velocidade_media_km_h,
        "horas_direcao_dia": horas_direcao_dia,
        "km_dia_operacional": km_dia_operacional,
        "distancia_km_nulos": int(df_carteira_enriquecida["distancia_km"].isna().sum()),
        "transit_time_nulos": int(df_carteira_enriquecida["transit_time_dias"].isna().sum()),
        "folga_nulos": int(df_carteira_enriquecida["folga_dias"].isna().sum()),
        "regiao_nulos": int(df_carteira_enriquecida["regiao"].isna().sum()),
        "subregiao_nulos": int(df_carteira_enriquecida["subregiao"].isna().sum()),
        "mesorregiao_nulos": int(df_carteira_enriquecida["mesorregiao"].isna().sum()),
        "status_geo_ok": int((df_carteira_enriquecida["status_geo"] == "ok").sum()),
        "status_geo_pendencia": int((df_carteira_enriquecida["status_geo"] == "pendencia_geo").sum()),
        "linhas_enriquecidas_por_fallback_regiao": int(
            (
                df_carteira_enriquecida["regiao"].notna()
                & df_carteira_tratada.reindex(df_carteira_enriquecida.index)["regiao"].isna()
            ).sum()
        ) if "regiao" in df_carteira_tratada.columns else None,
        "linhas_enriquecidas_por_fallback_subregiao": int(
            (
                df_carteira_enriquecida["subregiao"].notna()
                & df_carteira_tratada.reindex(df_carteira_enriquecida.index)["subregiao"].isna()
            ).sum()
        ) if "subregiao" in df_carteira_tratada.columns else None,
        "linhas_enriquecidas_por_fallback_mesorregiao": int(
            (
                df_carteira_enriquecida["mesorregiao"].notna()
                & df_carteira_tratada.reindex(df_carteira_enriquecida.index)["mesorregiao"].isna()
            ).sum()
        ) if "mesorregiao" in df_carteira_tratada.columns else None,
    }

    return df_carteira_enriquecida, resumo


def _normalizar_datetime_sem_timezone(value: Any) -> Any:
    if pd.isna(value):
        return pd.NaT

    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT

    if getattr(ts, "tzinfo", None) is not None:
        try:
            return ts.tz_localize(None)
        except TypeError:
            try:
                return ts.tz_convert(None)
            except Exception:
                return pd.Timestamp(ts).tz_localize(None)

    return ts


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

    colunas_minimas_geo_opcao_1 = ["nome", "uf", "mesorregiao", "microrregiao"]
    colunas_minimas_geo_opcao_2 = ["cidade", "uf", "mesorregiao", "microrregiao"]

    tem_opcao_1 = all(c in geo.columns for c in colunas_minimas_geo_opcao_1)
    tem_opcao_2 = all(c in geo.columns for c in colunas_minimas_geo_opcao_2)

    if not (tem_opcao_1 or tem_opcao_2):
        raise Exception(
            "Faltam colunas mínimas na base geográfica tratada para o M2.\n"
            "Aceito: ['nome','uf','mesorregiao','microrregiao'] "
            "ou ['cidade','uf','mesorregiao','microrregiao']"
        )


def _validar_saida_m2(df: pd.DataFrame) -> None:
    colunas_obrigatorias_saida = [
        "regiao",
        "subregiao",
        "mesorregiao",
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


def _limpar_vazio(valor: Any) -> Any:
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip()
    if texto == "":
        return np.nan
    if texto.lower() in {"nan", "none", "null", "<na>"}:
        return np.nan
    return texto


def _preencher_somente_se_vazio(valor_atual: Any, fallback: Any) -> Any:
    valor_atual = _limpar_vazio(valor_atual)
    fallback = _limpar_vazio(fallback)

    if pd.notna(valor_atual):
        return valor_atual
    return fallback
