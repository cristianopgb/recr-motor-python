# ============================================================
# MÓDULO 2 - ENRIQUECIMENTO GEOGRÁFICO E TEMPORAL
# (VERSÃO API - ADAPTADA DO NOTEBOOK)
# ============================================================

import math
import re
import unicodedata
from typing import Any, Dict

import numpy as np
import pandas as pd


def remover_acentos(texto):
    if pd.isna(texto):
        return np.nan
    texto = str(texto)
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )


def normalizar_texto(valor):
    if pd.isna(valor):
        return np.nan
    valor = str(valor).strip()
    valor = remover_acentos(valor)
    valor = re.sub(r"\s+", " ", valor)
    return valor.upper()


def to_num(x):
    return pd.to_numeric(x, errors="coerce")


def haversine_km(lat1, lon1, lat2, lon2):
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


def classificar_faixa_km(km):
    if pd.isna(km):
        return "sem_km"
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
    else:
        return "acima_300"


def classificar_quadrante(lat_origem, lon_origem, lat_destino, lon_destino):
    if any(pd.isna(v) for v in [lat_origem, lon_origem, lat_destino, lon_destino]):
        return "sem_coord"

    norte = lat_destino >= lat_origem
    leste = lon_destino >= lon_origem

    if norte and leste:
        return "NE"
    elif norte and not leste:
        return "NO"
    elif not norte and leste:
        return "SE"
    else:
        return "SO"


def calcular_transit_time_dias(km_rodoviario, km_dia):
    if pd.isna(km_rodoviario) or pd.isna(km_dia) or km_dia <= 0:
        return np.nan
    return int(np.ceil(km_rodoviario / km_dia))


def primeira_nao_nula(a, b):
    if pd.notna(a):
        return a
    return b


def _obter_parametros_dict(df_parametros: pd.DataFrame) -> Dict[str, Any]:
    if df_parametros.empty:
        return {}

    if "parametro" not in df_parametros.columns or "valor" not in df_parametros.columns:
        return {}

    base = df_parametros.copy()
    base["parametro"] = base["parametro"].astype(str).str.strip().str.lower()
    return dict(zip(base["parametro"], base["valor"]))


def _obter_parametro_num(parametros_dict: Dict[str, Any], chaves_possiveis, default):
    for chave in chaves_possiveis:
        if chave in parametros_dict:
            valor = pd.to_numeric(parametros_dict[chave], errors="coerce")
            if pd.notna(valor):
                return float(valor)
    return float(default)


def _obter_parametro_data(parametros_dict: Dict[str, Any], chaves_possiveis, default=None):
    for chave in chaves_possiveis:
        if chave in parametros_dict:
            valor = pd.to_datetime(parametros_dict[chave], errors="coerce", dayfirst=True)
            if pd.notna(valor):
                return valor
    return default


def executar_m2_enriquecimento(
    df_carteira_tratada: pd.DataFrame,
    df_geo_tratado: pd.DataFrame,
    df_parametros_tratados: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
    data_base: Any = None,
) -> Dict[str, Any]:
    """
    M2 adaptado para API.
    Recebe os outputs do M1 e devolve a carteira enriquecida.
    """

    # --------------------------------------------------------
    # 1) CÓPIAS
    # --------------------------------------------------------
    carteira = df_carteira_tratada.copy()
    geo = df_geo_tratado.copy()
    parametros = df_parametros_tratados.copy()
    _ = df_veiculos_tratados.copy()  # mantido por contrato, mesmo sem uso direto agora

    # --------------------------------------------------------
    # 2) CONSTANTES PADRÃO
    # --------------------------------------------------------
    FATOR_KM_RODOVIARIO_PADRAO = 1.20
    KM_MINIMO_OPERACIONAL = 5.0
    KM_MAX_DIA_PADRAO = 450.0

    # --------------------------------------------------------
    # 3) VALIDAÇÕES MÍNIMAS
    # --------------------------------------------------------
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

    colunas_minimas_geo = ["cidade", "uf", "mesorregiao", "microrregiao"]
    faltam_geo = [c for c in colunas_minimas_geo if c not in geo.columns]
    if faltam_geo:
        raise Exception(
            "Faltam colunas mínimas na base geográfica tratada para o M2:\n- " +
            "\n- ".join(faltam_geo)
        )

    # --------------------------------------------------------
    # 4) DATA BASE
    # --------------------------------------------------------
    parametros_dict = _obter_parametros_dict(parametros)

    data_base_param = _obter_parametro_data(
        parametros_dict,
        ["data_base_roteirizacao", "data_corte_referencia"],
        default=None
    )

    if data_base is not None:
        data_base_execucao = pd.to_datetime(data_base, errors="coerce", dayfirst=True)
    else:
        data_base_execucao = data_base_param

    if pd.isna(data_base_execucao):
        data_base_execucao = pd.Timestamp.now()

    # --------------------------------------------------------
    # 5) PARÂMETROS OPERACIONAIS
    # --------------------------------------------------------
    km_dia_max = _obter_parametro_num(
        parametros_dict,
        ["km_dia_max", "km_max_dia", "km por dia", "km_dia"],
        KM_MAX_DIA_PADRAO
    )

    fator_km_rodoviario = _obter_parametro_num(
        parametros_dict,
        ["fator_km_rodoviario", "fator_rodoviario", "fator_km_rodoviario_estimado"],
        FATOR_KM_RODOVIARIO_PADRAO
    )

    # --------------------------------------------------------
    # 6) GARANTIR TIPAGEM
    # --------------------------------------------------------
    for c in [
        "latitude_filial",
        "longitude_filial",
        "latitude_destinatario",
        "longitude_destinatario",
    ]:
        carteira[c] = pd.to_numeric(carteira[c], errors="coerce")

    for c in ["data_agenda", "data_leadtime", "data_descarga", "data_nf"]:
        if c in carteira.columns:
            carteira[c] = pd.to_datetime(carteira[c], errors="coerce", dayfirst=True)

    if "agendada" not in carteira.columns:
        carteira["agendada"] = False
    carteira["agendada"] = carteira["agendada"].fillna(False).astype(bool)

    # --------------------------------------------------------
    # 7) CONTRATO EXPLÍCITO DE ORIGEM
    # --------------------------------------------------------
    carteira["origem_latitude"] = carteira["latitude_filial"]
    carteira["origem_longitude"] = carteira["longitude_filial"]

    # --------------------------------------------------------
    # 8) ENRIQUECIMENTO GEO TEMPORAL
    # --------------------------------------------------------
    carteira["distancia_km"] = carteira.apply(
        lambda row: haversine_km(
            row["origem_latitude"],
            row["origem_longitude"],
            row["latitude_destinatario"],
            row["longitude_destinatario"],
        ),
        axis=1
    )

    carteira["distancia_km"] = carteira["distancia_km"].clip(lower=KM_MINIMO_OPERACIONAL)
    carteira["distancia_rodoviaria_est_km"] = (
        carteira["distancia_km"] * fator_km_rodoviario
    ).round(2)

    carteira["transit_time_dias"] = carteira["distancia_rodoviaria_est_km"].apply(
        lambda x: calcular_transit_time_dias(x, km_dia_max)
    )

    carteira["faixa_km_cd"] = carteira["distancia_rodoviaria_est_km"].apply(classificar_faixa_km)

    carteira["quadrante"] = carteira.apply(
        lambda row: classificar_quadrante(
            row["origem_latitude"],
            row["origem_longitude"],
            row["latitude_destinatario"],
            row["longitude_destinatario"],
        ),
        axis=1
    )

    # --------------------------------------------------------
    # 9) DATA LIMITE CONSIDERADA
    # --------------------------------------------------------
    carteira["data_limite_considerada"] = np.where(
        (carteira["agendada"] == True) & (carteira["data_agenda"].notna()),
        carteira["data_agenda"],
        carteira["data_leadtime"]
    )
    carteira["data_limite_considerada"] = pd.to_datetime(
        carteira["data_limite_considerada"],
        errors="coerce"
    )

    carteira["tipo_data_limite"] = np.where(
        (carteira["agendada"] == True) & (carteira["data_agenda"].notna()),
        "agenda",
        np.where(carteira["data_leadtime"].notna(), "leadtime", "sem_data")
    )

    # --------------------------------------------------------
    # 10) FOLGA OPERACIONAL
    # --------------------------------------------------------
    carteira["dias_ate_data_alvo"] = (
        carteira["data_limite_considerada"].dt.normalize() - data_base_execucao.normalize()
    ).dt.days

    carteira["folga_dias"] = carteira["dias_ate_data_alvo"] - carteira["transit_time_dias"]

    def classificar_status_folga(folga):
        if pd.isna(folga):
            return "sem_folga"
        if folga < 0:
            return "negativa"
        if folga == 0:
            return "zero"
        if folga == 1:
            return "um_dia"
        if folga == 2:
            return "dois_dias"
        return "maior_que_2"

    carteira["status_folga"] = carteira["folga_dias"].apply(classificar_status_folga)

    # --------------------------------------------------------
    # 11) ENRIQUECIMENTO DE MESO/MICRO
    # --------------------------------------------------------
    geo["_cidade_norm"] = geo["cidade"].apply(normalizar_texto)
    geo["_uf_norm"] = geo["uf"].apply(normalizar_texto)

    carteira["_cidade_norm"] = carteira["cidade"].apply(normalizar_texto)
    carteira["_uf_norm"] = carteira["uf"].apply(normalizar_texto)

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
        suffixes=("_carteira", "_geo")
    )

    col_meso_carteira = "mesorregiao_carteira" if "mesorregiao_carteira" in carteira.columns else None
    col_meso_geo = "mesorregiao_geo" if "mesorregiao_geo" in carteira.columns else None
    col_micro_carteira = "microrregiao_carteira" if "microrregiao_carteira" in carteira.columns else None
    col_micro_geo = "microrregiao_geo" if "microrregiao_geo" in carteira.columns else None

    if col_meso_carteira is not None or col_meso_geo is not None:
        carteira["mesorregiao"] = carteira.apply(
            lambda row: primeira_nao_nula(
                row[col_meso_carteira] if col_meso_carteira is not None else np.nan,
                row[col_meso_geo] if col_meso_geo is not None else np.nan
            ),
            axis=1
        )

    if col_micro_carteira is not None or col_micro_geo is not None:
        carteira["microrregiao"] = carteira.apply(
            lambda row: primeira_nao_nula(
                row[col_micro_carteira] if col_micro_carteira is not None else np.nan,
                row[col_micro_geo] if col_micro_geo is not None else np.nan
            ),
            axis=1
        )

    # --------------------------------------------------------
    # 12) SCORE E RANKING PRELIMINAR
    # --------------------------------------------------------
    def calcular_score(row):
        score = 0

        if row["agendada"] == True:
            score += 100

        folga = row["folga_dias"]
        if pd.notna(folga):
            if folga < 0:
                score += 80
            elif folga == 0:
                score += 60
            elif folga == 1:
                score += 40
            elif folga == 2:
                score += 10

        km = row["distancia_rodoviaria_est_km"]
        if pd.notna(km):
            if km > 300:
                score += 10
            elif km > 150:
                score += 5

        return score

    carteira["score_prioridade_preliminar"] = carteira.apply(calcular_score, axis=1)
    carteira["ranking_preliminar"] = (
        carteira["score_prioridade_preliminar"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )

    # --------------------------------------------------------
    # 13) PERFIL DE VEÍCULO DE REFERÊNCIA
    # --------------------------------------------------------
    def classificar_perfil_veiculo_referencia(km):
        if pd.isna(km):
            return "indefinido"
        if km <= 50:
            return "VUC"
        elif km <= 180:
            return "3/4"
        elif km <= 500:
            return "TOCO"
        elif km <= 2000:
            return "TRUCK"
        return "CARRETA"

    carteira["perfil_veiculo_referencia"] = carteira["distancia_rodoviaria_est_km"].apply(
        classificar_perfil_veiculo_referencia
    )

    # --------------------------------------------------------
    # 14) STATUS GEO
    # --------------------------------------------------------
    carteira["status_geo"] = np.where(
        carteira["mesorregiao"].notna() & carteira["microrregiao"].notna(),
        "ok",
        "pendencia_geo"
    )

    # --------------------------------------------------------
    # 15) LIMPEZA DE COLUNAS TÉCNICAS
    # --------------------------------------------------------
    colunas_descartar = [
        "_cidade_norm", "_uf_norm",
        "mesorregiao_carteira", "mesorregiao_geo",
        "microrregiao_carteira", "microrregiao_geo"
    ]
    carteira.drop(
        columns=[c for c in colunas_descartar if c in carteira.columns],
        inplace=True,
        errors="ignore"
    )

    # --------------------------------------------------------
    # 16) ORDENAÇÃO FINAL
    # --------------------------------------------------------
    colunas_ordem_preferencial = [
        "ranking_preliminar",
        "score_prioridade_preliminar",
        "filial_roteirizacao",
        "romaneio",
        "filial_origem",
        "serie",
        "nro_documento",
        "tomador",
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
        "dias_ate_data_alvo",
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

    # --------------------------------------------------------
    # 17) VALIDAÇÕES FINAIS
    # --------------------------------------------------------
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

    faltam_saida = [c for c in colunas_obrigatorias_saida if c not in df_carteira_enriquecida.columns]
    if faltam_saida:
        raise Exception(
            "A saída do M2 ficou incompleta. Faltam colunas obrigatórias:\n- " +
            "\n- ".join(faltam_saida)
        )

    # --------------------------------------------------------
    # 18) RESUMOS
    # --------------------------------------------------------
    df_resumo_modulo_2 = pd.DataFrame({
        "indicador": [
            "linhas_carteira",
            "colunas_carteira",
            "km_dia_max",
            "fator_km_rodoviario",
            "distancia_km_nulos",
            "distancia_rodoviaria_nulos",
            "transit_time_nulos",
            "data_limite_nulos",
            "folga_nulos",
            "mesorregiao_nulos",
            "microrregiao_nulos",
            "status_geo_ok",
            "status_geo_pendencia",
        ],
        "valor": [
            len(df_carteira_enriquecida),
            len(df_carteira_enriquecida.columns),
            km_dia_max,
            fator_km_rodoviario,
            int(df_carteira_enriquecida["distancia_km"].isna().sum()),
            int(df_carteira_enriquecida["distancia_rodoviaria_est_km"].isna().sum()),
            int(df_carteira_enriquecida["transit_time_dias"].isna().sum()),
            int(df_carteira_enriquecida
