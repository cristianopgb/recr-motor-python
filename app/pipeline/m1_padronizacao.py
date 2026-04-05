# ============================================================
# MÓDULO 1 - LIMPEZA, PADRONIZAÇÃO E TIPAGEM
# (VERSÃO API - SEM LEITURA DE ARQUIVO)
# ============================================================

import re
import unicodedata
from typing import Dict, Any

import pandas as pd
import numpy as np


# ============================================================
# FUNÇÕES BASE
# ============================================================

def normalizar_texto_basico(valor):
    if pd.isna(valor):
        return np.nan
    texto = str(valor).replace("\u00a0", " ")
    texto = texto.strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def remover_acentos(texto):
    if pd.isna(texto):
        return np.nan
    texto = str(texto)
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )


def padronizar_nome_coluna(col):
    col = normalizar_texto_basico(col)
    col = remover_acentos(col)
    col = col.lower()
    col = col.replace("/", "_")
    col = col.replace(".", "")
    col = col.replace("-", "_")
    col = col.replace("(", "_")
    col = col.replace(")", "_")
    col = col.replace("%", "perc")
    col = re.sub(r"[^a-z0-9_]+", "_", col)
    col = re.sub(r"_+", "_", col)
    col = col.strip("_")
    return col


def garantir_colunas_unicas(colunas):
    novas = []
    contador = {}
    for c in colunas:
        if c not in contador:
            contador[c] = 0
            novas.append(c)
        else:
            contador[c] += 1
            novas.append(f"{c}_{contador[c]}")
    return novas


# ============================================================
# CONVERSORES
# ============================================================

def converter_numerico_brasil(serie):
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")

    s = serie.astype(str).str.strip()

    def _conv(x):
        if pd.isna(x):
            return np.nan
        x = str(x).replace("R$", "").replace(" ", "")

        if "." in x and "," in x:
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        elif "," in x:
            x = x.replace(".", "").replace(",", ".")

        return pd.to_numeric(x, errors="coerce")

    return s.apply(_conv)


def converter_coordenada(serie):
    s = serie.astype(str).str.strip()

    def _coord(x):
        if pd.isna(x):
            return np.nan
        x = str(x).replace(" ", "")

        if "." in x and "," in x:
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        elif "," in x:
            x = x.replace(",", ".")

        return pd.to_numeric(x, errors="coerce")

    return s.apply(_coord)


def converter_data(serie):
    return pd.to_datetime(serie, errors="coerce", dayfirst=True)


def converter_flag(serie):
    def _f(x):
        if pd.isna(x):
            return False
        x = remover_acentos(str(x)).lower()
        return x in ["sim", "s", "true", "1", "ok"]

    return serie.apply(_f)


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def executar_m1_padronizacao(
    df_carteira_raw: pd.DataFrame,
    df_geo_raw: pd.DataFrame,
    df_parametros_raw: pd.DataFrame,
    df_veiculos_raw: pd.DataFrame
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # 1) CÓPIAS
    # --------------------------------------------------------
    carteira = df_carteira_raw.copy()
    geo = df_geo_raw.copy()
    parametros = df_parametros_raw.copy()
    veiculos = df_veiculos_raw.copy()

    # --------------------------------------------------------
    # 2) PADRONIZA COLUNAS
    # --------------------------------------------------------
    for df in [carteira, geo, parametros, veiculos]:
        cols = [padronizar_nome_coluna(c) for c in df.columns]
        df.columns = garantir_colunas_unicas(cols)

    # --------------------------------------------------------
    # 3) MAPA CARTEIRA
    # --------------------------------------------------------
    mapa = {
        "filial_roteirizacao": "filial_roteirizacao",
        "romane": "romaneio",
        "filial_origem": "filial_origem",
        "serie": "serie",
        "nro_doc": "nro_documento",
        "data_des": "data_descarga",
        "data_nf": "data_nf",
        "dle": "data_leadtime",
        "agendam": "data_agenda",
        "peso": "peso_kg",
        "peso_c": "vol_m3",
        "cida": "cidade",
        "uf": "uf",
        "lat": "latitude_destinatario",
        "lon": "longitude_destinatario"
    }

    carteira = carteira.rename(columns={k: v for k, v in mapa.items() if k in carteira.columns})

    # --------------------------------------------------------
    # 4) TIPAGEM
    # --------------------------------------------------------
    for c in ["peso_kg", "vol_m3"]:
        if c in carteira.columns:
            carteira[c] = converter_numerico_brasil(carteira[c])

    if "latitude_destinatario" in carteira:
        carteira["latitude_destinatario"] = converter_coordenada(carteira["latitude_destinatario"])

    if "longitude_destinatario" in carteira:
        carteira["longitude_destinatario"] = converter_coordenada(carteira["longitude_destinatario"])

    for c in ["data_descarga", "data_nf", "data_leadtime", "data_agenda"]:
        if c in carteira.columns:
            carteira[c] = converter_data(carteira[c])

    if "data_agenda" in carteira.columns:
        carteira["agendada"] = converter_flag(carteira["data_agenda"])

    # --------------------------------------------------------
    # 5) PARÂMETROS → ORIGEM
    # --------------------------------------------------------
    param_dict = dict(zip(parametros["parametro"], parametros["valor"]))

    carteira["origem_cidade"] = param_dict.get("origem_cidade")
    carteira["origem_uf"] = param_dict.get("origem_uf")
    carteira["latitude_filial"] = float(param_dict.get("origem_latitude", 0))
    carteira["longitude_filial"] = float(param_dict.get("origem_longitude", 0))

    # --------------------------------------------------------
    # 6) GEO NORMALIZA
    # --------------------------------------------------------
    geo["cidade_chave"] = geo["nome"].apply(lambda x: remover_acentos(str(x)).upper())
    geo["uf_chave"] = geo["uf"].str.upper()

    carteira["cidade_chave"] = carteira["cidade"].apply(lambda x: remover_acentos(str(x)).upper())
    carteira["uf_chave"] = carteira["uf"].str.upper()

    # --------------------------------------------------------
    # 7) VEÍCULOS
    # --------------------------------------------------------
    veiculos["ordem_porte"] = np.arange(1, len(veiculos) + 1)

    # --------------------------------------------------------
    # 8) OUTPUT
    # --------------------------------------------------------
    return {
        "df_carteira_tratada": carteira,
        "df_geo_tratado": geo,
        "df_parametros_tratados": parametros,
        "df_veiculos_tratados": veiculos
    }
