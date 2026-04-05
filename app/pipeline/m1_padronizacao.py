# ============================================================
# MÓDULO 1 - LIMPEZA, PADRONIZAÇÃO E TIPAGEM
# (VERSÃO API - AJUSTADA AO CONTRATO REAL DO SISTEMA 1)
# ============================================================

import re
import unicodedata
from typing import Dict, Any

import pandas as pd
import numpy as np


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


def converter_flag_agendamento(serie):
    def _f(x):
        if pd.isna(x):
            return False
        x = str(x).strip()
        return x != ""
    return serie.apply(_f)


def escolher_coluna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    for c in candidatos:
        if c in df.columns:
            return c
    return None


def normalizar_chave_texto(serie: pd.Series) -> pd.Series:
    return serie.apply(
        lambda x: remover_acentos(str(x)).upper().strip() if pd.notna(x) else np.nan
    )


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
    mapa_carteira = {
        "filial": "filial_roteirizacao",
        "romane": "romaneio",
        "filial_origem": "filial_origem",
        "serie": "serie",
        "nro_doc": "nro_documento",
        "data_des": "data_descarga",
        "data_nf": "data_nf",
        "dle": "data_leadtime",
        "agendam": "data_agenda",
        "palet": "qtd_pallet",
        "conf": "conferencia",
        "peso": "peso_kg",
        "vlr_merc": "valor_nf",
        "qtd": "qtd_volumes",
        "peso_c": "vol_m3",
        "classifi": "classifi",
        "tomador": "tomador",
        "destinatario": "destinatario",
        "bairro": "bairro",
        "cida": "cidade",
        "uf": "uf",
        "nf_serie": "nf_serie",
        "tipo_carga": "tipo_carga",
        "qtd_nf": "qtd_nf",
        "regiao": "regiao",
        "sub_regiao": "sub_regiao",
        "ocorrencias_nfs": "ocorrencias_nfs",
        "remetente": "remetente",
        "observacao_r": "observacao_r",
        "ref_cliente": "ref_cliente",
        "cidade_dest": "cidade_dest",
        "mesoregiao": "mesorregiao",
        "agenda": "agenda",
        "tipo_c": "tipo_c",
        "ultima": "ultima",
        "status": "status",
        "lat": "latitude_destinatario",
        "lon": "longitude_destinatario"
    }

    carteira = carteira.rename(
        columns={k: v for k, v in mapa_carteira.items() if k in carteira.columns}
    )

    # --------------------------------------------------------
    # 4) MAPA GEO / REGIONALIDADES
    # Aceita tanto base antiga quanto contrato do Sistema 1
    # --------------------------------------------------------
    mapa_geo = {
        "cidade": "cidade",
        "nome": "nome",
        "uf": "uf",
        "mesorregiao": "mesorregiao",
        "microrregiao": "microrregiao",
        "latitude": "latitude",
        "longitude": "longitude"
    }

    geo = geo.rename(columns={k: v for k, v in mapa_geo.items() if k in geo.columns})

    # se veio "cidade" e não veio "nome", cria "nome" espelhado
    if "nome" not in geo.columns and "cidade" in geo.columns:
        geo["nome"] = geo["cidade"]

    # se veio "nome" e não veio "cidade", cria "cidade" espelhado
    if "cidade" not in geo.columns and "nome" in geo.columns:
        geo["cidade"] = geo["nome"]

    # --------------------------------------------------------
    # 5) MAPA PARÂMETROS
    # --------------------------------------------------------
    if "parametro" not in parametros.columns and "chave" in parametros.columns:
        parametros = parametros.rename(columns={"chave": "parametro"})

    if "valor" not in parametros.columns:
        col_valor = escolher_coluna(parametros, ["valor", "value"])
        if col_valor and col_valor != "valor":
            parametros = parametros.rename(columns={col_valor: "valor"})

    # se parâmetros vierem como dict convertido em linhas/colunas estranhas, garante mínimo
    if "parametro" not in parametros.columns or "valor" not in parametros.columns:
        raise Exception("A base de parâmetros não contém as colunas obrigatórias 'parametro' e 'valor'.")

    # --------------------------------------------------------
    # 6) MAPA VEÍCULOS
    # --------------------------------------------------------
    mapa_veiculos = {
        "id": "id",
        "perfil": "perfil",
        "placa": "placa",
        "qtd_eixos": "qtd_eixos",
        "capacidade_peso_kg": "capacidade_peso_kg",
        "capacidade_vol_m3": "capacidade_vol_m3",
        "max_entregas": "max_entregas",
        "max_km_distancia": "max_km_distancia",
        "ocupacao_minima_perc": "ocupacao_minima_perc",
        "filial_id": "filial_id",
        "tipo_frota": "tipo_frota",
        "ativo": "ativo"
    }

    veiculos = veiculos.rename(
        columns={k: v for k, v in mapa_veiculos.items() if k in veiculos.columns}
    )

    # --------------------------------------------------------
    # 7) TIPAGEM CARTEIRA
    # --------------------------------------------------------
    colunas_num = [
        "filial_roteirizacao",
        "romaneio",
        "filial_origem",
        "serie",
        "nro_documento",
        "qtd_pallet",
        "peso_kg",
        "valor_nf",
        "qtd_volumes",
        "vol_m3",
        "qtd_nf"
    ]

    for c in colunas_num:
        if c in carteira.columns:
            carteira[c] = converter_numerico_brasil(carteira[c])

    for c in ["latitude_destinatario", "longitude_destinatario"]:
        if c in carteira.columns:
            carteira[c] = converter_coordenada(carteira[c])

    for c in ["data_descarga", "data_nf", "data_leadtime", "data_agenda"]:
        if c in carteira.columns:
            carteira[c] = converter_data(carteira[c])

    if "data_agenda" in carteira.columns:
        carteira["agendada"] = converter_flag_agendamento(carteira["data_agenda"])
    else:
        carteira["agendada"] = False

    # --------------------------------------------------------
    # 8) TIPAGEM GEO
    # --------------------------------------------------------
    if "cidade" in geo.columns:
        geo["cidade"] = geo["cidade"].apply(normalizar_texto_basico)

    if "nome" in geo.columns:
        geo["nome"] = geo["nome"].apply(normalizar_texto_basico)

    if "uf" in geo.columns:
        geo["uf"] = geo["uf"].apply(normalizar_texto_basico)

    if "mesorregiao" in geo.columns:
        geo["mesorregiao"] = geo["mesorregiao"].apply(normalizar_texto_basico)

    if "microrregiao" in geo.columns:
        geo["microrregiao"] = geo["microrregiao"].apply(normalizar_texto_basico)

    # --------------------------------------------------------
    # 9) TIPAGEM PARÂMETROS
    # --------------------------------------------------------
    parametros["parametro"] = parametros["parametro"].apply(normalizar_texto_basico)
    parametros["valor"] = parametros["valor"].apply(lambda x: x if pd.isna(x) else str(x).strip())

    param_dict = dict(zip(parametros["parametro"], parametros["valor"]))

    carteira["origem_cidade"] = param_dict.get("origem_cidade")
    carteira["origem_uf"] = param_dict.get("origem_uf")
    carteira["latitude_filial"] = float(param_dict.get("origem_latitude", 0))
    carteira["longitude_filial"] = float(param_dict.get("origem_longitude", 0))
    carteira["data_base_roteirizacao"] = param_dict.get("data_base_roteirizacao")

    # --------------------------------------------------------
    # 10) CHAVES GEO
    # --------------------------------------------------------
    if "cidade" not in carteira.columns:
        raise Exception("A carteira tratada não contém a coluna obrigatória 'cidade'.")

    if "uf" not in carteira.columns:
        raise Exception("A carteira tratada não contém a coluna obrigatória 'uf'.")

    if "cidade" not in geo.columns:
        raise Exception("A base de regionalidades não contém a coluna obrigatória 'cidade'.")

    if "uf" not in geo.columns:
        raise Exception("A base de regionalidades não contém a coluna obrigatória 'uf'.")

    geo["cidade_chave"] = normalizar_chave_texto(geo["cidade"])
    geo["uf_chave"] = normalizar_chave_texto(geo["uf"])

    carteira["cidade_chave"] = normalizar_chave_texto(carteira["cidade"])
    carteira["uf_chave"] = normalizar_chave_texto(carteira["uf"])

    # --------------------------------------------------------
    # 11) TIPAGEM VEÍCULOS
    # --------------------------------------------------------
    colunas_num_veiculos = [
        "id",
        "qtd_eixos",
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ocupacao_minima_perc"
    ]

    for c in colunas_num_veiculos:
        if c in veiculos.columns:
            veiculos[c] = converter_numerico_brasil(veiculos[c])

    if "perfil" in veiculos.columns:
        veiculos["perfil"] = veiculos["perfil"].apply(normalizar_texto_basico)

    veiculos["ordem_porte"] = np.arange(1, len(veiculos) + 1)

    # --------------------------------------------------------
    # 12) OUTPUT
    # --------------------------------------------------------
    return {
        "df_carteira_tratada": carteira,
        "df_geo_tratado": geo,
        "df_parametros_tratados": parametros,
        "df_veiculos_tratados": veiculos
    }
