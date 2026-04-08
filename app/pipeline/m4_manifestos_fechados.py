
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# MÓDULO 5.1 - MANIFESTOS COMPOSTOS - RODADA 1
#
# OBJETIVO:
# Receber SOMENTE o remanescente oficial do M4 e tentar gerar
# pré-manifestos compostos de forma simples, leve e auditável.
#
# REGRAS DE NEGÓCIO DESTA RODADA:
# 1) entrada = remanescente oficial do M4 (trava dura)
# 2) organiza fila por prioridade operacional:
#    - prioridade = "sim"
#    - agendadas
#    - menor folga -> maior folga
# 3) agrupamento simples por regionalidade, nesta ordem:
#    - mesmo cliente
#    - mesma cidade
#    - mesma sub-região
#    - mesma mesorregião
# 4) SEM ancoragem pesada / SEM confronto / SEM solver
# 5) tenta veículo do MAIOR para o MENOR
# 6) respeita:
#    - capacidade peso / volume
#    - ocupação mínima / máxima
#    - máximo de entregas
#    - raio / km
# 7) o que fechar gera pré-manifesto M5.1
# 8) o que não fechar fica no remanescente M5.1
# ============================================================

OCUPACAO_MINIMA_PADRAO = 0.70
OCUPACAO_MINIMA_SECUNDARIA_PADRAO = 0.20
OCUPACAO_MAXIMA_PADRAO = 1.00


def _agora() -> float:
    return time.perf_counter()


def _duracao_ms(inicio: float) -> float:
    return round((time.perf_counter() - inicio) * 1000, 2)


def _to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or len(df) == 0:
        return []
    df2 = df.copy()
    for col in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[col]):
            df2[col] = df2[col].astype(str)
    df2 = df2.where(pd.notnull(df2), None)
    return df2.to_dict(orient="records")


def _scalar_safe(x: Any) -> Any:
    if isinstance(x, pd.Series):
        return x.iloc[0] if len(x) > 0 else np.nan
    if isinstance(x, np.ndarray):
        return x[0] if len(x) > 0 else np.nan
    if isinstance(x, list):
        return x[0] if len(x) > 0 else np.nan
    if isinstance(x, tuple):
        return x[0] if len(x) > 0 else np.nan
    return x


def _bool_safe(x: Any) -> bool:
    x = _scalar_safe(x)

    try:
        if pd.isna(x):
            return False
    except Exception:
        pass

    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(int(x))

    txt = str(x).strip().lower()
    return txt in {"true", "1", "sim", "s", "yes", "y"}


def _num_safe(x: Any, default: float = np.nan) -> float:
    x = _scalar_safe(x)
    val = pd.to_numeric(x, errors="coerce")
    return float(val) if pd.notna(val) else default


def _int_safe(x: Any, default: int = 0) -> int:
    val = _num_safe(x, default=np.nan)
    if pd.isna(val):
        return default
    return int(val)


def _resolver_coluna_tipo_veiculo(df_veiculos: pd.DataFrame) -> str:
    if "tipo" in df_veiculos.columns:
        return "tipo"
    if "perfil" in df_veiculos.columns:
        return "perfil"
    raise Exception("Faltam colunas mínimas na base de veículos: esperado 'tipo' ou 'perfil'.")


def _normalizar_tipo_roteirizacao(valor: Any) -> str:
    txt = str(valor).strip().lower() if valor is not None else "carteira"
    return "frota" if txt == "frota" else "carteira"


def _garantir_coluna_por_alias(
    df: pd.DataFrame,
    coluna_destino: str,
    aliases: List[str],
    default: Any = None,
) -> pd.DataFrame:
    if coluna_destino in df.columns:
        return df

    for alias in aliases:
        if alias in df.columns:
            df[coluna_destino] = df[alias]
            return df

    df[coluna_destino] = default
    return df


def _normalizar_configuracao_frota(configuracao_frota: Any) -> pd.DataFrame:
    if configuracao_frota is None:
        return pd.DataFrame(columns=["perfil", "quantidade"])

    if isinstance(configuracao_frota, pd.DataFrame):
        cfg = configuracao_frota.copy()
    else:
        rows: List[Dict[str, Any]] = []
        for item in configuracao_frota:
            if hasattr(item, "model_dump"):
                rows.append(item.model_dump(exclude_none=False))
            elif isinstance(item, dict):
                rows.append(item)
            else:
                rows.append(dict(item))
        cfg = pd.DataFrame(rows)

    if len(cfg) == 0:
        return pd.DataFrame(columns=["perfil", "quantidade"])

    if "perfil" not in cfg.columns or "quantidade" not in cfg.columns:
        return pd.DataFrame(columns=["perfil", "quantidade"])

    cfg["perfil"] = cfg["perfil"].astype(str).str.strip()
    cfg["quantidade"] = pd.to_numeric(cfg["quantidade"], errors="coerce").fillna(0).astype(int)
    cfg = cfg.loc[cfg["perfil"].astype(str).str.strip() != ""].copy()
    cfg = cfg.loc[cfg["quantidade"] > 0].copy()

    if len(cfg) == 0:
        return pd.DataFrame(columns=["perfil", "quantidade"])

    cfg = cfg.groupby("perfil", as_index=False)["quantidade"].sum()
    return cfg.reset_index(drop=True)


def _normalizar_remanescente_m4(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    trabalho = df.copy()

    trabalho = _garantir_coluna_por_alias(trabalho, "id_linha_pipeline", ["id_linha_pipeline", "id", "linha_numero"])
    trabalho = _garantir_coluna_por_alias(trabalho, "nro_documento", ["nro_documento", "Nro Doc.", "nro_doc"])
    trabalho = _garantir_coluna_por_alias(trabalho, "destinatario", ["destinatario", "Destinatário"])
    trabalho = _garantir_coluna_por_alias(trabalho, "cliente", ["ref_cliente", "Ref Cliente", "tomador", "Tomador", "destinatario"])
    trabalho = _garantir_coluna_por_alias(trabalho, "cidade", ["cidade_dest", "Cidade Dest.", "cidade", "Cida"])
    trabalho = _garantir_coluna_por_alias(trabalho, "subregiao", ["subregiao", "sub_regiao", "Sub-Região"])
    trabalho = _garantir_coluna_por_alias(trabalho, "mesorregiao", ["mesorregiao", "Mesoregião"])
    trabalho = _garantir_coluna_por_alias(trabalho, "uf", ["uf", "UF"])
    trabalho = _garantir_coluna_por_alias(trabalho, "peso_kg", ["peso_kg", "Peso"], default=0)
    trabalho = _garantir_coluna_por_alias(trabalho, "vol_m3", ["vol_m3", "Peso C"], default=0)
    trabalho = _garantir_coluna_por_alias(trabalho, "prioridade", ["prioridade_embarque", "Prioridade"], default=np.nan)
    trabalho = _garantir_coluna_por_alias(trabalho, "data_agenda", ["data_agenda", "Agendam."], default=pd.NaT)
    trabalho = _garantir_coluna_por_alias(trabalho, "folga_dias", ["folga_dias"], default=np.nan)
    trabalho = _garantir_coluna_por_alias(
        trabalho,
        "distancia_rodoviaria_est_km",
        ["distancia_rodoviaria_est_km", "km_rodoviario", "km", "distancia_km"],
        default=np.nan,
    )

    trabalho["peso_kg"] = pd.to_numeric(trabalho["peso_kg"], errors="coerce").fillna(0.0)
    trabalho["vol_m3"] = pd.to_numeric(trabalho["vol_m3"], errors="coerce").fillna(0.0)
    trabalho["prioridade"] = trabalho["prioridade"].astype(str).str.strip().str.lower()
    trabalho["data_agenda"] = pd.to_datetime(trabalho["data_agenda"], errors="coerce")
    trabalho["folga_dias"] = pd.to_numeric(trabalho["folga_dias"], errors="coerce")
    trabalho["distancia_rodoviaria_est_km"] = pd.to_numeric(trabalho["distancia_rodoviaria_est_km"], errors="coerce")

    for col in ("cliente", "cidade", "subregiao", "mesorregiao", "uf", "destinatario", "nro_documento"):
        trabalho[col] = trabalho[col].astype(str).str.strip()
        trabalho[col] = trabalho[col].replace({"nan": "", "None": ""})

    if trabalho["id_linha_pipeline"].isna().any():
        faltantes = trabalho["id_linha_pipeline"].isna()
        trabalho.loc[faltantes, "id_linha_pipeline"] = [f"M5_1_LINE_{i+1}" for i in range(int(faltantes.sum()))]

    trabalho["id_linha_pipeline"] = trabalho["id_linha_pipeline"].astype(str)

    return trabalho.reset_index(drop=True)


def _normalizar_catalogo_veiculos(
    df_veiculos_tratados: pd.DataFrame,
    tipo_roteirizacao: str,
    configuracao_frota: Any,
    df_uso_frota_m4: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if df_veiculos_tratados is None or len(df_veiculos_tratados) == 0:
        return pd.DataFrame()

    catalogo = df_veiculos_tratados.copy()
    col_tipo = _resolver_coluna_tipo_veiculo(catalogo)

    catalogo["veiculo_tipo"] = catalogo[col_tipo].astype(str).str.strip()
    catalogo["capacidade_peso_kg"] = pd.to_numeric(catalogo.get("capacidade_peso_kg"), errors="coerce")
    catalogo["capacidade_vol_m3"] = pd.to_numeric(catalogo.get("capacidade_vol_m3"), errors="coerce")
    catalogo["max_entregas"] = pd.to_numeric(catalogo.get("max_entregas"), errors="coerce")
    catalogo["max_km_distancia"] = pd.to_numeric(catalogo.get("max_km_distancia"), errors="coerce")
    catalogo["ocupacao_minima_perc"] = pd.to_numeric(catalogo.get("ocupacao_minima_perc"), errors="coerce")

    if "ativo" in catalogo.columns:
        catalogo = catalogo.loc[catalogo["ativo"].fillna(True).astype(bool)].copy()

    catalogo["limite_manifestos"] = np.nan
    catalogo["manifestos_utilizados"] = 0

    if _normalizar_tipo_roteirizacao(tipo_roteirizacao) == "frota":
        cfg = _normalizar_configuracao_frota(configuracao_frota)
        limites = cfg.set_index("perfil")["quantidade"].to_dict() if len(cfg) > 0 else {}

        catalogo["limite_manifestos"] = catalogo["veiculo_tipo"].map(limites).astype(float)
        catalogo["limite_manifestos"] = catalogo["limite_manifestos"].fillna(0)

        if df_uso_frota_m4 is not None and len(df_uso_frota_m4) > 0:
            uso = df_uso_frota_m4.copy()
            col_uso = None
            for col in ("veiculo_tipo", "tipo", "perfil"):
                if col in uso.columns:
                    col_uso = col
                    break
            if col_uso is not None:
                uso[col_uso] = uso[col_uso].astype(str).str.strip()
                usados = uso.groupby(col_uso).size().to_dict()
                catalogo["manifestos_utilizados"] = catalogo["veiculo_tipo"].map(usados).fillna(0).astype(int)

        catalogo = catalogo.loc[catalogo["limite_manifestos"] > 0].copy()

    catalogo = catalogo.sort_values(
        by=["capacidade_peso_kg", "capacidade_vol_m3", "max_entregas", "max_km_distancia"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return catalogo


def _veiculo_tem_saldo(veiculo: pd.Series, tipo_roteirizacao: str) -> bool:
    if _normalizar_tipo_roteirizacao(tipo_roteirizacao) != "frota":
        return True

    limite = _num_safe(veiculo.get("limite_manifestos"), default=np.nan)
    usados = _num_safe(veiculo.get("manifestos_utilizados"), default=0)

    if pd.isna(limite):
        return True

    return int(usados) < int(limite)


def _consumir_veiculo(catalogo: pd.DataFrame, idx_catalogo: Optional[int], tipo_roteirizacao: str) -> None:
    if idx_catalogo is None:
        return
    if _normalizar_tipo_roteirizacao(tipo_roteirizacao) != "frota":
        return
    if idx_catalogo not in catalogo.index:
        return

    atual = _int_safe(catalogo.at[idx_catalogo, "manifestos_utilizados"], default=0)
    catalogo.at[idx_catalogo, "manifestos_utilizados"] = atual + 1


def _score_prioridade(row: pd.Series) -> Tuple[Any, ...]:
    prioridade_sim = 0 if str(row.get("prioridade", "")).strip().lower() in {"sim", "s", "1", "true"} else 1
    agendada = 0 if pd.notna(row.get("data_agenda")) else 1
    folga = _num_safe(row.get("folga_dias"), default=np.nan)
    folga_ordenacao = folga if pd.notna(folga) else 999999
    peso = _num_safe(row.get("peso_kg"), default=0)
    volume = _num_safe(row.get("vol_m3"), default=0)

    return (
        prioridade_sim,
        agendada,
        folga_ordenacao,
        -peso,
        -volume,
    )


def _ordenar_por_prioridade(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df.copy()

    trabalho = df.copy()
    trabalho["__ord__"] = trabalho.apply(_score_prioridade, axis=1)
    trabalho = trabalho.sort_values("__ord__").drop(columns="__ord__")
    return trabalho.reset_index(drop=True)


def _parada_key(row: pd.Series) -> str:
    destinatario = str(row.get("destinatario", "") or "").strip().upper()
    cidade = str(row.get("cidade", "") or "").strip().upper()
    uf = str(row.get("uf", "") or "").strip().upper()
    return f"{destinatario}|{cidade}|{uf}"


def _cliente_key(row: pd.Series) -> str:
    return str(row.get("cliente", "") or "").strip().upper()


def _filtrar_pool_por_criterio(df: pd.DataFrame, criterio: str, valor: str, uf_ref: str) -> pd.DataFrame:
    trabalho = df.copy()

    if criterio == "mesmo_cliente":
        filtrado = trabalho.loc[trabalho.apply(_cliente_key, axis=1) == valor].copy()
    elif criterio == "mesma_cidade":
        filtrado = trabalho.loc[
            (trabalho["cidade"].astype(str).str.strip().str.upper() == valor)
            & (trabalho["uf"].astype(str).str.strip().str.upper() == uf_ref)
        ].copy()
    elif criterio == "mesma_subregiao":
        filtrado = trabalho.loc[
            (trabalho["subregiao"].astype(str).str.strip().str.upper() == valor)
            & (trabalho["uf"].astype(str).str.strip().str.upper() == uf_ref)
        ].copy()
    elif criterio == "mesma_mesorregiao":
        filtrado = trabalho.loc[
            (trabalho["mesorregiao"].astype(str).str.strip().str.upper() == valor)
            & (trabalho["uf"].astype(str).str.strip().str.upper() == uf_ref)
        ].copy()
    else:
        filtrado = pd.DataFrame(columns=trabalho.columns)

    return _ordenar_por_prioridade(filtrado)


def _obter_grupos_ordenados(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    grupos: List[Tuple[str, str, str]] = []

    # 1) mesmo cliente
    if "cliente" in df.columns:
        base = (
            df.assign(__cliente=df["cliente"].astype(str).str.strip().str.upper())
            .loc[lambda x: x["__cliente"] != ""]
            .groupby("__cliente", as_index=False)
            .size()
            .sort_values(["size", "__cliente"], ascending=[False, True])
        )
        for _, row in base.iterrows():
            grupos.append(("mesmo_cliente", str(row["__cliente"]), ""))

    # 2) mesma cidade
    base = (
        df.assign(
            __cidade=df["cidade"].astype(str).str.strip().str.upper(),
            __uf=df["uf"].astype(str).str.strip().str.upper(),
        )
        .loc[lambda x: (x["__cidade"] != "") & (x["__uf"] != "")]
        .groupby(["__cidade", "__uf"], as_index=False)
        .size()
        .sort_values(["size", "__cidade", "__uf"], ascending=[False, True, True])
    )
    for _, row in base.iterrows():
        grupos.append(("mesma_cidade", str(row["__cidade"]), str(row["__uf"])))

    # 3) mesma sub-região
    base = (
        df.assign(
            __sub=df["subregiao"].astype(str).str.strip().str.upper(),
            __uf=df["uf"].astype(str).str.strip().str.upper(),
        )
        .loc[lambda x: (x["__sub"] != "") & (x["__uf"] != "")]
        .groupby(["__sub", "__uf"], as_index=False)
        .size()
        .sort_values(["size", "__sub", "__uf"], ascending=[False, True, True])
    )
    for _, row in base.iterrows():
        grupos.append(("mesma_subregiao", str(row["__sub"]), str(row["__uf"])))

    # 4) mesma mesorregião
    base = (
        df.assign(
            __meso=df["mesorregiao"].astype(str).str.strip().str.upper(),
            __uf=df["uf"].astype(str).str.strip().str.upper(),
        )
        .loc[lambda x: (x["__meso"] != "") & (x["__uf"] != "")]
        .groupby(["__meso", "__uf"], as_index=False)
        .size()
        .sort_values(["size", "__meso", "__uf"], ascending=[False, True, True])
    )
    for _, row in base.iterrows():
        grupos.append(("mesma_mesorregiao", str(row["__meso"]), str(row["__uf"])))

    return grupos


def _avaliar_combo_no_veiculo(df_combo: pd.DataFrame, veiculo: pd.Series) -> Dict[str, Any]:
    peso_total = float(pd.to_numeric(df_combo["peso_kg"], errors="coerce").fillna(0).sum())
    vol_total = float(pd.to_numeric(df_combo["vol_m3"], errors="coerce").fillna(0).sum())

    capacidade_peso = _num_safe(veiculo.get("capacidade_peso_kg"), default=np.nan)
    capacidade_vol = _num_safe(veiculo.get("capacidade_vol_m3"), default=np.nan)
    max_entregas = _num_safe(veiculo.get("max_entregas"), default=np.nan)
    max_km = _num_safe(veiculo.get("max_km_distancia"), default=np.nan)
    ocupacao_min = _num_safe(veiculo.get("ocupacao_minima_perc"), default=np.nan)

    if pd.isna(ocupacao_min):
        ocupacao_min = OCUPACAO_MINIMA_PADRAO * 100.0

    ocupacao_min = float(ocupacao_min) / 100.0 if ocupacao_min > 1 else float(ocupacao_min)

    ocup_peso = (peso_total / capacidade_peso) if pd.notna(capacidade_peso) and capacidade_peso > 0 else 0.0
    ocup_vol = (vol_total / capacidade_vol) if pd.notna(capacidade_vol) and capacidade_vol > 0 else 0.0

    if ocup_peso >= ocup_vol:
        ocupacao_oficial = ocup_peso
        ocupacao_secundaria = ocup_vol
        base_oficial = "peso"
    else:
        ocupacao_oficial = ocup_vol
        ocupacao_secundaria = ocup_peso
        base_oficial = "volume"

    qtd_paradas = int(df_combo.apply(_parada_key, axis=1).nunique())
    km_referencia = pd.to_numeric(df_combo["distancia_rodoviaria_est_km"], errors="coerce").max()

    cabe_peso = pd.isna(capacidade_peso) or peso_total <= capacidade_peso + 1e-9
    cabe_vol = pd.isna(capacidade_vol) or vol_total <= capacidade_vol + 1e-9
    cabe_entregas = pd.isna(max_entregas) or qtd_paradas <= max_entregas
    cabe_km = pd.isna(max_km) or pd.isna(km_referencia) or km_referencia <= max_km

    ocupa_minimo = ocupacao_oficial >= ocupacao_min - 1e-9
    ocupa_secundario = ocupacao_secundaria >= OCUPACAO_MINIMA_SECUNDARIA_PADRAO - 1e-9
    ocupa_maximo = ocupacao_oficial <= OCUPACAO_MAXIMA_PADRAO + 1e-9

    if not cabe_peso:
        motivo = "excede_capacidade_peso"
    elif not cabe_vol:
        motivo = "excede_capacidade_volume"
    elif not cabe_entregas:
        motivo = "excede_max_entregas"
    elif not cabe_km:
        motivo = "excede_raio_km"
    elif not ocupa_minimo:
        motivo = "abaixo_ocupacao_minima"
    elif not ocupa_secundario:
        motivo = "abaixo_ocupacao_secundaria"
    elif not ocupa_maximo:
        motivo = "acima_ocupacao_maxima"
    else:
        motivo = None

    return {
        "aceito": motivo is None,
        "motivo_reprovacao": motivo,
        "veiculo_tipo": str(veiculo.get("veiculo_tipo", "")),
        "peso_total_kg": round(peso_total, 4),
        "vol_total_m3": round(vol_total, 4),
        "qtd_itens": int(len(df_combo)),
        "qtd_paradas": int(qtd_paradas),
        "km_referencia": None if pd.isna(km_referencia) else round(float(km_referencia), 4),
        "ocupacao_peso_perc": round(float(ocup_peso), 6),
        "ocupacao_volume_perc": round(float(ocup_vol), 6),
        "ocupacao_oficial_perc": round(float(ocupacao_oficial), 6),
        "base_carga_oficial": base_oficial,
        "capacidade_peso_kg": None if pd.isna(capacidade_peso) else float(capacidade_peso),
        "capacidade_vol_m3": None if pd.isna(capacidade_vol) else float(capacidade_vol),
        "max_entregas": None if pd.isna(max_entregas) else float(max_entregas),
        "max_km_distancia": None if pd.isna(max_km) else float(max_km),
        "ocupacao_minima_aplicada": round(float(ocupacao_min), 6),
    }


def _gerar_combo_guloso(df_pool: pd.DataFrame, veiculo: pd.Series) -> pd.DataFrame:
    if df_pool is None or len(df_pool) == 0:
        return pd.DataFrame(columns=df_pool.columns if df_pool is not None else [])

    pool = _ordenar_por_prioridade(df_pool)
    combo = pd.DataFrame(columns=pool.columns)

    for _, row in pool.iterrows():
        candidato = pd.concat([combo, row.to_frame().T], ignore_index=True)
        avaliacao = _avaliar_combo_no_veiculo(candidato, veiculo)

        if avaliacao["motivo_reprovacao"] in {None, "abaixo_ocupacao_minima", "abaixo_ocupacao_secundaria"}:
            if avaliacao["ocupacao_oficial_perc"] <= OCUPACAO_MAXIMA_PADRAO + 1e-9:
                combo = candidato

    return combo.reset_index(drop=True)


def _gerar_manifesto_id(rodada_id: str, sequencial: int) -> str:
    base = str(rodada_id).replace("-", "").upper()[:10]
    return f"M5_1-{base}-{sequencial:05d}"


def _gerar_pre_manifesto(
    df_combo: pd.DataFrame,
    avaliacao: Dict[str, Any],
    manifesto_id: str,
    criterio_agrupamento: str,
    valor_agrupamento: str,
) -> Dict[str, Any]:
    cliente_ref = ""
    cidade_ref = ""
    subregiao_ref = ""
    mesorregiao_ref = ""
    uf_ref = ""

    if len(df_combo) > 0:
        cliente_ref = str(df_combo["cliente"].iloc[0] or "").strip()
        cidade_ref = str(df_combo["cidade"].iloc[0] or "").strip()
        subregiao_ref = str(df_combo["subregiao"].iloc[0] or "").strip()
        mesorregiao_ref = str(df_combo["mesorregiao"].iloc[0] or "").strip()
        uf_ref = str(df_combo["uf"].iloc[0] or "").strip()

    return {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": "pre_manifesto_composto",
        "origem_modulo": "m5.1_manifestos_compostos_Rodada_1",
        "criterio_agrupamento": criterio_agrupamento,
        "valor_agrupamento": valor_agrupamento,
        "veiculo_tipo": avaliacao["veiculo_tipo"],
        "qtd_itens": avaliacao["qtd_itens"],
        "qtd_paradas": avaliacao["qtd_paradas"],
        "peso_total_kg": avaliacao["peso_total_kg"],
        "vol_total_m3": avaliacao["vol_total_m3"],
        "km_referencia": avaliacao["km_referencia"],
        "ocupacao_peso_perc": avaliacao["ocupacao_peso_perc"],
        "ocupacao_volume_perc": avaliacao["ocupacao_volume_perc"],
        "ocupacao_oficial_perc": avaliacao["ocupacao_oficial_perc"],
        "base_carga_oficial": avaliacao["base_carga_oficial"],
        "capacidade_peso_kg_veiculo": avaliacao["capacidade_peso_kg"],
        "capacidade_vol_m3_veiculo": avaliacao["capacidade_vol_m3"],
        "max_entregas_veiculo": avaliacao["max_entregas"],
        "max_km_distancia_veiculo": avaliacao["max_km_distancia"],
        "ocupacao_minima_aplicada": avaliacao["ocupacao_minima_aplicada"],
        "cliente_referencia": cliente_ref,
        "cidade_referencia": cidade_ref,
        "subregiao_referencia": subregiao_ref,
        "mesorregiao_referencia": mesorregiao_ref,
        "uf_referencia": uf_ref,
    }


def _gerar_itens_pre_manifesto(
    df_combo: pd.DataFrame,
    manifesto_id: str,
    criterio_agrupamento: str,
    valor_agrupamento: str,
    veiculo_tipo: str,
) -> pd.DataFrame:
    itens = df_combo.copy().reset_index(drop=True)
    itens["manifesto_id"] = manifesto_id
    itens["origem_modulo"] = "m5.1_manifestos_compostos_Rodada_1"
    itens["criterio_agrupamento"] = criterio_agrupamento
    itens["valor_agrupamento"] = valor_agrupamento
    itens["veiculo_tipo_m5_1"] = veiculo_tipo
    itens["ordem_pre_manifesto"] = np.arange(1, len(itens) + 1)
    return itens


def executar_m5_1_manifestos_compostos_rodada_1(
    *,
    df_remanescente_roteirizavel_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
    rodada_id: str,
    data_base_roteirizacao: Any,
    tipo_roteirizacao: str,
    configuracao_frota: Any = None,
    caminhos_pipeline: Optional[Dict[str, str]] = None,
    df_uso_frota_m4: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    inicio = _agora()

    # trava dura: entrada desta rodada é somente o remanescente oficial do M4
    df_base = _normalizar_remanescente_m4(df_remanescente_roteirizavel_bloco_4)
    catalogo = _normalizar_catalogo_veiculos(
        df_veiculos_tratados=df_veiculos_tratados,
        tipo_roteirizacao=tipo_roteirizacao,
        configuracao_frota=configuracao_frota,
        df_uso_frota_m4=df_uso_frota_m4,
    )

    if len(df_base) == 0:
        outputs = {
            "df_pre_manifestos_bloco_5_1": pd.DataFrame(),
            "df_itens_pre_manifestos_bloco_5_1": pd.DataFrame(),
            "df_tentativas_bloco_5_1": pd.DataFrame(),
            "df_remanescente_roteirizavel_bloco_5_1": pd.DataFrame(),
            "df_uso_frota_m5_1": pd.DataFrame(),
        }
        meta = {
            "resumo_m5_1": {
                "modulo": "m5.1_manifestos_compostos_Rodada_1",
                "entrada_remanescente_m4": 0,
                "pre_manifestos_gerados_m5_1": 0,
                "itens_pre_manifestados_m5_1": 0,
                "remanescente_m5_1": 0,
                "tempo_execucao_ms": _duracao_ms(inicio),
            }
        }
        return outputs, meta

    if len(catalogo) == 0:
        outputs = {
            "df_pre_manifestos_bloco_5_1": pd.DataFrame(),
            "df_itens_pre_manifestos_bloco_5_1": pd.DataFrame(),
            "df_tentativas_bloco_5_1": pd.DataFrame(
                [
                    {
                        "criterio_agrupamento": "catalogo",
                        "valor_agrupamento": None,
                        "veiculo_tipo": None,
                        "qtd_pool": int(len(df_base)),
                        "qtd_combo": 0,
                        "aceito": False,
                        "motivo_reprovacao": "sem_veiculos_disponiveis",
                    }
                ]
            ),
            "df_remanescente_roteirizavel_bloco_5_1": df_base.copy(),
            "df_uso_frota_m5_1": pd.DataFrame(),
        }
        meta = {
            "resumo_m5_1": {
                "modulo": "m5.1_manifestos_compostos_Rodada_1",
                "entrada_remanescente_m4": int(len(df_base)),
                "pre_manifestos_gerados_m5_1": 0,
                "itens_pre_manifestados_m5_1": 0,
                "remanescente_m5_1": int(len(df_base)),
                "tempo_execucao_ms": _duracao_ms(inicio),
                "motivo_sem_execucao": "sem_veiculos_disponiveis",
            }
        }
        return outputs, meta

    usados: set[str] = set()
    pre_manifestos: List[Dict[str, Any]] = []
    itens_pre_manifestos: List[pd.DataFrame] = []
    tentativas: List[Dict[str, Any]] = []
    sequencial_manifesto = 1

    fila = _ordenar_por_prioridade(df_base)

    grupos = _obter_grupos_ordenados(fila)

    for criterio, valor, uf_ref in grupos:
        disponiveis = fila.loc[~fila["id_linha_pipeline"].astype(str).isin(usados)].copy()
        if len(disponiveis) < 2:
            break

        pool = _filtrar_pool_por_criterio(disponiveis, criterio=criterio, valor=valor, uf_ref=uf_ref)
        if len(pool) < 2:
            continue

        aceitou = False

        for idx_catalogo, veiculo in catalogo.iterrows():
            if not _veiculo_tem_saldo(veiculo, tipo_roteirizacao):
                tentativas.append(
                    {
                        "criterio_agrupamento": criterio,
                        "valor_agrupamento": valor,
                        "veiculo_tipo": str(veiculo.get("veiculo_tipo", "")),
                        "qtd_pool": int(len(pool)),
                        "qtd_combo": 0,
                        "aceito": False,
                        "motivo_reprovacao": "veiculo_sem_saldo_modo_frota",
                    }
                )
                continue

            combo = _gerar_combo_guloso(pool, veiculo)
            avaliacao = _avaliar_combo_no_veiculo(combo, veiculo)

            tentativas.append(
                {
                    "criterio_agrupamento": criterio,
                    "valor_agrupamento": valor,
                    "veiculo_tipo": str(veiculo.get("veiculo_tipo", "")),
                    "qtd_pool": int(len(pool)),
                    "qtd_combo": int(len(combo)),
                    "aceito": bool(avaliacao["aceito"]),
                    "motivo_reprovacao": avaliacao["motivo_reprovacao"],
                    "peso_total_kg": avaliacao["peso_total_kg"],
                    "vol_total_m3": avaliacao["vol_total_m3"],
                    "qtd_paradas": avaliacao["qtd_paradas"],
                    "km_referencia": avaliacao["km_referencia"],
                    "ocupacao_oficial_perc": avaliacao["ocupacao_oficial_perc"],
                    "base_carga_oficial": avaliacao["base_carga_oficial"],
                }
            )

            if not avaliacao["aceito"]:
                continue

            ids_combo = combo["id_linha_pipeline"].astype(str).tolist()
            if any(item_id in usados for item_id in ids_combo):
                continue

            manifesto_id = _gerar_manifesto_id(rodada_id, sequencial_manifesto)
            sequencial_manifesto += 1

            pre_manifestos.append(
                _gerar_pre_manifesto(
                    df_combo=combo,
                    avaliacao=avaliacao,
                    manifesto_id=manifesto_id,
                    criterio_agrupamento=criterio,
                    valor_agrupamento=valor,
                )
            )
            itens_pre_manifestos.append(
                _gerar_itens_pre_manifesto(
                    df_combo=combo,
                    manifesto_id=manifesto_id,
                    criterio_agrupamento=criterio,
                    valor_agrupamento=valor,
                    veiculo_tipo=avaliacao["veiculo_tipo"],
                )
            )

            usados.update(ids_combo)
            _consumir_veiculo(catalogo, idx_catalogo, tipo_roteirizacao)
            aceitou = True
            break

        if aceitou:
            continue

    df_pre_manifestos = pd.DataFrame(pre_manifestos)
    df_itens_pre_manifestos = (
        pd.concat(itens_pre_manifestos, ignore_index=True) if len(itens_pre_manifestos) > 0 else pd.DataFrame()
    )
    df_tentativas = pd.DataFrame(tentativas)
    df_remanescente_m5_1 = fila.loc[~fila["id_linha_pipeline"].astype(str).isin(usados)].copy().reset_index(drop=True)

    df_uso_frota_m5_1 = catalogo.copy()
    if len(df_uso_frota_m5_1) > 0:
        df_uso_frota_m5_1["origem_modulo"] = "m5.1_manifestos_compostos_Rodada_1"

    resumo = {
        "modulo": "m5.1_manifestos_compostos_Rodada_1",
        "entrada_remanescente_m4": int(len(df_base)),
        "pre_manifestos_gerados_m5_1": int(len(df_pre_manifestos)),
        "itens_pre_manifestados_m5_1": int(len(df_itens_pre_manifestos)),
        "remanescente_m5_1": int(len(df_remanescente_m5_1)),
        "tentativas_m5_1": int(len(df_tentativas)),
        "tempo_execucao_ms": _duracao_ms(inicio),
        "criterios_aplicados": [
            "mesmo_cliente",
            "mesma_cidade",
            "mesma_subregiao",
            "mesma_mesorregiao",
        ],
        "ordem_priorizacao": [
            "prioridade_sim",
            "agendada",
            "menor_folga_para_maior",
        ],
        "ordem_teste_veiculos": "maior_para_menor",
    }

    outputs = {
        "df_pre_manifestos_bloco_5_1": df_pre_manifestos,
        "df_itens_pre_manifestos_bloco_5_1": df_itens_pre_manifestos,
        "df_tentativas_bloco_5_1": df_tentativas,
        "df_remanescente_roteirizavel_bloco_5_1": df_remanescente_m5_1,
        "df_uso_frota_m5_1": df_uso_frota_m5_1,
    }

    meta = {
        "resumo_m5_1": resumo,
        "auditoria_m5_1": {
            "total_tentativas": int(len(df_tentativas)),
            "total_pre_manifestos": int(len(df_pre_manifestos)),
            "total_itens_pre_manifestados": int(len(df_itens_pre_manifestos)),
            "total_remanescentes": int(len(df_remanescente_m5_1)),
        },
        "amostras_m5_1": {
            "pre_manifestos": _to_records(df_pre_manifestos.head(10)),
            "remanescente": _to_records(df_remanescente_m5_1.head(10)),
        },
    }

    return outputs, meta
