from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# MÓDULO 4 - GERAÇÃO DE MANIFESTOS FECHADOS
# VERSÃO OTIMIZADA SEM ALTERAR REGRA DE NEGÓCIO
#
# Regras principais preservadas:
# - recebe somente a carteira roteirizável
# - usa peso_calculado como base oficial de ocupação/capacidade
# - exclusivo sai primeiro e não exige ocupação mínima
# - exclusivo pode puxar outros CTes do mesmo cliente
# - depois processa os demais por prioridade operacional:
#   1) prioridade_embarque = 1
#   2) agendadas
#   3) leadtimes com folga positiva (menor folga primeiro)
#   4) leadtimes vencidos
# - menor veículo é orientação, não trava: se não couber, sobe
# - modo carteira: usa catálogo aberto
# - modo frota: respeita configuracao_frota (perfil + quantidade)
# ============================================================

OCUPACAO_MINIMA_PADRAO = 0.70
CHAVES_PARADA = ["destinatario", "cidade", "uf"]


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


def _resolver_coluna_tipo_veiculo(df_veiculos: pd.DataFrame) -> str:
    if "tipo" in df_veiculos.columns:
        return "tipo"
    if "perfil" in df_veiculos.columns:
        return "perfil"
    raise Exception("Faltam colunas mínimas na base de veículos:\n- tipo ou perfil")


def _txt_norm(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def _scalar_safe(x: Any) -> Any:
    if isinstance(x, pd.Series):
        if len(x) == 0:
            return np.nan
        return x.iloc[0]
    if isinstance(x, np.ndarray):
        if len(x) == 0:
            return np.nan
        return x[0]
    if isinstance(x, list):
        if len(x) == 0:
            return np.nan
        return x[0]
    if isinstance(x, tuple):
        if len(x) == 0:
            return np.nan
        return x[0]
    return x


def _bool_safe(x: Any) -> bool:
    x = _scalar_safe(x)

    if pd.isna(x):
        return False

    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    if isinstance(x, (int, float, np.integer, np.floating)):
        if pd.isna(x):
            return False
        return bool(int(x))

    txt = str(x).strip().lower()
    return txt in {"true", "1", "sim", "s", "yes", "y"}


def _num_safe(x: Any, default: float = np.nan) -> float:
    x = _scalar_safe(x)
    val = pd.to_numeric(x, errors="coerce")
    return float(val) if pd.notna(val) else default


def _int_safe(x: Any, default: int = 0) -> int:
    x = _scalar_safe(x)
    val = pd.to_numeric(x, errors="coerce")
    if pd.isna(val):
        return default
    return int(val)


def _deduplicar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df.columns) == 0:
        return df.copy()

    if not df.columns.duplicated().any():
        return df.copy()

    return df.loc[:, ~df.columns.duplicated()].copy()


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


def _normalizar_flag_agendada(serie: pd.Series) -> pd.Series:
    if serie is None:
        return pd.Series(dtype=bool)
    return serie.apply(_bool_safe).astype(bool)


def _normalizar_tipo_roteirizacao(valor: Any) -> str:
    txt = str(valor).strip().lower() if valor is not None else "carteira"
    if txt not in {"carteira", "frota"}:
        return "carteira"
    return txt


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
    cfg = cfg.loc[(cfg["perfil"] != "") & (cfg["quantidade"] > 0)].copy()

    if len(cfg) == 0:
        return pd.DataFrame(columns=["perfil", "quantidade"])

    cfg = cfg.groupby("perfil", as_index=False)["quantidade"].sum()
    return cfg.reset_index(drop=True)


def _preparar_catalogo_veiculos(
    df_veic: pd.DataFrame,
    coluna_tipo_veiculo: str,
    tipo_roteirizacao: str,
    configuracao_frota: Any,
) -> pd.DataFrame:
    cat = df_veic.copy()

    colunas_min = [
        coluna_tipo_veiculo,
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
    ]
    cat = cat.loc[cat[colunas_min].notna().all(axis=1)].copy()

    cat["tipo"] = cat[coluna_tipo_veiculo].astype(str).str.strip()
    cat["capacidade_peso_kg"] = pd.to_numeric(cat["capacidade_peso_kg"], errors="coerce")
    cat["capacidade_vol_m3"] = pd.to_numeric(cat["capacidade_vol_m3"], errors="coerce")
    cat["max_entregas"] = pd.to_numeric(cat["max_entregas"], errors="coerce")
    cat["max_km_distancia"] = pd.to_numeric(cat["max_km_distancia"], errors="coerce")

    cat = (
        cat.groupby("tipo", as_index=False)
        .agg(
            {
                "capacidade_peso_kg": "max",
                "capacidade_vol_m3": "max",
                "max_entregas": "max",
                "max_km_distancia": "max",
            }
        )
        .sort_values(
            by=["capacidade_peso_kg", "capacidade_vol_m3", "max_entregas", "max_km_distancia"],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )

    tipo_roteirizacao = _normalizar_tipo_roteirizacao(tipo_roteirizacao)

    if tipo_roteirizacao == "frota":
        cfg = _normalizar_configuracao_frota(configuracao_frota)
        if len(cfg) == 0:
            raise Exception(
                "tipo_roteirizacao = 'frota', mas configuracao_frota está vazia ou inválida."
            )

        cat = cat.merge(cfg, how="inner", left_on="tipo", right_on="perfil")
        if len(cat) == 0:
            raise Exception(
                "Nenhum perfil da configuracao_frota foi encontrado no catálogo de veículos."
            )

        cat["limite_manifestos"] = pd.to_numeric(cat["quantidade"], errors="coerce").fillna(0).astype(int)
        cat.drop(columns=["perfil", "quantidade"], inplace=True, errors="ignore")
    else:
        cat["limite_manifestos"] = np.nan

    cat["manifestos_utilizados"] = 0
    cat["ordem_porte"] = np.arange(1, len(cat) + 1)
    return cat.reset_index(drop=True)


def _chave_parada_df(df_: pd.DataFrame) -> pd.Series:
    return (
        df_["destinatario"].astype(str).fillna("").str.strip().str.upper()
        + "|"
        + df_["cidade"].astype(str).fillna("").str.strip().str.upper()
        + "|"
        + df_["uf"].astype(str).fillna("").str.strip().str.upper()
    )


def _obter_base_carga_oficial(df_combo: pd.DataFrame) -> float:
    if "peso_calculado" not in df_combo.columns:
        return 0.0
    return float(pd.to_numeric(df_combo["peso_calculado"], errors="coerce").fillna(0).sum())


def _avaliar_combo_no_veiculo(
    df_combo: pd.DataFrame,
    veic: pd.Series,
    ignorar_ocupacao_minima: bool = False,
) -> Dict[str, Any]:
    base_carga_total = _obter_base_carga_oficial(df_combo)
    peso_total_kg = float(pd.to_numeric(df_combo["peso_kg"], errors="coerce").fillna(0).sum())
    vol_total_m3 = float(pd.to_numeric(df_combo["vol_m3"], errors="coerce").fillna(0).sum())
    km_combo = float(pd.to_numeric(df_combo["distancia_rodoviaria_est_km"], errors="coerce").max())

    col_cte = "cte" if "cte" in df_combo.columns else "id_linha_pipeline"
    qtd_ctes = int(df_combo[col_cte].astype(str).nunique())
    qtd_itens = int(len(df_combo))
    qtd_paradas = int(_chave_parada_df(df_combo).nunique())

    cap_peso = float(veic["capacidade_peso_kg"])
    cap_vol = float(veic["capacidade_vol_m3"])
    max_entregas = int(veic["max_entregas"])
    max_km = float(veic["max_km_distancia"])

    cabe_carga_oficial = base_carga_total <= cap_peso
    cabe_paradas = qtd_paradas <= max_entregas
    cabe_km = km_combo <= max_km if pd.notna(km_combo) else False

    ocupacao_oficial = base_carga_total / cap_peso if pd.notna(cap_peso) and cap_peso > 0 else np.nan
    passa_ocupacao = True if ignorar_ocupacao_minima else (
        pd.notna(ocupacao_oficial) and ocupacao_oficial >= OCUPACAO_MINIMA_PADRAO
    )

    return {
        "veiculo_tipo": veic["tipo"],
        "capacidade_peso_kg": cap_peso,
        "capacidade_vol_m3": cap_vol,
        "max_entregas": max_entregas,
        "max_km_distancia": max_km,
        "base_carga_oficial": round(base_carga_total, 3),
        "peso_total_kg": round(peso_total_kg, 3),
        "vol_total_m3": round(vol_total_m3, 3),
        "km_referencia": round(km_combo, 2) if pd.notna(km_combo) else np.nan,
        "qtd_itens": qtd_itens,
        "qtd_ctes": qtd_ctes,
        "qtd_paradas": qtd_paradas,
        "cabe_carga_oficial": cabe_carga_oficial,
        "cabe_paradas": cabe_paradas,
        "cabe_km": cabe_km,
        "ocupacao_oficial_perc": round(float(ocupacao_oficial * 100), 2) if pd.notna(ocupacao_oficial) else np.nan,
        "passa_ocupacao": passa_ocupacao,
        "ignorar_ocupacao_minima": bool(ignorar_ocupacao_minima),
        "aceito": bool(cabe_carga_oficial and cabe_paradas and cabe_km and passa_ocupacao),
    }


def _veiculo_disponivel_no_modo_frota(veic: pd.Series, tipo_roteirizacao: str) -> bool:
    tipo_roteirizacao = _normalizar_tipo_roteirizacao(tipo_roteirizacao)
    if tipo_roteirizacao == "carteira":
        return True

    limite = _num_safe(veic.get("limite_manifestos"), default=np.nan)
    usados = _num_safe(veic.get("manifestos_utilizados"), default=0)

    if pd.isna(limite):
        return True

    return int(usados) < int(limite)


def _avaliar_combo_catalogo(
    df_combo: pd.DataFrame,
    catalogo_veiculos: pd.DataFrame,
    tipo_roteirizacao: str,
    ignorar_ocupacao_minima: bool = False,
) -> Dict[str, Any]:
    tentativas = []

    for idx, veic in catalogo_veiculos.sort_values("ordem_porte").iterrows():
        if not _veiculo_disponivel_no_modo_frota(veic, tipo_roteirizacao):
            tentativas.append(
                {
                    "veiculo_tipo": veic["tipo"],
                    "resultado_teste": "rejeitado",
                    "motivo_reprovacao": "perfil_sem_disponibilidade_no_modo_frota",
                }
            )
            continue

        r = _avaliar_combo_no_veiculo(
            df_combo=df_combo,
            veic=veic,
            ignorar_ocupacao_minima=ignorar_ocupacao_minima,
        )
        r["resultado_teste"] = "aceito" if r["aceito"] else "rejeitado"

        if not r["aceito"]:
            motivos = []
            if not r.get("cabe_carga_oficial", True):
                motivos.append("excede_capacidade_peso_oficial")
            if not r.get("cabe_paradas", True):
                motivos.append("excede_max_entregas")
            if not r.get("cabe_km", True):
                motivos.append("excede_max_km")
            if not r.get("passa_ocupacao", True):
                motivos.append("nao_atinge_ocupacao_minima")
            r["motivo_reprovacao"] = "|".join(motivos) if motivos else "nao_cabe_ou_nao_atinge_regra_minima"

        tentativas.append(r)

        if r["aceito"]:
            return {
                **r,
                "catalogo_idx": idx,
                "tentativas": tentativas,
            }

    return {
        "aceito": False,
        "motivo_reprovacao": "nao_cabe_ou_nao_atinge_regra_minima",
        "tentativas": tentativas,
    }


def _consumir_veiculo_catalogo(
    catalogo_veiculos: pd.DataFrame,
    catalogo_idx: Optional[int],
    tipo_roteirizacao: str,
) -> None:
    if catalogo_idx is None:
        return

    tipo_roteirizacao = _normalizar_tipo_roteirizacao(tipo_roteirizacao)
    if tipo_roteirizacao != "frota":
        return

    if catalogo_idx not in catalogo_veiculos.index:
        return

    atual = _int_safe(catalogo_veiculos.at[catalogo_idx, "manifestos_utilizados"], default=0)
    catalogo_veiculos.at[catalogo_idx, "manifestos_utilizados"] = atual + 1


def _gerar_resumo_manifesto(
    df_combo: pd.DataFrame,
    avaliacao: Dict[str, Any],
    manifesto_id: str,
    tipo_manifesto: str,
    origem_etapa: str,
) -> Dict[str, Any]:
    linha = {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": tipo_manifesto,
        "veiculo_tipo": avaliacao["veiculo_tipo"],
        "qtd_itens": avaliacao["qtd_itens"],
        "qtd_ctes": avaliacao["qtd_ctes"],
        "qtd_paradas": avaliacao["qtd_paradas"],
        "base_carga_oficial": avaliacao["base_carga_oficial"],
        "peso_total_kg": avaliacao["peso_total_kg"],
        "vol_total_m3": avaliacao["vol_total_m3"],
        "km_referencia": avaliacao["km_referencia"],
        "ocupacao_oficial_perc": avaliacao["ocupacao_oficial_perc"],
        "capacidade_peso_kg_veiculo": avaliacao["capacidade_peso_kg"],
        "capacidade_vol_m3_veiculo": avaliacao["capacidade_vol_m3"],
        "max_entregas_veiculo": avaliacao["max_entregas"],
        "max_km_distancia_veiculo": avaliacao["max_km_distancia"],
        "ignorar_ocupacao_minima": avaliacao["ignorar_ocupacao_minima"],
        "ranking_prioridade_min": pd.to_numeric(
            df_combo.get("ranking_prioridade", pd.Series([np.nan] * len(df_combo))),
            errors="coerce",
        ).min(),
        "score_prioridade_max": pd.to_numeric(
            df_combo.get("score_prioridade_preliminar", pd.Series([np.nan] * len(df_combo))),
            errors="coerce",
        ).max(),
        "origem_modulo": 4,
        "origem_etapa": origem_etapa,
    }

    if avaliacao["qtd_paradas"] == 1:
        linha["destinatario"] = df_combo["destinatario"].iloc[0]
        linha["cidade"] = df_combo["cidade"].iloc[0]
        linha["uf"] = df_combo["uf"].iloc[0]
        linha["mesorregiao"] = df_combo["mesorregiao"].iloc[0] if "mesorregiao" in df_combo.columns else np.nan
        linha["subregiao"] = df_combo["subregiao"].iloc[0] if "subregiao" in df_combo.columns else np.nan
    else:
        linha["destinatario"] = df_combo["destinatario"].iloc[0]
        linha["cidade"] = "MULTICIDADE"
        linha["uf"] = (
            df_combo["uf"].mode().iloc[0]
            if "uf" in df_combo.columns and df_combo["uf"].notna().any()
            else np.nan
        )
        linha["mesorregiao"] = "MULTI"
        linha["subregiao"] = "MULTI"

    return linha


def _eh_exclusivo(row: pd.Series) -> bool:
    if "veiculo_exclusivo_flag" in row.index:
        return _bool_safe(row.get("veiculo_exclusivo_flag"))
    return _bool_safe(row.get("veiculo_exclusivo"))


def _score_ordem_fila(row: pd.Series) -> Tuple[Any, ...]:
    exclusivo = 0 if _eh_exclusivo(row) else 1

    prioridade_embarque = _num_safe(row.get("prioridade_embarque", np.nan), default=np.nan)
    data_agenda = _scalar_safe(row.get("data_agenda", pd.NaT))
    folga = _num_safe(row.get("folga_dias", np.nan), default=np.nan)
    score = _num_safe(row.get("score_prioridade_preliminar", 0), default=0)
    km = _num_safe(row.get("distancia_rodoviaria_est_km", np.nan), default=np.nan)
    base_carga = _num_safe(row.get("peso_calculado", 0), default=0)

    if pd.notna(prioridade_embarque) and int(prioridade_embarque) == 1:
        grupo = 1
    elif pd.notna(data_agenda):
        grupo = 2
    elif pd.notna(folga) and folga >= 0:
        grupo = 3
    else:
        grupo = 4

    folga_ordem = folga if pd.notna(folga) else 999999
    km_ordem = km if pd.notna(km) else 999999

    return (
        exclusivo,
        grupo,
        folga_ordem,
        -score,
        km_ordem,
        -base_carga,
    )


def _ordenar_fila(df_: pd.DataFrame) -> pd.DataFrame:
    if len(df_) == 0:
        return df_.copy()
    return (
        df_.assign(__ord__=df_.apply(_score_ordem_fila, axis=1))
        .sort_values("__ord__")
        .drop(columns="__ord__")
        .reset_index(drop=True)
    )


def _ordenar_mesmo_cliente(df_: pd.DataFrame) -> pd.DataFrame:
    if len(df_) == 0:
        return df_.copy()
    return _ordenar_fila(df_)


def _materializar_selecionados(selecionados: List[pd.Series], nova_linha: Optional[pd.Series] = None) -> pd.DataFrame:
    rows = list(selecionados)
    if nova_linha is not None:
        rows.append(nova_linha)
    if len(rows) == 0:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def _tentar_exclusivo_com_mesmo_cliente(
    linha_exclusiva: pd.Series,
    fila_disponivel: pd.DataFrame,
    catalogo_veiculos: pd.DataFrame,
    tipo_roteirizacao: str,
) -> Dict[str, Any]:
    cliente_ref = linha_exclusiva["destinatario"]

    base_cliente = fila_disponivel.loc[
        fila_disponivel["destinatario"].astype(str) == str(cliente_ref)
    ].copy()

    if len(base_cliente) == 0:
        base_cliente = pd.DataFrame([linha_exclusiva])

    base_cliente = _ordenar_mesmo_cliente(base_cliente)

    base_cliente["__is_exclusivo_anchor"] = base_cliente["id_linha_pipeline"].astype(str).eq(
        str(linha_exclusiva["id_linha_pipeline"])
    )
    base_cliente = base_cliente.sort_values(
        by=["__is_exclusivo_anchor"],
        ascending=[False],
    ).drop(columns="__is_exclusivo_anchor").reset_index(drop=True)

    melhor_resultado: Optional[Dict[str, Any]] = None

    for idx, veic in catalogo_veiculos.sort_values("ordem_porte").iterrows():
        if not _veiculo_disponivel_no_modo_frota(veic, tipo_roteirizacao):
            continue

        selecionados: List[pd.Series] = []
        for _, row in base_cliente.iterrows():
            teste = _materializar_selecionados(selecionados, row)

            aval = _avaliar_combo_no_veiculo(
                teste,
                veic=veic,
                ignorar_ocupacao_minima=True,
            )

            if aval["cabe_carga_oficial"] and aval["cabe_paradas"] and aval["cabe_km"]:
                selecionados.append(row)

        if len(selecionados) == 0:
            continue

        df_combo = _materializar_selecionados(selecionados)

        if str(linha_exclusiva["id_linha_pipeline"]) not in set(df_combo["id_linha_pipeline"].astype(str)):
            continue

        avaliacao = _avaliar_combo_no_veiculo(
            df_combo,
            veic=veic,
            ignorar_ocupacao_minima=True,
        )

        if avaliacao["aceito"]:
            resultado = {
                "df_combo": df_combo,
                "avaliacao": avaliacao,
                "catalogo_idx": idx,
                "veic_tipo": veic["tipo"],
            }
            melhor_resultado = resultado
            break

    if melhor_resultado is None:
        return {
            "aceito": False,
            "motivo_reprovacao": "exclusivo_sem_veiculo_viavel",
            "tentativas": [],
        }

    return {
        "aceito": True,
        **melhor_resultado,
    }


def _tentar_fechamento_direto_linha(
    linha_base: pd.Series,
    fila_disponivel: pd.DataFrame,
    catalogo_veiculos: pd.DataFrame,
    tipo_roteirizacao: str,
) -> Dict[str, Any]:
    df_combo = pd.DataFrame([linha_base]).reset_index(drop=True)
    avaliacao = _avaliar_combo_catalogo(
        df_combo=df_combo,
        catalogo_veiculos=catalogo_veiculos,
        tipo_roteirizacao=tipo_roteirizacao,
        ignorar_ocupacao_minima=False,
    )
    if avaliacao["aceito"]:
        return {
            "aceito": True,
            "df_combo": df_combo,
            "avaliacao": avaliacao,
            "catalogo_idx": avaliacao["catalogo_idx"],
        }

    return {
        "aceito": False,
        "motivo_reprovacao": avaliacao.get("motivo_reprovacao", "nao_fechou_direto"),
        "tentativas": avaliacao.get("tentativas", []),
    }


def _tentar_consolidar_mesmo_cliente(
    linha_ancora: pd.Series,
    fila_disponivel: pd.DataFrame,
    catalogo_veiculos: pd.DataFrame,
    tipo_roteirizacao: str,
) -> Dict[str, Any]:
    cliente_ref = linha_ancora["destinatario"]

    base_cliente = fila_disponivel.loc[
        fila_disponivel["destinatario"].astype(str) == str(cliente_ref)
    ].copy()

    if len(base_cliente) <= 1:
        return {
            "aceito": False,
            "motivo_reprovacao": "sem_massa_mesmo_cliente_para_consolidar",
            "tentativas": [],
        }

    base_cliente = _ordenar_mesmo_cliente(base_cliente)

    melhor: Optional[Dict[str, Any]] = None
    tentativas: List[Dict[str, Any]] = []

    for idx, veic in catalogo_veiculos.sort_values("ordem_porte").iterrows():
        if not _veiculo_disponivel_no_modo_frota(veic, tipo_roteirizacao):
            tentativas.append(
                {
                    "veiculo_tipo": veic["tipo"],
                    "resultado_teste": "rejeitado",
                    "motivo_reprovacao": "perfil_sem_disponibilidade_no_modo_frota",
                }
            )
            continue

        selecionados: List[pd.Series] = []
        for _, row in base_cliente.iterrows():
            teste = _materializar_selecionados(selecionados, row)

            aval_teste = _avaliar_combo_no_veiculo(
                teste,
                veic=veic,
                ignorar_ocupacao_minima=False,
            )

            if aval_teste["cabe_carga_oficial"] and aval_teste["cabe_paradas"] and aval_teste["cabe_km"]:
                selecionados.append(row)

        if len(selecionados) == 0:
            continue

        df_combo = _materializar_selecionados(selecionados)

        if str(linha_ancora["id_linha_pipeline"]) not in set(df_combo["id_linha_pipeline"].astype(str)):
            continue

        aval = _avaliar_combo_no_veiculo(
            df_combo,
            veic=veic,
            ignorar_ocupacao_minima=False,
        )

        tent = {**aval, "resultado_teste": "aceito" if aval["aceito"] else "rejeitado"}
        if not aval["aceito"]:
            motivos = []
            if not aval.get("cabe_carga_oficial", True):
                motivos.append("excede_capacidade_peso_oficial")
            if not aval.get("cabe_paradas", True):
                motivos.append("excede_max_entregas")
            if not aval.get("cabe_km", True):
                motivos.append("excede_max_km")
            if not aval.get("passa_ocupacao", True):
                motivos.append("nao_atinge_ocupacao_minima")
            tent["motivo_reprovacao"] = "|".join(motivos) if motivos else "mesmo_cliente_nao_fechou"

        tentativas.append(tent)

        if aval["aceito"]:
            candidato = {
                "aceito": True,
                "df_combo": df_combo,
                "avaliacao": aval,
                "catalogo_idx": idx,
                "tentativas": tentativas.copy(),
            }

            if melhor is None:
                melhor = candidato
            else:
                chave_nova = (
                    len(df_combo),
                    aval["base_carga_oficial"],
                    -int(veic["ordem_porte"]),
                )
                chave_atual = (
                    len(melhor["df_combo"]),
                    melhor["avaliacao"]["base_carga_oficial"],
                    -int(catalogo_veiculos.loc[melhor["catalogo_idx"], "ordem_porte"]),
                )
                if chave_nova > chave_atual:
                    melhor = candidato

    if melhor is None:
        return {
            "aceito": False,
            "motivo_reprovacao": "mesmo_cliente_nao_fechou",
            "tentativas": tentativas,
        }

    return melhor


def _motivo_final_remanescente(
    id_linha: str,
    cliente: str,
    df_tentativas: pd.DataFrame,
) -> str:
    if df_tentativas is None or df_tentativas.empty:
        return "sem_tentativa_registrada"

    base = df_tentativas.copy()

    if "linha_ancora" in base.columns:
        base = base.loc[base["linha_ancora"].astype(str) == str(id_linha)].copy()

    if base.empty and "cliente_referencia" in df_tentativas.columns:
        base = df_tentativas.loc[df_tentativas["cliente_referencia"].astype(str) == str(cliente)].copy()

    if base.empty:
        return "sem_tentativa_registrada"

    if "resultado_teste" in base.columns:
        base_rej = base.loc[base["resultado_teste"].astype(str) == "rejeitado"].copy()
        if not base_rej.empty:
            base = base_rej

    motivos = []
    if "motivo_reprovacao" in base.columns:
        motivos = [
            str(x).strip()
            for x in base["motivo_reprovacao"].dropna().astype(str).tolist()
            if str(x).strip() != ""
        ]

    if len(motivos) == 0:
        return "rejeitado_sem_motivo_detalhado"

    freq: Dict[str, int] = {}
    for motivo in motivos:
        freq[motivo] = freq.get(motivo, 0) + 1

    return max(freq.items(), key=lambda kv: kv[1])[0]


def executar_m4_manifestos_fechados(
    df_input_oficial_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
    rodada_id: str,
    data_base_roteirizacao: pd.Timestamp,
    tipo_roteirizacao: str = "carteira",
    configuracao_frota: Any = None,
    caminhos_pipeline: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    inicio_total = _agora()
    tempos_m4: Dict[str, float] = {}
    contadores_m4: Dict[str, Any] = {
        "qtd_anchors_exclusivos": 0,
        "qtd_anchors_direto": 0,
        "qtd_anchors_consolidacao": 0,
        "qtd_tentativas_total": 0,
        "qtd_tentativas_rejeitadas_ocupacao": 0,
        "qtd_tentativas_rejeitadas_km": 0,
        "qtd_tentativas_rejeitadas_paradas": 0,
        "qtd_tentativas_rejeitadas_capacidade": 0,
        "qtd_tentativas_sem_disponibilidade_frota": 0,
    }

    fila = _deduplicar_colunas(df_input_oficial_bloco_4.copy().reset_index(drop=True))
    veiculos = _deduplicar_colunas(df_veiculos_tratados.copy().reset_index(drop=True))
    caminhos_pipeline = caminhos_pipeline or {}
    tipo_roteirizacao = _normalizar_tipo_roteirizacao(tipo_roteirizacao)

    persistir_artefatos = bool(caminhos_pipeline.get("persistir_artefatos", False))

    # =========================================================================================
    # PREPARAÇÃO
    # =========================================================================================
    t0 = _agora()

    fila = _garantir_coluna_por_alias(fila, "destinatario", ["Destinatário", "cliente"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "cidade", ["Cida", "cidade_dest", "cidade_destino"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "uf", ["UF"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "peso_kg", ["Peso", "peso"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "vol_m3", ["Peso C", "peso_c", "cubagem_m3"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "peso_calculado", ["Peso Calculado", "peso_calc"], default=np.nan)
    fila = _garantir_coluna_por_alias(
        fila,
        "veiculo_exclusivo",
        ["Veiculo Exclusivo", "veiculo_dedicado"],
        default=np.nan,
    )
    fila = _garantir_coluna_por_alias(
        fila,
        "veiculo_exclusivo_flag",
        ["flag_veiculo_exclusivo", "veiculo_exclusivo_bool"],
        default=False,
    )
    fila = _garantir_coluna_por_alias(
        fila,
        "prioridade_embarque",
        ["Prioridade", "prioridade"],
        default=np.nan,
    )
    fila = _garantir_coluna_por_alias(
        fila,
        "distancia_rodoviaria_est_km",
        ["km_referencia", "distancia_km", "km_rota_referencia"],
        default=np.nan,
    )
    fila = _garantir_coluna_por_alias(
        fila,
        "status_triagem",
        ["status_roteirizacao", "status_fila"],
        default=np.nan,
    )
    fila = _garantir_coluna_por_alias(
        fila,
        "grupo_saida",
        ["grupo_pipeline", "grupo_status"],
        default=np.nan,
    )
    fila = _garantir_coluna_por_alias(
        fila,
        "agendada",
        ["flag_agendada", "agendada_flag"],
        default=False,
    )
    fila = _garantir_coluna_por_alias(
        fila,
        "data_agenda",
        ["Agendam.", "agenda_data", "data_agendamento"],
        default=pd.NaT,
    )
    fila = _garantir_coluna_por_alias(
        fila,
        "data_leadtime",
        ["D.L.E.", "dle", "leadtime_data_limite_entrega"],
        default=pd.NaT,
    )
    fila = _garantir_coluna_por_alias(
        fila,
        "ranking_prioridade",
        ["ranking_prioridade_operacional", "ranking_preliminar"],
        default=999999,
    )
    fila = _garantir_coluna_por_alias(
        fila,
        "score_prioridade_preliminar",
        ["score_prioridade", "score_operacional"],
        default=0.0,
    )
    fila = _garantir_coluna_por_alias(
        fila,
        "id_linha_pipeline",
        ["id", "id_linha", "hash_linha_pipeline"],
        default=np.nan,
    )
    fila = _garantir_coluna_por_alias(
        fila,
        "cte",
        ["nro_documento", "romaneio", "nro_doc", "Nro Doc."],
        default=np.nan,
    )

    coluna_tipo_veiculo = _resolver_coluna_tipo_veiculo(veiculos)

    colunas_minimas_fila = [
        "id_linha_pipeline",
        "destinatario",
        "cidade",
        "uf",
        "peso_calculado",
        "distancia_rodoviaria_est_km",
        "status_triagem",
        "grupo_saida",
    ]
    colunas_minimas_veiculos = [
        coluna_tipo_veiculo,
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
    ]

    faltam_fila = [c for c in colunas_minimas_fila if c not in fila.columns]
    faltam_veiculos = [c for c in colunas_minimas_veiculos if c not in veiculos.columns]

    if faltam_fila:
        raise Exception("Faltam colunas mínimas na fila oficial do Bloco 4:\n- " + "\n- ".join(faltam_fila))
    if faltam_veiculos:
        raise Exception("Faltam colunas mínimas na base de veículos:\n- " + "\n- ".join(faltam_veiculos))

    linhas_input_invalido = fila.loc[
        (fila["status_triagem"].astype(str) != "roteirizavel")
        | (fila["grupo_saida"].astype(str) != "df_carteira_roteirizavel")
    ].copy()
    if len(linhas_input_invalido) > 0:
        raise Exception(
            "O BLOCO 4 recebeu linhas incompatíveis com o estágio. "
            "Há registros com status_triagem != 'roteirizavel' ou grupo_saida inválido."
        )

    if fila["id_linha_pipeline"].isna().any():
        qtd_nulos = int(fila["id_linha_pipeline"].isna().sum())
        raise Exception(f"O input oficial do Bloco 4 possui id_linha_pipeline nulo: {qtd_nulos}")

    if fila["id_linha_pipeline"].astype(str).duplicated().any():
        qtd_dup = int(fila["id_linha_pipeline"].astype(str).duplicated().sum())
        raise Exception(f"O input oficial do Bloco 4 possui id_linha_pipeline duplicado: {qtd_dup}")

    for col in [
        "peso_kg",
        "vol_m3",
        "peso_calculado",
        "distancia_rodoviaria_est_km",
        "ranking_prioridade",
        "score_prioridade_preliminar",
        "folga_dias",
        "prioridade_embarque",
    ]:
        if col in fila.columns:
            fila[col] = pd.to_numeric(fila[col], errors="coerce")

    for col in ["capacidade_peso_kg", "capacidade_vol_m3", "max_entregas", "max_km_distancia"]:
        veiculos[col] = pd.to_numeric(veiculos[col], errors="coerce")

    for col in ["data_agenda", "data_leadtime"]:
        if col in fila.columns:
            fila[col] = pd.to_datetime(fila[col], errors="coerce")

    fila["agendada"] = fila["data_agenda"].notna()
    fila["veiculo_exclusivo_flag"] = fila.apply(_eh_exclusivo, axis=1)

    if fila["cte"].isna().all():
        fila["cte"] = fila["id_linha_pipeline"].astype(str)
    else:
        fila["cte"] = fila["cte"].fillna(fila["id_linha_pipeline"].astype(str))

    fila["ranking_prioridade"] = pd.to_numeric(fila["ranking_prioridade"], errors="coerce").fillna(999999)
    fila["score_prioridade_preliminar"] = pd.to_numeric(
        fila["score_prioridade_preliminar"], errors="coerce"
    ).fillna(0.0)
    fila["peso_calculado"] = pd.to_numeric(fila["peso_calculado"], errors="coerce")

    if fila["peso_calculado"].isna().any():
        qtd_nulos = int(fila["peso_calculado"].isna().sum())
        raise Exception(
            f"O M4 recebeu {qtd_nulos} linhas sem peso_calculado. "
            "Como peso_calculado é a base oficial de ocupação/capacidade, o input do bloco 4 precisa estar completo."
        )

    catalogo_veiculos = _preparar_catalogo_veiculos(
        df_veic=veiculos,
        coluna_tipo_veiculo=coluna_tipo_veiculo,
        tipo_roteirizacao=tipo_roteirizacao,
        configuracao_frota=configuracao_frota,
    )

    fila_ordenada = _ordenar_fila(fila)

    tempos_m4["preparacao_validacao_ms"] = _duracao_ms(t0)

    # =========================================================================================
    # EXECUÇÃO
    # =========================================================================================
    manifestos_fechados: List[Dict[str, Any]] = []
    itens_manifestos_fechados: List[pd.DataFrame] = []
    tentativas_fechamento: List[Dict[str, Any]] = []
    ids_alocados: set[str] = set()
    contador_manifesto = 1

    def _contabilizar_tentativa(tent: Dict[str, Any]) -> None:
        contadores_m4["qtd_tentativas_total"] += 1
        motivo = str(tent.get("motivo_reprovacao", "")).strip()
        if motivo == "perfil_sem_disponibilidade_no_modo_frota":
            contadores_m4["qtd_tentativas_sem_disponibilidade_frota"] += 1
        if "nao_atinge_ocupacao_minima" in motivo:
            contadores_m4["qtd_tentativas_rejeitadas_ocupacao"] += 1
        if "excede_max_km" in motivo:
            contadores_m4["qtd_tentativas_rejeitadas_km"] += 1
        if "excede_max_entregas" in motivo:
            contadores_m4["qtd_tentativas_rejeitadas_paradas"] += 1
        if "excede_capacidade_peso_oficial" in motivo:
            contadores_m4["qtd_tentativas_rejeitadas_capacidade"] += 1

    def registrar_manifesto(
        df_combo: pd.DataFrame,
        avaliacao: Dict[str, Any],
        origem_etapa: str,
        catalogo_idx: Optional[int],
    ) -> None:
        nonlocal contador_manifesto, manifestos_fechados, itens_manifestos_fechados, ids_alocados

        manifesto_id = f"MF4_{contador_manifesto:04d}"
        contador_manifesto += 1

        resumo = _gerar_resumo_manifesto(
            df_combo=df_combo,
            avaliacao=avaliacao,
            manifesto_id=manifesto_id,
            tipo_manifesto="fechado_bloco_4",
            origem_etapa=origem_etapa,
        )
        manifestos_fechados.append(resumo)

        itens = df_combo.copy()
        itens["manifesto_id"] = manifesto_id
        itens["tipo_manifesto"] = "fechado_bloco_4"
        itens["veiculo_tipo"] = avaliacao["veiculo_tipo"]
        itens["capacidade_peso_kg_veiculo"] = avaliacao["capacidade_peso_kg"]
        itens["capacidade_vol_m3_veiculo"] = avaliacao["capacidade_vol_m3"]
        itens["max_entregas_veiculo"] = avaliacao["max_entregas"]
        itens["max_km_distancia_veiculo"] = avaliacao["max_km_distancia"]
        itens["base_carga_oficial_manifesto"] = avaliacao["base_carga_oficial"]
        itens["ocupacao_oficial_perc_manifesto"] = avaliacao["ocupacao_oficial_perc"]
        itens["ignorar_ocupacao_minima_manifesto"] = avaliacao["ignorar_ocupacao_minima"]
        itens["origem_modulo"] = 4
        itens["origem_etapa"] = origem_etapa
        itens_manifestos_fechados.append(itens)

        ids_alocados.update(df_combo["id_linha_pipeline"].astype(str).tolist())
        _consumir_veiculo_catalogo(catalogo_veiculos, catalogo_idx, tipo_roteirizacao)

    # -----------------------------------------
    # 4B1 - EXCLUSIVOS
    # -----------------------------------------
    t0 = _agora()
    exclusivos = fila_ordenada.loc[fila_ordenada["veiculo_exclusivo_flag"] == True].copy()
    exclusivos = exclusivos.loc[~exclusivos["id_linha_pipeline"].astype(str).isin(ids_alocados)].reset_index(drop=True)
    contadores_m4["qtd_anchors_exclusivos"] = int(len(exclusivos))

    for _, linha_exclusiva in exclusivos.iterrows():
        id_anchor = str(linha_exclusiva["id_linha_pipeline"])
        if id_anchor in ids_alocados:
            continue

        fila_disponivel = fila_ordenada.loc[
            ~fila_ordenada["id_linha_pipeline"].astype(str).isin(ids_alocados)
        ].copy().reset_index(drop=True)

        resultado_exclusivo = _tentar_exclusivo_com_mesmo_cliente(
            linha_exclusiva=linha_exclusiva,
            fila_disponivel=fila_disponivel,
            catalogo_veiculos=catalogo_veiculos,
            tipo_roteirizacao=tipo_roteirizacao,
        )

        if resultado_exclusivo["aceito"]:
            tent = {
                "etapa_fechamento": "4B1_exclusivo",
                "tipo_tentativa": "exclusivo_com_mesmo_cliente",
                "cliente_referencia": linha_exclusiva["destinatario"],
                "linha_ancora": id_anchor,
                "veiculo_tipo": resultado_exclusivo["avaliacao"]["veiculo_tipo"],
                "qtd_linhas_grupo": int(len(resultado_exclusivo["df_combo"])),
                "base_carga_oficial": resultado_exclusivo["avaliacao"]["base_carga_oficial"],
                "ocupacao_oficial_perc": resultado_exclusivo["avaliacao"]["ocupacao_oficial_perc"],
                "resultado_teste": "aceito",
            }
            tentativas_fechamento.append(tent)
            _contabilizar_tentativa(tent)

            registrar_manifesto(
                df_combo=resultado_exclusivo["df_combo"],
                avaliacao=resultado_exclusivo["avaliacao"],
                origem_etapa="4B1_exclusivo",
                catalogo_idx=resultado_exclusivo["catalogo_idx"],
            )
        else:
            tent = {
                "etapa_fechamento": "4B1_exclusivo",
                "tipo_tentativa": "exclusivo_com_mesmo_cliente",
                "cliente_referencia": linha_exclusiva["destinatario"],
                "linha_ancora": id_anchor,
                "resultado_teste": "rejeitado",
                "motivo_reprovacao": resultado_exclusivo.get("motivo_reprovacao", "exclusivo_sem_veiculo_viavel"),
            }
            tentativas_fechamento.append(tent)
            _contabilizar_tentativa(tent)

    tempos_m4["4B1_exclusivos_ms"] = _duracao_ms(t0)

    # -----------------------------------------
    # 4B2 - FECHAMENTO DIRETO
    # -----------------------------------------
    t0 = _agora()
    fila_saldo_direto = fila_ordenada.loc[
        ~fila_ordenada["id_linha_pipeline"].astype(str).isin(ids_alocados)
    ].copy().reset_index(drop=True)
    contadores_m4["qtd_anchors_direto"] = int(len(fila_saldo_direto))

    for _, linha in fila_saldo_direto.iterrows():
        id_anchor = str(linha["id_linha_pipeline"])
        if id_anchor in ids_alocados:
            continue

        resultado_direto = _tentar_fechamento_direto_linha(
            linha_base=linha,
            fila_disponivel=fila_saldo_direto,
            catalogo_veiculos=catalogo_veiculos,
            tipo_roteirizacao=tipo_roteirizacao,
        )

        if resultado_direto["aceito"]:
            tent = {
                "etapa_fechamento": "4B2_fechamento_direto",
                "tipo_tentativa": "linha_direta",
                "cliente_referencia": linha["destinatario"],
                "linha_ancora": id_anchor,
                "veiculo_tipo": resultado_direto["avaliacao"]["veiculo_tipo"],
                "qtd_linhas_grupo": int(len(resultado_direto["df_combo"])),
                "base_carga_oficial": resultado_direto["avaliacao"]["base_carga_oficial"],
                "ocupacao_oficial_perc": resultado_direto["avaliacao"]["ocupacao_oficial_perc"],
                "resultado_teste": "aceito",
            }
            tentativas_fechamento.append(tent)
            _contabilizar_tentativa(tent)

            registrar_manifesto(
                df_combo=resultado_direto["df_combo"],
                avaliacao=resultado_direto["avaliacao"],
                origem_etapa="4B2_fechamento_direto",
                catalogo_idx=resultado_direto["catalogo_idx"],
            )
        else:
            for tent in resultado_direto.get("tentativas", []):
                tent_padrao = {
                    **tent,
                    "etapa_fechamento": "4B2_fechamento_direto",
                    "tipo_tentativa": "linha_direta",
                    "cliente_referencia": linha["destinatario"],
                    "linha_ancora": id_anchor,
                }
                tentativas_fechamento.append(tent_padrao)
                _contabilizar_tentativa(tent_padrao)

    tempos_m4["4B2_fechamento_direto_ms"] = _duracao_ms(t0)

    # -----------------------------------------
    # 4C - CONSOLIDAÇÃO MESMO CLIENTE
    # -----------------------------------------
    t0 = _agora()
    fila_saldo_consolidacao = fila_ordenada.loc[
        ~fila_ordenada["id_linha_pipeline"].astype(str).isin(ids_alocados)
    ].copy().reset_index(drop=True)

    fila_saldo_consolidacao = fila_saldo_consolidacao.loc[
        fila_saldo_consolidacao["veiculo_exclusivo_flag"] == False
    ].copy().reset_index(drop=True)

    fila_saldo_consolidacao = _ordenar_fila(fila_saldo_consolidacao)
    contadores_m4["qtd_anchors_consolidacao"] = int(len(fila_saldo_consolidacao))

    for _, linha_ancora in fila_saldo_consolidacao.iterrows():
        id_anchor = str(linha_ancora["id_linha_pipeline"])
        if id_anchor in ids_alocados:
            continue

        fila_disponivel = fila_ordenada.loc[
            ~fila_ordenada["id_linha_pipeline"].astype(str).isin(ids_alocados)
        ].copy().reset_index(drop=True)
        fila_disponivel = fila_disponivel.loc[
            fila_disponivel["veiculo_exclusivo_flag"] == False
        ].copy().reset_index(drop=True)

        resultado_consolidacao = _tentar_consolidar_mesmo_cliente(
            linha_ancora=linha_ancora,
            fila_disponivel=fila_disponivel,
            catalogo_veiculos=catalogo_veiculos,
            tipo_roteirizacao=tipo_roteirizacao,
        )

        if resultado_consolidacao["aceito"]:
            for tent in resultado_consolidacao.get("tentativas", []):
                tent_padrao = {
                    **tent,
                    "etapa_fechamento": "4C_consolidacao_mesmo_cliente",
                    "tipo_tentativa": "mesmo_cliente",
                    "cliente_referencia": linha_ancora["destinatario"],
                    "linha_ancora": id_anchor,
                }
                tentativas_fechamento.append(tent_padrao)
                _contabilizar_tentativa(tent_padrao)

            registrar_manifesto(
                df_combo=resultado_consolidacao["df_combo"],
                avaliacao=resultado_consolidacao["avaliacao"],
                origem_etapa="4C_consolidacao_mesmo_cliente",
                catalogo_idx=resultado_consolidacao["catalogo_idx"],
            )
        else:
            if len(resultado_consolidacao.get("tentativas", [])) > 0:
                for tent in resultado_consolidacao.get("tentativas", []):
                    tent_padrao = {
                        **tent,
                        "etapa_fechamento": "4C_consolidacao_mesmo_cliente",
                        "tipo_tentativa": "mesmo_cliente",
                        "cliente_referencia": linha_ancora["destinatario"],
                        "linha_ancora": id_anchor,
                    }
                    tentativas_fechamento.append(tent_padrao)
                    _contabilizar_tentativa(tent_padrao)
            else:
                tent_padrao = {
                    "etapa_fechamento": "4C_consolidacao_mesmo_cliente",
                    "tipo_tentativa": "mesmo_cliente",
                    "cliente_referencia": linha_ancora["destinatario"],
                    "linha_ancora": id_anchor,
                    "resultado_teste": "rejeitado",
                    "motivo_reprovacao": resultado_consolidacao.get(
                        "motivo_reprovacao", "mesmo_cliente_nao_fechou"
                    ),
                }
                tentativas_fechamento.append(tent_padrao)
                _contabilizar_tentativa(tent_padrao)

    tempos_m4["4C_consolidacao_mesmo_cliente_ms"] = _duracao_ms(t0)

    # =========================================================================================
    # MATERIALIZAÇÃO DOS OUTPUTS
    # =========================================================================================
    t0 = _agora()

    df_manifestos_fechados_bloco_4 = pd.DataFrame(manifestos_fechados)
    df_itens_manifestos_fechados_bloco_4 = (
        pd.concat(itens_manifestos_fechados, ignore_index=True)
        if len(itens_manifestos_fechados) > 0
        else pd.DataFrame()
    )
    df_tentativas_fechamento_bloco_4 = pd.DataFrame(tentativas_fechamento)

    df_remanescente_roteirizavel_bloco_4 = fila.loc[
        ~fila["id_linha_pipeline"].astype(str).isin(ids_alocados)
    ].copy().reset_index(drop=True)

    if len(df_remanescente_roteirizavel_bloco_4) > 0:
        df_remanescente_roteirizavel_bloco_4["motivo_final_remanescente_m4"] = df_remanescente_roteirizavel_bloco_4.apply(
            lambda row: _motivo_final_remanescente(
                id_linha=str(row["id_linha_pipeline"]),
                cliente=str(row["destinatario"]),
                df_tentativas=df_tentativas_fechamento_bloco_4,
            ),
            axis=1,
        )

    uso_frota = catalogo_veiculos[["tipo", "limite_manifestos", "manifestos_utilizados"]].copy()
    uso_frota["saldo_manifestos"] = uso_frota.apply(
        lambda row: (
            np.nan
            if pd.isna(row["limite_manifestos"])
            else int(row["limite_manifestos"]) - int(_int_safe(row["manifestos_utilizados"], default=0))
        ),
        axis=1,
    )

    roteirizavel_entrada_m4 = len(fila)
    itens_manifestados_m4 = len(df_itens_manifestos_fechados_bloco_4)
    remanescente_roteirizavel_m4 = len(df_remanescente_roteirizavel_bloco_4)

    tempos_m4["materializacao_outputs_ms"] = _duracao_ms(t0)

    # =========================================================================================
    # PERSISTÊNCIA OPCIONAL DE ARTEFATOS
    # =========================================================================================
    t0 = _agora()

    if persistir_artefatos:
        try:
            pasta_saida_base_str = caminhos_pipeline.get("pasta_saida_base")
            if pasta_saida_base_str:
                pasta_saida_base = Path(pasta_saida_base_str)
            else:
                pasta_saida_base = Path("/tmp/rec_roteirizador") / str(rodada_id)

            pasta_modulo_4 = pasta_saida_base / "bloco_4_manifestos_fechados"
            pasta_modulo_4.mkdir(parents=True, exist_ok=True)

            arq_manifestos_xlsx = pasta_modulo_4 / "df_manifestos_fechados_bloco_4.xlsx"
            arq_itens_csv = pasta_modulo_4 / "df_itens_manifestos_fechados_bloco_4.csv"
            arq_tentativas_csv = pasta_modulo_4 / "df_tentativas_fechamento_bloco_4.csv"
            arq_remanescente_csv = pasta_modulo_4 / "df_remanescente_roteirizavel_bloco_4.csv"
            arq_resumo_xlsx = pasta_modulo_4 / "resumo_modulo_4.xlsx"
            arq_metadata_json = pasta_modulo_4 / "metadata_modulo_4.json"

            if len(df_manifestos_fechados_bloco_4) > 0:
                df_manifestos_fechados_bloco_4.to_excel(arq_manifestos_xlsx, index=False)

            if len(df_itens_manifestos_fechados_bloco_4) > 0:
                df_itens_manifestos_fechados_bloco_4.to_csv(arq_itens_csv, index=False, encoding="utf-8-sig")

            if len(df_tentativas_fechamento_bloco_4) > 0:
                df_tentativas_fechamento_bloco_4.to_csv(arq_tentativas_csv, index=False, encoding="utf-8-sig")

            if len(df_remanescente_roteirizavel_bloco_4) > 0:
                df_remanescente_roteirizavel_bloco_4.to_csv(arq_remanescente_csv, index=False, encoding="utf-8-sig")

            with pd.ExcelWriter(arq_resumo_xlsx, engine="openpyxl") as writer:
                pd.DataFrame(
                    [
                        {
                            "roteirizavel_entrada_m4": int(roteirizavel_entrada_m4),
                            "manifestos_fechados_gerados_m4": int(len(df_manifestos_fechados_bloco_4)),
                            "itens_manifestados_m4": int(itens_manifestados_m4),
                            "remanescente_roteirizavel_m4": int(remanescente_roteirizavel_m4),
                            "tipo_roteirizacao": tipo_roteirizacao,
                        }
                    ]
                ).to_excel(writer, sheet_name="resumo", index=False)

                if len(uso_frota) > 0:
                    uso_frota.to_excel(writer, sheet_name="uso_frota", index=False)

            metadata = {
                "modulo": "4_manifestos_fechados_regra_nova",
                "data_execucao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_base_projeto": pd.Timestamp(data_base_roteirizacao).strftime("%Y-%m-%d"),
                "tipo_roteirizacao": tipo_roteirizacao,
                "regras": {
                    "peso_calculado_base_oficial": True,
                    "exclusivo_sem_ocupacao_minima": True,
                    "exclusivo_puxa_mesmo_cliente": True,
                    "prioridade_embarque_1_primeiro": True,
                    "agendadas_depois_da_prioridade_1": True,
                    "leadtimes_positivos_antes_dos_vencidos": True,
                    "consolidacao_mesmo_cliente_no_saldo": True,
                    "ocupacao_minima_padrao": OCUPACAO_MINIMA_PADRAO,
                    "modo_frota_respeita_quantidade_por_perfil": True,
                },
                "totais": {
                    "roteirizavel_entrada_m4": int(roteirizavel_entrada_m4),
                    "manifestos_fechados_gerados_m4": int(len(df_manifestos_fechados_bloco_4)),
                    "itens_manifestados_m4": int(itens_manifestados_m4),
                    "remanescente_roteirizavel_m4": int(remanescente_roteirizavel_m4),
                },
                "outputs": {
                    "df_manifestos_fechados_bloco_4_xlsx": str(arq_manifestos_xlsx),
                    "df_itens_manifestos_fechados_bloco_4_csv": str(arq_itens_csv),
                    "df_tentativas_fechamento_bloco_4_csv": str(arq_tentativas_csv),
                    "df_remanescente_roteirizavel_bloco_4_csv": str(arq_remanescente_csv),
                    "resumo_modulo_4_xlsx": str(arq_resumo_xlsx),
                },
            }

            with open(arq_metadata_json, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

            caminhos_pipeline["df_manifestos_fechados_bloco_4_xlsx"] = str(arq_manifestos_xlsx)
            caminhos_pipeline["df_itens_manifestos_fechados_bloco_4_csv"] = str(arq_itens_csv)
            caminhos_pipeline["df_tentativas_fechamento_bloco_4_csv"] = str(arq_tentativas_csv)
            caminhos_pipeline["df_remanescente_roteirizavel_bloco_4_csv"] = str(arq_remanescente_csv)
            caminhos_pipeline["resumo_modulo_4_xlsx"] = str(arq_resumo_xlsx)
            caminhos_pipeline["metadata_modulo_4_json"] = str(arq_metadata_json)
        except Exception:
            pass

    tempos_m4["persistencia_artefatos_ms"] = _duracao_ms(t0)
    tempos_m4["tempo_total_m4_ms"] = _duracao_ms(inicio_total)

    # =========================================================================================
    # RESUMOS E METADADOS
    # =========================================================================================
    resumo_m4 = {
        "modulo": "M4",
        "data_base_roteirizacao": pd.Timestamp(data_base_roteirizacao).isoformat(),
        "coluna_tipo_veiculo_utilizada": coluna_tipo_veiculo,
        "tipo_roteirizacao": tipo_roteirizacao,
        "roteirizavel_entrada_m4": int(roteirizavel_entrada_m4),
        "manifestos_fechados_gerados_m4": int(len(df_manifestos_fechados_bloco_4)),
        "itens_manifestados_m4": int(itens_manifestados_m4),
        "remanescente_roteirizavel_m4": int(remanescente_roteirizavel_m4),
        "exclusivos_entrada_m4": int((fila["veiculo_exclusivo_flag"] == True).sum()),
        "prioridade_embarque_1_entrada_m4": int((pd.to_numeric(fila["prioridade_embarque"], errors="coerce") == 1).sum()),
        "ocupacao_minima_padrao_perc": round(OCUPACAO_MINIMA_PADRAO * 100, 2),
        "persistiu_artefatos": persistir_artefatos,
        "caminhos_pipeline": caminhos_pipeline,
    }

    auditoria_m4 = {
        "motivos_remanescente_m4": (
            df_remanescente_roteirizavel_bloco_4["motivo_final_remanescente_m4"].value_counts(dropna=False).to_dict()
            if "motivo_final_remanescente_m4" in df_remanescente_roteirizavel_bloco_4.columns
            else {}
        )
    }

    outputs = {
        "df_manifestos_fechados_bloco_4": df_manifestos_fechados_bloco_4,
        "df_itens_manifestos_fechados_bloco_4": df_itens_manifestos_fechados_bloco_4,
        "df_tentativas_fechamento_bloco_4": df_tentativas_fechamento_bloco_4,
        "df_remanescente_roteirizavel_bloco_4": df_remanescente_roteirizavel_bloco_4,
        "df_uso_frota_m4": uso_frota,
    }

    meta = {
        "resumo_m4": resumo_m4,
        "auditoria_m4": auditoria_m4,
        "metricas_m4": {
            **tempos_m4,
            **contadores_m4,
        },
        "metadata_modulo_4": {
            "tipo_roteirizacao": tipo_roteirizacao,
            "catalogo_veiculos": _to_records(catalogo_veiculos),
            "uso_frota": _to_records(uso_frota),
        },
    }

    return outputs, meta
