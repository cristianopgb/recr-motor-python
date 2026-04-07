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
#
# REGRAS DE NEGÓCIO IMPLEMENTADAS:
# 1) entrada = somente carteira roteirizável
# 2) exclusivos:
#    - saem primeiro
#    - ignoram ocupação mínima
#    - respeitam capacidade / km / paradas
#    - podem puxar outros docs do MESMO cliente, se couberem
# 3) não exclusivos:
#    - NUNCA misturam clientes
#    - primeiro tentam consolidar o cluster do mesmo cliente
#      no MENOR veículo possível, respeitando ocupação entre 70% e 100%
#    - se não der para absorver tudo, tentam subconjuntos do mesmo cliente
#      ainda no menor veículo possível
#    - o restante do cliente tenta do MAIOR para o MENOR, ainda respeitando 70% a 100%
# 4) o que não fechar no M4 fica para o M5
#
# CORREÇÕES IMPORTANTES:
# - impede reutilização de CTE em mais de um manifesto
# - impede ocupação/base duplicada por linha
# - impede somatório artificial > 100%
# - resumo do manifesto calculado UMA única vez
# ============================================================

OCUPACAO_MINIMA_PADRAO = 0.70
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


def _resolver_coluna_tipo_veiculo(df_veiculos: pd.DataFrame) -> str:
    if "tipo" in df_veiculos.columns:
        return "tipo"
    if "perfil" in df_veiculos.columns:
        return "perfil"
    raise Exception("Faltam colunas mínimas na base de veículos:\n- tipo ou perfil")


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
            raise Exception("tipo_roteirizacao = 'frota', mas configuracao_frota está vazia ou inválida.")

        cat = cat.merge(cfg, how="inner", left_on="tipo", right_on="perfil")
        if len(cat) == 0:
            raise Exception("Nenhum perfil da configuracao_frota foi encontrado no catálogo de veículos.")

        cat["limite_manifestos"] = pd.to_numeric(cat["quantidade"], errors="coerce").fillna(0).astype(int)
        cat.drop(columns=["perfil", "quantidade"], inplace=True, errors="ignore")
    else:
        cat["limite_manifestos"] = np.nan

    cat["manifestos_utilizados"] = 0
    cat["ordem_porte"] = np.arange(1, len(cat) + 1)
    return cat.reset_index(drop=True)


def _obter_coluna_id_documento(df: pd.DataFrame) -> str:
    if "cte" in df.columns:
        return "cte"
    return "id_linha_pipeline"


def _normalizar_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def _eh_exclusivo(row: pd.Series) -> bool:
    if "veiculo_exclusivo_flag" in row.index:
        return _bool_safe(row.get("veiculo_exclusivo_flag"))
    return _bool_safe(row.get("veiculo_exclusivo"))


def _cliente_key(row: pd.Series) -> str:
    return _normalizar_str(row.get("destinatario"))


def _cliente_cidade_key(row: pd.Series) -> str:
    return f"{_normalizar_str(row.get('destinatario'))}|{_normalizar_str(row.get('cidade'))}|{_normalizar_str(row.get('uf'))}"


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

    col_doc = _obter_coluna_id_documento(df_combo)
    qtd_ctes = int(df_combo[col_doc].astype(str).nunique())
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

    if ignorar_ocupacao_minima:
        passa_ocupacao = True
    else:
        passa_ocupacao = (
            pd.notna(ocupacao_oficial)
            and ocupacao_oficial >= OCUPACAO_MINIMA_PADRAO
            and ocupacao_oficial <= OCUPACAO_MAXIMA_PADRAO
        )

    aceito = bool(cabe_carga_oficial and cabe_paradas and cabe_km and passa_ocupacao)

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
        "aceito": aceito,
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
        "origem_modulo": 4,
        "origem_etapa": origem_etapa,
    }

    if len(df_combo) > 0:
        linha["destinatario"] = df_combo["destinatario"].iloc[0]
        linha["cidade"] = df_combo["cidade"].iloc[0] if "cidade" in df_combo.columns else np.nan
        linha["uf"] = df_combo["uf"].iloc[0] if "uf" in df_combo.columns else np.nan
        linha["mesorregiao"] = df_combo["mesorregiao"].iloc[0] if "mesorregiao" in df_combo.columns else np.nan
        linha["subregiao"] = df_combo["subregiao"].iloc[0] if "subregiao" in df_combo.columns else np.nan

    return linha


def _materializar_df_por_indices(df_base: pd.DataFrame, indices: List[int]) -> pd.DataFrame:
    if len(indices) == 0:
        return pd.DataFrame(columns=df_base.columns)
    return df_base.loc[indices].copy().reset_index(drop=True)


def _gerar_subconjunto_guloso(
    base_cliente: pd.DataFrame,
    veic: pd.Series,
    anchor_id: Optional[str] = None,
    ignorar_ocupacao_minima: bool = False,
    ordem_desc: bool = True,
) -> pd.DataFrame:
    """
    Monta subconjunto guloso sem duplicar itens.
    Se anchor_id vier preenchido, garante que o anchor esteja no grupo.
    """
    if len(base_cliente) == 0:
        return pd.DataFrame(columns=base_cliente.columns)

    trabalho = base_cliente.copy().reset_index(drop=True)

    if anchor_id is not None:
        mask_anchor = trabalho["id_linha_pipeline"].astype(str) == str(anchor_id)
        if not mask_anchor.any():
            return pd.DataFrame(columns=trabalho.columns)

        idx_anchor = trabalho.index[mask_anchor][0]
        restantes = trabalho.drop(index=idx_anchor).copy()
        restantes = restantes.sort_values("peso_calculado", ascending=not ordem_desc).reset_index(drop=True)

        ordem_indices = [idx_anchor]
        # reindexar restantes em sequência separada
        restantes["_tmp_idx_local"] = np.arange(len(restantes))
        # vamos trabalhar por linhas, depois remontar pelo conteúdo
        grupo = trabalho.loc[[idx_anchor]].copy().reset_index(drop=True)

        for _, row in restantes.iterrows():
            teste = pd.concat([grupo, row.to_frame().T], ignore_index=True)
            aval = _avaliar_combo_no_veiculo(teste, veic=veic, ignorar_ocupacao_minima=ignorar_ocupacao_minima)
            if aval["cabe_carga_oficial"] and aval["cabe_paradas"] and aval["cabe_km"]:
                grupo = teste

        return grupo.reset_index(drop=True)

    trabalho = trabalho.sort_values("peso_calculado", ascending=not ordem_desc).reset_index(drop=True)
    grupo = pd.DataFrame(columns=trabalho.columns)

    for _, row in trabalho.iterrows():
        teste = pd.concat([grupo, row.to_frame().T], ignore_index=True)
        aval = _avaliar_combo_no_veiculo(teste, veic=veic, ignorar_ocupacao_minima=ignorar_ocupacao_minima)
        if aval["cabe_carga_oficial"] and aval["cabe_paradas"] and aval["cabe_km"]:
            grupo = teste

    return grupo.reset_index(drop=True)


def _tentar_exclusivo_com_mesmo_cliente(
    linha_exclusiva: pd.Series,
    fila_disponivel: pd.DataFrame,
    catalogo_veiculos: pd.DataFrame,
    tipo_roteirizacao: str,
) -> Dict[str, Any]:
    cliente_ref = _cliente_key(linha_exclusiva)
    base_cliente = fila_disponivel.loc[
        fila_disponivel.apply(_cliente_key, axis=1) == cliente_ref
    ].copy()

    if len(base_cliente) == 0:
        return {"aceito": False, "motivo_reprovacao": "exclusivo_sem_base_cliente", "tentativas": []}

    tentativas: List[Dict[str, Any]] = []
    anchor_id = str(linha_exclusiva["id_linha_pipeline"])

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

        grupo = _gerar_subconjunto_guloso(
            base_cliente=base_cliente,
            veic=veic,
            anchor_id=anchor_id,
            ignorar_ocupacao_minima=True,
            ordem_desc=True,
        )

        if len(grupo) == 0:
            tentativas.append(
                {
                    "veiculo_tipo": veic["tipo"],
                    "resultado_teste": "rejeitado",
                    "motivo_reprovacao": "exclusivo_sem_combo_viavel",
                }
            )
            continue

        aval = _avaliar_combo_no_veiculo(grupo, veic=veic, ignorar_ocupacao_minima=True)
        registro = {**aval, "resultado_teste": "aceito" if aval["aceito"] else "rejeitado"}
        tentativas.append(registro)

        if aval["aceito"]:
            return {
                "aceito": True,
                "df_combo": grupo,
                "avaliacao": aval,
                "catalogo_idx": idx,
                "tentativas": tentativas,
            }

    return {
        "aceito": False,
        "motivo_reprovacao": "exclusivo_sem_veiculo_viavel",
        "tentativas": tentativas,
    }


def _tentar_fechamento_direto_linha(
    linha_base: pd.Series,
    catalogo_veiculos: pd.DataFrame,
    tipo_roteirizacao: str,
) -> Dict[str, Any]:
    df_combo = pd.DataFrame([linha_base]).reset_index(drop=True)
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

        aval = _avaliar_combo_no_veiculo(df_combo, veic=veic, ignorar_ocupacao_minima=False)
        registro = {**aval, "resultado_teste": "aceito" if aval["aceito"] else "rejeitado"}

        if not aval["aceito"]:
            motivos = []
            if not aval.get("cabe_carga_oficial", True):
                motivos.append("excede_capacidade_peso_oficial")
            if not aval.get("cabe_paradas", True):
                motivos.append("excede_max_entregas")
            if not aval.get("cabe_km", True):
                motivos.append("excede_max_km")
            if not aval.get("passa_ocupacao", True):
                motivos.append("nao_atinge_faixa_ocupacao_70_100")
            registro["motivo_reprovacao"] = "|".join(motivos) if motivos else "nao_fechou_direto"

        tentativas.append(registro)

        if aval["aceito"]:
            return {
                "aceito": True,
                "df_combo": df_combo,
                "avaliacao": aval,
                "catalogo_idx": idx,
                "tentativas": tentativas,
            }

    return {
        "aceito": False,
        "motivo_reprovacao": "nao_fechou_direto",
        "tentativas": tentativas,
    }


def _tentar_cluster_mesmo_cliente_menor_para_maior(
    base_cliente: pd.DataFrame,
    catalogo_veiculos: pd.DataFrame,
    tipo_roteirizacao: str,
) -> Dict[str, Any]:
    """
    Tenta absorver o cluster inteiro do cliente no menor veículo possível.
    Se não couber tudo, tenta subconjunto do mesmo cliente ainda no menor veículo possível.
    """
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

        # tenta cluster inteiro primeiro
        aval_full = _avaliar_combo_no_veiculo(base_cliente, veic=veic, ignorar_ocupacao_minima=False)
        reg_full = {
            **aval_full,
            "resultado_teste": "aceito" if aval_full["aceito"] else "rejeitado",
            "tipo_busca": "cluster_completo",
        }
        if not aval_full["aceito"]:
            motivos = []
            if not aval_full.get("cabe_carga_oficial", True):
                motivos.append("excede_capacidade_peso_oficial")
            if not aval_full.get("cabe_paradas", True):
                motivos.append("excede_max_entregas")
            if not aval_full.get("cabe_km", True):
                motivos.append("excede_max_km")
            if not aval_full.get("passa_ocupacao", True):
                motivos.append("nao_atinge_faixa_ocupacao_70_100")
            reg_full["motivo_reprovacao"] = "|".join(motivos) if motivos else "cluster_completo_nao_fechou"

        tentativas.append(reg_full)

        if aval_full["aceito"]:
            return {
                "aceito": True,
                "df_combo": base_cliente.copy().reset_index(drop=True),
                "avaliacao": aval_full,
                "catalogo_idx": idx,
                "tentativas": tentativas,
            }

        # se o cluster inteiro não fechou, tenta subconjunto guloso do mesmo cliente
        grupo = _gerar_subconjunto_guloso(
            base_cliente=base_cliente,
            veic=veic,
            anchor_id=None,
            ignorar_ocupacao_minima=False,
            ordem_desc=True,
        )

        if len(grupo) == 0:
            continue

        aval_sub = _avaliar_combo_no_veiculo(grupo, veic=veic, ignorar_ocupacao_minima=False)
        reg_sub = {
            **aval_sub,
            "resultado_teste": "aceito" if aval_sub["aceito"] else "rejeitado",
            "tipo_busca": "subconjunto_mesmo_cliente",
        }
        if not aval_sub["aceito"]:
            motivos = []
            if not aval_sub.get("cabe_carga_oficial", True):
                motivos.append("excede_capacidade_peso_oficial")
            if not aval_sub.get("cabe_paradas", True):
                motivos.append("excede_max_entregas")
            if not aval_sub.get("cabe_km", True):
                motivos.append("excede_max_km")
            if not aval_sub.get("passa_ocupacao", True):
                motivos.append("nao_atinge_faixa_ocupacao_70_100")
            reg_sub["motivo_reprovacao"] = "|".join(motivos) if motivos else "subconjunto_nao_fechou"

        tentativas.append(reg_sub)

        if aval_sub["aceito"]:
            return {
                "aceito": True,
                "df_combo": grupo,
                "avaliacao": aval_sub,
                "catalogo_idx": idx,
                "tentativas": tentativas,
            }

    return {
        "aceito": False,
        "motivo_reprovacao": "cliente_nao_fechou_menor_para_maior",
        "tentativas": tentativas,
    }


def _tentar_restante_cliente_maior_para_menor(
    base_cliente: pd.DataFrame,
    catalogo_veiculos: pd.DataFrame,
    tipo_roteirizacao: str,
) -> Dict[str, Any]:
    """
    Para o restante do cliente, tenta do maior para o menor, buscando ainda 70% a 100%.
    """
    tentativas: List[Dict[str, Any]] = []

    for idx, veic in catalogo_veiculos.sort_values("ordem_porte", ascending=False).iterrows():
        if not _veiculo_disponivel_no_modo_frota(veic, tipo_roteirizacao):
            tentativas.append(
                {
                    "veiculo_tipo": veic["tipo"],
                    "resultado_teste": "rejeitado",
                    "motivo_reprovacao": "perfil_sem_disponibilidade_no_modo_frota",
                }
            )
            continue

        aval_full = _avaliar_combo_no_veiculo(base_cliente, veic=veic, ignorar_ocupacao_minima=False)
        reg_full = {
            **aval_full,
            "resultado_teste": "aceito" if aval_full["aceito"] else "rejeitado",
            "tipo_busca": "restante_full_maior_para_menor",
        }
        if not aval_full["aceito"]:
            motivos = []
            if not aval_full.get("cabe_carga_oficial", True):
                motivos.append("excede_capacidade_peso_oficial")
            if not aval_full.get("cabe_paradas", True):
                motivos.append("excede_max_entregas")
            if not aval_full.get("cabe_km", True):
                motivos.append("excede_max_km")
            if not aval_full.get("passa_ocupacao", True):
                motivos.append("nao_atinge_faixa_ocupacao_70_100")
            reg_full["motivo_reprovacao"] = "|".join(motivos) if motivos else "restante_nao_fechou"

        tentativas.append(reg_full)

        if aval_full["aceito"]:
            return {
                "aceito": True,
                "df_combo": base_cliente.copy().reset_index(drop=True),
                "avaliacao": aval_full,
                "catalogo_idx": idx,
                "tentativas": tentativas,
            }

        grupo = _gerar_subconjunto_guloso(
            base_cliente=base_cliente,
            veic=veic,
            anchor_id=None,
            ignorar_ocupacao_minima=False,
            ordem_desc=True,
        )

        if len(grupo) == 0:
            continue

        aval_sub = _avaliar_combo_no_veiculo(grupo, veic=veic, ignorar_ocupacao_minima=False)
        reg_sub = {
            **aval_sub,
            "resultado_teste": "aceito" if aval_sub["aceito"] else "rejeitado",
            "tipo_busca": "restante_subconjunto_maior_para_menor",
        }
        if not aval_sub["aceito"]:
            motivos = []
            if not aval_sub.get("cabe_carga_oficial", True):
                motivos.append("excede_capacidade_peso_oficial")
            if not aval_sub.get("cabe_paradas", True):
                motivos.append("excede_max_entregas")
            if not aval_sub.get("cabe_km", True):
                motivos.append("excede_max_km")
            if not aval_sub.get("passa_ocupacao", True):
                motivos.append("nao_atinge_faixa_ocupacao_70_100")
            reg_sub["motivo_reprovacao"] = "|".join(motivos) if motivos else "restante_subconjunto_nao_fechou"

        tentativas.append(reg_sub)

        if aval_sub["aceito"]:
            return {
                "aceito": True,
                "df_combo": grupo,
                "avaliacao": aval_sub,
                "catalogo_idx": idx,
                "tentativas": tentativas,
            }

    return {
        "aceito": False,
        "motivo_reprovacao": "restante_cliente_nao_fechou",
        "tentativas": tentativas,
    }


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
        "qtd_clientes_nao_exclusivos": 0,
        "qtd_tentativas_total": 0,
        "qtd_tentativas_rejeitadas_ocupacao": 0,
        "qtd_tentativas_rejeitadas_km": 0,
        "qtd_tentativas_rejeitadas_paradas": 0,
        "qtd_tentativas_rejeitadas_capacidade": 0,
        "qtd_tentativas_sem_disponibilidade_frota": 0,
        "qtd_manifestos_exclusivos": 0,
        "qtd_manifestos_cliente_menor_maior": 0,
        "qtd_manifestos_cliente_maior_menor": 0,
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
    fila = _garantir_coluna_por_alias(fila, "cidade", ["Cidade Dest.", "Cida", "cidade_dest", "cidade_destino"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "uf", ["UF"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "peso_kg", ["Peso", "peso"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "vol_m3", ["Peso C", "peso_c", "cubagem_m3"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "peso_calculado", ["Peso Calculado", "peso_calc"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "veiculo_exclusivo", ["Veiculo Exclusivo", "veiculo_dedicado"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "veiculo_exclusivo_flag", ["flag_veiculo_exclusivo", "veiculo_exclusivo_bool"], default=False)
    fila = _garantir_coluna_por_alias(fila, "prioridade_embarque", ["Prioridade", "prioridade"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "distancia_rodoviaria_est_km", ["km_referencia", "distancia_km", "km_rota_referencia"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "status_triagem", ["status_roteirizacao", "status_fila"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "grupo_saida", ["grupo_pipeline", "grupo_status"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "data_agenda", ["Agendam.", "agenda_data", "data_agendamento"], default=pd.NaT)
    fila = _garantir_coluna_por_alias(fila, "data_leadtime", ["D.L.E.", "dle", "leadtime_data_limite_entrega"], default=pd.NaT)
    fila = _garantir_coluna_por_alias(fila, "ranking_prioridade", ["ranking_prioridade_operacional", "ranking_preliminar"], default=999999)
    fila = _garantir_coluna_por_alias(fila, "score_prioridade_preliminar", ["score_prioridade", "score_operacional"], default=0.0)
    fila = _garantir_coluna_por_alias(fila, "id_linha_pipeline", ["id", "id_linha", "hash_linha_pipeline"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "cte", ["nro_documento", "romaneio", "nro_doc", "Nro Doc."], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "mesorregiao", ["Mesoregião", "mesorregiao"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "subregiao", ["Sub-Região", "subregiao"], default=np.nan)

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

    fila["veiculo_exclusivo_flag"] = fila.apply(_eh_exclusivo, axis=1)

    if fila["cte"].isna().all():
        fila["cte"] = fila["id_linha_pipeline"].astype(str)
    else:
        fila["cte"] = fila["cte"].fillna(fila["id_linha_pipeline"].astype(str))

    fila["ranking_prioridade"] = pd.to_numeric(fila["ranking_prioridade"], errors="coerce").fillna(999999)
    fila["score_prioridade_preliminar"] = pd.to_numeric(fila["score_prioridade_preliminar"], errors="coerce").fillna(0.0)
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
    docs_alocados: set[str] = set()
    contador_manifesto = 1

    def _contabilizar_tentativa(tent: Dict[str, Any]) -> None:
        contadores_m4["qtd_tentativas_total"] += 1
        motivo = str(tent.get("motivo_reprovacao", "")).strip()
        if motivo == "perfil_sem_disponibilidade_no_modo_frota":
            contadores_m4["qtd_tentativas_sem_disponibilidade_frota"] += 1
        if "nao_atinge_faixa_ocupacao_70_100" in motivo:
            contadores_m4["qtd_tentativas_rejeitadas_ocupacao"] += 1
        if "excede_max_km" in motivo:
            contadores_m4["qtd_tentativas_rejeitadas_km"] += 1
        if "excede_max_entregas" in motivo:
            contadores_m4["qtd_tentativas_rejeitadas_paradas"] += 1
        if "excede_capacidade_peso_oficial" in motivo:
            contadores_m4["qtd_tentativas_rejeitadas_capacidade"] += 1

    def _filtrar_nao_alocados(df_base: pd.DataFrame) -> pd.DataFrame:
        col_doc = _obter_coluna_id_documento(df_base)
        mask = (
            ~df_base["id_linha_pipeline"].astype(str).isin(ids_alocados)
            & ~df_base[col_doc].astype(str).isin(docs_alocados)
        )
        return df_base.loc[mask].copy().reset_index(drop=True)

    def registrar_manifesto(
        df_combo: pd.DataFrame,
        avaliacao: Dict[str, Any],
        origem_etapa: str,
        catalogo_idx: Optional[int],
    ) -> None:
        nonlocal contador_manifesto, manifestos_fechados, itens_manifestos_fechados, ids_alocados, docs_alocados

        if len(df_combo) == 0:
            raise Exception("Tentativa de registrar manifesto vazio.")

        col_doc = _obter_coluna_id_documento(df_combo)

        ids_combo = set(df_combo["id_linha_pipeline"].astype(str).tolist())
        docs_combo = set(df_combo[col_doc].astype(str).tolist())

        if ids_combo & ids_alocados:
            raise Exception("Manifesto inválido: há id_linha_pipeline já alocado em outro manifesto.")

        if docs_combo & docs_alocados:
            raise Exception("Manifesto inválido: há documento/CTE já alocado em outro manifesto.")

        if not avaliacao.get("ignorar_ocupacao_minima", False):
            ocup = _num_safe(avaliacao.get("ocupacao_oficial_perc"), default=np.nan)
            if pd.isna(ocup) or ocup < 70 or ocup > 100:
                raise Exception(
                    f"Manifesto inválido: ocupação fora da faixa 70%-100%. "
                    f"Valor encontrado: {ocup}"
                )

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

        itens = df_combo.copy().reset_index(drop=True)
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

        ids_alocados.update(ids_combo)
        docs_alocados.update(docs_combo)

        _consumir_veiculo_catalogo(catalogo_veiculos, catalogo_idx, tipo_roteirizacao)

    # -----------------------------------------
    # 4B1 - EXCLUSIVOS
    # -----------------------------------------
    t0 = _agora()
    exclusivos = _filtrar_nao_alocados(fila_ordenada)
    exclusivos = exclusivos.loc[exclusivos["veiculo_exclusivo_flag"] == True].copy().reset_index(drop=True)
    contadores_m4["qtd_anchors_exclusivos"] = int(len(exclusivos))

    for _, linha_exclusiva in exclusivos.iterrows():
        id_anchor = str(linha_exclusiva["id_linha_pipeline"])
        if id_anchor in ids_alocados:
            continue

        fila_disponivel = _filtrar_nao_alocados(fila_ordenada)

        resultado_exclusivo = _tentar_exclusivo_com_mesmo_cliente(
            linha_exclusiva=linha_exclusiva,
            fila_disponivel=fila_disponivel,
            catalogo_veiculos=catalogo_veiculos,
            tipo_roteirizacao=tipo_roteirizacao,
        )

        for tent in resultado_exclusivo.get("tentativas", []):
            tent_padrao = {
                **tent,
                "etapa_fechamento": "4B1_exclusivo",
                "tipo_tentativa": "exclusivo_com_mesmo_cliente",
                "cliente_referencia": linha_exclusiva["destinatario"],
                "linha_ancora": id_anchor,
            }
            tentativas_fechamento.append(tent_padrao)
            _contabilizar_tentativa(tent_padrao)

        if resultado_exclusivo["aceito"]:
            registrar_manifesto(
                df_combo=resultado_exclusivo["df_combo"],
                avaliacao=resultado_exclusivo["avaliacao"],
                origem_etapa="4B1_exclusivo",
                catalogo_idx=resultado_exclusivo["catalogo_idx"],
            )
            contadores_m4["qtd_manifestos_exclusivos"] += 1

    tempos_m4["4B1_exclusivos_ms"] = _duracao_ms(t0)

    # -----------------------------------------
    # 4B2/4C - CLIENTES NÃO EXCLUSIVOS
    # -----------------------------------------
    t0 = _agora()

    fila_nao_exclusiva = _filtrar_nao_alocados(fila_ordenada)
    fila_nao_exclusiva = fila_nao_exclusiva.loc[fila_nao_exclusiva["veiculo_exclusivo_flag"] == False].copy()

    # agrupamento por cliente puro, nunca mistura cliente diferente
    clientes = list(dict.fromkeys(fila_nao_exclusiva["destinatario"].astype(str).tolist()))
    contadores_m4["qtd_clientes_nao_exclusivos"] = int(len(clientes))

    for cliente in clientes:
        while True:
            fila_cliente = _filtrar_nao_alocados(fila_ordenada)
            fila_cliente = fila_cliente.loc[
                (fila_cliente["veiculo_exclusivo_flag"] == False)
                & (fila_cliente["destinatario"].astype(str) == str(cliente))
            ].copy().reset_index(drop=True)

            if len(fila_cliente) == 0:
                break

            # 1) tenta cluster completo / subconjunto no menor -> maior
            resultado_primario = _tentar_cluster_mesmo_cliente_menor_para_maior(
                base_cliente=fila_cliente,
                catalogo_veiculos=catalogo_veiculos,
                tipo_roteirizacao=tipo_roteirizacao,
            )

            for tent in resultado_primario.get("tentativas", []):
                tent_padrao = {
                    **tent,
                    "etapa_fechamento": "4B2_cliente_menor_para_maior",
                    "tipo_tentativa": "cluster_mesmo_cliente",
                    "cliente_referencia": cliente,
                    "linha_ancora": fila_cliente["id_linha_pipeline"].astype(str).iloc[0],
                }
                tentativas_fechamento.append(tent_padrao)
                _contabilizar_tentativa(tent_padrao)

            if resultado_primario["aceito"]:
                registrar_manifesto(
                    df_combo=resultado_primario["df_combo"],
                    avaliacao=resultado_primario["avaliacao"],
                    origem_etapa="4B2_cliente_menor_para_maior",
                    catalogo_idx=resultado_primario["catalogo_idx"],
                )
                contadores_m4["qtd_manifestos_cliente_menor_maior"] += 1
                continue

            # 2) tenta o restante do cliente do maior -> menor
            resultado_restante = _tentar_restante_cliente_maior_para_menor(
                base_cliente=fila_cliente,
                catalogo_veiculos=catalogo_veiculos,
                tipo_roteirizacao=tipo_roteirizacao,
            )

            for tent in resultado_restante.get("tentativas", []):
                tent_padrao = {
                    **tent,
                    "etapa_fechamento": "4C_restante_cliente_maior_para_menor",
                    "tipo_tentativa": "restante_mesmo_cliente",
                    "cliente_referencia": cliente,
                    "linha_ancora": fila_cliente["id_linha_pipeline"].astype(str).iloc[0],
                }
                tentativas_fechamento.append(tent_padrao)
                _contabilizar_tentativa(tent_padrao)

            if resultado_restante["aceito"]:
                registrar_manifesto(
                    df_combo=resultado_restante["df_combo"],
                    avaliacao=resultado_restante["avaliacao"],
                    origem_etapa="4C_restante_cliente_maior_para_menor",
                    catalogo_idx=resultado_restante["catalogo_idx"],
                )
                contadores_m4["qtd_manifestos_cliente_maior_menor"] += 1
                continue

            # nada mais fecha para esse cliente no M4
            break

    tempos_m4["4B2_4C_clientes_ms"] = _duracao_ms(t0)

    # -----------------------------------------
    # VALIDAÇÃO DURA PÓS-M4
    # -----------------------------------------
    t0 = _agora()

    df_manifestos_fechados_bloco_4 = pd.DataFrame(manifestos_fechados)

    df_itens_manifestos_fechados_bloco_4 = (
        pd.concat(itens_manifestos_fechados, ignore_index=True)
        if len(itens_manifestos_fechados) > 0
        else pd.DataFrame()
    )

    df_tentativas_fechamento_bloco_4 = pd.DataFrame(tentativas_fechamento)

    df_remanescente_roteirizavel_bloco_4 = _filtrar_nao_alocados(fila).copy().reset_index(drop=True)

    if len(df_itens_manifestos_fechados_bloco_4) > 0:
        col_doc = _obter_coluna_id_documento(df_itens_manifestos_fechados_bloco_4)

        if df_itens_manifestos_fechados_bloco_4["id_linha_pipeline"].astype(str).duplicated().any():
            raise Exception("Validação pós-M4 falhou: id_linha_pipeline repetido em mais de um item manifesto.")

        if df_itens_manifestos_fechados_bloco_4[col_doc].astype(str).duplicated().any():
            raise Exception("Validação pós-M4 falhou: documento/CTE repetido em mais de um manifesto.")

        # validar ocupação por manifesto pelo resumo, não somando linhas
        if len(df_manifestos_fechados_bloco_4) > 0:
            base_invalidos = df_manifestos_fechados_bloco_4.loc[
                (~df_manifestos_fechados_bloco_4["ignorar_ocupacao_minima"])
                & (
                    (pd.to_numeric(df_manifestos_fechados_bloco_4["ocupacao_oficial_perc"], errors="coerce") < 70)
                    | (pd.to_numeric(df_manifestos_fechados_bloco_4["ocupacao_oficial_perc"], errors="coerce") > 100)
                )
            ].copy()
            if len(base_invalidos) > 0:
                raise Exception(
                    "Validação pós-M4 falhou: manifesto não exclusivo com ocupação fora de 70%-100%."
                )

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

    tempos_m4["validacao_pos_m4_ms"] = _duracao_ms(t0)

    # =========================================================================================
    # PERSISTÊNCIA OPCIONAL
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
                "modulo": "4_manifestos_fechados_corrigido",
                "data_execucao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_base_projeto": pd.Timestamp(data_base_roteirizacao).strftime("%Y-%m-%d"),
                "tipo_roteirizacao": tipo_roteirizacao,
                "regras": {
                    "peso_calculado_base_oficial": True,
                    "exclusivo_sem_ocupacao_minima": True,
                    "nao_mistura_clientes": True,
                    "cluster_cliente_menor_para_maior": True,
                    "restante_cliente_maior_para_menor": True,
                    "ocupacao_minima_padrao": OCUPACAO_MINIMA_PADRAO,
                    "ocupacao_maxima_padrao": OCUPACAO_MAXIMA_PADRAO,
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
        "ocupacao_minima_padrao_perc": round(OCUPACAO_MINIMA_PADRAO * 100, 2),
        "ocupacao_maxima_padrao_perc": round(OCUPACAO_MAXIMA_PADRAO * 100, 2),
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
