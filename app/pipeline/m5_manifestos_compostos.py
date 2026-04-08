from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

OCUPACAO_MINIMA_DOMINANTE = 0.70
OCUPACAO_MAXIMA_PADRAO = 1.00
OCUPACAO_MINIMA_SECUNDARIA = 0.20
MIN_PARADAS_COMPOSTO = 2


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


def _deduplicar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df.columns) == 0:
        return df.copy()
    if not df.columns.duplicated().any():
        return df.copy()
    return df.loc[:, ~df.columns.duplicated()].copy()


def _resolver_coluna_tipo_veiculo(df_veiculos: pd.DataFrame) -> str:
    if "tipo" in df_veiculos.columns:
        return "tipo"
    if "perfil" in df_veiculos.columns:
        return "perfil"
    raise Exception("Faltam colunas mínimas na base de veículos: tipo ou perfil.")


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


def _normalizar_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def _cliente_key(row: pd.Series) -> str:
    return _normalizar_str(row.get("destinatario"))


def _cidade_key(row: pd.Series) -> str:
    return f"{_normalizar_str(row.get('cidade'))}|{_normalizar_str(row.get('uf'))}"


def _subregiao_key(row: pd.Series) -> str:
    return f"{_normalizar_str(row.get('subregiao'))}|{_normalizar_str(row.get('uf'))}"


def _mesorregiao_key(row: pd.Series) -> str:
    return f"{_normalizar_str(row.get('mesorregiao'))}|{_normalizar_str(row.get('uf'))}"


def _chave_parada_df(df_: pd.DataFrame) -> pd.Series:
    return (
        df_["destinatario"].astype(str).fillna("").str.strip().str.upper()
        + "|"
        + df_["cidade"].astype(str).fillna("").str.strip().str.upper()
        + "|"
        + df_["uf"].astype(str).fillna("").str.strip().str.upper()
    )


def _eh_exclusivo(row: pd.Series) -> bool:
    if "veiculo_exclusivo_flag" in row.index:
        return _bool_safe(row.get("veiculo_exclusivo_flag"))
    return _bool_safe(row.get("veiculo_exclusivo"))


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


def _preparar_catalogo_veiculos_m5(
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


def _veiculo_disponivel_no_modo_frota(veic: pd.Series, tipo_roteirizacao: str) -> bool:
    tipo_roteirizacao = _normalizar_tipo_roteirizacao(tipo_roteirizacao)
    if tipo_roteirizacao == "carteira":
        return True

    limite = _num_safe(veic.get("limite_manifestos"), default=np.nan)
    usados = _num_safe(veic.get("manifestos_utilizados"), default=0)

    if pd.isna(limite):
        return True

    return int(usados) < int(limite)


def _consumir_veiculo_catalogo(catalogo_veiculos: pd.DataFrame, catalogo_idx: Optional[int], tipo_roteirizacao: str) -> None:
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


def _obter_base_carga_oficial(df_combo: pd.DataFrame) -> float:
    if "peso_calculado" not in df_combo.columns:
        return 0.0
    return float(pd.to_numeric(df_combo["peso_calculado"], errors="coerce").fillna(0).sum())


def _avaliar_combo_no_veiculo(df_combo: pd.DataFrame, veic: pd.Series) -> Dict[str, Any]:
    base_carga_total = _obter_base_carga_oficial(df_combo)
    peso_total_kg = float(pd.to_numeric(df_combo["peso_kg"], errors="coerce").fillna(0).sum())
    vol_total_m3 = float(pd.to_numeric(df_combo["vol_m3"], errors="coerce").fillna(0).sum())
    km_combo = float(pd.to_numeric(df_combo["distancia_rodoviaria_est_km"], errors="coerce").max())
    qtd_itens = int(len(df_combo))
    qtd_paradas = int(_chave_parada_df(df_combo).nunique())

    cap_peso = float(veic["capacidade_peso_kg"])
    cap_vol = float(veic["capacidade_vol_m3"])
    max_entregas = int(veic["max_entregas"])
    max_km = float(veic["max_km_distancia"])

    cabe_peso = base_carga_total <= cap_peso
    cabe_volume = vol_total_m3 <= cap_vol if pd.notna(cap_vol) and cap_vol > 0 else True
    cabe_paradas = qtd_paradas <= max_entregas
    cabe_km = km_combo <= max_km if pd.notna(km_combo) else False

    ocupacao = base_carga_total / cap_peso if pd.notna(cap_peso) and cap_peso > 0 else np.nan

    passa_ocupacao = (
        pd.notna(ocupacao)
        and ocupacao >= OCUPACAO_MINIMA_DOMINANTE
        and ocupacao <= OCUPACAO_MAXIMA_PADRAO
    )
    passa_paradas_minimas = qtd_paradas >= MIN_PARADAS_COMPOSTO

    if qtd_paradas > 0:
        base_por_parada = (
            df_combo.assign(_chave_parada=_chave_parada_df(df_combo))
            .groupby("_chave_parada", as_index=False)["peso_calculado"]
            .sum()
        )
        menor_parada = float(pd.to_numeric(base_por_parada["peso_calculado"], errors="coerce").fillna(0).min())
        ocupacao_secundaria = menor_parada / cap_peso if cap_peso > 0 else np.nan
    else:
        ocupacao_secundaria = np.nan

    passa_ocupacao_secundaria = (
        pd.notna(ocupacao_secundaria)
        and ocupacao_secundaria >= OCUPACAO_MINIMA_SECUNDARIA
    )

    aceito = bool(
        cabe_peso
        and cabe_volume
        and cabe_paradas
        and cabe_km
        and passa_ocupacao
        and passa_paradas_minimas
        and passa_ocupacao_secundaria
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
        "qtd_paradas": qtd_paradas,
        "cabe_peso": cabe_peso,
        "cabe_volume": cabe_volume,
        "cabe_paradas": cabe_paradas,
        "cabe_km": cabe_km,
        "ocupacao_dominante_perc": round(float(ocupacao * 100), 2) if pd.notna(ocupacao) else np.nan,
        "passa_ocupacao_dominante": passa_ocupacao,
        "ocupacao_secundaria_perc": round(float(ocupacao_secundaria * 100), 2) if pd.notna(ocupacao_secundaria) else np.nan,
        "passa_ocupacao_secundaria": passa_ocupacao_secundaria,
        "passa_paradas_minimas": passa_paradas_minimas,
        "aceito": aceito,
    }


def _motivos_reprovacao(avaliacao: Dict[str, Any]) -> str:
    motivos: List[str] = []
    if not avaliacao.get("cabe_peso", True):
        motivos.append("excede_capacidade_peso")
    if not avaliacao.get("cabe_volume", True):
        motivos.append("excede_capacidade_volume")
    if not avaliacao.get("cabe_paradas", True):
        motivos.append("excede_max_entregas")
    if not avaliacao.get("cabe_km", True):
        motivos.append("excede_max_km")
    if not avaliacao.get("passa_paradas_minimas", True):
        motivos.append("menos_de_2_paradas")
    if not avaliacao.get("passa_ocupacao_dominante", True):
        motivos.append("nao_atinge_ocupacao_dominante_70_100")
    if not avaliacao.get("passa_ocupacao_secundaria", True):
        motivos.append("nao_atinge_ocupacao_secundaria_20")
    return "|".join(motivos) if motivos else "nao_fechou"


def _validar_input_oficial_bloco_5(
    df_input_oficial_bloco_5: pd.DataFrame,
    df_manifestos_fechados_bloco_4: pd.DataFrame,
    df_itens_manifestos_fechados_bloco_4: pd.DataFrame,
) -> pd.DataFrame:
    fila = _deduplicar_colunas(df_input_oficial_bloco_5.copy().reset_index(drop=True))
    fila = _garantir_coluna_por_alias(fila, "id_linha_pipeline", ["id", "id_linha", "hash_linha_pipeline"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "status_triagem", ["status_roteirizacao", "status_fila"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "grupo_saida", ["grupo_pipeline", "grupo_status"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "manifesto_id", ["manifesto"], default=np.nan)

    if fila["id_linha_pipeline"].isna().any():
        qtd_nulos = int(fila["id_linha_pipeline"].isna().sum())
        raise Exception(f"O input oficial do Bloco 5 possui id_linha_pipeline nulo: {qtd_nulos}")

    if fila["id_linha_pipeline"].astype(str).duplicated().any():
        qtd_dup = int(fila["id_linha_pipeline"].astype(str).duplicated().sum())
        raise Exception(f"O input oficial do Bloco 5 possui id_linha_pipeline duplicado: {qtd_dup}")

    if len(df_manifestos_fechados_bloco_4) > 0 and "manifesto_id" in fila.columns:
        if fila["manifesto_id"].notna().any():
            raise Exception("O input oficial do Bloco 5 não pode receber manifestos fechados do M4.")

    if len(df_itens_manifestos_fechados_bloco_4) > 0 and "id_linha_pipeline" in df_itens_manifestos_fechados_bloco_4.columns:
        ids_m4 = set(df_itens_manifestos_fechados_bloco_4["id_linha_pipeline"].astype(str).tolist())
        ids_m5 = set(fila["id_linha_pipeline"].astype(str).tolist())
        if ids_m4 & ids_m5:
            raise Exception("O input oficial do Bloco 5 contém itens já manifestados no M4.")

    linhas_input_invalido = fila.loc[
        (fila["status_triagem"].astype(str) != "roteirizavel")
        | (fila["grupo_saida"].astype(str) != "df_carteira_roteirizavel")
    ].copy()
    if len(linhas_input_invalido) > 0:
        raise Exception(
            "O BLOCO 5 recebeu linhas incompatíveis com o estágio. "
            "Há registros com status_triagem != 'roteirizavel' ou grupo_saida inválido."
        )

    return fila


def _preparar_fila_operacional(df_base: pd.DataFrame) -> pd.DataFrame:
    fila = df_base.copy()

    fila = _garantir_coluna_por_alias(fila, "destinatario", ["Destinatário", "cliente"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "cidade", ["Cidade Dest.", "Cida", "cidade_dest", "cidade_destino"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "uf", ["UF"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "subregiao", ["Sub-Região", "sub_regiao"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "mesorregiao", ["Mesoregião", "mesoregiao"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "peso_kg", ["Peso", "peso"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "vol_m3", ["Peso C", "peso_c", "cubagem_m3"], default=0.0)
    fila = _garantir_coluna_por_alias(fila, "peso_calculado", ["Peso Calculado", "peso_calc"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "veiculo_exclusivo", ["Veiculo Exclusivo", "veiculo_dedicado"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "veiculo_exclusivo_flag", ["flag_veiculo_exclusivo", "veiculo_exclusivo_bool"], default=False)
    fila = _garantir_coluna_por_alias(fila, "prioridade_embarque", ["Prioridade", "prioridade"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "distancia_rodoviaria_est_km", ["km_referencia", "distancia_km", "km_rota_referencia"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "data_agenda", ["Agendam.", "agenda_data", "data_agendamento"], default=pd.NaT)
    fila = _garantir_coluna_por_alias(fila, "data_leadtime", ["D.L.E.", "dle", "leadtime_data_limite_entrega"], default=pd.NaT)
    fila = _garantir_coluna_por_alias(fila, "score_prioridade_preliminar", ["score_prioridade", "score_operacional"], default=0.0)
    fila = _garantir_coluna_por_alias(fila, "folga_dias", ["folga", "folga_dia"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "nro_documento", ["Nro Doc.", "nro_doc"], default=np.nan)
    fila = _garantir_coluna_por_alias(fila, "cte", ["nro_documento", "Nro Doc."], default=np.nan)

    for col in [
        "peso_kg", "vol_m3", "peso_calculado", "distancia_rodoviaria_est_km",
        "prioridade_embarque", "score_prioridade_preliminar", "folga_dias"
    ]:
        if col in fila.columns:
            fila[col] = pd.to_numeric(fila[col], errors="coerce")

    for col in ["data_agenda", "data_leadtime"]:
        if col in fila.columns:
            fila[col] = pd.to_datetime(fila[col], errors="coerce")

    fila["veiculo_exclusivo_flag"] = fila.apply(_eh_exclusivo, axis=1)
    fila["cte"] = fila["cte"].fillna(fila["id_linha_pipeline"].astype(str))
    fila["cliente_chave_m5"] = fila["destinatario"].astype(str).fillna("").str.strip().str.upper()
    fila["cidade_chave_m5"] = fila["cidade"].astype(str).fillna("").str.strip().str.upper()
    fila["subregiao_chave_m5"] = fila["subregiao"].astype(str).fillna("").str.strip().str.upper()
    fila["mesorregiao_chave_m5"] = fila["mesorregiao"].astype(str).fillna("").str.strip().str.upper()
    fila["bucket_temporal_m5"] = np.where(
        fila["data_agenda"].notna(),
        "agendada",
        np.where(fila["data_leadtime"].notna(), "leadtime", "sem_data")
    )

    return _ordenar_fila(fila)


def _selecionar_ancoras_unicas_por_cliente(df_fila_operacional: pd.DataFrame) -> pd.DataFrame:
    if len(df_fila_operacional) == 0:
        return df_fila_operacional.head(0).copy()

    ancoras_idx: List[int] = []
    clientes_reservados: set[str] = set()

    for idx, row in df_fila_operacional.iterrows():
        cliente = _cliente_key(row)
        if cliente == "":
            continue
        if cliente in clientes_reservados:
            continue
        ancoras_idx.append(idx)
        clientes_reservados.add(cliente)

    df_ancoras = df_fila_operacional.loc[ancoras_idx].copy().reset_index(drop=True)
    df_ancoras["anchor_id_m5"] = df_ancoras["id_linha_pipeline"].astype(str)
    return df_ancoras


def _buscar_candidatos_hierarquicos(df_disponivel: pd.DataFrame, anchor: pd.Series) -> pd.DataFrame:
    if len(df_disponivel) == 0:
        return df_disponivel.head(0).copy()

    cliente = _cliente_key(anchor)
    cidade = _cidade_key(anchor)
    subregiao = _subregiao_key(anchor)
    mesorregiao = _mesorregiao_key(anchor)

    df = df_disponivel.copy()

    mask_cliente = df.apply(_cliente_key, axis=1) == cliente
    mask_cidade = df.apply(_cidade_key, axis=1) == cidade
    mask_subregiao = df.apply(_subregiao_key, axis=1) == subregiao
    mask_mesorregiao = df.apply(_mesorregiao_key, axis=1) == mesorregiao

    grupos = []

    def _marcar(df_part: pd.DataFrame, escopo: str, prioridade_escopo: int) -> pd.DataFrame:
        if len(df_part) == 0:
            return df_part
        out = df_part.copy()
        out["escopo_composicao_m5"] = escopo
        out["prioridade_escopo_m5"] = prioridade_escopo
        return out

    grupos.append(_marcar(df.loc[mask_cliente].copy(), "mesmo_cliente", 1))
    grupos.append(_marcar(df.loc[~mask_cliente & mask_cidade].copy(), "mesma_cidade", 2))
    grupos.append(_marcar(df.loc[~mask_cliente & ~mask_cidade & mask_subregiao].copy(), "mesma_subregiao", 3))
    grupos.append(_marcar(df.loc[~mask_cliente & ~mask_cidade & ~mask_subregiao & mask_mesorregiao].copy(), "mesma_mesorregiao", 4))

    grupos_validos = [g for g in grupos if len(g) > 0]
    if len(grupos_validos) == 0:
        return df.head(0).copy()

    candidatos = pd.concat(grupos_validos, ignore_index=True)
    candidatos = _ordenar_fila(candidatos)
    candidatos = candidatos.sort_values(
        by=["prioridade_escopo_m5", "folga_dias", "distancia_rodoviaria_est_km", "peso_calculado"],
        ascending=[True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    return candidatos


def _tentar_montar_premanifesto_anchor(
    anchor: pd.Series,
    df_disponivel: pd.DataFrame,
    catalogo_veiculos: pd.DataFrame,
    tipo_roteirizacao: str,
) -> Dict[str, Any]:
    anchor_id = str(anchor["id_linha_pipeline"])
    candidatos = _buscar_candidatos_hierarquicos(df_disponivel, anchor)
    tentativas: List[Dict[str, Any]] = []

    if len(candidatos) == 0:
        return {
            "aceito": False,
            "motivo_reprovacao": "sem_candidatos_hierarquicos",
            "tentativas": tentativas,
        }

    candidatos = candidatos.loc[
        ~candidatos["id_linha_pipeline"].astype(str).duplicated()
    ].copy().reset_index(drop=True)

    if anchor_id not in set(candidatos["id_linha_pipeline"].astype(str).tolist()):
        candidatos = pd.concat([pd.DataFrame([anchor]), candidatos], ignore_index=True)

    for idx_veic, veic in catalogo_veiculos.sort_values("ordem_porte", ascending=False).iterrows():
        if not _veiculo_disponivel_no_modo_frota(veic, tipo_roteirizacao):
            tentativas.append(
                {
                    "anchor_id_m5": anchor_id,
                    "cliente_referencia": anchor.get("destinatario"),
                    "veiculo_tipo": veic["tipo"],
                    "resultado_teste": "rejeitado",
                    "motivo_reprovacao": "perfil_sem_disponibilidade_no_modo_frota",
                    "etapa_fechamento": "5B_premanifesto_anchor",
                }
            )
            continue

        grupo = pd.DataFrame(columns=candidatos.columns)
        usados_ids: set[str] = set()

        anchor_df = candidatos.loc[candidatos["id_linha_pipeline"].astype(str) == anchor_id].head(1).copy()
        if len(anchor_df) == 0:
            continue
        grupo = pd.concat([grupo, anchor_df], ignore_index=True)
        usados_ids.add(anchor_id)

        restantes = candidatos.loc[candidatos["id_linha_pipeline"].astype(str) != anchor_id].copy().reset_index(drop=True)

        for _, row in restantes.iterrows():
            row_id = str(row["id_linha_pipeline"])
            if row_id in usados_ids:
                continue

            teste = pd.concat([grupo, row.to_frame().T], ignore_index=True)
            aval_teste = _avaliar_combo_no_veiculo(teste, veic)

            pode_continuar = (
                aval_teste["cabe_peso"]
                and aval_teste["cabe_volume"]
                and aval_teste["cabe_paradas"]
                and aval_teste["cabe_km"]
            )
            if pode_continuar:
                grupo = teste
                usados_ids.add(row_id)

        avaliacao_final = _avaliar_combo_no_veiculo(grupo, veic)
        registro = {
            **avaliacao_final,
            "anchor_id_m5": anchor_id,
            "cliente_referencia": anchor.get("destinatario"),
            "resultado_teste": "aceito" if avaliacao_final["aceito"] else "rejeitado",
            "motivo_reprovacao": None if avaliacao_final["aceito"] else _motivos_reprovacao(avaliacao_final),
            "etapa_fechamento": "5B_premanifesto_anchor",
        }
        tentativas.append(registro)

        if avaliacao_final["aceito"]:
            grupo = grupo.copy().reset_index(drop=True)
            grupo["anchor_id_m5"] = anchor_id
            return {
                "aceito": True,
                "df_combo": grupo,
                "avaliacao": avaliacao_final,
                "catalogo_idx": idx_veic,
                "tentativas": tentativas,
            }

    return {
        "aceito": False,
        "motivo_reprovacao": "anchor_nao_fechou_em_ningum_perfil",
        "tentativas": tentativas,
    }


def _sequenciar_manifesto(df_combo: pd.DataFrame) -> pd.DataFrame:
    if len(df_combo) == 0:
        return df_combo.copy()

    df = df_combo.copy().reset_index(drop=True)
    sort_cols = []
    ascending = []

    if "data_agenda" in df.columns:
        sort_cols.append("data_agenda")
        ascending.append(True)
    if "data_leadtime" in df.columns:
        sort_cols.append("data_leadtime")
        ascending.append(True)
    if "distancia_rodoviaria_est_km" in df.columns:
        sort_cols.append("distancia_rodoviaria_est_km")
        ascending.append(True)
    if "peso_calculado" in df.columns:
        sort_cols.append("peso_calculado")
        ascending.append(False)

    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)

    df["ordem_descarga_m5"] = np.arange(1, len(df) + 1)
    df["ordem_carregamento_m5"] = np.arange(len(df), 0, -1)
    return df


def _gerar_resumo_manifesto_m5(
    df_combo: pd.DataFrame,
    avaliacao: Dict[str, Any],
    manifesto_id: str,
    origem_etapa: str,
) -> Dict[str, Any]:
    linha = {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": "composto_bloco_5",
        "veiculo_tipo": avaliacao["veiculo_tipo"],
        "qtd_itens": int(len(df_combo)),
        "qtd_paradas": int(_chave_parada_df(df_combo).nunique()),
        "base_carga_oficial": avaliacao["base_carga_oficial"],
        "peso_total_kg": avaliacao["peso_total_kg"],
        "vol_total_m3": avaliacao["vol_total_m3"],
        "km_referencia": avaliacao["km_referencia"],
        "ocupacao_dominante_perc": avaliacao["ocupacao_dominante_perc"],
        "ocupacao_secundaria_perc": avaliacao["ocupacao_secundaria_perc"],
        "capacidade_peso_kg_veiculo": avaliacao["capacidade_peso_kg"],
        "capacidade_vol_m3_veiculo": avaliacao["capacidade_vol_m3"],
        "max_entregas_veiculo": avaliacao["max_entregas"],
        "max_km_distancia_veiculo": avaliacao["max_km_distancia"],
        "origem_modulo": 5,
        "origem_etapa": origem_etapa,
    }

    if len(df_combo) > 0:
        linha["destinatario_anchor"] = df_combo["destinatario"].iloc[0]
        linha["cidade_anchor"] = df_combo["cidade"].iloc[0] if "cidade" in df_combo.columns else np.nan
        linha["uf_anchor"] = df_combo["uf"].iloc[0] if "uf" in df_combo.columns else np.nan
        linha["mesorregiao_anchor"] = df_combo["mesorregiao"].iloc[0] if "mesorregiao" in df_combo.columns else np.nan
        linha["subregiao_anchor"] = df_combo["subregiao"].iloc[0] if "subregiao" in df_combo.columns else np.nan
        linha["anchor_id_m5"] = df_combo["anchor_id_m5"].iloc[0] if "anchor_id_m5" in df_combo.columns else np.nan

    return linha


def _motivo_final_remanescente_m5(id_linha: str, df_tentativas: pd.DataFrame) -> str:
    if df_tentativas is None or df_tentativas.empty:
        return "sem_tentativa_registrada_m5"

    base = df_tentativas.copy()
    if "linha_ancora" in base.columns:
        base = base.loc[base["linha_ancora"].astype(str) == str(id_linha)].copy()

    if base.empty and "motivo_reprovacao" in df_tentativas.columns:
        motivos = [
            str(x).strip()
            for x in df_tentativas["motivo_reprovacao"].dropna().astype(str).tolist()
            if str(x).strip() != ""
        ]
        if len(motivos) == 0:
            return "sem_tentativa_registrada_m5"
        return motivos[0]

    if base.empty or "motivo_reprovacao" not in base.columns:
        return "rejeitado_sem_motivo_detalhado_m5"

    motivos = [
        str(x).strip()
        for x in base["motivo_reprovacao"].dropna().astype(str).tolist()
        if str(x).strip() != ""
    ]
    if len(motivos) == 0:
        return "rejeitado_sem_motivo_detalhado_m5"

    freq: Dict[str, int] = {}
    for motivo in motivos:
        freq[motivo] = freq.get(motivo, 0) + 1
    return max(freq.items(), key=lambda kv: kv[1])[0]


def executar_m5_manifestos_compostos(
    df_input_oficial_bloco_5: pd.DataFrame,
    df_manifestos_fechados_bloco_4: pd.DataFrame,
    df_itens_manifestos_fechados_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
    rodada_id: str,
    data_base_roteirizacao: pd.Timestamp,
    tipo_roteirizacao: str = "carteira",
    configuracao_frota: Any = None,
    caminhos_pipeline: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    inicio_total = _agora()
    tempos_m5: Dict[str, float] = {}
    contadores_m5: Dict[str, Any] = {
        "qtd_anchors_m5": 0,
        "qtd_tentativas_total": 0,
        "qtd_manifestos_compostos": 0,
        "qtd_itens_compostos": 0,
        "qtd_trocas_aceitas_5D": 0,
    }

    caminhos_pipeline = caminhos_pipeline or {}
    tipo_roteirizacao = _normalizar_tipo_roteirizacao(tipo_roteirizacao)

    # 5A
    t0 = _agora()
    fila_bruta = _validar_input_oficial_bloco_5(
        df_input_oficial_bloco_5=df_input_oficial_bloco_5,
        df_manifestos_fechados_bloco_4=df_manifestos_fechados_bloco_4,
        df_itens_manifestos_fechados_bloco_4=df_itens_manifestos_fechados_bloco_4,
    )
    fila_operacional = _preparar_fila_operacional(fila_bruta)
    coluna_tipo_veiculo = _resolver_coluna_tipo_veiculo(df_veiculos_tratados)
    catalogo_veiculos = _preparar_catalogo_veiculos_m5(
        df_veic=_deduplicar_colunas(df_veiculos_tratados.copy()),
        coluna_tipo_veiculo=coluna_tipo_veiculo,
        tipo_roteirizacao=tipo_roteirizacao,
        configuracao_frota=configuracao_frota,
    )
    df_ancoras_m5 = _selecionar_ancoras_unicas_por_cliente(fila_operacional)
    contadores_m5["qtd_anchors_m5"] = int(len(df_ancoras_m5))
    tempos_m5["5A_preparacao_fila_operacional_ms"] = _duracao_ms(t0)

    # 5B / 5C
    t0 = _agora()
    ids_alocados: set[str] = set()
    premanifestos: List[Dict[str, Any]] = []
    itens_premanifestos: List[pd.DataFrame] = []
    tentativas_bloco_5: List[Dict[str, Any]] = []
    contador_premanifesto = 1

    def _filtrar_nao_alocados(df_base: pd.DataFrame) -> pd.DataFrame:
        return df_base.loc[~df_base["id_linha_pipeline"].astype(str).isin(ids_alocados)].copy().reset_index(drop=True)

    for _, anchor in df_ancoras_m5.iterrows():
        anchor_id = str(anchor["id_linha_pipeline"])
        if anchor_id in ids_alocados:
            continue

        fila_disponivel = _filtrar_nao_alocados(fila_operacional)
        resultado = _tentar_montar_premanifesto_anchor(
            anchor=anchor,
            df_disponivel=fila_disponivel,
            catalogo_veiculos=catalogo_veiculos,
            tipo_roteirizacao=tipo_roteirizacao,
        )

        for tent in resultado.get("tentativas", []):
            tent["linha_ancora"] = anchor_id
            tentativas_bloco_5.append(tent)
            contadores_m5["qtd_tentativas_total"] += 1

        if not resultado["aceito"]:
            continue

        df_combo = resultado["df_combo"].copy().reset_index(drop=True)
        ids_combo = set(df_combo["id_linha_pipeline"].astype(str).tolist())
        if ids_combo & ids_alocados:
            raise Exception("Premanifesto inválido no M5: há id_linha_pipeline já alocado em outro pré-manifesto.")

        premanifesto_id = f"PM5_{contador_premanifesto:04d}"
        contador_premanifesto += 1

        resumo = _gerar_resumo_manifesto_m5(
            df_combo=df_combo,
            avaliacao=resultado["avaliacao"],
            manifesto_id=premanifesto_id,
            origem_etapa="5B_premanifesto_anchor",
        )
        resumo["tipo_manifesto"] = "pre_manifesto_bloco_5"
        premanifestos.append(resumo)

        itens = df_combo.copy().reset_index(drop=True)
        itens["pre_manifesto_id"] = premanifesto_id
        itens["tipo_manifesto"] = "pre_manifesto_bloco_5"
        itens["veiculo_tipo"] = resultado["avaliacao"]["veiculo_tipo"]
        itens["capacidade_peso_kg_veiculo"] = resultado["avaliacao"]["capacidade_peso_kg"]
        itens["capacidade_vol_m3_veiculo"] = resultado["avaliacao"]["capacidade_vol_m3"]
        itens["max_entregas_veiculo"] = resultado["avaliacao"]["max_entregas"]
        itens["max_km_distancia_veiculo"] = resultado["avaliacao"]["max_km_distancia"]
        itens["base_carga_oficial_manifesto"] = resultado["avaliacao"]["base_carga_oficial"]
        itens["ocupacao_dominante_perc_manifesto"] = resultado["avaliacao"]["ocupacao_dominante_perc"]
        itens["ocupacao_secundaria_perc_manifesto"] = resultado["avaliacao"]["ocupacao_secundaria_perc"]
        itens["origem_modulo"] = 5
        itens["origem_etapa"] = "5B_premanifesto_anchor"

        itens_premanifestos.append(itens)
        ids_alocados.update(ids_combo)
        _consumir_veiculo_catalogo(catalogo_veiculos, resultado["catalogo_idx"], tipo_roteirizacao)

    df_premanifestos_bloco_5 = pd.DataFrame(premanifestos)
    df_itens_premanifestos_bloco_5 = (
        pd.concat(itens_premanifestos, ignore_index=True)
        if len(itens_premanifestos) > 0
        else pd.DataFrame()
    )
    df_nao_premanifestadas_m5 = _filtrar_nao_alocados(fila_operacional)
    tempos_m5["5B_5C_premanifestos_e_separacao_ms"] = _duracao_ms(t0)

    # 5D
    t0 = _agora()
    df_premanifestos_otimizados_bloco_5 = df_premanifestos_bloco_5.copy()
    df_itens_premanifestos_otimizados_bloco_5 = df_itens_premanifestos_bloco_5.copy()

    if len(df_itens_premanifestos_otimizados_bloco_5) > 0 and "pre_manifesto_id" in df_itens_premanifestos_otimizados_bloco_5.columns:
        grupos_pm = list(df_itens_premanifestos_otimizados_bloco_5["pre_manifesto_id"].dropna().astype(str).unique())

        for i in range(len(grupos_pm)):
            for j in range(i + 1, len(grupos_pm)):
                pm_a = grupos_pm[i]
                pm_b = grupos_pm[j]

                df_a = df_itens_premanifestos_otimizados_bloco_5.loc[
                    df_itens_premanifestos_otimizados_bloco_5["pre_manifesto_id"].astype(str) == pm_a
                ].copy()
                df_b = df_itens_premanifestos_otimizados_bloco_5.loc[
                    df_itens_premanifestos_otimizados_bloco_5["pre_manifesto_id"].astype(str) == pm_b
                ].copy()

                if len(df_a) == 0 or len(df_b) == 0:
                    continue

                veiculo_a = df_a["veiculo_tipo"].iloc[0]
                veiculo_b = df_b["veiculo_tipo"].iloc[0]

                cat_a = catalogo_veiculos.loc[catalogo_veiculos["tipo"].astype(str) == str(veiculo_a)].head(1)
                cat_b = catalogo_veiculos.loc[catalogo_veiculos["tipo"].astype(str) == str(veiculo_b)].head(1)
                if len(cat_a) == 0 or len(cat_b) == 0:
                    continue

                row_a = cat_a.iloc[0]
                row_b = cat_b.iloc[0]

                aval_a_original = _avaliar_combo_no_veiculo(df_a, row_a)
                aval_b_original = _avaliar_combo_no_veiculo(df_b, row_b)
                score_original = (
                    _num_safe(aval_a_original.get("ocupacao_dominante_perc"), 0)
                    + _num_safe(aval_b_original.get("ocupacao_dominante_perc"), 0)
                    - _num_safe(aval_a_original.get("km_referencia"), 0)
                    - _num_safe(aval_b_original.get("km_referencia"), 0)
                )

                idx_a = df_a["peso_calculado"].astype(float).idxmin()
                idx_b = df_b["peso_calculado"].astype(float).idxmax()

                linha_a = df_a.loc[[idx_a]].copy()
                linha_b = df_b.loc[[idx_b]].copy()

                novo_a = pd.concat([df_a.drop(index=idx_a), linha_b], ignore_index=True)
                novo_b = pd.concat([df_b.drop(index=idx_b), linha_a], ignore_index=True)

                aval_a_novo = _avaliar_combo_no_veiculo(novo_a, row_a)
                aval_b_novo = _avaliar_combo_no_veiculo(novo_b, row_b)

                if aval_a_novo["aceito"] and aval_b_novo["aceito"]:
                    score_novo = (
                        _num_safe(aval_a_novo.get("ocupacao_dominante_perc"), 0)
                        + _num_safe(aval_b_novo.get("ocupacao_dominante_perc"), 0)
                        - _num_safe(aval_a_novo.get("km_referencia"), 0)
                        - _num_safe(aval_b_novo.get("km_referencia"), 0)
                    )
                    if score_novo > score_original:
                        novo_a["pre_manifesto_id"] = pm_a
                        novo_b["pre_manifesto_id"] = pm_b

                        df_itens_premanifestos_otimizados_bloco_5 = df_itens_premanifestos_otimizados_bloco_5.loc[
                            ~df_itens_premanifestos_otimizados_bloco_5["pre_manifesto_id"].astype(str).isin([pm_a, pm_b])
                        ].copy()
                        df_itens_premanifestos_otimizados_bloco_5 = pd.concat(
                            [df_itens_premanifestos_otimizados_bloco_5, novo_a, novo_b],
                            ignore_index=True,
                        )
                        contadores_m5["qtd_trocas_aceitas_5D"] += 1

    tempos_m5["5D_reotimizacao_premanifestos_ms"] = _duracao_ms(t0)

    # 5E
    t0 = _agora()
    sequenciados: List[pd.DataFrame] = []
    if len(df_itens_premanifestos_otimizados_bloco_5) > 0:
        for pm_id, df_pm in df_itens_premanifestos_otimizados_bloco_5.groupby("pre_manifesto_id", dropna=False):
            df_seq = _sequenciar_manifesto(df_pm)
            df_seq["pre_manifesto_id"] = pm_id
            df_seq["origem_etapa"] = "5E_sequenciamento"
            sequenciados.append(df_seq)

    df_sequenciamento_manifestos_m5 = (
        pd.concat(sequenciados, ignore_index=True)
        if len(sequenciados) > 0
        else pd.DataFrame()
    )
    tempos_m5["5E_sequenciamento_ms"] = _duracao_ms(t0)

    # 5F
    t0 = _agora()
    manifestos_compostos: List[Dict[str, Any]] = []
    itens_manifestos_compostos: List[pd.DataFrame] = []
    contador_manifesto_final = 1

    if len(df_sequenciamento_manifestos_m5) > 0:
        for _, df_pm in df_sequenciamento_manifestos_m5.groupby("pre_manifesto_id", dropna=False):
            veiculo_tipo = df_pm["veiculo_tipo"].iloc[0]
            cat = catalogo_veiculos.loc[catalogo_veiculos["tipo"].astype(str) == str(veiculo_tipo)].head(1)
            if len(cat) == 0:
                continue
            aval_final = _avaliar_combo_no_veiculo(df_pm, cat.iloc[0])
            if not aval_final["aceito"]:
                continue

            manifesto_id = f"MC5_{contador_manifesto_final:04d}"
            contador_manifesto_final += 1

            resumo = _gerar_resumo_manifesto_m5(
                df_combo=df_pm,
                avaliacao=aval_final,
                manifesto_id=manifesto_id,
                origem_etapa="5F_consolidacao_final",
            )
            resumo["tipo_manifesto"] = "composto_bloco_5"
            manifestos_compostos.append(resumo)

            itens = df_pm.copy().reset_index(drop=True)
            itens["manifesto_id"] = manifesto_id
            itens["tipo_manifesto"] = "composto_bloco_5"
            itens["origem_modulo"] = 5
            itens["origem_etapa"] = "5F_consolidacao_final"
            itens_manifestos_compostos.append(itens)

    df_manifestos_compostos_bloco_5 = pd.DataFrame(manifestos_compostos)
    df_itens_manifestos_compostos_bloco_5 = (
        pd.concat(itens_manifestos_compostos, ignore_index=True)
        if len(itens_manifestos_compostos) > 0
        else pd.DataFrame()
    )

    ids_compostos = set(df_itens_manifestos_compostos_bloco_5.get("id_linha_pipeline", pd.Series(dtype=str)).astype(str).tolist())
    df_remanescente_roteirizavel_bloco_5 = fila_operacional.loc[
        ~fila_operacional["id_linha_pipeline"].astype(str).isin(ids_compostos)
    ].copy().reset_index(drop=True)

    df_tentativas_bloco_5 = pd.DataFrame(tentativas_bloco_5)
    if len(df_remanescente_roteirizavel_bloco_5) > 0:
        df_remanescente_roteirizavel_bloco_5["motivo_final_remanescente_m5"] = df_remanescente_roteirizavel_bloco_5.apply(
            lambda row: _motivo_final_remanescente_m5(
                id_linha=str(row["id_linha_pipeline"]),
                df_tentativas=df_tentativas_bloco_5,
            ),
            axis=1,
        )

    mapa_resumo = {
        "documento": "nro_documento",
        "cliente": "destinatario",
        "cidade": "cidade",
        "exclusividade": "veiculo_exclusivo_flag",
        "data_agenda": "data_agenda",
        "data_leadtime": "data_leadtime",
        "peso_calculado": "peso_calculado",
        "motivo_final_remanescente_m5": "motivo_final_remanescente_m5",
    }
    df_remanescente_roteirizavel_resumido_bloco_5 = pd.DataFrame()
    if len(df_remanescente_roteirizavel_bloco_5) > 0:
        for destino, origem in mapa_resumo.items():
            if origem in df_remanescente_roteirizavel_bloco_5.columns:
                df_remanescente_roteirizavel_resumido_bloco_5[destino] = df_remanescente_roteirizavel_bloco_5[origem]
            else:
                df_remanescente_roteirizavel_resumido_bloco_5[destino] = np.nan

    uso_frota = catalogo_veiculos[["tipo", "limite_manifestos", "manifestos_utilizados"]].copy()
    uso_frota["saldo_manifestos"] = uso_frota.apply(
        lambda row: (
            np.nan
            if pd.isna(row["limite_manifestos"])
            else int(row["limite_manifestos"]) - int(_int_safe(row["manifestos_utilizados"], default=0))
        ),
        axis=1,
    )

    if len(df_itens_manifestos_compostos_bloco_5) > 0 and df_itens_manifestos_compostos_bloco_5["id_linha_pipeline"].astype(str).duplicated().any():
        raise Exception("Validação pós-M5 falhou: id_linha_pipeline repetido em mais de um item composto.")

    contadores_m5["qtd_manifestos_compostos"] = int(len(df_manifestos_compostos_bloco_5))
    contadores_m5["qtd_itens_compostos"] = int(len(df_itens_manifestos_compostos_bloco_5))
    tempos_m5["5F_consolidacao_final_ms"] = _duracao_ms(t0)
    tempos_m5["tempo_total_m5_ms"] = _duracao_ms(inicio_total)

    resumo_m5 = {
        "modulo": "M5",
        "data_base_roteirizacao": pd.Timestamp(data_base_roteirizacao).isoformat(),
        "tipo_roteirizacao": tipo_roteirizacao,
        "remanescente_entrada_m5": int(len(fila_operacional)),
        "anchors_m5": int(len(df_ancoras_m5)),
        "premanifestos_gerados_m5": int(len(df_premanifestos_bloco_5)),
        "manifestos_compostos_gerados_m5": int(len(df_manifestos_compostos_bloco_5)),
        "itens_manifestados_m5": int(len(df_itens_manifestos_compostos_bloco_5)),
        "remanescente_roteirizavel_m5": int(len(df_remanescente_roteirizavel_bloco_5)),
        "ocupacao_minima_dominante_perc": round(OCUPACAO_MINIMA_DOMINANTE * 100, 2),
        "ocupacao_minima_secundaria_perc": round(OCUPACAO_MINIMA_SECUNDARIA * 100, 2),
    }

    auditoria_m5 = {
        "motivos_remanescente_m5": (
            df_remanescente_roteirizavel_bloco_5["motivo_final_remanescente_m5"].value_counts(dropna=False).to_dict()
            if "motivo_final_remanescente_m5" in df_remanescente_roteirizavel_bloco_5.columns
            else {}
        )
    }

    outputs = {
        "df_fila_operacional_m5": fila_operacional,
        "df_ancoras_m5": df_ancoras_m5,
        "df_premanifestos_bloco_5": df_premanifestos_bloco_5,
        "df_itens_premanifestos_bloco_5": df_itens_premanifestos_bloco_5,
        "df_premanifestos_otimizados_bloco_5": df_premanifestos_otimizados_bloco_5,
        "df_itens_premanifestos_otimizados_bloco_5": df_itens_premanifestos_otimizados_bloco_5,
        "df_sequenciamento_manifestos_m5": df_sequenciamento_manifestos_m5,
        "df_manifestos_compostos_bloco_5": df_manifestos_compostos_bloco_5,
        "df_itens_manifestos_compostos_bloco_5": df_itens_manifestos_compostos_bloco_5,
        "df_tentativas_bloco_5": df_tentativas_bloco_5,
        "df_nao_premanifestadas_m5": df_nao_premanifestadas_m5,
        "df_remanescente_roteirizavel_bloco_5": df_remanescente_roteirizavel_bloco_5,
        "df_remanescente_roteirizavel_resumido_bloco_5": df_remanescente_roteirizavel_resumido_bloco_5,
        "df_uso_frota_m5": uso_frota,
    }

    meta = {
        "resumo_m5": resumo_m5,
        "auditoria_m5": auditoria_m5,
        "metricas_m5": {
            **tempos_m5,
            **contadores_m5,
        },
        "metadata_modulo_5": {
            "tipo_roteirizacao": tipo_roteirizacao,
            "catalogo_veiculos": _to_records(catalogo_veiculos),
            "uso_frota": _to_records(uso_frota),
        },
    }

    return outputs, meta
