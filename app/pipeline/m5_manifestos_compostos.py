from __future__ import annotations

import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# MÓDULO 5.1 - PRÉ-MANIFESTOS COMPOSTOS (RODADA SIMPLES)
#
# OBJETIVO:
# - receber somente o remanescente do M4
# - tentar fechamento de grupos simples por regionalidade
# - testar veículos do maior para o menor
# - gerar pré-manifestos válidos
# - devolver saldo remanescente para a próxima rodada
#
# REGRAS DESTA ETAPA:
# 1) entrada = somente remanescente roteirizável do M4
# 2) agrupamento em camadas:
#    - mesmo cliente
#    - mesma cidade
#    - mesma subregião
#    - mesma mesorregião
# 3) tentativa de veículo:
#    - do maior para o menor
#    - se falhar em um perfil, tenta o próximo
#    - não descarta o grupo só porque um perfil falhou
# 4) travas obrigatórias:
#    - ocupação mínima e máxima por peso_calculado
#    - máximo de entregas
#    - km máximo do perfil
#    - capacidade de peso
# 5) esta versão NÃO faz:
#    - âncora
#    - confronto entre pré-manifestos
#    - recombinação global
#    - busca combinatória de subconjunto
# 6) saída:
#    - pré-manifestos M5.1
#    - itens dos pré-manifestos M5.1
#    - tentativas auditáveis
#    - remanescente M5.1
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
        try:
            if pd.isna(x):
                return False
        except Exception:
            pass
        return bool(int(x))

    txt = str(x).strip().lower()
    return txt in {"true", "1", "sim", "s", "yes", "y"}


def _normalizar_texto(valor: Any) -> str:
    if valor is None:
        return ""
    txt = str(valor).strip()
    if txt == "" or txt.lower() == "nan":
        return ""
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    return txt.strip().lower()


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


def _resolver_coluna_tipo_veiculo(df_veiculos: pd.DataFrame) -> str:
    if "tipo" in df_veiculos.columns:
        return "tipo"
    if "perfil" in df_veiculos.columns:
        return "perfil"
    raise Exception("Faltam colunas mínimas na base de veículos: tipo ou perfil.")


def _normalizar_configuracao_frota(configuracao_frota: Any) -> pd.DataFrame:
    if configuracao_frota is None:
        return pd.DataFrame(columns=["perfil", "quantidade"])

    if isinstance(configuracao_frota, pd.DataFrame):
        cfg = configuracao_frota.copy()
    else:
        rows: List[Dict[str, Any]] = []
        for item in configuracao_frota:
            if hasattr(item, "model_dump"):
                row = item.model_dump()
            elif isinstance(item, dict):
                row = item
            else:
                row = vars(item)
            rows.append(row)
        cfg = pd.DataFrame(rows)

    if cfg.empty:
        return pd.DataFrame(columns=["perfil", "quantidade"])

    cfg = _deduplicar_colunas(cfg)
    cfg = _garantir_coluna_por_alias(cfg, "perfil", ["tipo", "veiculo_tipo"], default=None)
    cfg = _garantir_coluna_por_alias(cfg, "quantidade", ["qtd", "limite_manifestos"], default=np.nan)
    cfg["perfil"] = cfg["perfil"].astype(str).str.strip()
    cfg["quantidade"] = pd.to_numeric(cfg["quantidade"], errors="coerce")
    cfg = cfg.loc[cfg["perfil"].astype(str).str.strip() != ""].copy()
    return cfg[["perfil", "quantidade"]].reset_index(drop=True)


def _normalizar_tipo_roteirizacao(valor: Any) -> str:
    txt = str(valor).strip().lower() if valor is not None else "carteira"
    if txt not in {"carteira", "frota"}:
        return "carteira"
    return txt


def _preparar_df_entrada(df_remanescente_m4: pd.DataFrame) -> pd.DataFrame:
    if df_remanescente_m4 is None or len(df_remanescente_m4) == 0:
        return pd.DataFrame()

    df = _deduplicar_colunas(df_remanescente_m4).copy()

    colunas_alias = [
        ("id_linha_pipeline", ["id_linha_pipeline"]),
        ("destinatario", ["destinatario"]),
        ("cidade", ["cidade", "cidade_dest"]),
        ("uf", ["uf", "uf_chave"]),
        ("subregiao", ["subregiao", "sub_regiao"]),
        ("mesorregiao", ["mesorregiao"]),
        ("peso_calculado", ["peso_calculado", "Peso Calculo", "peso_c"]),
        ("distancia_rodoviaria_est_km", ["distancia_rodoviaria_est_km", "distancia_km"]),
        ("cte", ["cte", "nro_documento"]),
        ("nro_documento", ["nro_documento", "cte"]),
        ("restricao_veiculo", ["restricao_veiculo", "Restrição Veículo"]),
        ("veiculo_exclusivo", ["veiculo_exclusivo", "veiculo_exclusivo_flag", "Carro Dedicado"]),
        ("ranking_prioridade_operacional", ["ranking_prioridade_operacional", "ranking_prioridade"]),
        ("prioridade_embarque_num", ["prioridade_embarque_num", "prioridade_embarque"]),
        ("score_prioridade_preliminar", ["score_prioridade_preliminar"]),
        ("ranking_preliminar", ["ranking_preliminar"]),
    ]

    for destino, aliases in colunas_alias:
        df = _garantir_coluna_por_alias(df, destino, aliases, default=None)

    if "id_linha_pipeline" not in df.columns:
        raise Exception("M5.1 exige a coluna id_linha_pipeline no remanescente recebido do M4.")

    df["id_linha_pipeline"] = df["id_linha_pipeline"].astype(str).str.strip()
    if df["id_linha_pipeline"].eq("").any():
        raise Exception("M5.1 encontrou id_linha_pipeline vazio.")

    if df["id_linha_pipeline"].duplicated().any():
        raise Exception("M5.1 recebeu remanescente com id_linha_pipeline duplicado.")

    df["destinatario"] = df["destinatario"].astype(str).fillna("").str.strip()
    df["cidade"] = df["cidade"].astype(str).fillna("").str.strip()
    df["uf"] = df["uf"].astype(str).fillna("").str.strip()
    df["subregiao"] = df["subregiao"].astype(str).fillna("").str.strip()
    df["mesorregiao"] = df["mesorregiao"].astype(str).fillna("").str.strip()

    df["peso_calculado"] = pd.to_numeric(df["peso_calculado"], errors="coerce").fillna(0.0)
    df["distancia_rodoviaria_est_km"] = pd.to_numeric(
        df["distancia_rodoviaria_est_km"], errors="coerce"
    ).fillna(0.0)

    df["veiculo_exclusivo"] = df["veiculo_exclusivo"].apply(_bool_safe)

    df["ranking_prioridade_operacional"] = pd.to_numeric(
        df["ranking_prioridade_operacional"], errors="coerce"
    ).fillna(9999)

    df["prioridade_embarque_num"] = pd.to_numeric(
        df["prioridade_embarque_num"], errors="coerce"
    ).fillna(9999)

    df["score_prioridade_preliminar"] = pd.to_numeric(
        df["score_prioridade_preliminar"], errors="coerce"
    ).fillna(0)

    df["ranking_preliminar"] = pd.to_numeric(
        df["ranking_preliminar"], errors="coerce"
    ).fillna(999999)

    df["cliente_chave_m5"] = df["destinatario"].apply(_normalizar_texto)
    df["cidade_chave_m5"] = (df["cidade"].astype(str) + "|" + df["uf"].astype(str)).apply(_normalizar_texto)
    df["subregiao_chave_m5"] = df["subregiao"].apply(_normalizar_texto)
    df["mesorregiao_chave_m5"] = df["mesorregiao"].apply(_normalizar_texto)

    return df.reset_index(drop=True)


def _preparar_catalogo_veiculos(
    df_veiculos_tratados: pd.DataFrame,
    tipo_roteirizacao: str,
    configuracao_frota: Any = None,
    df_uso_frota_m4: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if df_veiculos_tratados is None or len(df_veiculos_tratados) == 0:
        raise Exception("M5.1 exige df_veiculos_tratados preenchido.")

    df = _deduplicar_colunas(df_veiculos_tratados).copy()
    coluna_tipo = _resolver_coluna_tipo_veiculo(df)

    df = _garantir_coluna_por_alias(df, "tipo", [coluna_tipo], default=None)
    df = _garantir_coluna_por_alias(df, "capacidade_peso_kg", ["capacidade_peso_kg"], default=np.nan)
    df = _garantir_coluna_por_alias(df, "capacidade_vol_m3", ["capacidade_vol_m3"], default=np.nan)
    df = _garantir_coluna_por_alias(df, "max_entregas", ["max_entregas"], default=np.nan)
    df = _garantir_coluna_por_alias(df, "max_km_distancia", ["max_km_distancia"], default=np.nan)
    df = _garantir_coluna_por_alias(
        df,
        "ocupacao_minima_perc",
        ["ocupacao_minima_perc", "ocupacao_minima", "ocup_min"],
        default=np.nan,
    )

    df["tipo"] = df["tipo"].astype(str).str.strip()
    df["capacidade_peso_kg"] = pd.to_numeric(df["capacidade_peso_kg"], errors="coerce")
    df["capacidade_vol_m3"] = pd.to_numeric(df["capacidade_vol_m3"], errors="coerce")
    df["max_entregas"] = pd.to_numeric(df["max_entregas"], errors="coerce")
    df["max_km_distancia"] = pd.to_numeric(df["max_km_distancia"], errors="coerce")
    df["ocupacao_minima_perc"] = pd.to_numeric(df["ocupacao_minima_perc"], errors="coerce")

    df = df.loc[df["tipo"].astype(str).str.strip() != ""].copy()
    df = df.loc[df["capacidade_peso_kg"].fillna(0) > 0].copy()
    df = df.drop_duplicates(subset=["tipo"]).reset_index(drop=True)

    cfg = _normalizar_configuracao_frota(configuracao_frota)
    uso_m4 = pd.DataFrame()
    if df_uso_frota_m4 is not None and len(df_uso_frota_m4) > 0:
        uso_m4 = _deduplicar_colunas(df_uso_frota_m4).copy()
        uso_m4 = _garantir_coluna_por_alias(uso_m4, "tipo", ["tipo", "perfil"], default=None)
        uso_m4 = _garantir_coluna_por_alias(
            uso_m4, "manifestos_utilizados", ["manifestos_utilizados"], default=0
        )
        uso_m4["tipo"] = uso_m4["tipo"].astype(str).str.strip()
        uso_m4["manifestos_utilizados"] = pd.to_numeric(
            uso_m4["manifestos_utilizados"], errors="coerce"
        ).fillna(0)

    if tipo_roteirizacao == "frota":
        if cfg.empty:
            raise Exception(
                "M5.1 recebeu tipo_roteirizacao='frota', mas configuracao_frota veio vazia."
            )

        cfg["perfil"] = cfg["perfil"].astype(str).str.strip()
        cfg["quantidade"] = pd.to_numeric(cfg["quantidade"], errors="coerce")

        catalogo = df.merge(cfg, left_on="tipo", right_on="perfil", how="inner").copy()
        catalogo["limite_manifestos"] = catalogo["quantidade"]
    else:
        catalogo = df.copy()
        catalogo["limite_manifestos"] = np.nan

    if len(uso_m4) > 0:
        catalogo = catalogo.merge(
            uso_m4[["tipo", "manifestos_utilizados"]].drop_duplicates(subset=["tipo"]),
            on="tipo",
            how="left",
        )
    else:
        catalogo["manifestos_utilizados"] = 0

    catalogo["manifestos_utilizados"] = pd.to_numeric(
        catalogo["manifestos_utilizados"], errors="coerce"
    ).fillna(0)

    catalogo["manifestos_gerados_m5_1"] = 0

    def _saldo(row: pd.Series) -> float:
        if pd.isna(row["limite_manifestos"]):
            return np.nan
        return float(row["limite_manifestos"]) - float(row["manifestos_utilizados"]) - float(
            row["manifestos_gerados_m5_1"]
        )

    catalogo["saldo_manifestos"] = catalogo.apply(_saldo, axis=1)

    catalogo["ocupacao_minima_decimal"] = catalogo["ocupacao_minima_perc"].apply(
        lambda x: (float(x) / 100.0)
        if pd.notna(x)
        else OCUPACAO_MINIMA_PADRAO
    )
    catalogo["ocupacao_maxima_decimal"] = OCUPACAO_MAXIMA_PADRAO

    catalogo["tipo_norm"] = catalogo["tipo"].apply(_normalizar_texto)

    catalogo = catalogo.sort_values(
        by=["capacidade_peso_kg", "max_entregas", "max_km_distancia"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return catalogo


def _ordenar_fila(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df.copy()

    return df.sort_values(
        by=[
            "ranking_prioridade_operacional",
            "prioridade_embarque_num",
            "ranking_preliminar",
            "score_prioridade_preliminar",
            "destinatario",
            "cidade",
            "id_linha_pipeline",
        ],
        ascending=[True, True, True, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _montar_lista_grupos(
    df_saldo: pd.DataFrame,
    coluna_chave: str,
) -> List[Tuple[str, List[str]]]:
    if df_saldo is None or len(df_saldo) == 0:
        return []

    base = df_saldo.copy()
    base = base.loc[base[coluna_chave].astype(str).str.strip() != ""].copy()
    if len(base) == 0:
        return []

    agrupado = (
        base.groupby(coluna_chave, dropna=False)
        .agg(
            qtd_itens=("id_linha_pipeline", "count"),
            peso_total=("peso_calculado", "sum"),
            prioridade_min=("ranking_prioridade_operacional", "min"),
        )
        .reset_index()
    )

    agrupado = agrupado.sort_values(
        by=["qtd_itens", "peso_total", "prioridade_min", coluna_chave],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    grupos: List[Tuple[str, List[str]]] = []
    for _, row in agrupado.iterrows():
        chave = row[coluna_chave]
        ids = base.loc[base[coluna_chave] == chave, "id_linha_pipeline"].astype(str).tolist()
        if len(ids) > 0:
            grupos.append((str(chave), ids))

    return grupos


def _qtd_paradas(df_grupo: pd.DataFrame) -> int:
    if "destinatario" in df_grupo.columns:
        qtd = df_grupo["destinatario"].astype(str).str.strip().nunique()
        return int(qtd)
    return int(len(df_grupo))


def _km_grupo(df_grupo: pd.DataFrame) -> float:
    if "distancia_rodoviaria_est_km" not in df_grupo.columns or len(df_grupo) == 0:
        return 0.0
    val = pd.to_numeric(df_grupo["distancia_rodoviaria_est_km"], errors="coerce").fillna(0.0).max()
    return float(val)


def _ocupacao_peso(peso_total: float, capacidade_peso_kg: float) -> float:
    if capacidade_peso_kg is None or pd.isna(capacidade_peso_kg) or capacidade_peso_kg <= 0:
        return 0.0
    return float(peso_total) / float(capacidade_peso_kg)


def _filtrar_catalogo_por_restricao(
    df_grupo: pd.DataFrame,
    catalogo_veiculos: pd.DataFrame,
) -> pd.DataFrame:
    if "restricao_veiculo" not in df_grupo.columns:
        return catalogo_veiculos.copy()

    restricoes = (
        df_grupo["restricao_veiculo"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    if len(restricoes) == 0:
        return catalogo_veiculos.copy()

    restricoes_norm = {_normalizar_texto(x) for x in restricoes if str(x).strip() != ""}
    if len(restricoes_norm) == 0:
        return catalogo_veiculos.copy()

    return catalogo_veiculos.loc[catalogo_veiculos["tipo_norm"].isin(restricoes_norm)].copy()


def _validar_grupo_contra_veiculo(
    df_grupo: pd.DataFrame,
    veiculo: pd.Series,
) -> Tuple[bool, Dict[str, Any]]:
    if len(df_grupo) == 0:
        return False, {"motivo": "grupo_vazio"}

    if df_grupo["veiculo_exclusivo"].fillna(False).any():
        return False, {"motivo": "grupo_contem_exclusivo"}

    peso_total = float(pd.to_numeric(df_grupo["peso_calculado"], errors="coerce").fillna(0.0).sum())
    qtd_itens = int(len(df_grupo))
    qtd_ctes = int(df_grupo["cte"].astype(str).nunique()) if "cte" in df_grupo.columns else qtd_itens
    qtd_paradas = _qtd_paradas(df_grupo)
    km_ref = _km_grupo(df_grupo)

    capacidade_peso = _num_safe(veiculo.get("capacidade_peso_kg"), default=np.nan)
    max_entregas = _num_safe(veiculo.get("max_entregas"), default=np.nan)
    max_km = _num_safe(veiculo.get("max_km_distancia"), default=np.nan)
    ocup_min = _num_safe(
        veiculo.get("ocupacao_minima_decimal"),
        default=OCUPACAO_MINIMA_PADRAO,
    )
    ocup_max = _num_safe(
        veiculo.get("ocupacao_maxima_decimal"),
        default=OCUPACAO_MAXIMA_PADRAO,
    )

    ocupacao = _ocupacao_peso(peso_total, capacidade_peso)

    if pd.isna(capacidade_peso) or capacidade_peso <= 0:
        return False, {
            "motivo": "veiculo_sem_capacidade_peso",
            "peso_total": peso_total,
            "ocupacao": round(ocupacao * 100, 2),
            "qtd_itens": qtd_itens,
            "qtd_ctes": qtd_ctes,
            "qtd_paradas": qtd_paradas,
            "km_referencia": round(km_ref, 2),
        }

    if peso_total > capacidade_peso:
        return False, {
            "motivo": "excede_capacidade_peso",
            "peso_total": peso_total,
            "capacidade_peso_kg": capacidade_peso,
            "ocupacao": round(ocupacao * 100, 2),
            "qtd_itens": qtd_itens,
            "qtd_ctes": qtd_ctes,
            "qtd_paradas": qtd_paradas,
            "km_referencia": round(km_ref, 2),
        }

    if pd.notna(max_entregas) and qtd_paradas > max_entregas:
        return False, {
            "motivo": "excede_max_entregas",
            "peso_total": peso_total,
            "capacidade_peso_kg": capacidade_peso,
            "ocupacao": round(ocupacao * 100, 2),
            "qtd_itens": qtd_itens,
            "qtd_ctes": qtd_ctes,
            "qtd_paradas": qtd_paradas,
            "max_entregas_veiculo": int(max_entregas),
            "km_referencia": round(km_ref, 2),
        }

    if pd.notna(max_km) and km_ref > max_km:
        return False, {
            "motivo": "excede_max_km",
            "peso_total": peso_total,
            "capacidade_peso_kg": capacidade_peso,
            "ocupacao": round(ocupacao * 100, 2),
            "qtd_itens": qtd_itens,
            "qtd_ctes": qtd_ctes,
            "qtd_paradas": qtd_paradas,
            "km_referencia": round(km_ref, 2),
            "max_km_distancia_veiculo": float(max_km),
        }

    if ocupacao < ocup_min:
        return False, {
            "motivo": "abaixo_ocupacao_minima",
            "peso_total": peso_total,
            "capacidade_peso_kg": capacidade_peso,
            "ocupacao": round(ocupacao * 100, 2),
            "ocupacao_minima_perc_veiculo": round(float(ocup_min) * 100, 2),
            "qtd_itens": qtd_itens,
            "qtd_ctes": qtd_ctes,
            "qtd_paradas": qtd_paradas,
            "km_referencia": round(km_ref, 2),
        }

    if ocupacao > ocup_max:
        return False, {
            "motivo": "acima_ocupacao_maxima",
            "peso_total": peso_total,
            "capacidade_peso_kg": capacidade_peso,
            "ocupacao": round(ocupacao * 100, 2),
            "ocupacao_maxima_perc_veiculo": round(float(ocup_max) * 100, 2),
            "qtd_itens": qtd_itens,
            "qtd_ctes": qtd_ctes,
            "qtd_paradas": qtd_paradas,
            "km_referencia": round(km_ref, 2),
        }

    return True, {
        "motivo": "grupo_fechou",
        "peso_total": peso_total,
        "capacidade_peso_kg": capacidade_peso,
        "ocupacao": round(ocupacao * 100, 2),
        "qtd_itens": qtd_itens,
        "qtd_ctes": qtd_ctes,
        "qtd_paradas": qtd_paradas,
        "km_referencia": round(km_ref, 2),
    }


def _gerar_manifesto_id(indice: int) -> str:
    return f"PM51_{int(indice):04d}"


def _montar_manifesto_resumo(
    df_grupo: pd.DataFrame,
    veiculo: pd.Series,
    manifesto_id: str,
    criterio_agrupamento: str,
    chave_agrupamento: str,
) -> Dict[str, Any]:
    peso_total = float(pd.to_numeric(df_grupo["peso_calculado"], errors="coerce").fillna(0.0).sum())
    km_ref = _km_grupo(df_grupo)
    qtd_paradas = _qtd_paradas(df_grupo)
    qtd_itens = int(len(df_grupo))
    qtd_ctes = int(df_grupo["cte"].astype(str).nunique()) if "cte" in df_grupo.columns else qtd_itens
    capacidade_peso = _num_safe(veiculo.get("capacidade_peso_kg"), default=np.nan)
    ocupacao = _ocupacao_peso(peso_total, capacidade_peso)

    cidade = None
    uf = None
    subregiao = None
    mesorregiao = None
    destinatario = None

    if len(df_grupo) > 0:
        cidade = _scalar_safe(df_grupo["cidade"].iloc[0]) if "cidade" in df_grupo.columns else None
        uf = _scalar_safe(df_grupo["uf"].iloc[0]) if "uf" in df_grupo.columns else None
        subregiao = _scalar_safe(df_grupo["subregiao"].iloc[0]) if "subregiao" in df_grupo.columns else None
        mesorregiao = _scalar_safe(df_grupo["mesorregiao"].iloc[0]) if "mesorregiao" in df_grupo.columns else None
        if criterio_agrupamento == "mesmo_cliente" and "destinatario" in df_grupo.columns:
            destinatario = _scalar_safe(df_grupo["destinatario"].iloc[0])

    return {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": "pre_manifesto_bloco_5_1",
        "veiculo_tipo": veiculo["tipo"],
        "qtd_itens": qtd_itens,
        "qtd_ctes": qtd_ctes,
        "qtd_paradas": qtd_paradas,
        "base_carga_oficial": round(peso_total, 3),
        "peso_total_kg": round(peso_total, 3),
        "vol_total_m3": 0.0,
        "km_referencia": round(km_ref, 2),
        "ocupacao_oficial_perc": round(float(ocupacao) * 100, 2),
        "capacidade_peso_kg_veiculo": _num_safe(veiculo.get("capacidade_peso_kg"), default=np.nan),
        "capacidade_vol_m3_veiculo": _num_safe(veiculo.get("capacidade_vol_m3"), default=np.nan),
        "max_entregas_veiculo": _num_safe(veiculo.get("max_entregas"), default=np.nan),
        "max_km_distancia_veiculo": _num_safe(veiculo.get("max_km_distancia"), default=np.nan),
        "ignorar_ocupacao_minima": False,
        "origem_modulo": 5,
        "origem_etapa": "5.1_trava_dura_regionalidade",
        "criterio_agrupamento": criterio_agrupamento,
        "chave_agrupamento": chave_agrupamento,
        "destinatario": destinatario,
        "cidade": cidade,
        "uf": uf,
        "mesorregiao": mesorregiao,
        "subregiao": subregiao,
    }


def _montar_itens_manifesto(
    df_grupo: pd.DataFrame,
    manifesto_resumo: Dict[str, Any],
) -> pd.DataFrame:
    itens = df_grupo.copy()

    itens["manifesto_id"] = manifesto_resumo["manifesto_id"]
    itens["tipo_manifesto"] = manifesto_resumo["tipo_manifesto"]
    itens["veiculo_tipo"] = manifesto_resumo["veiculo_tipo"]
    itens["capacidade_peso_kg_veiculo"] = manifesto_resumo["capacidade_peso_kg_veiculo"]
    itens["capacidade_vol_m3_veiculo"] = manifesto_resumo["capacidade_vol_m3_veiculo"]
    itens["max_entregas_veiculo"] = manifesto_resumo["max_entregas_veiculo"]
    itens["max_km_distancia_veiculo"] = manifesto_resumo["max_km_distancia_veiculo"]
    itens["base_carga_oficial_manifesto"] = manifesto_resumo["base_carga_oficial"]
    itens["ocupacao_oficial_perc_manifesto"] = manifesto_resumo["ocupacao_oficial_perc"]
    itens["ignorar_ocupacao_minima_manifesto"] = manifesto_resumo["ignorar_ocupacao_minima"]
    itens["origem_modulo"] = manifesto_resumo["origem_modulo"]
    itens["origem_etapa"] = manifesto_resumo["origem_etapa"]
    itens["criterio_agrupamento_m5_1"] = manifesto_resumo["criterio_agrupamento"]
    itens["chave_agrupamento_m5_1"] = manifesto_resumo["chave_agrupamento"]
    itens["status_roteirizacao"] = "pre_manifestado_m5_1"
    itens["origem_bloco"] = "M5.1"
    itens["segue_para_proximo_bloco"] = False
    itens["motivo_nao_roteirizado"] = None

    return itens.reset_index(drop=True)


def _montar_df_nao_roteirizados_bloco_5_1(df_remanescente: pd.DataFrame) -> pd.DataFrame:
    if df_remanescente is None or len(df_remanescente) == 0:
        return pd.DataFrame()

    df = df_remanescente.copy()
    if "status_roteirizacao" not in df.columns:
        df["status_roteirizacao"] = "remanescente_bloco_5_1"
    if "origem_bloco" not in df.columns:
        df["origem_bloco"] = "M5.1"
    if "segue_para_proximo_bloco" not in df.columns:
        df["segue_para_proximo_bloco"] = True
    if "motivo_nao_roteirizado" not in df.columns:
        df["motivo_nao_roteirizado"] = "remanescente_nao_fechado_no_m5_1"

    return df.reset_index(drop=True)


def executar_m5_manifestos_compostos(
    df_remanescente_roteirizavel_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
    rodada_id: Optional[str] = None,
    data_base_roteirizacao: Any = None,
    tipo_roteirizacao: str = "carteira",
    configuracao_frota: Any = None,
    df_uso_frota_m4: Optional[pd.DataFrame] = None,
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    tempos_m5_1: Dict[str, float] = {}

    t0_total = _agora()

    # ============================================================
    # PREPARAÇÃO
    # ============================================================
    t0 = _agora()
    tipo_roteirizacao = _normalizar_tipo_roteirizacao(tipo_roteirizacao)

    fila = _preparar_df_entrada(df_remanescente_roteirizavel_bloco_4)
    fila = _ordenar_fila(fila)

    catalogo_veiculos = _preparar_catalogo_veiculos(
        df_veiculos_tratados=df_veiculos_tratados,
        tipo_roteirizacao=tipo_roteirizacao,
        configuracao_frota=configuracao_frota,
        df_uso_frota_m4=df_uso_frota_m4,
    )

    tempos_m5_1["preparacao_ms"] = _duracao_ms(t0)

    if len(fila) == 0:
        outputs_vazios = {
            "df_premanifestos_m5_1": pd.DataFrame(),
            "df_itens_premanifestos_m5_1": pd.DataFrame(),
            "df_tentativas_m5_1": pd.DataFrame(),
            "df_remanescente_m5_1": pd.DataFrame(),
            "df_nao_roteirizados_bloco_5_1": pd.DataFrame(),
            "df_uso_frota_m5_1": catalogo_veiculos.copy(),
        }
        meta_vazia = {
            "resumo_m5_1": {
                "modulo": "M5.1",
                "data_base_roteirizacao": (
                    data_base_roteirizacao.isoformat()
                    if hasattr(data_base_roteirizacao, "isoformat")
                    else str(data_base_roteirizacao)
                ),
                "tipo_roteirizacao": tipo_roteirizacao,
                "remanescente_entrada_m5_1": 0,
                "pre_manifestos_gerados_m5_1": 0,
                "itens_pre_manifestados_m5_1": 0,
                "remanescente_saida_m5_1": 0,
                "coluna_tipo_veiculo_utilizada": "tipo",
                "criterios_agrupamento_m5_1": [
                    "mesmo_cliente",
                    "mesma_cidade",
                    "mesma_subregiao",
                    "mesma_mesorregiao",
                ],
                "persistiu_artefatos": False,
                "caminhos_pipeline": caminhos_pipeline or {},
            },
            "auditoria_m5_1": {
                "total_tentativas": 0,
                "total_grupos_fechados": 0,
                "total_remanescentes": 0,
            },
            "metricas_m5_1": tempos_m5_1,
        }
        return outputs_vazios, meta_vazia

    # ============================================================
    # EXECUÇÃO EM CAMADAS
    # ============================================================
    t0 = _agora()

    tentativas: List[Dict[str, Any]] = []
    premanifestos: List[Dict[str, Any]] = []
    itens_premanifestos: List[pd.DataFrame] = []

    ids_alocados: set[str] = set()
    contador_manifestos = 1

    criterios = [
        ("mesmo_cliente", "cliente_chave_m5"),
        ("mesma_cidade", "cidade_chave_m5"),
        ("mesma_subregiao", "subregiao_chave_m5"),
        ("mesma_mesorregiao", "mesorregiao_chave_m5"),
    ]

    for criterio_nome, coluna_chave in criterios:
        saldo_atual = fila.loc[~fila["id_linha_pipeline"].isin(ids_alocados)].copy()
        grupos = _montar_lista_grupos(saldo_atual, coluna_chave)

        for chave_agrupamento, ids_grupo in grupos:
            ids_grupo_set = set(ids_grupo)

            if len(ids_grupo_set) == 0:
                continue

            # garante que o grupo ainda está inteiro disponível
            if len(ids_grupo_set.intersection(ids_alocados)) > 0:
                continue

            df_grupo = saldo_atual.loc[saldo_atual["id_linha_pipeline"].isin(ids_grupo_set)].copy()
            if len(df_grupo) == 0:
                continue

            # se grupo virar unitário, não fecha nesta etapa de composição
            if len(df_grupo) <= 1:
                tentativas.append(
                    {
                        "criterio_agrupamento": criterio_nome,
                        "chave_agrupamento": chave_agrupamento,
                        "veiculo_tipo_tentado": None,
                        "status_tentativa": "falha",
                        "motivo": "grupo_unitario_nao_composto_no_m5_1",
                        "qtd_itens_grupo": int(len(df_grupo)),
                        "qtd_ctes_grupo": int(df_grupo["cte"].astype(str).nunique()) if "cte" in df_grupo.columns else int(len(df_grupo)),
                        "qtd_paradas_grupo": _qtd_paradas(df_grupo),
                        "peso_total_grupo": round(float(df_grupo["peso_calculado"].sum()), 3),
                        "km_referencia_grupo": round(_km_grupo(df_grupo), 2),
                        "manifesto_id": None,
                    }
                )
                continue

            catalogo_candidato = _filtrar_catalogo_por_restricao(df_grupo, catalogo_veiculos)

            if len(catalogo_candidato) == 0:
                tentativas.append(
                    {
                        "criterio_agrupamento": criterio_nome,
                        "chave_agrupamento": chave_agrupamento,
                        "veiculo_tipo_tentado": None,
                        "status_tentativa": "falha",
                        "motivo": "nenhum_veiculo_compativel_com_restricao",
                        "qtd_itens_grupo": int(len(df_grupo)),
                        "qtd_ctes_grupo": int(df_grupo["cte"].astype(str).nunique()) if "cte" in df_grupo.columns else int(len(df_grupo)),
                        "qtd_paradas_grupo": _qtd_paradas(df_grupo),
                        "peso_total_grupo": round(float(df_grupo["peso_calculado"].sum()), 3),
                        "km_referencia_grupo": round(_km_grupo(df_grupo), 2),
                        "manifesto_id": None,
                    }
                )
                continue

            fechou = False

            for idx_veic, veiculo in catalogo_candidato.iterrows():
                saldo_manifestos = veiculo.get("saldo_manifestos", np.nan)
                if pd.notna(saldo_manifestos) and float(saldo_manifestos) <= 0:
                    tentativas.append(
                        {
                            "criterio_agrupamento": criterio_nome,
                            "chave_agrupamento": chave_agrupamento,
                            "veiculo_tipo_tentado": veiculo["tipo"],
                            "status_tentativa": "falha",
                            "motivo": "saldo_frota_esgotado",
                            "qtd_itens_grupo": int(len(df_grupo)),
                            "qtd_ctes_grupo": int(df_grupo["cte"].astype(str).nunique()) if "cte" in df_grupo.columns else int(len(df_grupo)),
                            "qtd_paradas_grupo": _qtd_paradas(df_grupo),
                            "peso_total_grupo": round(float(df_grupo["peso_calculado"].sum()), 3),
                            "km_referencia_grupo": round(_km_grupo(df_grupo), 2),
                            "manifesto_id": None,
                        }
                    )
                    continue

                ok, detalhe = _validar_grupo_contra_veiculo(df_grupo, veiculo)

                tentativa = {
                    "criterio_agrupamento": criterio_nome,
                    "chave_agrupamento": chave_agrupamento,
                    "veiculo_tipo_tentado": veiculo["tipo"],
                    "status_tentativa": "sucesso" if ok else "falha",
                    "motivo": detalhe.get("motivo"),
                    "qtd_itens_grupo": detalhe.get("qtd_itens"),
                    "qtd_ctes_grupo": detalhe.get("qtd_ctes"),
                    "qtd_paradas_grupo": detalhe.get("qtd_paradas"),
                    "peso_total_grupo": round(float(detalhe.get("peso_total", 0.0)), 3),
                    "capacidade_peso_kg_veiculo": detalhe.get("capacidade_peso_kg"),
                    "ocupacao_perc_grupo": detalhe.get("ocupacao"),
                    "km_referencia_grupo": detalhe.get("km_referencia"),
                    "max_entregas_veiculo": detalhe.get("max_entregas_veiculo"),
                    "max_km_distancia_veiculo": detalhe.get("max_km_distancia_veiculo"),
                    "manifesto_id": None,
                }

                if ok:
                    manifesto_id = _gerar_manifesto_id(contador_manifestos)
                    contador_manifestos += 1

                    resumo_manifesto = _montar_manifesto_resumo(
                        df_grupo=df_grupo,
                        veiculo=veiculo,
                        manifesto_id=manifesto_id,
                        criterio_agrupamento=criterio_nome,
                        chave_agrupamento=chave_agrupamento,
                    )
                    itens_manifesto = _montar_itens_manifesto(df_grupo, resumo_manifesto)

                    premanifestos.append(resumo_manifesto)
                    itens_premanifestos.append(itens_manifesto)

                    tentativa["manifesto_id"] = manifesto_id

                    ids_alocados.update(df_grupo["id_linha_pipeline"].astype(str).tolist())

                    pos_catalogo = catalogo_veiculos.index[catalogo_veiculos["tipo"] == veiculo["tipo"]].tolist()
                    if len(pos_catalogo) > 0:
                        pos = pos_catalogo[0]
                        catalogo_veiculos.loc[pos, "manifestos_gerados_m5_1"] = (
                            _num_safe(catalogo_veiculos.loc[pos, "manifestos_gerados_m5_1"], default=0) + 1
                        )
                        if pd.notna(catalogo_veiculos.loc[pos, "limite_manifestos"]):
                            catalogo_veiculos.loc[pos, "saldo_manifestos"] = (
                                _num_safe(catalogo_veiculos.loc[pos, "limite_manifestos"], default=np.nan)
                                - _num_safe(catalogo_veiculos.loc[pos, "manifestos_utilizados"], default=0)
                                - _num_safe(catalogo_veiculos.loc[pos, "manifestos_gerados_m5_1"], default=0)
                            )

                    fechou = True
                    tentativas.append(tentativa)
                    break

                tentativas.append(tentativa)

            if not fechou:
                # grupo permanece no saldo para a próxima camada / próxima rodada
                pass

        saldo_atual = fila.loc[~fila["id_linha_pipeline"].isin(ids_alocados)].copy()

    tempos_m5_1["execucao_camadas_ms"] = _duracao_ms(t0)

    # ============================================================
    # CONSOLIDAÇÃO FINAL
    # ============================================================
    t0 = _agora()

    df_premanifestos_m5_1 = pd.DataFrame(premanifestos)

    df_itens_premanifestos_m5_1 = (
        pd.concat(itens_premanifestos, ignore_index=True)
        if len(itens_premanifestos) > 0
        else pd.DataFrame()
    )

    df_tentativas_m5_1 = pd.DataFrame(tentativas)

    df_remanescente_m5_1 = (
        fila.loc[~fila["id_linha_pipeline"].isin(ids_alocados)].copy().reset_index(drop=True)
    )

    if len(df_itens_premanifestos_m5_1) > 0:
        if df_itens_premanifestos_m5_1["id_linha_pipeline"].astype(str).duplicated().any():
            raise Exception(
                "Validação pós-M5.1 falhou: id_linha_pipeline repetido em mais de um item pré-manifesto."
            )

    if len(df_premanifestos_m5_1) > 0:
        invalidos = df_premanifestos_m5_1.loc[
            (pd.to_numeric(df_premanifestos_m5_1["ocupacao_oficial_perc"], errors="coerce") < 70)
            | (pd.to_numeric(df_premanifestos_m5_1["ocupacao_oficial_perc"], errors="coerce") > 100)
        ].copy()
        if len(invalidos) > 0:
            raise Exception(
                "Validação pós-M5.1 falhou: pré-manifesto com ocupação fora de 70%-100%."
            )

    if len(df_remanescente_m5_1) > 0:
        df_remanescente_m5_1["status_roteirizacao"] = "remanescente_bloco_5_1"
        df_remanescente_m5_1["origem_bloco"] = "M5.1"
        df_remanescente_m5_1["segue_para_proximo_bloco"] = True
        if "motivo_nao_roteirizado" not in df_remanescente_m5_1.columns:
            df_remanescente_m5_1["motivo_nao_roteirizado"] = "remanescente_nao_fechado_no_m5_1"

    df_nao_roteirizados_bloco_5_1 = _montar_df_nao_roteirizados_bloco_5_1(
        df_remanescente_m5_1
    )

    df_uso_frota_m5_1 = catalogo_veiculos[
        [
            "tipo",
            "capacidade_peso_kg",
            "max_entregas",
            "max_km_distancia",
            "limite_manifestos",
            "manifestos_utilizados",
            "manifestos_gerados_m5_1",
            "saldo_manifestos",
        ]
    ].copy()

    tempos_m5_1["consolidacao_ms"] = _duracao_ms(t0)
    tempos_m5_1["tempo_total_m5_1_ms"] = _duracao_ms(t0_total)

    resumo_m5_1 = {
        "modulo": "M5.1",
        "data_base_roteirizacao": (
            data_base_roteirizacao.isoformat()
            if hasattr(data_base_roteirizacao, "isoformat")
            else str(data_base_roteirizacao)
        ),
        "tipo_roteirizacao": tipo_roteirizacao,
        "remanescente_entrada_m5_1": int(len(fila)),
        "pre_manifestos_gerados_m5_1": int(len(df_premanifestos_m5_1)),
        "itens_pre_manifestados_m5_1": int(len(df_itens_premanifestos_m5_1)),
        "remanescente_saida_m5_1": int(len(df_remanescente_m5_1)),
        "nao_roteirizados_bloco_5_1": int(len(df_nao_roteirizados_bloco_5_1)),
        "coluna_tipo_veiculo_utilizada": "tipo",
        "criterios_agrupamento_m5_1": [
            "mesmo_cliente",
            "mesma_cidade",
            "mesma_subregiao",
            "mesma_mesorregiao",
        ],
        "ocupacao_minima_padrao_perc": 70,
        "ocupacao_maxima_padrao_perc": 100,
        "persistiu_artefatos": False,
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    auditoria_m5_1 = {
        "total_tentativas": int(len(df_tentativas_m5_1)),
        "total_grupos_fechados": int(len(df_premanifestos_m5_1)),
        "total_itens_pre_manifestados": int(len(df_itens_premanifestos_m5_1)),
        "total_remanescentes": int(len(df_remanescente_m5_1)),
    }

    outputs = {
        "df_premanifestos_m5_1": df_premanifestos_m5_1,
        "df_itens_premanifestos_m5_1": df_itens_premanifestos_m5_1,
        "df_tentativas_m5_1": df_tentativas_m5_1,
        "df_remanescente_m5_1": df_remanescente_m5_1,
        "df_nao_roteirizados_bloco_5_1": df_nao_roteirizados_bloco_5_1,
        "df_uso_frota_m5_1": df_uso_frota_m5_1,
    }

    meta = {
        "resumo_m5_1": resumo_m5_1,
        "auditoria_m5_1": auditoria_m5_1,
        "metricas_m5_1": tempos_m5_1,
    }

    return outputs, meta
