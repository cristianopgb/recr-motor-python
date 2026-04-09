from __future__ import annotations

import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# MÓDULO 5.1 - PRÉ-MANIFESTOS COMPOSTOS (RODADA SIMPLES COM ÂNCORA CONTROLADA)
#
# OBJETIVO:
# - receber somente o remanescente do M4
# - tentar fechamento integral por mesmo cliente
# - se não fechar, gerar UMA âncora por grupo de cliente
# - tentar composição da âncora por regionalidade:
#   mesma cidade -> mesma subregião -> mesma mesorregião
# - testar veículos do maior para o menor
# - respeitar capacidade, restrição, max entregas, km e ocupação
# - devolver pré-manifestos e saldo final
#
# REGRAS DESTA ETAPA:
# 1) entrada = somente remanescente roteirizável do M4
# 2) fase 1 = consolidar por mesmo cliente
# 3) ordenação interna do grupo:
#    - prioridade operacional
#    - agendada folga 0
#    - agendada folga 1
#    - leadtime do menor para o maior
#    - leadtime negativo por último
# 4) tentativa de veículo:
#    - do maior para o menor
#    - ocupação mínima é critério de otimização entre perfis
#    - se não fechar no maior, tenta os menores
# 5) se mesmo cliente não fechar:
#    - item mais prioritário vira âncora
#    - demais itens voltam ao saldo elegível
# 6) fase 2 = composição da âncora por regionalidade:
#    - mesma cidade
#    - mesma subregião
#    - mesma mesorregião
# 7) cada grupo de cliente gera no máximo UMA âncora
#    - evita explosão combinatória do M5 anterior
# ============================================================

OCUPACAO_MINIMA_PADRAO = 0.70
OCUPACAO_MAXIMA_PADRAO = 1.00


def _agora() -> float:
    return time.perf_counter()


def _duracao_ms(inicio: float) -> float:
    return round((time.perf_counter() - inicio) * 1000, 2)


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
        ("mesorregiao", ["mesorregiao", "mesoregiao"]),
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
        ("agendada", ["agendada"]),
        ("folga_dias", ["folga_dias"]),
        ("data_limite_considerada", ["data_limite_considerada"]),
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

    for col in ["destinatario", "cidade", "uf", "subregiao", "mesorregiao"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

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
    df["ranking_preliminar"] = pd.to_numeric(df["ranking_preliminar"], errors="coerce").fillna(999999)
    df["agendada"] = df["agendada"].apply(_bool_safe)
    df["folga_dias"] = pd.to_numeric(df["folga_dias"], errors="coerce")

    # classificação interna do M5.1 para ordenar grupos e candidatos
    cond_ag0 = df["agendada"] & (df["folga_dias"] == 0)
    cond_ag1 = df["agendada"] & (df["folga_dias"] == 1)
    cond_lead_nonneg = (~df["agendada"]) & (df["folga_dias"].fillna(-999999) >= 0)
    cond_lead_neg = (~df["agendada"]) & (df["folga_dias"].fillna(-999999) < 0)

    df["faixa_temporal_m5"] = np.select(
        [cond_ag0, cond_ag1, cond_lead_nonneg, cond_lead_neg],
        [1, 2, 3, 4],
        default=5,
    )

    # leadtime do menor para o maior; leadtime negativo fica por último
    df["leadtime_ordem_m5"] = np.where(
        cond_lead_nonneg,
        df["folga_dias"].fillna(999999),
        np.where(cond_lead_neg, 999999 + df["folga_dias"].abs().fillna(999999), 999998),
    )

    df["cliente_chave_m5"] = df["destinatario"].apply(_normalizar_texto)
    df["cidade_chave_m5"] = (
        df["cidade"].astype(str).str.strip() + "|" + df["uf"].astype(str).str.strip()
    ).apply(_normalizar_texto)
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
        lambda x: (float(x) / 100.0) if pd.notna(x) else OCUPACAO_MINIMA_PADRAO
    )
    catalogo["ocupacao_maxima_decimal"] = OCUPACAO_MAXIMA_PADRAO
    catalogo["tipo_norm"] = catalogo["tipo"].apply(_normalizar_texto)

    catalogo = catalogo.sort_values(
        by=["capacidade_peso_kg", "max_entregas", "max_km_distancia"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return catalogo


def _ordenar_df_operacional(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df.copy()

    return df.sort_values(
        by=[
            "ranking_prioridade_operacional",
            "faixa_temporal_m5",
            "leadtime_ordem_m5",
            "prioridade_embarque_num",
            "ranking_preliminar",
            "score_prioridade_preliminar",
            "peso_calculado",
            "id_linha_pipeline",
        ],
        ascending=[True, True, True, True, True, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _ordenar_catalogo_veiculos(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["capacidade_peso_kg", "max_entregas", "max_km_distancia"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


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
    ocup_min = _num_safe(veiculo.get("ocupacao_minima_decimal"), default=OCUPACAO_MINIMA_PADRAO)
    ocup_max = _num_safe(veiculo.get("ocupacao_maxima_decimal"), default=OCUPACAO_MAXIMA_PADRAO)

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
    anchor_id: Optional[str] = None,
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
        "origem_etapa": "5.1_ancora_controlada_regionalidade",
        "criterio_agrupamento": criterio_agrupamento,
        "chave_agrupamento": chave_agrupamento,
        "anchor_id_linha_pipeline": anchor_id,
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
    itens["anchor_id_linha_pipeline_m5_1"] = manifesto_resumo.get("anchor_id_linha_pipeline")
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


def _registrar_tentativa(
    tentativas: List[Dict[str, Any]],
    etapa: str,
    criterio: str,
    chave: str,
    veiculo_tipo: Optional[str],
    status: str,
    motivo: str,
    df_grupo: Optional[pd.DataFrame] = None,
    detalhe: Optional[Dict[str, Any]] = None,
    anchor_id: Optional[str] = None,
    manifesto_id: Optional[str] = None,
) -> None:
    detalhe = detalhe or {}
    qtd_itens = 0
    qtd_ctes = 0
    qtd_paradas = 0
    peso_total = 0.0
    km_ref = 0.0

    if df_grupo is not None and len(df_grupo) > 0:
        qtd_itens = int(len(df_grupo))
        qtd_ctes = int(df_grupo["cte"].astype(str).nunique()) if "cte" in df_grupo.columns else qtd_itens
        qtd_paradas = _qtd_paradas(df_grupo)
        peso_total = round(float(pd.to_numeric(df_grupo["peso_calculado"], errors="coerce").fillna(0.0).sum()), 3)
        km_ref = round(_km_grupo(df_grupo), 2)

    tentativas.append(
        {
            "etapa_m5_1": etapa,
            "criterio_agrupamento": criterio,
            "chave_agrupamento": chave,
            "anchor_id_linha_pipeline": anchor_id,
            "veiculo_tipo_tentado": veiculo_tipo,
            "status_tentativa": status,
            "motivo": motivo,
            "qtd_itens_grupo": int(detalhe.get("qtd_itens", qtd_itens)),
            "qtd_ctes_grupo": int(detalhe.get("qtd_ctes", qtd_ctes)) if detalhe.get("qtd_ctes", qtd_ctes) == detalhe.get("qtd_ctes", qtd_ctes) else qtd_ctes,
            "qtd_paradas_grupo": int(detalhe.get("qtd_paradas", qtd_paradas)) if detalhe.get("qtd_paradas", qtd_paradas) == detalhe.get("qtd_paradas", qtd_paradas) else qtd_paradas,
            "peso_total_grupo": round(float(detalhe.get("peso_total", peso_total)), 3),
            "capacidade_peso_kg_veiculo": detalhe.get("capacidade_peso_kg"),
            "ocupacao_perc_grupo": detalhe.get("ocupacao"),
            "km_referencia_grupo": detalhe.get("km_referencia", km_ref),
            "max_entregas_veiculo": detalhe.get("max_entregas_veiculo"),
            "max_km_distancia_veiculo": detalhe.get("max_km_distancia_veiculo"),
            "manifesto_id": manifesto_id,
        }
    )


def _atualizar_uso_frota(catalogo_veiculos: pd.DataFrame, veiculo_tipo: str) -> pd.DataFrame:
    pos_catalogo = catalogo_veiculos.index[catalogo_veiculos["tipo"] == veiculo_tipo].tolist()
    if len(pos_catalogo) == 0:
        return catalogo_veiculos

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
    return catalogo_veiculos


def _gerar_grupo_guloso_com_ancora(
    df_pool: pd.DataFrame,
    anchor_id: str,
    veiculo: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    pool = _ordenar_df_operacional(df_pool.copy())
    pool = pool.drop_duplicates(subset=["id_linha_pipeline"]).reset_index(drop=True)

    if anchor_id not in set(pool["id_linha_pipeline"].astype(str).tolist()):
        return pd.DataFrame(), {"motivo": "anchor_nao_encontrada_no_pool"}

    anchor = pool.loc[pool["id_linha_pipeline"].astype(str) == str(anchor_id)].copy()
    candidatos = pool.loc[pool["id_linha_pipeline"].astype(str) != str(anchor_id)].copy()

    grupo_ids = [str(anchor_id)]
    grupo = anchor.copy()

    ok_anchor, detalhe_anchor = _validar_grupo_contra_veiculo(grupo, veiculo)
    if ok_anchor:
        # âncora sozinha já atende tudo
        return grupo.reset_index(drop=True), detalhe_anchor

    motivo_anchor = detalhe_anchor.get("motivo")
    if motivo_anchor in {
        "grupo_contem_exclusivo",
        "veiculo_sem_capacidade_peso",
        "excede_capacidade_peso",
        "excede_max_entregas",
        "excede_max_km",
        "acima_ocupacao_maxima",
    }:
        return pd.DataFrame(), detalhe_anchor

    # abaixo ocupação mínima é esperado neste ponto; tenta completar
    for _, item in candidatos.iterrows():
        candidato = pd.concat([grupo, item.to_frame().T], ignore_index=True)
        ok_parcial, detalhe_parcial = _validar_grupo_contra_veiculo(candidato, veiculo)

        if ok_parcial:
            grupo = candidato.copy()
            continue

        motivo = detalhe_parcial.get("motivo")
        if motivo == "abaixo_ocupacao_minima":
            grupo = candidato.copy()
            continue

        # para hard stop, simplesmente não adiciona o item e segue para o próximo
        continue

    ok_final, detalhe_final = _validar_grupo_contra_veiculo(grupo, veiculo)
    if ok_final:
        return grupo.reset_index(drop=True), detalhe_final

    return pd.DataFrame(), detalhe_final


def _filtrar_pool_camadas_para_ancora(
    saldo_disponivel: pd.DataFrame,
    anchor_row: pd.Series,
    criterio: str,
) -> pd.DataFrame:
    if saldo_disponivel is None or len(saldo_disponivel) == 0:
        return pd.DataFrame()

    anchor_id = str(anchor_row["id_linha_pipeline"])
    anchor_cidade = _normalizar_texto(f"{anchor_row.get('cidade', '')}|{anchor_row.get('uf', '')}")
    anchor_sub = _normalizar_texto(anchor_row.get("subregiao"))
    anchor_meso = _normalizar_texto(anchor_row.get("mesorregiao"))

    if criterio == "mesma_cidade":
        mask = saldo_disponivel["cidade_chave_m5"] == anchor_cidade
    elif criterio == "mesma_subregiao":
        if anchor_sub == "":
            return pd.DataFrame()
        mask = saldo_disponivel["subregiao_chave_m5"] == anchor_sub
    elif criterio == "mesma_mesorregiao":
        if anchor_meso == "":
            return pd.DataFrame()
        mask = saldo_disponivel["mesorregiao_chave_m5"] == anchor_meso
    else:
        return pd.DataFrame()

    pool = saldo_disponivel.loc[mask].copy()
    if len(pool) == 0:
        return pd.DataFrame()

    # mantém a âncora no topo; demais ficam ordenados pela regra do M5.1
    pool = _ordenar_df_operacional(pool)
    pool["anchor_topo_sort"] = np.where(
        pool["id_linha_pipeline"].astype(str) == anchor_id,
        0,
        1,
    )
    pool = pool.sort_values(
        by=["anchor_topo_sort", "ranking_prioridade_operacional", "faixa_temporal_m5", "leadtime_ordem_m5", "peso_calculado"],
        ascending=[True, True, True, True, False],
        na_position="last",
    ).drop(columns=["anchor_topo_sort"]).reset_index(drop=True)
    return pool


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
    fila = _ordenar_df_operacional(fila)

    catalogo_veiculos = _preparar_catalogo_veiculos(
        df_veiculos_tratados=df_veiculos_tratados,
        tipo_roteirizacao=tipo_roteirizacao,
        configuracao_frota=configuracao_frota,
        df_uso_frota_m4=df_uso_frota_m4,
    )
    catalogo_veiculos = _ordenar_catalogo_veiculos(catalogo_veiculos)

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
                "estrategia_m5_1": [
                    "fase_1_mesmo_cliente",
                    "fase_2_ancora_controlada",
                    "camadas_cidade_subregiao_mesorregiao",
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
    # EXECUÇÃO
    # ============================================================
    t0 = _agora()

    tentativas: List[Dict[str, Any]] = []
    premanifestos: List[Dict[str, Any]] = []
    itens_premanifestos: List[pd.DataFrame] = []

    ids_alocados: set[str] = set()
    ids_bloqueados_ancora: set[str] = set()
    contador_manifestos = 1
    anchors: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # FASE 1 - TENTATIVA INTEGRAL POR MESMO CLIENTE
    # ------------------------------------------------------------
    grupos_cliente = (
        fila.loc[fila["cliente_chave_m5"].astype(str).str.strip() != ""]
        .groupby("cliente_chave_m5", dropna=False)
        .agg(
            qtd_itens=("id_linha_pipeline", "count"),
            peso_total=("peso_calculado", "sum"),
            prioridade_min=("ranking_prioridade_operacional", "min"),
            faixa_temporal_min=("faixa_temporal_m5", "min"),
        )
        .reset_index()
        .sort_values(
            by=["prioridade_min", "faixa_temporal_min", "qtd_itens", "peso_total", "cliente_chave_m5"],
            ascending=[True, True, False, False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    for _, grupo_meta in grupos_cliente.iterrows():
        cliente_chave = str(grupo_meta["cliente_chave_m5"])
        saldo_atual = fila.loc[
            (~fila["id_linha_pipeline"].isin(ids_alocados))
            & (~fila["id_linha_pipeline"].isin(ids_bloqueados_ancora))
        ].copy()

        df_cliente = saldo_atual.loc[saldo_atual["cliente_chave_m5"] == cliente_chave].copy()
        if len(df_cliente) == 0:
            continue

        df_cliente = _ordenar_df_operacional(df_cliente)
        anchor_row = df_cliente.iloc[0].copy()
        anchor_id = str(anchor_row["id_linha_pipeline"])

        catalogo_candidato = _filtrar_catalogo_por_restricao(df_cliente, catalogo_veiculos)
        catalogo_candidato = _ordenar_catalogo_veiculos(catalogo_candidato)

        if len(catalogo_candidato) == 0:
            _registrar_tentativa(
                tentativas=tentativas,
                etapa="fase_1_cliente",
                criterio="mesmo_cliente",
                chave=cliente_chave,
                veiculo_tipo=None,
                status="falha",
                motivo="nenhum_veiculo_compativel_com_restricao",
                df_grupo=df_cliente,
                anchor_id=anchor_id,
            )
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "cliente_chave": cliente_chave,
                    "origem": "mesmo_cliente_falhou_sem_veiculo",
                }
            )
            ids_bloqueados_ancora.add(anchor_id)
            continue

        fechou_cliente = False

        for _, veiculo in catalogo_candidato.iterrows():
            saldo_manifestos = veiculo.get("saldo_manifestos", np.nan)
            if pd.notna(saldo_manifestos) and float(saldo_manifestos) <= 0:
                _registrar_tentativa(
                    tentativas=tentativas,
                    etapa="fase_1_cliente",
                    criterio="mesmo_cliente",
                    chave=cliente_chave,
                    veiculo_tipo=veiculo["tipo"],
                    status="falha",
                    motivo="saldo_frota_esgotado",
                    df_grupo=df_cliente,
                    anchor_id=anchor_id,
                )
                continue

            ok, detalhe = _validar_grupo_contra_veiculo(df_cliente, veiculo)
            if ok:
                manifesto_id = _gerar_manifesto_id(contador_manifestos)
                contador_manifestos += 1

                resumo_manifesto = _montar_manifesto_resumo(
                    df_grupo=df_cliente,
                    veiculo=veiculo,
                    manifesto_id=manifesto_id,
                    criterio_agrupamento="mesmo_cliente",
                    chave_agrupamento=cliente_chave,
                    anchor_id=anchor_id,
                )
                itens_manifesto = _montar_itens_manifesto(df_cliente, resumo_manifesto)

                premanifestos.append(resumo_manifesto)
                itens_premanifestos.append(itens_manifesto)

                ids_alocados.update(df_cliente["id_linha_pipeline"].astype(str).tolist())
                catalogo_veiculos = _atualizar_uso_frota(catalogo_veiculos, veiculo["tipo"])

                _registrar_tentativa(
                    tentativas=tentativas,
                    etapa="fase_1_cliente",
                    criterio="mesmo_cliente",
                    chave=cliente_chave,
                    veiculo_tipo=veiculo["tipo"],
                    status="sucesso",
                    motivo="grupo_fechou",
                    df_grupo=df_cliente,
                    detalhe=detalhe,
                    anchor_id=anchor_id,
                    manifesto_id=manifesto_id,
                )
                fechou_cliente = True
                break

            _registrar_tentativa(
                tentativas=tentativas,
                etapa="fase_1_cliente",
                criterio="mesmo_cliente",
                chave=cliente_chave,
                veiculo_tipo=veiculo["tipo"],
                status="falha",
                motivo=detalhe.get("motivo", "falha_validacao_grupo"),
                df_grupo=df_cliente,
                detalhe=detalhe,
                anchor_id=anchor_id,
            )

        if not fechou_cliente:
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "cliente_chave": cliente_chave,
                    "origem": "mesmo_cliente_nao_fechou",
                }
            )
            ids_bloqueados_ancora.add(anchor_id)

    tempos_m5_1["fase_1_cliente_ms"] = _duracao_ms(t0)

    # ------------------------------------------------------------
    # FASE 2 - ÂNCORA CONTROLADA POR REGIONALIDADE
    # ------------------------------------------------------------
    t0 = _agora()

    criterios_regionais = [
        ("mesma_cidade", "cidade"),
        ("mesma_subregiao", "subregiao"),
        ("mesma_mesorregiao", "mesorregiao"),
    ]

    for anchor_info in anchors:
        anchor_id = str(anchor_info["anchor_id"])

        if anchor_id in ids_alocados:
            continue

        saldo_disponivel = fila.loc[~fila["id_linha_pipeline"].isin(ids_alocados)].copy()
        anchor_df = saldo_disponivel.loc[saldo_disponivel["id_linha_pipeline"].astype(str) == anchor_id].copy()

        if len(anchor_df) == 0:
            continue

        anchor_row = anchor_df.iloc[0].copy()
        fechou_ancora = False

        for criterio_nome, _label in criterios_regionais:
            saldo_disponivel = fila.loc[~fila["id_linha_pipeline"].isin(ids_alocados)].copy()
            anchor_df = saldo_disponivel.loc[
                saldo_disponivel["id_linha_pipeline"].astype(str) == anchor_id
            ].copy()
            if len(anchor_df) == 0:
                break

            anchor_row = anchor_df.iloc[0].copy()
            pool = _filtrar_pool_camadas_para_ancora(saldo_disponivel, anchor_row, criterio_nome)

            if len(pool) == 0:
                _registrar_tentativa(
                    tentativas=tentativas,
                    etapa="fase_2_ancora",
                    criterio=criterio_nome,
                    chave=str(anchor_row.get(criterio_nome.replace("mesma_", "") if criterio_nome != "mesma_cidade" else "cidade")),
                    veiculo_tipo=None,
                    status="falha",
                    motivo="pool_regional_vazio",
                    df_grupo=anchor_df,
                    anchor_id=anchor_id,
                )
                continue

            catalogo_candidato = _filtrar_catalogo_por_restricao(pool, catalogo_veiculos)
            catalogo_candidato = _ordenar_catalogo_veiculos(catalogo_candidato)

            if len(catalogo_candidato) == 0:
                _registrar_tentativa(
                    tentativas=tentativas,
                    etapa="fase_2_ancora",
                    criterio=criterio_nome,
                    chave=str(pool.iloc[0][
                        "cidade_chave_m5" if criterio_nome == "mesma_cidade" else
                        "subregiao_chave_m5" if criterio_nome == "mesma_subregiao" else
                        "mesorregiao_chave_m5"
                    ]),
                    veiculo_tipo=None,
                    status="falha",
                    motivo="nenhum_veiculo_compativel_com_restricao",
                    df_grupo=pool,
                    anchor_id=anchor_id,
                )
                continue

            chave_agrupamento = str(pool.iloc[0][
                "cidade_chave_m5" if criterio_nome == "mesma_cidade" else
                "subregiao_chave_m5" if criterio_nome == "mesma_subregiao" else
                "mesorregiao_chave_m5"
            ])

            for _, veiculo in catalogo_candidato.iterrows():
                saldo_manifestos = veiculo.get("saldo_manifestos", np.nan)
                if pd.notna(saldo_manifestos) and float(saldo_manifestos) <= 0:
                    _registrar_tentativa(
                        tentativas=tentativas,
                        etapa="fase_2_ancora",
                        criterio=criterio_nome,
                        chave=chave_agrupamento,
                        veiculo_tipo=veiculo["tipo"],
                        status="falha",
                        motivo="saldo_frota_esgotado",
                        df_grupo=pool,
                        anchor_id=anchor_id,
                    )
                    continue

                grupo_guloso, detalhe = _gerar_grupo_guloso_com_ancora(pool, anchor_id, veiculo)

                if len(grupo_guloso) == 0:
                    _registrar_tentativa(
                        tentativas=tentativas,
                        etapa="fase_2_ancora",
                        criterio=criterio_nome,
                        chave=chave_agrupamento,
                        veiculo_tipo=veiculo["tipo"],
                        status="falha",
                        motivo=detalhe.get("motivo", "falha_montagem_gulosa"),
                        df_grupo=pool,
                        detalhe=detalhe,
                        anchor_id=anchor_id,
                    )
                    continue

                manifesto_id = _gerar_manifesto_id(contador_manifestos)
                contador_manifestos += 1

                resumo_manifesto = _montar_manifesto_resumo(
                    df_grupo=grupo_guloso,
                    veiculo=veiculo,
                    manifesto_id=manifesto_id,
                    criterio_agrupamento=criterio_nome,
                    chave_agrupamento=chave_agrupamento,
                    anchor_id=anchor_id,
                )
                itens_manifesto = _montar_itens_manifesto(grupo_guloso, resumo_manifesto)

                premanifestos.append(resumo_manifesto)
                itens_premanifestos.append(itens_manifesto)

                ids_alocados.update(grupo_guloso["id_linha_pipeline"].astype(str).tolist())
                catalogo_veiculos = _atualizar_uso_frota(catalogo_veiculos, veiculo["tipo"])

                _registrar_tentativa(
                    tentativas=tentativas,
                    etapa="fase_2_ancora",
                    criterio=criterio_nome,
                    chave=chave_agrupamento,
                    veiculo_tipo=veiculo["tipo"],
                    status="sucesso",
                    motivo="grupo_fechou",
                    df_grupo=grupo_guloso,
                    detalhe=detalhe,
                    anchor_id=anchor_id,
                    manifesto_id=manifesto_id,
                )

                fechou_ancora = True
                break

            if fechou_ancora:
                break

    tempos_m5_1["fase_2_ancora_ms"] = _duracao_ms(t0)

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

    df_nao_roteirizados_bloco_5_1 = _montar_df_nao_roteirizados_bloco_5_1(df_remanescente_m5_1)

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
        "estrategia_m5_1": [
            "fase_1_mesmo_cliente",
            "fase_2_ancora_controlada",
            "camadas_cidade_subregiao_mesorregiao",
        ],
        "ocupacao_minima_padrao_perc": 70,
        "ocupacao_maxima_padrao_perc": 100,
        "anchors_geradas_m5_1": int(len(anchors)),
        "persistiu_artefatos": False,
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    auditoria_m5_1 = {
        "total_tentativas": int(len(df_tentativas_m5_1)),
        "total_grupos_fechados": int(len(df_premanifestos_m5_1)),
        "total_itens_pre_manifestados": int(len(df_itens_premanifestos_m5_1)),
        "total_remanescentes": int(len(df_remanescente_m5_1)),
        "total_anchors_geradas": int(len(anchors)),
        "tentativas_fase_1_cliente": int((df_tentativas_m5_1["etapa_m5_1"] == "fase_1_cliente").sum()) if len(df_tentativas_m5_1) > 0 else 0,
        "tentativas_fase_2_ancora": int((df_tentativas_m5_1["etapa_m5_1"] == "fase_2_ancora").sum()) if len(df_tentativas_m5_1) > 0 else 0,
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

