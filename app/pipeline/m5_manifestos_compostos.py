from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import pandas as pd


# ============================================================
# M5.1 - Manifestos compostos leves
# ------------------------------------------------------------
# Estratégia oficial desta versão:
# 1) Entrada dura: recebe somente o remanescente do M4
# 2) Fase 1: consolidação por mesmo cliente
# 3) Fase 2: composição regional por fila global de prioridade,
#    com uma âncora por vez (cidade -> subregião -> mesorregião)
# 4) Saída: pré-manifestos, itens, tentativas, remanescente e
#    resumo auditável
# ============================================================


@dataclass
class VehicleCheck:
    ok: bool
    motivo: str
    ocupacao_perc: float
    paradas: int
    peso_total: float
    volume_total: float
    km_ref: float


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "sim", "s", "yes", "y"}


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _first_existing(row: pd.Series, columns: List[str], default: Any = None) -> Any:
    for col in columns:
        if col in row.index:
            value = row[col]
            if not pd.isna(value):
                return value
    return default


def _column(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for col in options:
        if col in df.columns:
            return col
    return None


def _normalize_inputs(
    df_remanescente_m4: pd.DataFrame,
    df_veiculos: pd.DataFrame,
    df_parametros: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = df_remanescente_m4.copy() if df_remanescente_m4 is not None else pd.DataFrame()
    veic = df_veiculos.copy() if df_veiculos is not None else pd.DataFrame()

    if df.empty:
        return df, veic, {
            "ocupacao_minima_padrao_perc": 70.0,
            "ocupacao_maxima_padrao_perc": 100.0,
        }

    # Padronização mínima de colunas do remanescente
    rename_map = {}
    if "sub_regiao" in df.columns and "subregiao" not in df.columns:
        rename_map["sub_regiao"] = "subregiao"
    if "mesoregiao" in df.columns and "mesorregiao" not in df.columns:
        rename_map["mesoregiao"] = "mesorregiao"
    if "tipo" in veic.columns and "perfil" not in veic.columns:
        # Mantém as duas se vierem diferentes, mas cria perfil quando faltar.
        pass
    if rename_map:
        df = df.rename(columns=rename_map)

    # Colunas obrigatórias mínimas
    if "id_linha_pipeline" not in df.columns:
        raise ValueError("M5.1 exige a coluna 'id_linha_pipeline' no remanescente do M4.")

    if "peso_calculado" not in df.columns:
        if "peso_c" in df.columns:
            df["peso_calculado"] = df["peso_c"]
        elif "peso_kg" in df.columns:
            df["peso_calculado"] = df["peso_kg"]
        else:
            df["peso_calculado"] = 0.0

    if "peso_kg" not in df.columns:
        df["peso_kg"] = df["peso_calculado"]

    if "vol_m3" not in df.columns:
        df["vol_m3"] = 0.0

    if "distancia_rodoviaria_est_km" not in df.columns:
        if "distancia_km" in df.columns:
            df["distancia_rodoviaria_est_km"] = df["distancia_km"]
        else:
            df["distancia_rodoviaria_est_km"] = 0.0

    if "cidade" not in df.columns:
        cidade_alt = _column(df, ["cidade_dest", "cidade_chave"])
        if cidade_alt:
            df["cidade"] = df[cidade_alt]
        else:
            df["cidade"] = ""

    if "subregiao" not in df.columns:
        df["subregiao"] = ""

    if "mesorregiao" not in df.columns:
        df["mesorregiao"] = ""

    if "destinatario" not in df.columns:
        df["destinatario"] = ""

    if "restricao_veiculo" not in df.columns:
        df["restricao_veiculo"] = None

    if "agendada" not in df.columns:
        df["agendada"] = False

    if "folga_dias" not in df.columns:
        df["folga_dias"] = 999

    if "prioridade_embarque_num" not in df.columns:
        if "prioridade_embarque" in df.columns:
            df["prioridade_embarque_num"] = pd.to_numeric(df["prioridade_embarque"], errors="coerce")
        else:
            df["prioridade_embarque_num"] = pd.NA

    if "ranking_prioridade_operacional" not in df.columns:
        df["ranking_prioridade_operacional"] = pd.NA

    # Padronização de tipos
    numeric_cols = [
        "peso_calculado",
        "peso_kg",
        "vol_m3",
        "distancia_rodoviaria_est_km",
        "folga_dias",
        "prioridade_embarque_num",
        "ranking_prioridade_operacional",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = ["agendada", "veiculo_exclusivo", "veiculo_exclusivo_flag"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(_to_bool)

    # Veículos
    if veic.empty:
        raise ValueError("M5.1 exige o dataframe de veículos.")

    if "tipo" not in veic.columns and "perfil" in veic.columns:
        veic["tipo"] = veic["perfil"]
    if "perfil" not in veic.columns and "tipo" in veic.columns:
        veic["perfil"] = veic["tipo"]

    for col in [
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ocupacao_minima_perc",
        "ocupacao_maxima_perc",
    ]:
        if col not in veic.columns:
            veic[col] = pd.NA
        veic[col] = pd.to_numeric(veic[col], errors="coerce")

    veic["tipo"] = veic["tipo"].astype(str)
    veic["perfil"] = veic["perfil"].astype(str)

    params: Dict[str, Any] = {
        "ocupacao_minima_padrao_perc": 70.0,
        "ocupacao_maxima_padrao_perc": 100.0,
    }

    if df_parametros is not None and not df_parametros.empty:
        for row in df_parametros.to_dict(orient="records"):
            chave = _safe_text(row.get("chave"))
            valor = row.get("valor")
            if chave:
                params[chave] = valor

    for key in ["ocupacao_minima_padrao_perc", "ocupacao_maxima_padrao_perc"]:
        params[key] = _to_float(params.get(key), 70.0 if "minima" in key else 100.0)

    return df, veic, params


def _priority_bucket(row: pd.Series) -> int:
    prioridade = _to_float(_first_existing(row, ["prioridade_embarque_num", "prioridade_embarque"], default=math.nan), math.nan)
    agendada = _to_bool(row.get("agendada"))
    folga = _to_float(row.get("folga_dias"), 999)

    if not math.isnan(prioridade) and prioridade > 0:
        return 0
    if agendada and folga == 0:
        return 1
    if agendada and folga == 1:
        return 2
    if not agendada and 0 <= folga <= 1:
        return 3
    if not agendada and folga < 0:
        return 9
    return 8


def _priority_score(row: pd.Series) -> Tuple:
    prioridade = _to_float(_first_existing(row, ["prioridade_embarque_num", "prioridade_embarque"], default=math.nan), math.nan)
    prioridade_sort = prioridade if not math.isnan(prioridade) else 999
    bucket = _priority_bucket(row)
    folga = _to_float(row.get("folga_dias"), 999)
    leadtime_sort = folga if 0 <= folga <= 1 else 999
    ranking_op = _to_float(row.get("ranking_prioridade_operacional"), 999)
    dist = _to_float(row.get("distancia_rodoviaria_est_km"), 999999)
    peso = -_to_float(row.get("peso_calculado"), 0)
    return (bucket, prioridade_sort, leadtime_sort, ranking_op, dist, peso, str(row.get("id_linha_pipeline")))


def _phase2_eligible(row: pd.Series) -> bool:
    agendada = _to_bool(row.get("agendada"))
    folga = _to_float(row.get("folga_dias"), 999)
    prioridade = _to_float(_first_existing(row, ["prioridade_embarque_num", "prioridade_embarque"], default=math.nan), math.nan)

    if not math.isnan(prioridade) and prioridade > 0:
        return True
    if agendada and folga in (0, 1):
        return True
    if (not agendada) and (0 <= folga <= 1):
        return True
    return False


def _sort_rows_operational(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    tmp = df.copy()
    tmp["_priority_key"] = tmp.apply(_priority_score, axis=1)
    tmp = tmp.sort_values("_priority_key").drop(columns=["_priority_key"])
    return tmp.reset_index(drop=True)


def _client_group_key(row: pd.Series) -> str:
    # Chave do cliente precisa ser auditável e estável.
    return "||".join(
        [
            _safe_text(_first_existing(row, ["destinatario"], "")),
            _safe_text(_first_existing(row, ["cidade"], "")),
            _safe_text(_first_existing(row, ["subregiao"], "")),
            _safe_text(_first_existing(row, ["mesorregiao"], "")),
        ]
    )


def _vehicle_rows(df_veiculos: pd.DataFrame) -> pd.DataFrame:
    # Tenta do maior para o menor.
    tmp = df_veiculos.copy()
    tmp["sort_cap_peso"] = tmp["capacidade_peso_kg"].fillna(0)
    tmp["sort_cap_vol"] = tmp["capacidade_vol_m3"].fillna(0)
    tmp = tmp.sort_values(["sort_cap_peso", "sort_cap_vol"], ascending=[False, False]).drop(
        columns=["sort_cap_peso", "sort_cap_vol"]
    )
    return tmp.reset_index(drop=True)


def _restriction_allows(df_items: pd.DataFrame, vehicle_tipo: str, vehicle_perfil: str) -> bool:
    if "restricao_veiculo" not in df_items.columns:
        return True
    restricoes = set()
    for value in df_items["restricao_veiculo"].tolist():
        text = _safe_text(value)
        if text:
            restricoes.add(text.upper())

    if not restricoes:
        return True

    vehicle_tipo_u = _safe_text(vehicle_tipo).upper()
    vehicle_perfil_u = _safe_text(vehicle_perfil).upper()

    for restr in restricoes:
        if restr not in {vehicle_tipo_u, vehicle_perfil_u}:
            return False
    return True


def _validate_vehicle(
    df_items: pd.DataFrame,
    vehicle_row: pd.Series,
    params: Dict[str, Any],
) -> VehicleCheck:
    if df_items.empty:
        return VehicleCheck(False, "grupo_vazio", 0.0, 0, 0.0, 0.0, 0.0)

    vehicle_tipo = _safe_text(vehicle_row.get("tipo"))
    vehicle_perfil = _safe_text(vehicle_row.get("perfil"))

    if not _restriction_allows(df_items, vehicle_tipo, vehicle_perfil):
        return VehicleCheck(False, "restricao_veiculo_incompativel", 0.0, 0, 0.0, 0.0, 0.0)

    peso_total = float(df_items["peso_calculado"].fillna(0).sum())
    volume_total = float(df_items["vol_m3"].fillna(0).sum())
    km_ref = float(df_items["distancia_rodoviaria_est_km"].fillna(0).max())
    paradas = int(df_items["destinatario"].fillna("").astype(str).nunique())

    capacidade_peso = _to_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    capacidade_vol = _to_float(vehicle_row.get("capacidade_vol_m3"), 0.0)
    max_entregas = _to_int(vehicle_row.get("max_entregas"), 0)
    max_km = _to_float(vehicle_row.get("max_km_distancia"), 0.0)

    ocupacao_min = _to_float(vehicle_row.get("ocupacao_minima_perc"), params.get("ocupacao_minima_padrao_perc", 70.0))
    ocupacao_max = _to_float(vehicle_row.get("ocupacao_maxima_perc"), params.get("ocupacao_maxima_padrao_perc", 100.0))

    if capacidade_peso > 0 and peso_total > capacidade_peso:
        return VehicleCheck(False, "excede_capacidade_peso", 0.0, paradas, peso_total, volume_total, km_ref)

    if capacidade_vol > 0 and volume_total > capacidade_vol:
        return VehicleCheck(False, "excede_capacidade_volume", 0.0, paradas, peso_total, volume_total, km_ref)

    if max_entregas > 0 and paradas > max_entregas:
        return VehicleCheck(False, "excede_max_entregas", 0.0, paradas, peso_total, volume_total, km_ref)

    if max_km > 0 and km_ref > max_km:
        return VehicleCheck(False, "excede_max_km", 0.0, paradas, peso_total, volume_total, km_ref)

    ocupacao_perc = 0.0
    if capacidade_peso > 0:
        ocupacao_perc = (peso_total / capacidade_peso) * 100.0

    if ocupacao_perc > ocupacao_max:
        return VehicleCheck(False, "excede_ocupacao_maxima", ocupacao_perc, paradas, peso_total, volume_total, km_ref)

    if ocupacao_perc < ocupacao_min:
        return VehicleCheck(False, "abaixo_ocupacao_minima", ocupacao_perc, paradas, peso_total, volume_total, km_ref)

    return VehicleCheck(True, "ok", ocupacao_perc, paradas, peso_total, volume_total, km_ref)


def _build_greedy_candidate(anchor_row: pd.Series, candidate_pool: pd.DataFrame, vehicle_row: pd.Series) -> pd.DataFrame:
    if candidate_pool.empty:
        return pd.DataFrame(columns=candidate_pool.columns)

    ordered = _sort_rows_operational(candidate_pool)
    anchor_id = str(anchor_row["id_linha_pipeline"])
    ordered["_is_anchor"] = ordered["id_linha_pipeline"].astype(str) == anchor_id
    ordered = ordered.sort_values(["_is_anchor"], ascending=[False]).drop(columns=["_is_anchor"]).reset_index(drop=True)

    selected_rows: List[Dict[str, Any]] = []
    seen_ids = set()

    capacidade_peso = _to_float(vehicle_row.get("capacidade_peso_kg"), 0.0)
    capacidade_vol = _to_float(vehicle_row.get("capacidade_vol_m3"), 0.0)
    max_entregas = _to_int(vehicle_row.get("max_entregas"), 0)
    max_km = _to_float(vehicle_row.get("max_km_distancia"), 0.0)

    current_peso = 0.0
    current_vol = 0.0
    current_ids_dest = set()
    current_km = 0.0

    for _, row in ordered.iterrows():
        row_id = str(row["id_linha_pipeline"])
        if row_id in seen_ids:
            continue

        # Respeita restrição de veículo desde a montagem
        if not _restriction_allows(pd.DataFrame([row]), _safe_text(vehicle_row.get("tipo")), _safe_text(vehicle_row.get("perfil"))):
            continue

        row_peso = _to_float(row.get("peso_calculado"), 0.0)
        row_vol = _to_float(row.get("vol_m3"), 0.0)
        row_dest = _safe_text(row.get("destinatario"))
        row_km = _to_float(row.get("distancia_rodoviaria_est_km"), 0.0)

        next_peso = current_peso + row_peso
        next_vol = current_vol + row_vol
        next_dest_set = set(current_ids_dest)
        if row_dest:
            next_dest_set.add(row_dest)
        next_paradas = len(next_dest_set)
        next_km = max(current_km, row_km)

        if capacidade_peso > 0 and next_peso > capacidade_peso:
            continue
        if capacidade_vol > 0 and next_vol > capacidade_vol:
            continue
        if max_entregas > 0 and next_paradas > max_entregas:
            continue
        if max_km > 0 and next_km > max_km:
            continue

        selected_rows.append(row.to_dict())
        seen_ids.add(row_id)
        current_peso = next_peso
        current_vol = next_vol
        current_ids_dest = next_dest_set
        current_km = next_km

    if not selected_rows:
        return pd.DataFrame(columns=candidate_pool.columns)

    selected = pd.DataFrame(selected_rows)
    # Garante que a âncora fique no candidato, se ela couber.
    if anchor_id not in set(selected["id_linha_pipeline"].astype(str).tolist()):
        anchor_df = candidate_pool[candidate_pool["id_linha_pipeline"].astype(str) == anchor_id]
        if not anchor_df.empty:
            anchor = anchor_df.iloc[[0]]
            test = pd.concat([anchor, selected], ignore_index=True).drop_duplicates(subset=["id_linha_pipeline"], keep="first")
            check = _validate_vehicle(test, vehicle_row, {"ocupacao_minima_padrao_perc": 0, "ocupacao_maxima_padrao_perc": 100})
            if check.motivo not in {"excede_capacidade_peso", "excede_capacidade_volume", "excede_max_entregas", "excede_max_km", "restricao_veiculo_incompativel"}:
                selected = test

    return selected.reset_index(drop=True)


def _manifesto_meta(manifesto_id: str, tipo_manifesto: str, vehicle_row: pd.Series, check: VehicleCheck, fase: str, camada: str) -> Dict[str, Any]:
    return {
        "manifesto_id": manifesto_id,
        "tipo_manifesto": tipo_manifesto,
        "veiculo_tipo": _safe_text(vehicle_row.get("tipo")),
        "qtd_itens": None,
        "qtd_ctes": None,
        "qtd_paradas": check.paradas,
        "base_carga_oficial": round(check.peso_total, 3),
        "peso_total_kg": round(check.peso_total, 3),
        "vol_total_m3": round(check.volume_total, 3),
        "km_referencia": round(check.km_ref, 2),
        "ocupacao_oficial_perc": round(check.ocupacao_perc, 2),
        "capacidade_peso_kg_veiculo": _to_float(vehicle_row.get("capacidade_peso_kg"), 0.0),
        "capacidade_vol_m3_veiculo": _to_float(vehicle_row.get("capacidade_vol_m3"), 0.0),
        "max_entregas_veiculo": _to_int(vehicle_row.get("max_entregas"), 0),
        "max_km_distancia_veiculo": _to_float(vehicle_row.get("max_km_distancia"), 0.0),
        "ignorar_ocupacao_minima": False,
        "origem_modulo": 5,
        "origem_etapa": f"{fase}_{camada}",
    }


def _tentativa_row(
    fase: str,
    camada: str,
    anchor_id: Optional[str],
    vehicle_row: Optional[pd.Series],
    resultado: str,
    motivo: str,
    df_candidate: Optional[pd.DataFrame] = None,
    check: Optional[VehicleCheck] = None,
    cliente_chave: Optional[str] = None,
) -> Dict[str, Any]:
    qtd_itens = 0 if df_candidate is None else int(len(df_candidate))
    qtd_paradas = 0
    peso_total = 0.0
    km_ref = 0.0
    ocupacao = 0.0
    vehicle_tipo = None

    if check is not None:
        qtd_paradas = int(check.paradas)
        peso_total = round(float(check.peso_total), 3)
        km_ref = round(float(check.km_ref), 2)
        ocupacao = round(float(check.ocupacao_perc), 2)

    if vehicle_row is not None:
        vehicle_tipo = _safe_text(vehicle_row.get("tipo"))

    return {
        "fase": fase,
        "camada": camada,
        "anchor_id_linha_pipeline": anchor_id,
        "cliente_chave": cliente_chave,
        "veiculo_tipo_tentado": vehicle_tipo,
        "resultado": resultado,
        "motivo": motivo,
        "qtd_itens_candidato": qtd_itens,
        "qtd_paradas_candidato": qtd_paradas,
        "peso_total_candidato": peso_total,
        "km_referencia_candidato": km_ref,
        "ocupacao_perc_candidato": ocupacao,
    }


def _build_pre_manifesto(
    df_items: pd.DataFrame,
    vehicle_row: pd.Series,
    manifesto_id: str,
    fase: str,
    camada: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    check = _validate_vehicle(df_items, vehicle_row, {"ocupacao_minima_padrao_perc": 0, "ocupacao_maxima_padrao_perc": 100})
    meta = _manifesto_meta(manifesto_id, "pre_manifesto_bloco_5_1", vehicle_row, check, fase, camada)

    df_manifesto = pd.DataFrame([meta])
    df_manifesto["qtd_itens"] = int(len(df_items))
    if "cte" in df_items.columns:
        qtd_ctes = df_items["cte"].nunique(dropna=True)
    elif "nro_documento" in df_items.columns:
        qtd_ctes = df_items["nro_documento"].nunique(dropna=True)
    else:
        qtd_ctes = len(df_items)
    df_manifesto["qtd_ctes"] = int(qtd_ctes)

    df_itens = df_items.copy()
    for key, value in meta.items():
        df_itens[key] = value

    return df_manifesto, df_itens


def _empty_outputs(input_df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "df_pre_manifestos_m5_1": pd.DataFrame(),
        "df_itens_pre_manifestos_m5_1": pd.DataFrame(),
        "df_tentativas_m5_1": pd.DataFrame(),
        "df_remanescente_m5_1": input_df.copy(),
        "df_frota_utilizada_m5_1": pd.DataFrame(),
        "resumo_m5_1": {
            "modulo": "M5.1",
            "remanescente_entrada_m5_1": int(len(input_df)),
            "pre_manifestos_gerados_m5_1": 0,
            "itens_pre_manifestados_m5_1": 0,
            "remanescente_saida_m5_1": int(len(input_df)),
            "nao_roteirizados_bloco_5_1": int(len(input_df)),
            "coluna_tipo_veiculo_utilizada": "tipo",
            "estrategia_m5_1": [
                "fase_1_mesmo_cliente",
                "fase_2_fila_global_uma_ancora_por_vez",
                "camadas_cidade_subregiao_mesorregiao",
            ],
            "ocupacao_minima_padrao_perc": params.get("ocupacao_minima_padrao_perc", 70.0),
            "ocupacao_maxima_padrao_perc": params.get("ocupacao_maxima_padrao_perc", 100.0),
            "anchors_geradas_m5_1": 0,
        },
    }


def executar_m5_1_manifestos_compostos(
    df_remanescente_m4: pd.DataFrame,
    df_veiculos: pd.DataFrame,
    df_parametros: Optional[pd.DataFrame] = None,
    *,
    tipo_roteirizacao: str = "carteira",
    data_base_roteirizacao: Optional[str] = None,
    rodada_id: Optional[str] = None,
    pasta_saida_base: Optional[str] = None,
    persistiu_artefatos: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    df_input, df_veiculos_norm, params = _normalize_inputs(df_remanescente_m4, df_veiculos, df_parametros)

    if df_input.empty:
        return _empty_outputs(df_input, params)

    veiculos = _vehicle_rows(df_veiculos_norm)
    saldo = _sort_rows_operational(df_input).copy()
    saldo["_anchor_attempted_m5_1"] = False

    pre_manifestos: List[pd.DataFrame] = []
    itens_manifestados: List[pd.DataFrame] = []
    tentativas: List[Dict[str, Any]] = []
    manifest_counter = 1
    anchors_phase2 = 0

    # ========================================================
    # FASE 1 - Consolidar por mesmo cliente
    # ========================================================
    saldo["cliente_chave_m5_1"] = saldo.apply(_client_group_key, axis=1)

    groups = []
    for cliente_chave, group_df in saldo.groupby("cliente_chave_m5_1", sort=False):
        groups.append((cliente_chave, _sort_rows_operational(group_df)))

    processed_ids_client = set()
    ids_consumidos_fase1 = set()

    for cliente_chave, group_df in groups:
        group_ids = set(group_df["id_linha_pipeline"].astype(str).tolist())
        if group_ids & ids_consumidos_fase1:
            continue

        if group_df.empty:
            continue

        fechado = False
        melhor_motivo = "nenhum_veiculo_compativel"
        for _, vehicle_row in veiculos.iterrows():
            candidate_df = group_df.copy()
            check = _validate_vehicle(candidate_df, vehicle_row, params)
            tentativas.append(
                _tentativa_row(
                    fase="fase_1_cliente",
                    camada="mesmo_cliente",
                    anchor_id=None,
                    vehicle_row=vehicle_row,
                    resultado="fechado" if check.ok else "falhou",
                    motivo=check.motivo,
                    df_candidate=candidate_df,
                    check=check,
                    cliente_chave=cliente_chave,
                )
            )
            melhor_motivo = check.motivo
            if check.ok:
                manifesto_id = f"PM51_{manifest_counter:04d}"
                manifest_counter += 1
                df_manifesto, df_itens = _build_pre_manifesto(candidate_df, vehicle_row, manifesto_id, "fase_1_cliente", "mesmo_cliente")
                pre_manifestos.append(df_manifesto)
                itens_manifestados.append(df_itens)
                ids_consumidos_fase1.update(group_ids)
                fechado = True
                break

        if not fechado:
            processed_ids_client.update(group_ids)
            tentativas.append(
                _tentativa_row(
                    fase="fase_1_cliente",
                    camada="mesmo_cliente",
                    anchor_id=None,
                    vehicle_row=None,
                    resultado="saldo",
                    motivo=melhor_motivo,
                    df_candidate=group_df,
                    check=None,
                    cliente_chave=cliente_chave,
                )
            )

    if ids_consumidos_fase1:
        saldo = saldo[~saldo["id_linha_pipeline"].astype(str).isin(ids_consumidos_fase1)].copy()

    # ========================================================
    # FASE 2 - Fila global de prioridade, uma âncora por vez
    # ========================================================
    # Reordena saldo global depois da fase 1
    saldo = _sort_rows_operational(saldo)
    while True:
        elegiveis = saldo[(saldo["_anchor_attempted_m5_1"] == False) & saldo.apply(_phase2_eligible, axis=1)].copy()  # noqa: E712
        if elegiveis.empty:
            break

        anchor_row = _sort_rows_operational(elegiveis).iloc[0]
        anchor_id = str(anchor_row["id_linha_pipeline"])
        anchors_phase2 += 1

        # Marca a âncora como tentada neste ciclo global.
        saldo.loc[saldo["id_linha_pipeline"].astype(str) == anchor_id, "_anchor_attempted_m5_1"] = True

        anchor_fechada = False
        for camada in ["cidade", "subregiao", "mesorregiao"]:
            valor = _safe_text(anchor_row.get(camada))
            if not valor:
                tentativas.append(
                    _tentativa_row(
                        fase="fase_2_fila_global",
                        camada=camada,
                        anchor_id=anchor_id,
                        vehicle_row=None,
                        resultado="falhou",
                        motivo=f"{camada}_vazia",
                        df_candidate=None,
                        check=None,
                        cliente_chave=_safe_text(anchor_row.get("cliente_chave_m5_1")),
                    )
                )
                continue

            candidate_pool = saldo[saldo[camada].fillna("").astype(str).str.strip() == valor].copy()
            if candidate_pool.empty:
                tentativas.append(
                    _tentativa_row(
                        fase="fase_2_fila_global",
                        camada=camada,
                        anchor_id=anchor_id,
                        vehicle_row=None,
                        resultado="falhou",
                        motivo="pool_vazio",
                        df_candidate=None,
                        check=None,
                        cliente_chave=_safe_text(anchor_row.get("cliente_chave_m5_1")),
                    )
                )
                continue

            # Ordena o pool todo, mas só monta um candidato por veículo.
            candidate_pool = _sort_rows_operational(candidate_pool)
            melhor_motivo = "nenhum_veiculo_compativel"

            for _, vehicle_row in veiculos.iterrows():
                candidate_df = _build_greedy_candidate(anchor_row, candidate_pool, vehicle_row)
                if candidate_df.empty:
                    tentativas.append(
                        _tentativa_row(
                            fase="fase_2_fila_global",
                            camada=camada,
                            anchor_id=anchor_id,
                            vehicle_row=vehicle_row,
                            resultado="falhou",
                            motivo="sem_candidato_viavel_no_pool",
                            df_candidate=candidate_df,
                            check=None,
                            cliente_chave=_safe_text(anchor_row.get("cliente_chave_m5_1")),
                        )
                    )
                    melhor_motivo = "sem_candidato_viavel_no_pool"
                    continue

                check = _validate_vehicle(candidate_df, vehicle_row, params)
                tentativas.append(
                    _tentativa_row(
                        fase="fase_2_fila_global",
                        camada=camada,
                        anchor_id=anchor_id,
                        vehicle_row=vehicle_row,
                        resultado="fechado" if check.ok else "falhou",
                        motivo=check.motivo,
                        df_candidate=candidate_df,
                        check=check,
                        cliente_chave=_safe_text(anchor_row.get("cliente_chave_m5_1")),
                    )
                )
                melhor_motivo = check.motivo

                if check.ok:
                    manifesto_id = f"PM51_{manifest_counter:04d}"
                    manifest_counter += 1
                    df_manifesto, df_itens = _build_pre_manifesto(candidate_df, vehicle_row, manifesto_id, "fase_2_fila_global", camada)
                    pre_manifestos.append(df_manifesto)
                    itens_manifestados.append(df_itens)

                    consumed_ids = set(candidate_df["id_linha_pipeline"].astype(str).tolist())
                    saldo = saldo[~saldo["id_linha_pipeline"].astype(str).isin(consumed_ids)].copy()
                    saldo = _sort_rows_operational(saldo)
                    anchor_fechada = True
                    break

            if anchor_fechada:
                break

            tentativas.append(
                _tentativa_row(
                    fase="fase_2_fila_global",
                    camada=camada,
                    anchor_id=anchor_id,
                    vehicle_row=None,
                    resultado="saldo",
                    motivo=melhor_motivo,
                    df_candidate=candidate_pool,
                    check=None,
                    cliente_chave=_safe_text(anchor_row.get("cliente_chave_m5_1")),
                )
            )

        # Se não fechou, a âncora continua no saldo, apenas marcada como tentada.
        if anchor_fechada:
            continue

    # Limpeza final de colunas técnicas
    remanescente = saldo.drop(columns=[c for c in ["_anchor_attempted_m5_1", "cliente_chave_m5_1"] if c in saldo.columns]).reset_index(drop=True)

    df_pre_manifestos = pd.concat(pre_manifestos, ignore_index=True) if pre_manifestos else pd.DataFrame()
    df_itens_pre_manifestos = pd.concat(itens_manifestados, ignore_index=True) if itens_manifestados else pd.DataFrame()
    df_tentativas = pd.DataFrame(tentativas)

    if not df_itens_pre_manifestos.empty:
        frota = (
            df_itens_pre_manifestos.groupby("veiculo_tipo", dropna=False)
            .agg(
                pre_manifestos=("manifesto_id", "nunique"),
                itens=("id_linha_pipeline", "count"),
                peso_total_kg=("peso_calculado", "sum"),
            )
            .reset_index()
        )
    else:
        frota = pd.DataFrame(columns=["veiculo_tipo", "pre_manifestos", "itens", "peso_total_kg"])

    resumo = {
        "modulo": "M5.1",
        "data_base_roteirizacao": data_base_roteirizacao,
        "tipo_roteirizacao": tipo_roteirizacao,
        "remanescente_entrada_m5_1": int(len(df_input)),
        "pre_manifestos_gerados_m5_1": int(df_pre_manifestos["manifesto_id"].nunique()) if not df_pre_manifestos.empty else 0,
        "itens_pre_manifestados_m5_1": int(len(df_itens_pre_manifestos)),
        "remanescente_saida_m5_1": int(len(remanescente)),
        "nao_roteirizados_bloco_5_1": int(len(remanescente)),
        "coluna_tipo_veiculo_utilizada": "tipo",
        "estrategia_m5_1": [
            "fase_1_mesmo_cliente",
            "fase_2_fila_global_uma_ancora_por_vez",
            "camadas_cidade_subregiao_mesorregiao",
        ],
        "ocupacao_minima_padrao_perc": params.get("ocupacao_minima_padrao_perc", 70.0),
        "ocupacao_maxima_padrao_perc": params.get("ocupacao_maxima_padrao_perc", 100.0),
        "anchors_geradas_m5_1": int(anchors_phase2),
        "persistiu_artefatos": bool(persistiu_artefatos),
        "caminhos_pipeline": {
            "pasta_saida_base": pasta_saida_base,
            "rodada_id": rodada_id,
        },
    }

    result = {
        "df_pre_manifestos_m5_1": df_pre_manifestos,
        "df_itens_pre_manifestos_m5_1": df_itens_pre_manifestos,
        "df_tentativas_m5_1": df_tentativas,
        "df_remanescente_m5_1": remanescente,
        "df_frota_utilizada_m5_1": frota,
        "resumo_m5_1": resumo,
    }

    # Aliases defensivos para compatibilidade com services legados
    result["df_pre_manifestos"] = df_pre_manifestos
    result["df_itens_pre_manifestos"] = df_itens_pre_manifestos
    result["df_tentativas"] = df_tentativas
    result["df_remanescente"] = remanescente
    result["df_frota_utilizada"] = frota
    result["resumo"] = resumo

    return result


def executar_m5_1(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return executar_m5_1_manifestos_compostos(*args, **kwargs)


def processar_m5_1(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return executar_m5_1_manifestos_compostos(*args, **kwargs)


def rodar_m5_1(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return executar_m5_1_manifestos_compostos(*args, **kwargs)

