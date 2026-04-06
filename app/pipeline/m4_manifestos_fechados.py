from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


OCUPACAO_DOMINANTE_MIN = 0.70
OCUPACAO_SECUNDARIA_MIN = 0.20
MAX_ITERACOES_VARREDURA_4C = 10

CHAVES_PARADA = ["destinatario", "cidade", "uf"]
CHAVE_CLIENTE = "destinatario"


def _normalizar_texto(valor: Any) -> str:
    if pd.isna(valor):
        return ""
    return str(valor).strip()


def _to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df2 = df.copy()
    for col in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[col]):
            df2[col] = df2[col].astype(str)
    df2 = df2.where(pd.notnull(df2), None)
    return df2.to_dict(orient="records")


def _validar_input(
    df_input_oficial_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
) -> None:
    colunas_minimas_fila = [
        "id_linha_pipeline",
        "destinatario",
        "cidade",
        "uf",
        "peso_kg",
        "vol_m3",
        "distancia_rodoviaria_est_km",
        "status_triagem",
        "grupo_saida",
    ]

    colunas_minimas_veiculos = [
        "tipo",
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
    ]

    faltam_fila = [c for c in colunas_minimas_fila if c not in df_input_oficial_bloco_4.columns]
    faltam_veiculos = [c for c in colunas_minimas_veiculos if c not in df_veiculos_tratados.columns]

    if faltam_fila:
        raise Exception(
            "Faltam colunas mínimas no input oficial do Bloco 4:\n- " + "\n- ".join(faltam_fila)
        )

    if faltam_veiculos:
        raise Exception(
            "Faltam colunas mínimas na base de veículos tratados:\n- " + "\n- ".join(faltam_veiculos)
        )

    linhas_invalidas = df_input_oficial_bloco_4.loc[
        (df_input_oficial_bloco_4["status_triagem"].astype(str) != "roteirizavel")
        | (df_input_oficial_bloco_4["grupo_saida"].astype(str) != "df_carteira_roteirizavel")
    ].copy()

    if len(linhas_invalidas) > 0:
        raise Exception(
            "O M4 recebeu linhas incompatíveis com o estágio. "
            "Há registros com status_triagem != 'roteirizavel' "
            "ou grupo_saida != 'df_carteira_roteirizavel'."
        )

    if df_input_oficial_bloco_4["id_linha_pipeline"].astype(str).duplicated().any():
        qtd_dup = int(df_input_oficial_bloco_4["id_linha_pipeline"].astype(str).duplicated().sum())
        raise Exception(f"O input oficial do Bloco 4 possui id_linha_pipeline duplicado: {qtd_dup}")


def _padronizar_bases(
    df_input_oficial_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fila = df_input_oficial_bloco_4.copy().reset_index(drop=True)
    veiculos = df_veiculos_tratados.copy().reset_index(drop=True)

    for col in [
        "peso_kg",
        "vol_m3",
        "distancia_rodoviaria_est_km",
        "folga_dias",
        "ranking_preliminar",
        "score_prioridade_preliminar",
        "ranking_prioridade_operacional",
    ]:
        if col in fila.columns:
            fila[col] = pd.to_numeric(fila[col], errors="coerce")

    for col in [
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ordem_porte",
    ]:
        if col in veiculos.columns:
            veiculos[col] = pd.to_numeric(veiculos[col], errors="coerce")

    for col in ["data_agenda", "data_leadtime", "data_limite_considerada", "data_descarga", "data_nf"]:
        if col in fila.columns:
            fila[col] = pd.to_datetime(fila[col], errors="coerce")

    fila["destinatario"] = fila["destinatario"].apply(_normalizar_texto)
    fila["cidade"] = fila["cidade"].apply(_normalizar_texto)
    fila["uf"] = fila["uf"].apply(_normalizar_texto)

    if "agendada" in fila.columns:
        fila["agendada"] = fila["agendada"].fillna(False).astype(bool)
    else:
        fila["agendada"] = False

    if "ordem_porte" in veiculos.columns:
        veiculos = veiculos.sort_values(
            by=["ordem_porte", "capacidade_peso_kg", "capacidade_vol_m3", "max_entregas", "max_km_distancia"],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)
    else:
        veiculos = veiculos.sort_values(
            by=["capacidade_peso_kg", "capacidade_vol_m3", "max_entregas", "max_km_distancia"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    return fila, veiculos


def _criar_bucket_temporal(df: pd.DataFrame) -> pd.Series:
    """
    Bucket temporal para o 4A:
    - agendada -> AGENDA_YYYY-MM-DD
    - não agendada com leadtime -> LEADTIME_YYYY-MM-DD
    - sem data -> SEM_DATA
    """
    bucket = pd.Series(index=df.index, dtype="object")

    mask_agendada = df["agendada"].fillna(False)
    if "data_agenda" in df.columns:
        bucket.loc[mask_agendada] = (
            "AGENDA_" + df.loc[mask_agendada, "data_agenda"].dt.strftime("%Y-%m-%d").fillna("SEM_DATA")
        )

    mask_nao_agendada = ~mask_agendada
    if "data_leadtime" in df.columns:
        bucket.loc[mask_nao_agendada] = (
            "LEADTIME_" + df.loc[mask_nao_agendada, "data_leadtime"].dt.strftime("%Y-%m-%d").fillna("SEM_DATA")
        )
    else:
        bucket.loc[mask_nao_agendada] = "SEM_DATA"

    bucket = bucket.fillna("SEM_DATA")
    return bucket


def _calcular_metricas_grupo(df_grupo: pd.DataFrame, veiculo: pd.Series) -> Dict[str, Any]:
    peso_total = float(df_grupo["peso_kg"].fillna(0).sum())
    vol_total = float(df_grupo["vol_m3"].fillna(0).sum())

    qtd_paradas = int(
        df_grupo[CHAVES_PARADA]
        .fillna("")
        .astype(str)
        .agg("|".join, axis=1)
        .nunique()
    )

    km_max = float(df_grupo["distancia_rodoviaria_est_km"].fillna(0).max())

    cap_peso = float(veiculo["capacidade_peso_kg"])
    cap_vol = float(veiculo["capacidade_vol_m3"])
    max_entregas = int(veiculo["max_entregas"])
    max_km = float(veiculo["max_km_distancia"])

    ocup_peso = 0.0 if cap_peso <= 0 else peso_total / cap_peso
    ocup_vol = 0.0 if cap_vol <= 0 else vol_total / cap_vol

    ocup_dominante = max(ocup_peso, ocup_vol)
    ocup_secundaria = min(ocup_peso, ocup_vol)

    return {
        "peso_total_kg": round(peso_total, 3),
        "vol_total_m3": round(vol_total, 3),
        "qtd_paradas": qtd_paradas,
        "km_rota_referencia": round(km_max, 3),
        "ocupacao_peso_perc": round(ocup_peso * 100, 2),
        "ocupacao_vol_perc": round(ocup_vol * 100, 2),
        "ocupacao_dominante_perc": round(ocup_dominante * 100, 2),
        "ocupacao_secundaria_perc": round(ocup_secundaria * 100, 2),
        "capacidade_ok": peso_total <= cap_peso + 1e-9 and vol_total <= cap_vol + 1e-9,
        "entregas_ok": qtd_paradas <= max_entregas,
        "distancia_ok": km_max <= max_km + 1e-9,
        "ocupacao_ok": (
            ocup_dominante >= OCUPACAO_DOMINANTE_MIN
            and ocup_secundaria >= OCUPACAO_SECUNDARIA_MIN
        ),
    }


def _escolher_menor_veiculo_viavel(df_grupo: pd.DataFrame, veiculos: pd.DataFrame) -> Tuple[pd.Series | None, Dict[str, Any] | None]:
    for _, veiculo in veiculos.iterrows():
        metricas = _calcular_metricas_grupo(df_grupo, veiculo)

        if (
            metricas["capacidade_ok"]
            and metricas["entregas_ok"]
            and metricas["distancia_ok"]
            and metricas["ocupacao_ok"]
        ):
            return veiculo, metricas

    return None, None


def _gerar_manifesto_id(rodada_id: str, origem_etapa: str, seq: int) -> str:
    base = f"{rodada_id}|{origem_etapa}|{seq}"
    token = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"M4-{seq:05d}-{token}"


def _selecionar_campos_manifesto(df_grupo: pd.DataFrame, veiculo: pd.Series, metricas: Dict[str, Any], manifesto_id: str, origem_etapa: str) -> Dict[str, Any]:
    primeira = df_grupo.iloc[0]

    qtd_itens = int(len(df_grupo))
    qtd_paradas = int(metricas["qtd_paradas"])

    datas_limite = df_grupo["data_limite_considerada"].dropna() if "data_limite_considerada" in df_grupo.columns else pd.Series(dtype="datetime64[ns]")
    menor_data_limite = datas_limite.min() if len(datas_limite) > 0 else pd.NaT

    return {
        "manifesto_id": manifesto_id,
        "origem_etapa": origem_etapa,
        "veiculo_tipo": veiculo["tipo"],
        "qtd_itens": qtd_itens,
        "qtd_paradas": qtd_paradas,
        "destinatario_referencia": primeira.get("destinatario"),
        "cidade_referencia": primeira.get("cidade"),
        "uf_referencia": primeira.get("uf"),
        "regiao_referencia": primeira.get("regiao"),
        "subregiao_referencia": primeira.get("subregiao"),
        "mesorregiao_referencia": primeira.get("mesorregiao"),
        "agendada_flag": bool(df_grupo["agendada"].fillna(False).any()),
        "menor_data_limite_considerada": menor_data_limite,
        "peso_total_kg": metricas["peso_total_kg"],
        "vol_total_m3": metricas["vol_total_m3"],
        "ocupacao_peso_perc": metricas["ocupacao_peso_perc"],
        "ocupacao_vol_perc": metricas["ocupacao_vol_perc"],
        "ocupacao_dominante_perc": metricas["ocupacao_dominante_perc"],
        "ocupacao_secundaria_perc": metricas["ocupacao_secundaria_perc"],
        "km_rota_referencia": metricas["km_rota_referencia"],
        "capacidade_peso_kg_veiculo": float(veiculo["capacidade_peso_kg"]),
        "capacidade_vol_m3_veiculo": float(veiculo["capacidade_vol_m3"]),
        "max_entregas_veiculo": int(veiculo["max_entregas"]),
        "max_km_distancia_veiculo": float(veiculo["max_km_distancia"]),
    }


def _gerar_itens_manifesto(df_grupo: pd.DataFrame, manifesto_id: str, origem_etapa: str, veiculo_tipo: str) -> pd.DataFrame:
    itens = df_grupo.copy()
    itens["manifesto_id"] = manifesto_id
    itens["origem_etapa"] = origem_etapa
    itens["veiculo_tipo_manifesto"] = veiculo_tipo
    return itens


def _registrar_tentativa(
    origem_etapa: str,
    chave_grupo: str,
    qtd_itens: int,
    veiculo_testado: str | None,
    aprovado: bool,
    motivo: str,
    metricas: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    base = {
        "origem_etapa": origem_etapa,
        "chave_grupo": chave_grupo,
        "qtd_itens_grupo": int(qtd_itens),
        "veiculo_testado": veiculo_testado,
        "aprovado": bool(aprovado),
        "motivo": motivo,
    }
    if metricas:
        base.update(metricas)
    return base


def _processar_grupos(
    fila: pd.DataFrame,
    veiculos: pd.DataFrame,
    origem_etapa: str,
    colunas_groupby: List[str],
    rodada_id: str,
    seq_manifesto_inicio: int,
    somente_multiplos: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    if len(fila) == 0:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            seq_manifesto_inicio,
        )

    manifestos: List[Dict[str, Any]] = []
    itens_manifestos: List[pd.DataFrame] = []
    tentativas: List[Dict[str, Any]] = []
    ids_aprovados: set[str] = set()

    seq_manifesto = seq_manifesto_inicio

    grupos = fila.groupby(colunas_groupby, dropna=False, sort=False)

    for chave, df_grupo in grupos:
        df_grupo = df_grupo.copy().reset_index(drop=True)

        if somente_multiplos and len(df_grupo) <= 1:
            continue

        if df_grupo["id_linha_pipeline"].astype(str).isin(ids_aprovados).any():
            continue

        chave_str = str(chave)
        veiculo, metricas = _escolher_menor_veiculo_viavel(df_grupo, veiculos)

        if veiculo is None:
            tentativas.append(
                _registrar_tentativa(
                    origem_etapa=origem_etapa,
                    chave_grupo=chave_str,
                    qtd_itens=len(df_grupo),
                    veiculo_testado=None,
                    aprovado=False,
                    motivo="nenhum_veiculo_viavel",
                )
            )
            continue

        manifesto_id = _gerar_manifesto_id(rodada_id=rodada_id, origem_etapa=origem_etapa, seq=seq_manifesto)
        seq_manifesto += 1

        linha_manifesto = _selecionar_campos_manifesto(
            df_grupo=df_grupo,
            veiculo=veiculo,
            metricas=metricas,
            manifesto_id=manifesto_id,
            origem_etapa=origem_etapa,
        )
        manifestos.append(linha_manifesto)

        itens_df = _gerar_itens_manifesto(
            df_grupo=df_grupo,
            manifesto_id=manifesto_id,
            origem_etapa=origem_etapa,
            veiculo_tipo=str(veiculo["tipo"]),
        )
        itens_manifestos.append(itens_df)

        ids_aprovados.update(df_grupo["id_linha_pipeline"].astype(str).tolist())

        tentativas.append(
            _registrar_tentativa(
                origem_etapa=origem_etapa,
                chave_grupo=chave_str,
                qtd_itens=len(df_grupo),
                veiculo_testado=str(veiculo["tipo"]),
                aprovado=True,
                motivo="grupo_fechado_com_sucesso",
                metricas=metricas,
            )
        )

    if len(ids_aprovados) > 0:
        fila_remanescente = fila.loc[
            ~fila["id_linha_pipeline"].astype(str).isin(ids_aprovados)
        ].copy().reset_index(drop=True)
    else:
        fila_remanescente = fila.copy().reset_index(drop=True)

    df_manifestos = pd.DataFrame(manifestos)
    df_itens = pd.concat(itens_manifestos, ignore_index=True) if itens_manifestos else pd.DataFrame()
    df_tentativas = pd.DataFrame(tentativas)

    return df_manifestos, df_itens, df_tentativas, seq_manifesto, fila_remanescente


def executar_m4_manifestos_fechados(
    df_input_oficial_bloco_4: pd.DataFrame,
    df_veiculos_tratados: pd.DataFrame,
    rodada_id: str,
    data_base_roteirizacao: pd.Timestamp,
    caminhos_pipeline: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    M4 - Geração de manifestos fechados
    Regra:
    - Recebe SOMENTE o df_input_oficial_bloco_4
    - Tenta esgotar fechamentos naturais antes do M5
    """
    _validar_input(df_input_oficial_bloco_4, df_veiculos_tratados)
    fila, veiculos = _padronizar_bases(df_input_oficial_bloco_4, df_veiculos_tratados)

    fila["bucket_temporal_4a"] = _criar_bucket_temporal(fila)

    # 4A - mesmo cliente + bucket temporal
    df_manifestos_4a, df_itens_4a, df_tentativas_4a, seq_manifesto, fila_apos_4a = _processar_grupos(
        fila=fila,
        veiculos=veiculos,
        origem_etapa="4A_mesmo_cliente_bucket",
        colunas_groupby=[CHAVE_CLIENTE, "bucket_temporal_4a"],
        rodada_id=rodada_id,
        seq_manifesto_inicio=1,
        somente_multiplos=True,
    )

    # 4B - mesma parada natural no saldo remanescente
    df_manifestos_4b, df_itens_4b, df_tentativas_4b, seq_manifesto, fila_apos_4b = _processar_grupos(
        fila=fila_apos_4a,
        veiculos=veiculos,
        origem_etapa="4B_parada_natural",
        colunas_groupby=CHAVES_PARADA,
        rodada_id=rodada_id,
        seq_manifesto_inicio=seq_manifesto,
        somente_multiplos=True,
    )

    # 4C - varredura final do remanescente
    fila_varredura = fila_apos_4b.copy()
    manifestos_4c: List[pd.DataFrame] = []
    itens_4c: List[pd.DataFrame] = []
    tentativas_4c: List[pd.DataFrame] = []
    iteracoes_4c = 0

    while iteracoes_4c < MAX_ITERACOES_VARREDURA_4C:
        iteracoes_4c += 1
        df_manifestos_loop, df_itens_loop, df_tentativas_loop, seq_manifesto, fila_pos_loop = _processar_grupos(
            fila=fila_varredura,
            veiculos=veiculos,
            origem_etapa="4C_varredura_final",
            colunas_groupby=[CHAVE_CLIENTE],
            rodada_id=rodada_id,
            seq_manifesto_inicio=seq_manifesto,
            somente_multiplos=True,
        )

        tentativas_4c.append(df_tentativas_loop)

        if len(df_manifestos_loop) == 0:
            fila_varredura = fila_pos_loop
            break

        manifestos_4c.append(df_manifestos_loop)
        itens_4c.append(df_itens_loop)
        fila_varredura = fila_pos_loop

    frames_manifestos = [df for df in [df_manifestos_4a, df_manifestos_4b] if len(df) > 0]
    frames_itens = [df for df in [df_itens_4a, df_itens_4b] if len(df) > 0]
    frames_tentativas = [df for df in [df_tentativas_4a, df_tentativas_4b] if len(df) > 0]

    frames_manifestos.extend([df for df in manifestos_4c if len(df) > 0])
    frames_itens.extend([df for df in itens_4c if len(df) > 0])
    frames_tentativas.extend([df for df in tentativas_4c if len(df) > 0])

    df_manifestos_fechados_bloco_4 = (
        pd.concat(frames_manifestos, ignore_index=True) if frames_manifestos else pd.DataFrame()
    )
    df_itens_manifestos_fechados_bloco_4 = (
        pd.concat(frames_itens, ignore_index=True) if frames_itens else pd.DataFrame()
    )
    df_tentativas_fechamento_bloco_4 = (
        pd.concat(frames_tentativas, ignore_index=True) if frames_tentativas else pd.DataFrame()
    )
    df_remanescente_roteirizavel_bloco_4 = fila_varredura.copy().reset_index(drop=True)

    if len(df_manifestos_fechados_bloco_4) > 0:
        df_manifestos_fechados_bloco_4 = df_manifestos_fechados_bloco_4.sort_values(
            by=["origem_etapa", "manifesto_id"], ascending=[True, True]
        ).reset_index(drop=True)

    if len(df_itens_manifestos_fechados_bloco_4) > 0:
        df_itens_manifestos_fechados_bloco_4 = df_itens_manifestos_fechados_bloco_4.sort_values(
            by=["manifesto_id", "ranking_prioridade_operacional", "ranking_preliminar"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    resumo_por_veiculo = pd.DataFrame()
    if len(df_manifestos_fechados_bloco_4) > 0:
        resumo_por_veiculo = (
            df_manifestos_fechados_bloco_4.groupby("veiculo_tipo", dropna=False)
            .agg(
                qtd_manifestos=("manifesto_id", "nunique"),
                peso_total_kg=("peso_total_kg", "sum"),
                vol_total_m3=("vol_total_m3", "sum"),
                ocupacao_dominante_perc=("ocupacao_dominante_perc", "mean"),
                ocupacao_secundaria_perc=("ocupacao_secundaria_perc", "mean"),
            )
            .reset_index()
        )

    resumo_por_etapa = pd.DataFrame()
    if len(df_manifestos_fechados_bloco_4) > 0:
        resumo_por_etapa = (
            df_manifestos_fechados_bloco_4.groupby("origem_etapa", dropna=False)
            .agg(
                qtd_manifestos=("manifesto_id", "nunique"),
                peso_total_kg=("peso_total_kg", "sum"),
                vol_total_m3=("vol_total_m3", "sum"),
                qtd_paradas=("qtd_paradas", "sum"),
            )
            .reset_index()
        )

    resumo_m4 = {
        "modulo": "M4",
        "data_base_roteirizacao": pd.Timestamp(data_base_roteirizacao).isoformat(),
        "roteirizavel_entrada_m4": int(len(df_input_oficial_bloco_4)),
        "manifestos_fechados_gerados_m4": int(df_manifestos_fechados_bloco_4["manifesto_id"].nunique()) if len(df_manifestos_fechados_bloco_4) > 0 else 0,
        "itens_manifestados_m4": int(len(df_itens_manifestos_fechados_bloco_4)),
        "remanescente_roteirizavel_m4": int(len(df_remanescente_roteirizavel_bloco_4)),
        "clientes_macro_testados_4a": int(fila[[CHAVE_CLIENTE, "bucket_temporal_4a"]].drop_duplicates().shape[0]),
        "grupos_naturais_testados_4b": int(fila_apos_4a[CHAVES_PARADA].drop_duplicates().shape[0]) if len(fila_apos_4a) > 0 else 0,
        "iteracoes_varredura_4c_executadas": int(iteracoes_4c),
        "novos_manifestos_gerados_4c": int(
            sum(df["manifesto_id"].nunique() for df in manifestos_4c if len(df) > 0)
        ),
        "ocupacao_dominante_min_perc": round(OCUPACAO_DOMINANTE_MIN * 100, 2),
        "ocupacao_secundaria_min_perc": round(OCUPACAO_SECUNDARIA_MIN * 100, 2),
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    meta_m4 = {
        "resumo_m4": resumo_m4,
        "resumo_por_veiculo": _to_records(resumo_por_veiculo),
        "resumo_por_etapa": _to_records(resumo_por_etapa),
        "outputs_m4": {
            "df_manifestos_fechados_bloco_4": df_manifestos_fechados_bloco_4,
            "df_itens_manifestos_fechados_bloco_4": df_itens_manifestos_fechados_bloco_4,
            "df_tentativas_fechamento_bloco_4": df_tentativas_fechamento_bloco_4,
            "df_remanescente_roteirizavel_bloco_4": df_remanescente_roteirizavel_bloco_4,
        },
    }

    outputs = {
        "df_manifestos_fechados_bloco_4": df_manifestos_fechados_bloco_4,
        "df_itens_manifestos_fechados_bloco_4": df_itens_manifestos_fechados_bloco_4,
        "df_tentativas_fechamento_bloco_4": df_tentativas_fechamento_bloco_4,
        "df_remanescente_roteirizavel_bloco_4": df_remanescente_roteirizavel_bloco_4,
    }

    return outputs, meta_m4
