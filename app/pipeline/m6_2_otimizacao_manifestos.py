from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CHAVES_PARADA = ["destinatario", "cidade", "uf"]

MANIFESTO_ID_ALIASES = [
    "manifesto_id",
    "id_manifesto",
    "id_manifesto_final",
    "id_manifesto_base",
    "manifesto",
    "manifesto_codigo",
]

MESORREGIAO_ALIASES = [
    "mesorregiao_operacional",
    "mesorregiao",
    "mesorregiao_destino",
    "mesoregiao",
]

PERFIL_ALIASES = [
    "perfil_veiculo",
    "perfil",
    "tipo_veiculo",
    "veiculo_perfil",
]

CAP_PESO_ALIASES = [
    "capacidade_peso_kg",
    "cap_peso_kg",
    "peso_capacidade_kg",
]

CAP_VOL_ALIASES = [
    "capacidade_vol_m3",
    "cap_vol_m3",
    "volume_capacidade_m3",
]

MAX_ENTREGAS_ALIASES = [
    "max_entregas",
    "maximo_entregas",
    "limite_entregas",
]

MAX_KM_ALIASES = [
    "max_km_distancia",
    "max_km",
    "raio_max_km",
    "distancia_maxima_km",
]

OCUP_MIN_ALIASES = [
    "ocupacao_minima_perc",
    "ocup_min_perc",
]

OCUP_MAX_ALIASES = [
    "ocupacao_maxima_perc",
    "ocup_max_perc",
]

DISTANCIA_ALIASES = [
    "distancia_rodoviaria_est_km",
    "distancia_km",
    "km",
    "km_est",
]

PESO_ITEM_ALIASES = [
    "peso_calculado",
    "peso_kg",
]

VOL_ITEM_ALIASES = [
    "vol_m3",
    "volume_m3",
]

RESTRICAO_VEICULO_ALIASES = [
    "restricao_veiculo",
    "perfil_restrito",
    "veiculo_restrito",
]

CLIENTE_ALIASES = [
    "destinatario",
    "cliente",
    "cliente_entrega",
]

CIDADE_ALIASES = [
    "cidade",
    "cidade_destino",
]

UF_ALIASES = [
    "uf",
    "estado",
]


@dataclass
class ManifestoStats:
    manifesto_id: str
    mesorregiao: str
    perfil: str
    capacidade_peso_kg: float
    capacidade_vol_m3: float
    max_entregas: int
    max_km_distancia: float
    ocupacao_minima_perc: float
    ocupacao_maxima_perc: float
    peso_total: float
    vol_total: float
    qtd_entregas: int
    distancia_total_km: float
    ocupacao_peso: float
    ocupacao_vol: float
    ocupacao_dominante: float
    ocupacao_secundaria: float


def executar_m6_2_otimizacao_manifestos(
    df_manifestos_base_m6: pd.DataFrame,
    df_itens_manifestos_base_m6: pd.DataFrame,
    df_pares_elegiveis_otimizacao_m6: pd.DataFrame,
    data_base_roteirizacao: datetime,
    df_veiculos_disponiveis: Optional[pd.DataFrame] = None,
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    df_manifestos = _normalizar_manifestos(df_manifestos_base_m6)
    df_itens = _normalizar_itens(df_itens_manifestos_base_m6)
    df_pares = _normalizar_pares(df_pares_elegiveis_otimizacao_m6)
    df_veiculos = _normalizar_veiculos(df_veiculos_disponiveis)

    _validar_entrada_m6_2(df_manifestos, df_itens, df_pares)

    if len(df_itens) == 0 or len(df_manifestos) == 0 or len(df_pares) == 0:
        df_manifestos_saida = _reconstruir_todos_manifestos(df_manifestos, df_itens, df_veiculos)
        return {
            "outputs_m6_2": {
                "df_manifestos_otimizados_m6_2": df_manifestos_saida,
                "df_itens_manifestos_otimizados_m6_2": df_itens,
                "df_movimentos_otimizacao_m6_2": pd.DataFrame(),
                "df_tentativas_otimizacao_m6_2": pd.DataFrame(),
                "df_estatisticas_antes_depois_m6_2": _montar_estatisticas_antes_depois(
                    df_manifestos_antes=df_manifestos,
                    df_manifestos_depois=df_manifestos_saida,
                    df_movimentos=pd.DataFrame(),
                    df_itens_antes=df_itens,
                    df_itens_depois=df_itens,
                ),
            },
            "resumo_m6_2": _montar_resumo_m6_2(
                df_manifestos_antes=df_manifestos,
                df_manifestos_depois=df_manifestos_saida,
                df_itens_antes=df_itens,
                df_itens_depois=df_itens,
                df_pares=df_pares,
                df_movimentos=pd.DataFrame(),
                df_tentativas=pd.DataFrame(),
                data_base_roteirizacao=data_base_roteirizacao,
                caminhos_pipeline=caminhos_pipeline,
            ),
        }

    tentativas: List[Dict[str, Any]] = []
    movimentos_aceitos: List[Dict[str, Any]] = []

    pares_ordenados = _ordenar_pares_para_otimizacao(df_pares)

    for _, row_par in pares_ordenados.iterrows():
        id_a = str(row_par["manifesto_id_a"])
        id_b = str(row_par["manifesto_id_b"])

        if id_a == id_b:
            continue

        manifestos_ids_atuais = set(df_itens["manifesto_id"].astype(str).tolist())
        if id_a not in manifestos_ids_atuais or id_b not in manifestos_ids_atuais:
            tentativas.append(
                _registrar_tentativa(
                    tipo_movimento="par_invalido_estado_atual",
                    manifesto_origem=id_a,
                    manifesto_destino=id_b,
                    aceito=False,
                    motivo="Um dos manifestos não existe mais no estado atual da otimização.",
                )
            )
            continue

        itens_a = df_itens.loc[df_itens["manifesto_id"] == id_a].copy()
        itens_b = df_itens.loc[df_itens["manifesto_id"] == id_b].copy()

        if len(itens_a) == 0 or len(itens_b) == 0:
            tentativas.append(
                _registrar_tentativa(
                    tipo_movimento="par_sem_itens",
                    manifesto_origem=id_a,
                    manifesto_destino=id_b,
                    aceito=False,
                    motivo="Um dos manifestos do par ficou sem itens no estado atual.",
                )
            )
            continue

        meso_a = _txt_norm(itens_a["mesorregiao_operacional"].iloc[0])
        meso_b = _txt_norm(itens_b["mesorregiao_operacional"].iloc[0])

        if meso_a != meso_b:
            tentativas.append(
                _registrar_tentativa(
                    tipo_movimento="mesorregiao_divergente",
                    manifesto_origem=id_a,
                    manifesto_destino=id_b,
                    aceito=False,
                    motivo="Par descartado porque os manifestos não pertencem à mesma mesorregião.",
                )
            )
            continue

        melhor_movimento = _buscar_melhor_movimento_no_par(
            manifesto_a_id=id_a,
            manifesto_b_id=id_b,
            df_manifestos_estado=df_manifestos,
            df_itens_estado=df_itens,
            df_veiculos=df_veiculos,
            tentativas=tentativas,
        )

        if melhor_movimento is None:
            continue

        df_itens, df_manifestos = _aplicar_movimento_aceito(
            df_itens_estado=df_itens,
            df_manifestos_estado=df_manifestos,
            movimento=melhor_movimento,
            df_veiculos=df_veiculos,
        )
        movimentos_aceitos.append(melhor_movimento)

    df_manifestos_final = _reconstruir_todos_manifestos(df_manifestos, df_itens, df_veiculos)

    _validar_integridade_final(
        df_itens_manifestos_base_m6=df_itens_manifestos_base_m6,
        df_itens_final=df_itens,
    )

    df_movimentos = pd.DataFrame(movimentos_aceitos)
    df_tentativas = pd.DataFrame(tentativas)
    df_estatisticas = _montar_estatisticas_antes_depois(
        df_manifestos_antes=df_manifestos_base_m6,
        df_manifestos_depois=df_manifestos_final,
        df_movimentos=df_movimentos,
        df_itens_antes=df_itens_manifestos_base_m6,
        df_itens_depois=df_itens,
    )

    return {
        "outputs_m6_2": {
            "df_manifestos_otimizados_m6_2": df_manifestos_final,
            "df_itens_manifestos_otimizados_m6_2": df_itens,
            "df_movimentos_otimizacao_m6_2": df_movimentos,
            "df_tentativas_otimizacao_m6_2": df_tentativas,
            "df_estatisticas_antes_depois_m6_2": df_estatisticas,
        },
        "resumo_m6_2": _montar_resumo_m6_2(
            df_manifestos_antes=df_manifestos_base_m6,
            df_manifestos_depois=df_manifestos_final,
            df_itens_antes=df_itens_manifestos_base_m6,
            df_itens_depois=df_itens,
            df_pares=df_pares,
            df_movimentos=df_movimentos,
            df_tentativas=df_tentativas,
            data_base_roteirizacao=data_base_roteirizacao,
            caminhos_pipeline=caminhos_pipeline,
        ),
    }


def _normalizar_manifestos(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=[
                "manifesto_id",
                "mesorregiao_operacional",
                "perfil",
                "capacidade_peso_kg",
                "capacidade_vol_m3",
                "max_entregas",
                "max_km_distancia",
                "ocupacao_minima_perc",
                "ocupacao_maxima_perc",
            ]
        )

    out = _deduplicar_colunas(df.copy())

    out["manifesto_id"] = _resolver_coluna(out, MANIFESTO_ID_ALIASES, obrigatoria=True).astype(str)
    out["mesorregiao_operacional"] = _resolver_coluna(out, MESORREGIAO_ALIASES, obrigatoria=False, default="").astype(str)
    out["perfil"] = _resolver_coluna(out, PERFIL_ALIASES, obrigatoria=False, default="").astype(str)
    out["capacidade_peso_kg"] = pd.to_numeric(_resolver_coluna(out, CAP_PESO_ALIASES, obrigatoria=False, default=np.nan), errors="coerce")
    out["capacidade_vol_m3"] = pd.to_numeric(_resolver_coluna(out, CAP_VOL_ALIASES, obrigatoria=False, default=np.nan), errors="coerce")
    out["max_entregas"] = pd.to_numeric(_resolver_coluna(out, MAX_ENTREGAS_ALIASES, obrigatoria=False, default=np.nan), errors="coerce")
    out["max_km_distancia"] = pd.to_numeric(_resolver_coluna(out, MAX_KM_ALIASES, obrigatoria=False, default=np.nan), errors="coerce")
    out["ocupacao_minima_perc"] = pd.to_numeric(_resolver_coluna(out, OCUP_MIN_ALIASES, obrigatoria=False, default=70), errors="coerce").fillna(70)
    out["ocupacao_maxima_perc"] = pd.to_numeric(_resolver_coluna(out, OCUP_MAX_ALIASES, obrigatoria=False, default=100), errors="coerce").fillna(100)

    base_cols = [
        "manifesto_id",
        "mesorregiao_operacional",
        "perfil",
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ocupacao_minima_perc",
        "ocupacao_maxima_perc",
    ]
    for col in base_cols:
        if col not in out.columns:
            out[col] = np.nan

    return out.reset_index(drop=True)


def _normalizar_itens(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=[
                "manifesto_id",
                "id_linha_pipeline",
                "mesorregiao_operacional",
                "destinatario",
                "cidade",
                "uf",
                "peso_calculado",
                "vol_m3",
                "distancia_rodoviaria_est_km",
                "restricao_veiculo",
            ]
        )

    out = _deduplicar_colunas(df.copy())

    out["manifesto_id"] = _resolver_coluna(out, MANIFESTO_ID_ALIASES, obrigatoria=True).astype(str)

    if "id_linha_pipeline" not in out.columns:
        out["id_linha_pipeline"] = out.index.astype(str)
    out["id_linha_pipeline"] = out["id_linha_pipeline"].astype(str)

    out["mesorregiao_operacional"] = _resolver_coluna(out, MESORREGIAO_ALIASES, obrigatoria=False, default="").astype(str)
    out["destinatario"] = _resolver_coluna(out, CLIENTE_ALIASES, obrigatoria=False, default="").astype(str)
    out["cidade"] = _resolver_coluna(out, CIDADE_ALIASES, obrigatoria=False, default="").astype(str)
    out["uf"] = _resolver_coluna(out, UF_ALIASES, obrigatoria=False, default="").astype(str)
    out["peso_calculado"] = pd.to_numeric(_resolver_coluna(out, PESO_ITEM_ALIASES, obrigatoria=True), errors="coerce").fillna(0)
    out["vol_m3"] = pd.to_numeric(_resolver_coluna(out, VOL_ITEM_ALIASES, obrigatoria=False, default=0), errors="coerce").fillna(0)
    out["distancia_rodoviaria_est_km"] = pd.to_numeric(_resolver_coluna(out, DISTANCIA_ALIASES, obrigatoria=False, default=0), errors="coerce").fillna(0)
    out["restricao_veiculo"] = _resolver_coluna(out, RESTRICAO_VEICULO_ALIASES, obrigatoria=False, default="").astype(str)

    return out.reset_index(drop=True)


def _normalizar_pares(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["manifesto_id_a", "manifesto_id_b", "score_prioridade"])

    out = _deduplicar_colunas(df.copy())

    col_a = _resolver_primeira_coluna_existente(out, ["manifesto_id_a", "manifesto_a", "id_manifesto_a", "origem_manifesto"])
    col_b = _resolver_primeira_coluna_existente(out, ["manifesto_id_b", "manifesto_b", "id_manifesto_b", "destino_manifesto"])

    if col_a is None or col_b is None:
        raise Exception(
            "O M6.2 não encontrou as colunas mínimas de pares elegíveis. "
            "Esperado algo como manifesto_id_a e manifesto_id_b."
        )

    out["manifesto_id_a"] = out[col_a].astype(str)
    out["manifesto_id_b"] = out[col_b].astype(str)

    score_col = _resolver_primeira_coluna_existente(
        out,
        ["score_prioridade", "score_criticidade", "score", "prioridade_score"],
    )
    if score_col is None:
        out["score_prioridade"] = 0.0
    else:
        out["score_prioridade"] = pd.to_numeric(out[score_col], errors="coerce").fillna(0)

    return out[["manifesto_id_a", "manifesto_id_b", "score_prioridade"]].drop_duplicates().reset_index(drop=True)


def _normalizar_veiculos(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=[
                "perfil",
                "capacidade_peso_kg",
                "capacidade_vol_m3",
                "max_entregas",
                "max_km_distancia",
                "ocupacao_minima_perc",
                "ocupacao_maxima_perc",
            ]
        )

    out = _deduplicar_colunas(df.copy())

    out["perfil"] = _resolver_coluna(out, PERFIL_ALIASES, obrigatoria=True).astype(str)
    out["capacidade_peso_kg"] = pd.to_numeric(_resolver_coluna(out, CAP_PESO_ALIASES, obrigatoria=False, default=np.nan), errors="coerce")
    out["capacidade_vol_m3"] = pd.to_numeric(_resolver_coluna(out, CAP_VOL_ALIASES, obrigatoria=False, default=np.nan), errors="coerce")
    out["max_entregas"] = pd.to_numeric(_resolver_coluna(out, MAX_ENTREGAS_ALIASES, obrigatoria=False, default=np.nan), errors="coerce")
    out["max_km_distancia"] = pd.to_numeric(_resolver_coluna(out, MAX_KM_ALIASES, obrigatoria=False, default=np.nan), errors="coerce")
    out["ocupacao_minima_perc"] = pd.to_numeric(_resolver_coluna(out, OCUP_MIN_ALIASES, obrigatoria=False, default=70), errors="coerce").fillna(70)
    out["ocupacao_maxima_perc"] = pd.to_numeric(_resolver_coluna(out, OCUP_MAX_ALIASES, obrigatoria=False, default=100), errors="coerce").fillna(100)

    cols = [
        "perfil",
        "capacidade_peso_kg",
        "capacidade_vol_m3",
        "max_entregas",
        "max_km_distancia",
        "ocupacao_minima_perc",
        "ocupacao_maxima_perc",
    ]
    return out[cols].drop_duplicates().reset_index(drop=True)


def _validar_entrada_m6_2(
    df_manifestos: pd.DataFrame,
    df_itens: pd.DataFrame,
    df_pares: pd.DataFrame,
) -> None:
    if "manifesto_id" not in df_manifestos.columns:
        raise Exception("M6.2 sem manifesto_id na base de manifestos.")
    if "manifesto_id" not in df_itens.columns:
        raise Exception("M6.2 sem manifesto_id na base de itens.")
    if "id_linha_pipeline" not in df_itens.columns:
        raise Exception("M6.2 sem id_linha_pipeline na base de itens.")
    if df_itens["id_linha_pipeline"].duplicated().any():
        qtd = int(df_itens["id_linha_pipeline"].duplicated().sum())
        raise Exception(f"M6.2 recebeu itens duplicados na entrada: {qtd} duplicidades.")
    if len(df_pares) == 0:
        return


def _validar_integridade_final(
    df_itens_manifestos_base_m6: pd.DataFrame,
    df_itens_final: pd.DataFrame,
) -> None:
    base_ids = set(df_itens_manifestos_base_m6["id_linha_pipeline"].astype(str))
    final_ids = set(df_itens_final["id_linha_pipeline"].astype(str))

    if base_ids != final_ids:
        faltando = list(base_ids - final_ids)[:10]
        sobrando = list(final_ids - base_ids)[:10]
        raise Exception(
            "M6.2 violou a integridade do pool consolidado do M6.1. "
            f"Faltando={faltando} | Sobrando={sobrando}"
        )

    if df_itens_final["id_linha_pipeline"].duplicated().any():
        qtd = int(df_itens_final["id_linha_pipeline"].duplicated().sum())
        raise Exception(f"M6.2 gerou itens duplicados no resultado final: {qtd} duplicidades.")


def _ordenar_pares_para_otimizacao(df_pares: pd.DataFrame) -> pd.DataFrame:
    out = df_pares.copy()
    out["score_prioridade"] = pd.to_numeric(out["score_prioridade"], errors="coerce").fillna(0)
    return out.sort_values(by=["score_prioridade"], ascending=False).reset_index(drop=True)


def _buscar_melhor_movimento_no_par(
    manifesto_a_id: str,
    manifesto_b_id: str,
    df_manifestos_estado: pd.DataFrame,
    df_itens_estado: pd.DataFrame,
    df_veiculos: pd.DataFrame,
    tentativas: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    itens_a = df_itens_estado.loc[df_itens_estado["manifesto_id"] == manifesto_a_id].copy()
    itens_b = df_itens_estado.loc[df_itens_estado["manifesto_id"] == manifesto_b_id].copy()

    candidatos_aceitos: List[Dict[str, Any]] = []

    for origem, destino in [(manifesto_a_id, manifesto_b_id), (manifesto_b_id, manifesto_a_id)]:
        mov = _simular_absorcao_total(
            manifesto_origem=origem,
            manifesto_destino=destino,
            df_manifestos_estado=df_manifestos_estado,
            df_itens_estado=df_itens_estado,
            df_veiculos=df_veiculos,
        )
        tentativas.append(mov)
        if mov["aceito"]:
            candidatos_aceitos.append(mov)

    grupos_cliente_a = _listar_grupos_moviveis(itens_a, chave="destinatario")
    grupos_cliente_b = _listar_grupos_moviveis(itens_b, chave="destinatario")

    for grupo in grupos_cliente_a:
        mov = _simular_transferencia_grupo(
            manifesto_origem=manifesto_a_id,
            manifesto_destino=manifesto_b_id,
            ids_grupo=grupo["ids"],
            regra="mesmo_cliente",
            df_manifestos_estado=df_manifestos_estado,
            df_itens_estado=df_itens_estado,
            df_veiculos=df_veiculos,
        )
        tentativas.append(mov)
        if mov["aceito"]:
            candidatos_aceitos.append(mov)

    for grupo in grupos_cliente_b:
        mov = _simular_transferencia_grupo(
            manifesto_origem=manifesto_b_id,
            manifesto_destino=manifesto_a_id,
            ids_grupo=grupo["ids"],
            regra="mesmo_cliente",
            df_manifestos_estado=df_manifestos_estado,
            df_itens_estado=df_itens_estado,
            df_veiculos=df_veiculos,
        )
        tentativas.append(mov)
        if mov["aceito"]:
            candidatos_aceitos.append(mov)

    grupos_cidade_a = _listar_grupos_moviveis(itens_a, chave="cidade")
    grupos_cidade_b = _listar_grupos_moviveis(itens_b, chave="cidade")

    for grupo in grupos_cidade_a:
        mov = _simular_transferencia_grupo(
            manifesto_origem=manifesto_a_id,
            manifesto_destino=manifesto_b_id,
            ids_grupo=grupo["ids"],
            regra="mesma_cidade",
            df_manifestos_estado=df_manifestos_estado,
            df_itens_estado=df_itens_estado,
            df_veiculos=df_veiculos,
        )
        tentativas.append(mov)
        if mov["aceito"]:
            candidatos_aceitos.append(mov)

    for grupo in grupos_cidade_b:
        mov = _simular_transferencia_grupo(
            manifesto_origem=manifesto_b_id,
            manifesto_destino=manifesto_a_id,
            ids_grupo=grupo["ids"],
            regra="mesma_cidade",
            df_manifestos_estado=df_manifestos_estado,
            df_itens_estado=df_itens_estado,
            df_veiculos=df_veiculos,
        )
        tentativas.append(mov)
        if mov["aceito"]:
            candidatos_aceitos.append(mov)

    trocas_cliente = _listar_trocas_possiveis(itens_a, itens_b, chave="destinatario")
    for troca in trocas_cliente:
        mov = _simular_troca_grupos(
            manifesto_a=manifesto_a_id,
            manifesto_b=manifesto_b_id,
            ids_a=troca["ids_a"],
            ids_b=troca["ids_b"],
            regra="troca_mesmo_cliente",
            df_manifestos_estado=df_manifestos_estado,
            df_itens_estado=df_itens_estado,
            df_veiculos=df_veiculos,
        )
        tentativas.append(mov)
        if mov["aceito"]:
            candidatos_aceitos.append(mov)

    trocas_cidade = _listar_trocas_possiveis(itens_a, itens_b, chave="cidade")
    for troca in trocas_cidade:
        mov = _simular_troca_grupos(
            manifesto_a=manifesto_a_id,
            manifesto_b=manifesto_b_id,
            ids_a=troca["ids_a"],
            ids_b=troca["ids_b"],
            regra="troca_mesma_cidade",
            df_manifestos_estado=df_manifestos_estado,
            df_itens_estado=df_itens_estado,
            df_veiculos=df_veiculos,
        )
        tentativas.append(mov)
        if mov["aceito"]:
            candidatos_aceitos.append(mov)

    if len(candidatos_aceitos) == 0:
        return None

    candidatos_aceitos = sorted(
        candidatos_aceitos,
        key=lambda x: (
            x["delta_manifestos_reduzidos"],
            x["ganho_ocupacao_media_perc"],
            x["ganho_distancia_total_km"],
            x["ganho_balanceamento_perc"],
        ),
        reverse=True,
    )
    return candidatos_aceitos[0]


def _simular_absorcao_total(
    manifesto_origem: str,
    manifesto_destino: str,
    df_manifestos_estado: pd.DataFrame,
    df_itens_estado: pd.DataFrame,
    df_veiculos: pd.DataFrame,
) -> Dict[str, Any]:
    itens_origem = df_itens_estado.loc[df_itens_estado["manifesto_id"] == manifesto_origem].copy()
    itens_destino = df_itens_estado.loc[df_itens_estado["manifesto_id"] == manifesto_destino].copy()

    if len(itens_origem) == 0 or len(itens_destino) == 0:
        return _registrar_tentativa(
            tipo_movimento="absorcao_total",
            manifesto_origem=manifesto_origem,
            manifesto_destino=manifesto_destino,
            aceito=False,
            motivo="Origem ou destino sem itens.",
        )

    meso_origem = _txt_norm(itens_origem["mesorregiao_operacional"].iloc[0])
    meso_destino = _txt_norm(itens_destino["mesorregiao_operacional"].iloc[0])
    if meso_origem != meso_destino:
        return _registrar_tentativa(
            tipo_movimento="absorcao_total",
            manifesto_origem=manifesto_origem,
            manifesto_destino=manifesto_destino,
            aceito=False,
            motivo="Mesorregiões diferentes.",
        )

    before_a = _calcular_stats_manifesto(manifesto_origem, df_manifestos_estado, itens_origem, df_veiculos)
    before_b = _calcular_stats_manifesto(manifesto_destino, df_manifestos_estado, itens_destino, df_veiculos)

    if not _stats_base_validos(before_a, before_b):
        return _registrar_tentativa(
            tipo_movimento="absorcao_total",
            manifesto_origem=manifesto_origem,
            manifesto_destino=manifesto_destino,
            aceito=False,
            motivo="Par descartado porque as métricas-base do manifesto estão incompletas/NaN.",
        )

    itens_resultado = pd.concat([itens_destino, itens_origem], ignore_index=True)
    melhor_destino = _escolher_melhor_configuracao_para_manifesto(
        manifesto_id=manifesto_destino,
        df_manifestos_estado=df_manifestos_estado,
        itens_manifesto=itens_resultado,
        df_veiculos=df_veiculos,
        perfis_preferenciais=[before_b.perfil, before_a.perfil],
    )

    if melhor_destino is None:
        return _registrar_tentativa(
            tipo_movimento="absorcao_total",
            manifesto_origem=manifesto_origem,
            manifesto_destino=manifesto_destino,
            aceito=False,
            motivo="A absorção total viola capacidade, entregas, raio, ocupação ou restrição de veículo.",
        )

    ok_melhoria, metricas = _comparar_par_antes_depois(
        before_a=before_a,
        before_b=before_b,
        after_a=None,
        after_b=melhor_destino,
    )
    if not ok_melhoria:
        return _registrar_tentativa(
            tipo_movimento="absorcao_total",
            manifesto_origem=manifesto_origem,
            manifesto_destino=manifesto_destino,
            aceito=False,
            motivo="A absorção total não melhorou o par em ocupação/distância dentro da regra aceita.",
            extra=metricas,
        )

    return _registrar_tentativa(
        tipo_movimento="absorcao_total",
        manifesto_origem=manifesto_origem,
        manifesto_destino=manifesto_destino,
        aceito=True,
        motivo="Absorção total aceita.",
        extra={
            **metricas,
            "ids_movidos": itens_origem["id_linha_pipeline"].astype(str).tolist(),
            "perfil_final_destino": melhor_destino.perfil,
            "manifesto_removido": manifesto_origem,
            "manifesto_mantido": manifesto_destino,
            "delta_manifestos_reduzidos": 1,
        },
    )


def _simular_transferencia_grupo(
    manifesto_origem: str,
    manifesto_destino: str,
    ids_grupo: List[str],
    regra: str,
    df_manifestos_estado: pd.DataFrame,
    df_itens_estado: pd.DataFrame,
    df_veiculos: pd.DataFrame,
) -> Dict[str, Any]:
    itens_origem = df_itens_estado.loc[df_itens_estado["manifesto_id"] == manifesto_origem].copy()
    itens_destino = df_itens_estado.loc[df_itens_estado["manifesto_id"] == manifesto_destino].copy()

    if len(itens_origem) == 0 or len(itens_destino) == 0:
        return _registrar_tentativa(
            tipo_movimento=f"transferencia_{regra}",
            manifesto_origem=manifesto_origem,
            manifesto_destino=manifesto_destino,
            aceito=False,
            motivo="Origem ou destino sem itens.",
        )

    grupo = itens_origem.loc[itens_origem["id_linha_pipeline"].astype(str).isin([str(x) for x in ids_grupo])].copy()
    if len(grupo) == 0 or len(grupo) == len(itens_origem):
        return _registrar_tentativa(
            tipo_movimento=f"transferencia_{regra}",
            manifesto_origem=manifesto_origem,
            manifesto_destino=manifesto_destino,
            aceito=False,
            motivo="Grupo inválido para transferência parcial.",
        )

    before_a = _calcular_stats_manifesto(manifesto_origem, df_manifestos_estado, itens_origem, df_veiculos)
    before_b = _calcular_stats_manifesto(manifesto_destino, df_manifestos_estado, itens_destino, df_veiculos)

    if not _stats_base_validos(before_a, before_b):
        return _registrar_tentativa(
            tipo_movimento=f"transferencia_{regra}",
            manifesto_origem=manifesto_origem,
            manifesto_destino=manifesto_destino,
            aceito=False,
            motivo="Par descartado porque as métricas-base do manifesto estão incompletas/NaN.",
        )

    itens_origem_novo = itens_origem.loc[~itens_origem["id_linha_pipeline"].isin(grupo["id_linha_pipeline"])].copy()
    itens_destino_novo = pd.concat([itens_destino, grupo], ignore_index=True)

    after_a = _escolher_melhor_configuracao_para_manifesto(
        manifesto_id=manifesto_origem,
        df_manifestos_estado=df_manifestos_estado,
        itens_manifesto=itens_origem_novo,
        df_veiculos=df_veiculos,
        perfis_preferenciais=[before_a.perfil],
    )
    after_b = _escolher_melhor_configuracao_para_manifesto(
        manifesto_id=manifesto_destino,
        df_manifestos_estado=df_manifestos_estado,
        itens_manifesto=itens_destino_novo,
        df_veiculos=df_veiculos,
        perfis_preferenciais=[before_b.perfil, before_a.perfil],
    )

    if after_a is None or after_b is None:
        return _registrar_tentativa(
            tipo_movimento=f"transferencia_{regra}",
            manifesto_origem=manifesto_origem,
            manifesto_destino=manifesto_destino,
            aceito=False,
            motivo="Transferência parcial viola restrições no doador ou receptor.",
        )

    ok_melhoria, metricas = _comparar_par_antes_depois(
        before_a=before_a,
        before_b=before_b,
        after_a=after_a,
        after_b=after_b,
    )
    if not ok_melhoria:
        return _registrar_tentativa(
            tipo_movimento=f"transferencia_{regra}",
            manifesto_origem=manifesto_origem,
            manifesto_destino=manifesto_destino,
            aceito=False,
            motivo="Transferência parcial sem ganho líquido aceito.",
            extra=metricas,
        )

    return _registrar_tentativa(
        tipo_movimento=f"transferencia_{regra}",
        manifesto_origem=manifesto_origem,
        manifesto_destino=manifesto_destino,
        aceito=True,
        motivo="Transferência parcial aceita.",
        extra={
            **metricas,
            "ids_movidos": grupo["id_linha_pipeline"].astype(str).tolist(),
            "perfil_final_origem": after_a.perfil,
            "perfil_final_destino": after_b.perfil,
            "delta_manifestos_reduzidos": 0,
        },
    )


def _simular_troca_grupos(
    manifesto_a: str,
    manifesto_b: str,
    ids_a: List[str],
    ids_b: List[str],
    regra: str,
    df_manifestos_estado: pd.DataFrame,
    df_itens_estado: pd.DataFrame,
    df_veiculos: pd.DataFrame,
) -> Dict[str, Any]:
    itens_a = df_itens_estado.loc[df_itens_estado["manifesto_id"] == manifesto_a].copy()
    itens_b = df_itens_estado.loc[df_itens_estado["manifesto_id"] == manifesto_b].copy()

    grupo_a = itens_a.loc[itens_a["id_linha_pipeline"].astype(str).isin([str(x) for x in ids_a])].copy()
    grupo_b = itens_b.loc[itens_b["id_linha_pipeline"].astype(str).isin([str(x) for x in ids_b])].copy()

    if len(grupo_a) == 0 or len(grupo_b) == 0:
        return _registrar_tentativa(
            tipo_movimento=regra,
            manifesto_origem=manifesto_a,
            manifesto_destino=manifesto_b,
            aceito=False,
            motivo="Troca inválida sem grupos válidos.",
        )

    before_a = _calcular_stats_manifesto(manifesto_a, df_manifestos_estado, itens_a, df_veiculos)
    before_b = _calcular_stats_manifesto(manifesto_b, df_manifestos_estado, itens_b, df_veiculos)

    if not _stats_base_validos(before_a, before_b):
        return _registrar_tentativa(
            tipo_movimento=regra,
            manifesto_origem=manifesto_a,
            manifesto_destino=manifesto_b,
            aceito=False,
            motivo="Par descartado porque as métricas-base do manifesto estão incompletas/NaN.",
        )

    itens_a_novo = pd.concat(
        [
            itens_a.loc[~itens_a["id_linha_pipeline"].isin(grupo_a["id_linha_pipeline"])],
            grupo_b,
        ],
        ignore_index=True,
    )
    itens_b_novo = pd.concat(
        [
            itens_b.loc[~itens_b["id_linha_pipeline"].isin(grupo_b["id_linha_pipeline"])],
            grupo_a,
        ],
        ignore_index=True,
    )

    after_a = _escolher_melhor_configuracao_para_manifesto(
        manifesto_id=manifesto_a,
        df_manifestos_estado=df_manifestos_estado,
        itens_manifesto=itens_a_novo,
        df_veiculos=df_veiculos,
        perfis_preferenciais=[before_a.perfil, before_b.perfil],
    )
    after_b = _escolher_melhor_configuracao_para_manifesto(
        manifesto_id=manifesto_b,
        df_manifestos_estado=df_manifestos_estado,
        itens_manifesto=itens_b_novo,
        df_veiculos=df_veiculos,
        perfis_preferenciais=[before_b.perfil, before_a.perfil],
    )

    if after_a is None or after_b is None:
        return _registrar_tentativa(
            tipo_movimento=regra,
            manifesto_origem=manifesto_a,
            manifesto_destino=manifesto_b,
            aceito=False,
            motivo="Troca violou restrições em pelo menos um dos dois manifestos.",
        )

    ok_melhoria, metricas = _comparar_par_antes_depois(
        before_a=before_a,
        before_b=before_b,
        after_a=after_a,
        after_b=after_b,
    )
    if not ok_melhoria:
        return _registrar_tentativa(
            tipo_movimento=regra,
            manifesto_origem=manifesto_a,
            manifesto_destino=manifesto_b,
            aceito=False,
            motivo="Troca sem ganho líquido aceito.",
            extra=metricas,
        )

    return _registrar_tentativa(
        tipo_movimento=regra,
        manifesto_origem=manifesto_a,
        manifesto_destino=manifesto_b,
        aceito=True,
        motivo="Troca aceita.",
        extra={
            **metricas,
            "ids_movidos_a_para_b": grupo_a["id_linha_pipeline"].astype(str).tolist(),
            "ids_movidos_b_para_a": grupo_b["id_linha_pipeline"].astype(str).tolist(),
            "perfil_final_a": after_a.perfil,
            "perfil_final_b": after_b.perfil,
            "delta_manifestos_reduzidos": 0,
        },
    )


def _aplicar_movimento_aceito(
    df_itens_estado: pd.DataFrame,
    df_manifestos_estado: pd.DataFrame,
    movimento: Dict[str, Any],
    df_veiculos: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_itens = df_itens_estado.copy()
    out_manifestos = df_manifestos_estado.copy()

    tipo = movimento["tipo_movimento"]
    origem = str(movimento["manifesto_origem"])
    destino = str(movimento["manifesto_destino"])

    manifestos_afetados = {origem, destino}

    if tipo == "absorcao_total":
        ids_movidos = set([str(x) for x in movimento.get("ids_movidos", [])])
        out_itens.loc[out_itens["id_linha_pipeline"].astype(str).isin(ids_movidos), "manifesto_id"] = destino
        out_manifestos = out_manifestos.loc[out_manifestos["manifesto_id"] != origem].copy()
        manifestos_afetados = {destino}

    elif tipo.startswith("transferencia_"):
        ids_movidos = set([str(x) for x in movimento.get("ids_movidos", [])])
        out_itens.loc[out_itens["id_linha_pipeline"].astype(str).isin(ids_movidos), "manifesto_id"] = destino

    elif tipo.startswith("troca_"):
        ids_a_para_b = set([str(x) for x in movimento.get("ids_movidos_a_para_b", [])])
        ids_b_para_a = set([str(x) for x in movimento.get("ids_movidos_b_para_a", [])])
        out_itens.loc[out_itens["id_linha_pipeline"].astype(str).isin(ids_a_para_b), "manifesto_id"] = destino
        out_itens.loc[out_itens["id_linha_pipeline"].astype(str).isin(ids_b_para_a), "manifesto_id"] = origem

    out_manifestos = _reconstruir_manifestos_afetados(
        df_manifestos_estado=out_manifestos,
        df_itens_estado=out_itens,
        df_veiculos=df_veiculos,
        manifestos_afetados=manifestos_afetados,
    )

    return out_itens.reset_index(drop=True), out_manifestos.reset_index(drop=True)


def _calcular_stats_manifesto(
    manifesto_id: str,
    df_manifestos_estado: pd.DataFrame,
    itens_manifesto: pd.DataFrame,
    df_veiculos: pd.DataFrame,
) -> ManifestoStats:
    manifesto_row = df_manifestos_estado.loc[df_manifestos_estado["manifesto_id"] == manifesto_id].head(1)
    if len(manifesto_row) == 0:
        raise Exception(f"M6.2 não encontrou cabeçalho para o manifesto {manifesto_id}.")

    manifesto_row = manifesto_row.iloc[0].to_dict()

    perfil_atual = _txt_norm(manifesto_row.get("perfil", ""))
    cap_peso = _num_safe(manifesto_row.get("capacidade_peso_kg", np.nan))
    cap_vol = _num_safe(manifesto_row.get("capacidade_vol_m3", np.nan))
    max_entregas = int(_num_safe(manifesto_row.get("max_entregas", np.nan), default=999999))
    max_km = _num_safe(manifesto_row.get("max_km_distancia", np.nan), default=np.nan)
    ocup_min = _num_safe(manifesto_row.get("ocupacao_minima_perc", 70), default=70)
    ocup_max = _num_safe(manifesto_row.get("ocupacao_maxima_perc", 100), default=100)
    mesorregiao = _txt_norm(manifesto_row.get("mesorregiao_operacional", ""))

    if (pd.isna(cap_peso) or pd.isna(cap_vol) or pd.isna(max_km) or perfil_atual == "") and len(df_veiculos) > 0:
        match = df_veiculos.loc[df_veiculos["perfil"].astype(str).str.upper() == perfil_atual]
        if len(match) > 0:
            v = match.iloc[0]
            if pd.isna(cap_peso):
                cap_peso = _num_safe(v["capacidade_peso_kg"])
            if pd.isna(cap_vol):
                cap_vol = _num_safe(v["capacidade_vol_m3"])
            if pd.isna(max_km):
                max_km = _num_safe(v["max_km_distancia"], default=np.nan)
            if pd.isna(max_entregas):
                max_entregas = int(_num_safe(v["max_entregas"], default=999999))
            ocup_min = _num_safe(v["ocupacao_minima_perc"], default=ocup_min)
            ocup_max = _num_safe(v["ocupacao_maxima_perc"], default=ocup_max)

    peso_total = float(pd.to_numeric(itens_manifesto["peso_calculado"], errors="coerce").fillna(0).sum())
    vol_total = float(pd.to_numeric(itens_manifesto["vol_m3"], errors="coerce").fillna(0).sum())
    qtd_entregas = int(_contar_entregas(itens_manifesto))
    distancia_total_km = float(pd.to_numeric(itens_manifesto["distancia_rodoviaria_est_km"], errors="coerce").fillna(0).max())

    ocup_peso = peso_total / cap_peso if pd.notna(cap_peso) and cap_peso > 0 else np.nan
    ocup_vol = vol_total / cap_vol if pd.notna(cap_vol) and cap_vol > 0 else np.nan

    if pd.notna(ocup_peso) and pd.notna(ocup_vol):
        ocup_dominante = float(max(ocup_peso, ocup_vol))
        ocup_secundaria = float(min(ocup_peso, ocup_vol))
    elif pd.notna(ocup_peso):
        ocup_dominante = float(ocup_peso)
        ocup_secundaria = float(ocup_peso)
    elif pd.notna(ocup_vol):
        ocup_dominante = float(ocup_vol)
        ocup_secundaria = float(ocup_vol)
    else:
        ocup_dominante = np.nan
        ocup_secundaria = np.nan

    return ManifestoStats(
        manifesto_id=str(manifesto_id),
        mesorregiao=mesorregiao,
        perfil=perfil_atual,
        capacidade_peso_kg=float(cap_peso) if pd.notna(cap_peso) else np.nan,
        capacidade_vol_m3=float(cap_vol) if pd.notna(cap_vol) else np.nan,
        max_entregas=int(max_entregas),
        max_km_distancia=float(max_km) if pd.notna(max_km) else np.nan,
        ocupacao_minima_perc=float(ocup_min),
        ocupacao_maxima_perc=float(ocup_max),
        peso_total=peso_total,
        vol_total=vol_total,
        qtd_entregas=qtd_entregas,
        distancia_total_km=distancia_total_km,
        ocupacao_peso=float(ocup_peso) if pd.notna(ocup_peso) else np.nan,
        ocupacao_vol=float(ocup_vol) if pd.notna(ocup_vol) else np.nan,
        ocupacao_dominante=float(ocup_dominante) if pd.notna(ocup_dominante) else np.nan,
        ocupacao_secundaria=float(ocup_secundaria) if pd.notna(ocup_secundaria) else np.nan,
    )


def _escolher_melhor_configuracao_para_manifesto(
    manifesto_id: str,
    df_manifestos_estado: pd.DataFrame,
    itens_manifesto: pd.DataFrame,
    df_veiculos: pd.DataFrame,
    perfis_preferenciais: Optional[List[str]] = None,
) -> Optional[ManifestoStats]:
    if len(itens_manifesto) == 0:
        return None

    base = df_manifestos_estado.loc[df_manifestos_estado["manifesto_id"] == manifesto_id].head(1)
    if len(base) == 0:
        return None

    base = base.iloc[0].to_dict()

    opcoes: List[Dict[str, Any]] = []
    perfil_atual = _txt_norm(base.get("perfil", ""))

    opcoes.append(
        {
            "perfil": perfil_atual,
            "capacidade_peso_kg": _num_safe(base.get("capacidade_peso_kg", np.nan)),
            "capacidade_vol_m3": _num_safe(base.get("capacidade_vol_m3", np.nan)),
            "max_entregas": int(_num_safe(base.get("max_entregas", 999999), default=999999)),
            "max_km_distancia": _num_safe(base.get("max_km_distancia", np.nan), default=np.nan),
            "ocupacao_minima_perc": _num_safe(base.get("ocupacao_minima_perc", 70), default=70),
            "ocupacao_maxima_perc": _num_safe(base.get("ocupacao_maxima_perc", 100), default=100),
        }
    )

    if len(df_veiculos) > 0:
        for _, row in df_veiculos.iterrows():
            opcoes.append(
                {
                    "perfil": _txt_norm(row["perfil"]),
                    "capacidade_peso_kg": _num_safe(row["capacidade_peso_kg"]),
                    "capacidade_vol_m3": _num_safe(row["capacidade_vol_m3"]),
                    "max_entregas": int(_num_safe(row["max_entregas"], default=999999)),
                    "max_km_distancia": _num_safe(row["max_km_distancia"], default=np.nan),
                    "ocupacao_minima_perc": _num_safe(row["ocupacao_minima_perc"], default=70),
                    "ocupacao_maxima_perc": _num_safe(row["ocupacao_maxima_perc"], default=100),
                }
            )

    opcoes_df = pd.DataFrame(opcoes).drop_duplicates(
        subset=["perfil", "capacidade_peso_kg", "capacidade_vol_m3", "max_entregas", "max_km_distancia"]
    )
    if len(opcoes_df) == 0:
        return None

    candidatos_validos: List[ManifestoStats] = []

    for _, opcao in opcoes_df.iterrows():
        stats = _calcular_stats_com_configuracao(
            manifesto_id=manifesto_id,
            itens_manifesto=itens_manifesto,
            mesorregiao=_txt_norm(base.get("mesorregiao_operacional", "")),
            perfil=_txt_norm(opcao["perfil"]),
            capacidade_peso_kg=_num_safe(opcao["capacidade_peso_kg"]),
            capacidade_vol_m3=_num_safe(opcao["capacidade_vol_m3"]),
            max_entregas=int(_num_safe(opcao["max_entregas"], default=999999)),
            max_km_distancia=_num_safe(opcao["max_km_distancia"], default=np.nan),
            ocupacao_minima_perc=_num_safe(opcao["ocupacao_minima_perc"], default=70),
            ocupacao_maxima_perc=_num_safe(opcao["ocupacao_maxima_perc"], default=100),
        )
        if _configuracao_valida_para_itens(stats, itens_manifesto):
            candidatos_validos.append(stats)

    if len(candidatos_validos) == 0:
        return None

    prefs = [_txt_norm(x) for x in (perfis_preferenciais or []) if _txt_norm(x) != ""]
    candidatos_validos = sorted(
        candidatos_validos,
        key=lambda s: (
            1 if s.perfil in prefs else 0,
            -abs((s.ocupacao_dominante * 100) - 85) if pd.notna(s.ocupacao_dominante) else -9999,
            -(s.capacidade_peso_kg if pd.notna(s.capacidade_peso_kg) else -9999),
        ),
        reverse=True,
    )
    return candidatos_validos[0]


def _calcular_stats_com_configuracao(
    manifesto_id: str,
    itens_manifesto: pd.DataFrame,
    mesorregiao: str,
    perfil: str,
    capacidade_peso_kg: float,
    capacidade_vol_m3: float,
    max_entregas: int,
    max_km_distancia: float,
    ocupacao_minima_perc: float,
    ocupacao_maxima_perc: float,
) -> ManifestoStats:
    peso_total = float(pd.to_numeric(itens_manifesto["peso_calculado"], errors="coerce").fillna(0).sum())
    vol_total = float(pd.to_numeric(itens_manifesto["vol_m3"], errors="coerce").fillna(0).sum())
    qtd_entregas = int(_contar_entregas(itens_manifesto))
    distancia_total_km = float(pd.to_numeric(itens_manifesto["distancia_rodoviaria_est_km"], errors="coerce").fillna(0).max())

    ocup_peso = peso_total / capacidade_peso_kg if pd.notna(capacidade_peso_kg) and capacidade_peso_kg > 0 else np.nan
    ocup_vol = vol_total / capacidade_vol_m3 if pd.notna(capacidade_vol_m3) and capacidade_vol_m3 > 0 else np.nan

    if pd.notna(ocup_peso) and pd.notna(ocup_vol):
        ocup_dominante = float(max(ocup_peso, ocup_vol))
        ocup_secundaria = float(min(ocup_peso, ocup_vol))
    elif pd.notna(ocup_peso):
        ocup_dominante = float(ocup_peso)
        ocup_secundaria = float(ocup_peso)
    elif pd.notna(ocup_vol):
        ocup_dominante = float(ocup_vol)
        ocup_secundaria = float(ocup_vol)
    else:
        ocup_dominante = np.nan
        ocup_secundaria = np.nan

    return ManifestoStats(
        manifesto_id=str(manifesto_id),
        mesorregiao=mesorregiao,
        perfil=perfil,
        capacidade_peso_kg=float(capacidade_peso_kg) if pd.notna(capacidade_peso_kg) else np.nan,
        capacidade_vol_m3=float(capacidade_vol_m3) if pd.notna(capacidade_vol_m3) else np.nan,
        max_entregas=int(max_entregas),
        max_km_distancia=float(max_km_distancia) if pd.notna(max_km_distancia) else np.nan,
        ocupacao_minima_perc=float(ocupacao_minima_perc),
        ocupacao_maxima_perc=float(ocupacao_maxima_perc),
        peso_total=peso_total,
        vol_total=vol_total,
        qtd_entregas=qtd_entregas,
        distancia_total_km=distancia_total_km,
        ocupacao_peso=float(ocup_peso) if pd.notna(ocup_peso) else np.nan,
        ocupacao_vol=float(ocup_vol) if pd.notna(ocup_vol) else np.nan,
        ocupacao_dominante=float(ocup_dominante) if pd.notna(ocup_dominante) else np.nan,
        ocupacao_secundaria=float(ocup_secundaria) if pd.notna(ocup_secundaria) else np.nan,
    )


def _configuracao_valida_para_itens(stats: ManifestoStats, itens_manifesto: pd.DataFrame) -> bool:
    if len(itens_manifesto) == 0:
        return False

    mesorregioes = {_txt_norm(x) for x in itens_manifesto["mesorregiao_operacional"].astype(str).tolist()}
    mesorregioes.discard("")
    if len(mesorregioes) > 1:
        return False

    restricoes = {_txt_norm(x) for x in itens_manifesto["restricao_veiculo"].astype(str).tolist()}
    restricoes.discard("")
    if len(restricoes) > 1:
        return False
    if len(restricoes) == 1 and stats.perfil != list(restricoes)[0]:
        return False

    if pd.isna(stats.capacidade_peso_kg) or stats.capacidade_peso_kg <= 0:
        return False
    if pd.isna(stats.capacidade_vol_m3) or stats.capacidade_vol_m3 <= 0:
        return False
    if pd.isna(stats.max_km_distancia) or stats.max_km_distancia <= 0:
        return False
    if stats.peso_total > stats.capacidade_peso_kg + 1e-9:
        return False
    if stats.vol_total > stats.capacidade_vol_m3 + 1e-9:
        return False
    if stats.qtd_entregas > stats.max_entregas:
        return False
    if stats.distancia_total_km > stats.max_km_distancia + 1e-9:
        return False
    if pd.isna(stats.ocupacao_dominante):
        return False

    ocup_min = stats.ocupacao_minima_perc / 100.0
    ocup_max = stats.ocupacao_maxima_perc / 100.0

    if stats.ocupacao_dominante < ocup_min - 1e-9:
        return False
    if stats.ocupacao_dominante > ocup_max + 1e-9:
        return False

    return True


def _stats_base_validos(before_a: ManifestoStats, before_b: ManifestoStats) -> bool:
    campos = [
        before_a.ocupacao_dominante,
        before_b.ocupacao_dominante,
        before_a.distancia_total_km,
        before_b.distancia_total_km,
    ]
    return all(pd.notna(x) for x in campos)


def _comparar_par_antes_depois(
    before_a: ManifestoStats,
    before_b: ManifestoStats,
    after_a: Optional[ManifestoStats],
    after_b: Optional[ManifestoStats],
) -> Tuple[bool, Dict[str, Any]]:
    if not _stats_base_validos(before_a, before_b):
        metricas = {
            "ocupacao_media_antes_perc": 0.0,
            "ocupacao_media_depois_perc": 0.0,
            "ganho_ocupacao_media_perc": 0.0,
            "distancia_total_antes_km": 0.0,
            "distancia_total_depois_km": 0.0,
            "ganho_distancia_total_km": 0.0,
            "balanceamento_antes_perc": 0.0,
            "balanceamento_depois_perc": 0.0,
            "ganho_balanceamento_perc": 0.0,
            "delta_manifestos_reduzidos": 0,
        }
        return False, metricas

    before_ocup_media = float(np.mean([before_a.ocupacao_dominante, before_b.ocupacao_dominante]) * 100)
    before_dist_total = float(before_a.distancia_total_km + before_b.distancia_total_km)
    before_balanceamento = float(min(before_a.ocupacao_dominante, before_b.ocupacao_dominante) * 100)

    after_stats = [x for x in [after_a, after_b] if x is not None]
    if len(after_stats) == 0:
        metricas = {
            "ocupacao_media_antes_perc": round(before_ocup_media, 4),
            "ocupacao_media_depois_perc": 0.0,
            "ganho_ocupacao_media_perc": 0.0,
            "distancia_total_antes_km": round(before_dist_total, 4),
            "distancia_total_depois_km": 0.0,
            "ganho_distancia_total_km": 0.0,
            "balanceamento_antes_perc": round(before_balanceamento, 4),
            "balanceamento_depois_perc": 0.0,
            "ganho_balanceamento_perc": 0.0,
            "delta_manifestos_reduzidos": 0,
        }
        return False, metricas

    campos_after = [x.ocupacao_dominante for x in after_stats] + [x.distancia_total_km for x in after_stats]
    if not all(pd.notna(x) for x in campos_after):
        metricas = {
            "ocupacao_media_antes_perc": round(before_ocup_media, 4),
            "ocupacao_media_depois_perc": 0.0,
            "ganho_ocupacao_media_perc": 0.0,
            "distancia_total_antes_km": round(before_dist_total, 4),
            "distancia_total_depois_km": 0.0,
            "ganho_distancia_total_km": 0.0,
            "balanceamento_antes_perc": round(before_balanceamento, 4),
            "balanceamento_depois_perc": 0.0,
            "ganho_balanceamento_perc": 0.0,
            "delta_manifestos_reduzidos": 0,
        }
        return False, metricas

    after_ocup_media = float(np.mean([x.ocupacao_dominante for x in after_stats]) * 100)
    after_dist_total = float(sum([x.distancia_total_km for x in after_stats]))
    after_balanceamento = float(min([x.ocupacao_dominante for x in after_stats]) * 100)

    delta_ocup = float(after_ocup_media - before_ocup_media)
    delta_dist_reducao = float(before_dist_total - after_dist_total)
    delta_balanceamento = float(after_balanceamento - before_balanceamento)

    melhora_aceita = (
        ((delta_ocup > 1e-9) and (delta_dist_reducao >= -1e-9))
        or ((delta_dist_reducao > 1e-9) and (delta_ocup >= -1e-9))
    )

    metricas = {
        "ocupacao_media_antes_perc": round(before_ocup_media, 4),
        "ocupacao_media_depois_perc": round(after_ocup_media, 4),
        "ganho_ocupacao_media_perc": round(delta_ocup, 4),
        "distancia_total_antes_km": round(before_dist_total, 4),
        "distancia_total_depois_km": round(after_dist_total, 4),
        "ganho_distancia_total_km": round(delta_dist_reducao, 4),
        "balanceamento_antes_perc": round(before_balanceamento, 4),
        "balanceamento_depois_perc": round(after_balanceamento, 4),
        "ganho_balanceamento_perc": round(delta_balanceamento, 4),
        "delta_manifestos_reduzidos": 0,
    }
    return melhora_aceita, metricas


def _reconstruir_manifestos_afetados(
    df_manifestos_estado: pd.DataFrame,
    df_itens_estado: pd.DataFrame,
    df_veiculos: pd.DataFrame,
    manifestos_afetados: set[str],
) -> pd.DataFrame:
    out = df_manifestos_estado.copy()
    ids_atuais = set(df_itens_estado["manifesto_id"].astype(str).unique().tolist())

    for manifesto_id in list(manifestos_afetados):
        if manifesto_id not in ids_atuais:
            out = out.loc[out["manifesto_id"] != manifesto_id].copy()
            continue

        itens = df_itens_estado.loc[df_itens_estado["manifesto_id"] == manifesto_id].copy()
        best = _escolher_melhor_configuracao_para_manifesto(
            manifesto_id=manifesto_id,
            df_manifestos_estado=out,
            itens_manifesto=itens,
            df_veiculos=df_veiculos,
            perfis_preferenciais=[
                _txt_norm(out.loc[out["manifesto_id"] == manifesto_id, "perfil"].head(1).iloc[0])
                if len(out.loc[out["manifesto_id"] == manifesto_id]) > 0
                else ""
            ],
        )
        if best is None:
            raise Exception(
                f"M6.2 deixou o manifesto {manifesto_id} em estado inválido após aplicação de movimento."
            )

        out = _atualizar_header_manifesto(out, best)

    return out.reset_index(drop=True)


def _reconstruir_todos_manifestos(
    df_manifestos_estado: pd.DataFrame,
    df_itens_estado: pd.DataFrame,
    df_veiculos: pd.DataFrame,
) -> pd.DataFrame:
    out = df_manifestos_estado.copy()
    ids_atuais = sorted(df_itens_estado["manifesto_id"].astype(str).unique().tolist())

    out = out.loc[out["manifesto_id"].astype(str).isin(ids_atuais)].copy()

    for manifesto_id in ids_atuais:
        itens = df_itens_estado.loc[df_itens_estado["manifesto_id"] == manifesto_id].copy()
        best = _escolher_melhor_configuracao_para_manifesto(
            manifesto_id=manifesto_id,
            df_manifestos_estado=out,
            itens_manifesto=itens,
            df_veiculos=df_veiculos,
            perfis_preferenciais=[
                _txt_norm(out.loc[out["manifesto_id"] == manifesto_id, "perfil"].head(1).iloc[0])
                if len(out.loc[out["manifesto_id"] == manifesto_id]) > 0
                else ""
            ],
        )
        if best is None:
            raise Exception(
                f"M6.2 não conseguiu reconstruir o manifesto {manifesto_id} no fechamento final do módulo."
            )
        out = _atualizar_header_manifesto(out, best)

    return out.reset_index(drop=True)


def _atualizar_header_manifesto(df_manifestos_estado: pd.DataFrame, best: ManifestoStats) -> pd.DataFrame:
    out = df_manifestos_estado.copy()
    idx = out.index[out["manifesto_id"] == best.manifesto_id]
    if len(idx) == 0:
        raise Exception(f"M6.2 não encontrou cabeçalho para atualizar o manifesto {best.manifesto_id}.")
    i = idx[0]

    out.loc[i, "manifesto_id"] = best.manifesto_id
    out.loc[i, "mesorregiao_operacional"] = best.mesorregiao
    out.loc[i, "perfil"] = best.perfil
    out.loc[i, "capacidade_peso_kg"] = best.capacidade_peso_kg
    out.loc[i, "capacidade_vol_m3"] = best.capacidade_vol_m3
    out.loc[i, "max_entregas"] = best.max_entregas
    out.loc[i, "max_km_distancia"] = best.max_km_distancia
    out.loc[i, "ocupacao_minima_perc"] = best.ocupacao_minima_perc
    out.loc[i, "ocupacao_maxima_perc"] = best.ocupacao_maxima_perc
    out.loc[i, "peso_total_kg"] = best.peso_total
    out.loc[i, "vol_total_m3"] = best.vol_total
    out.loc[i, "qtd_entregas"] = best.qtd_entregas
    out.loc[i, "distancia_total_km"] = best.distancia_total_km
    out.loc[i, "ocupacao_peso_perc"] = round(best.ocupacao_peso * 100, 4) if pd.notna(best.ocupacao_peso) else np.nan
    out.loc[i, "ocupacao_vol_perc"] = round(best.ocupacao_vol * 100, 4) if pd.notna(best.ocupacao_vol) else np.nan
    out.loc[i, "ocupacao_dominante_perc"] = round(best.ocupacao_dominante * 100, 4) if pd.notna(best.ocupacao_dominante) else np.nan
    out.loc[i, "ocupacao_secundaria_perc"] = round(best.ocupacao_secundaria * 100, 4) if pd.notna(best.ocupacao_secundaria) else np.nan
    return out


def _listar_grupos_moviveis(df_itens_manifesto: pd.DataFrame, chave: str) -> List[Dict[str, Any]]:
    if chave not in df_itens_manifesto.columns or len(df_itens_manifesto) == 0:
        return []

    grupos: List[Dict[str, Any]] = []
    serie = df_itens_manifesto[chave].astype(str).str.strip()
    valores = [x for x in sorted(serie.unique().tolist()) if x != ""]
    for valor in valores:
        sub = df_itens_manifesto.loc[serie == valor].copy()
        if len(sub) == 0 or len(sub) == len(df_itens_manifesto):
            continue
        grupos.append(
            {
                "chave": chave,
                "valor": valor,
                "ids": sub["id_linha_pipeline"].astype(str).tolist(),
                "peso_total": float(sub["peso_calculado"].sum()),
                "vol_total": float(sub["vol_m3"].sum()),
            }
        )

    grupos = sorted(
        grupos,
        key=lambda x: (x["peso_total"], x["vol_total"], len(x["ids"])),
        reverse=True,
    )
    return grupos


def _listar_trocas_possiveis(
    itens_a: pd.DataFrame,
    itens_b: pd.DataFrame,
    chave: str,
) -> List[Dict[str, Any]]:
    if chave not in itens_a.columns or chave not in itens_b.columns:
        return []

    vals_a = set([str(x).strip() for x in itens_a[chave].astype(str).tolist() if str(x).strip() != ""])
    vals_b = set([str(x).strip() for x in itens_b[chave].astype(str).tolist() if str(x).strip() != ""])
    comuns = sorted(vals_a.intersection(vals_b))
    trocas: List[Dict[str, Any]] = []

    for valor in comuns:
        grupo_a = itens_a.loc[itens_a[chave].astype(str).str.strip() == valor].copy()
        grupo_b = itens_b.loc[itens_b[chave].astype(str).str.strip() == valor].copy()
        if len(grupo_a) == 0 or len(grupo_b) == 0:
            continue
        trocas.append(
            {
                "valor": valor,
                "ids_a": grupo_a["id_linha_pipeline"].astype(str).tolist(),
                "ids_b": grupo_b["id_linha_pipeline"].astype(str).tolist(),
                "peso_total_a": float(grupo_a["peso_calculado"].sum()),
                "peso_total_b": float(grupo_b["peso_calculado"].sum()),
            }
        )

    trocas = sorted(
        trocas,
        key=lambda x: abs(x["peso_total_a"] - x["peso_total_b"]),
    )
    return trocas


def _montar_estatisticas_antes_depois(
    df_manifestos_antes: pd.DataFrame,
    df_manifestos_depois: pd.DataFrame,
    df_movimentos: pd.DataFrame,
    df_itens_antes: pd.DataFrame,
    df_itens_depois: pd.DataFrame,
) -> pd.DataFrame:
    def _safe_mean(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns or len(df) == 0:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())

    reg = {
        "manifestos_antes": int(len(df_manifestos_antes)),
        "manifestos_depois": int(len(df_manifestos_depois)),
        "itens_antes": int(len(df_itens_antes)),
        "itens_depois": int(len(df_itens_depois)),
        "ocupacao_media_antes_perc": round(_safe_mean(df_manifestos_antes, "ocupacao_dominante_perc"), 4),
        "ocupacao_media_depois_perc": round(_safe_mean(df_manifestos_depois, "ocupacao_dominante_perc"), 4),
        "distancia_media_antes_km": round(_safe_mean(df_manifestos_antes, "distancia_total_km"), 4),
        "distancia_media_depois_km": round(_safe_mean(df_manifestos_depois, "distancia_total_km"), 4),
        "movimentos_aceitos": int(len(df_movimentos)),
    }
    reg["ganho_ocupacao_media_perc"] = round(reg["ocupacao_media_depois_perc"] - reg["ocupacao_media_antes_perc"], 4)
    reg["ganho_distancia_media_km"] = round(reg["distancia_media_antes_km"] - reg["distancia_media_depois_km"], 4)
    return pd.DataFrame([reg])


def _montar_resumo_m6_2(
    df_manifestos_antes: pd.DataFrame,
    df_manifestos_depois: pd.DataFrame,
    df_itens_antes: pd.DataFrame,
    df_itens_depois: pd.DataFrame,
    df_pares: pd.DataFrame,
    df_movimentos: pd.DataFrame,
    df_tentativas: pd.DataFrame,
    data_base_roteirizacao: datetime,
    caminhos_pipeline: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    movimentos_aceitos = int(len(df_movimentos))
    tentativas_total = int(len(df_tentativas))

    def _contar_tipo(df: pd.DataFrame, prefixo: str) -> int:
        if len(df) == 0 or "tipo_movimento" not in df.columns:
            return 0
        return int(df["tipo_movimento"].astype(str).str.startswith(prefixo).sum())

    ocup_antes = float(pd.to_numeric(df_manifestos_antes.get("ocupacao_dominante_perc", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if len(df_manifestos_antes) > 0 else 0.0
    ocup_depois = float(pd.to_numeric(df_manifestos_depois.get("ocupacao_dominante_perc", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if len(df_manifestos_depois) > 0 else 0.0
    dist_antes = float(pd.to_numeric(df_manifestos_antes.get("distancia_total_km", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(df_manifestos_antes) > 0 else 0.0
    dist_depois = float(pd.to_numeric(df_manifestos_depois.get("distancia_total_km", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(df_manifestos_depois) > 0 else 0.0

    return {
        "modulo": "M6.2",
        "data_base_roteirizacao": data_base_roteirizacao.isoformat(),
        "manifestos_entrada": int(len(df_manifestos_antes)),
        "manifestos_saida": int(len(df_manifestos_depois)),
        "itens_entrada": int(len(df_itens_antes)),
        "itens_saida": int(len(df_itens_depois)),
        "pares_avaliados": int(len(df_pares)),
        "tentativas_total": tentativas_total,
        "movimentos_aceitos": movimentos_aceitos,
        "absorcoes_totais_aceitas": _contar_tipo(df_movimentos, "absorcao_total"),
        "transferencias_aceitas": int(
            _contar_tipo(df_movimentos, "transferencia_mesmo_cliente")
            + _contar_tipo(df_movimentos, "transferencia_mesma_cidade")
        ),
        "trocas_aceitas": int(
            _contar_tipo(df_movimentos, "troca_mesmo_cliente")
            + _contar_tipo(df_movimentos, "troca_mesma_cidade")
        ),
        "ganho_ocupacao_media_perc": round(ocup_depois - ocup_antes, 4),
        "ganho_distancia_total_km": round(dist_antes - dist_depois, 4),
        "remanescente_gerado": 0,
        "estrategia_m6_2": [
            "pares_somente_mesma_mesorregiao",
            "sem_recomposicao_global",
            "sem_gerar_remanescente",
            "prioriza_absorcao_total",
            "prioriza_mesmo_cliente",
            "prioriza_mesma_cidade",
            "aceita_so_ganho_liquido_ocupacao_distancia",
        ],
        "caminhos_pipeline": caminhos_pipeline or {},
    }


def _registrar_tentativa(
    tipo_movimento: str,
    manifesto_origem: str,
    manifesto_destino: str,
    aceito: bool,
    motivo: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reg = {
        "tipo_movimento": tipo_movimento,
        "manifesto_origem": str(manifesto_origem),
        "manifesto_destino": str(manifesto_destino),
        "aceito": bool(aceito),
        "motivo": motivo,
    }
    if extra:
        reg.update(extra)

    reg.setdefault("delta_manifestos_reduzidos", 0)
    reg.setdefault("ganho_ocupacao_media_perc", 0.0)
    reg.setdefault("ganho_distancia_total_km", 0.0)
    reg.setdefault("ganho_balanceamento_perc", 0.0)
    return reg


def _resolver_coluna(
    df: pd.DataFrame,
    aliases: List[str],
    obrigatoria: bool,
    default: Any = None,
) -> pd.Series:
    col = _resolver_primeira_coluna_existente(df, aliases)
    if col is not None:
        return df[col]
    if obrigatoria:
        raise Exception("M6.2 não encontrou coluna obrigatória. Aliases esperados: " + ", ".join(aliases))
    return pd.Series([default] * len(df), index=df.index)


def _resolver_primeira_coluna_existente(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def _deduplicar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df.columns) == 0:
        return df.copy()
    if not df.columns.duplicated().any():
        return df.copy()
    return df.loc[:, ~df.columns.duplicated()].copy()


def _txt_norm(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def _num_safe(x: Any, default: float = np.nan) -> float:
    val = pd.to_numeric(x, errors="coerce")
    return float(val) if pd.notna(val) else default


def _contar_entregas(df: pd.DataFrame) -> int:
    cols = [c for c in CHAVES_PARADA if c in df.columns]
    if len(cols) < 3:
        return int(len(df))
    chave = (
        df["destinatario"].astype(str).str.strip().str.upper()
        + "|"
        + df["cidade"].astype(str).str.strip().str.upper()
        + "|"
        + df["uf"].astype(str).str.strip().str.upper()
    )
    return int(chave.nunique())
