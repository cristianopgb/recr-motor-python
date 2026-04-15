from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


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

CHAVES_PARADA = ["destinatario", "cidade", "uf"]


def executar_m6_2_complemento_ocupacao(
    df_manifestos_base_m6: pd.DataFrame,
    df_itens_manifestos_base_m6: pd.DataFrame,
    df_remanescente_m5: pd.DataFrame,
    data_base_roteirizacao: datetime,
    tipo_roteirizacao: str,
    caminhos_pipeline: Optional[Dict[str, Any]] = None,
    ocupacao_alvo_perc: float = 85.0,
) -> Dict[str, Any]:
    df_manifestos = _normalizar_manifestos(df_manifestos_base_m6)
    df_itens = _normalizar_itens_manifestos(df_itens_manifestos_base_m6)
    df_remanescente = _normalizar_itens_remanescente(df_remanescente_m5)

    _validar_entrada(df_manifestos, df_itens, df_remanescente)

    df_remanescente_original = df_remanescente.copy()

    if "flag_otimizado_m6_2" not in df_itens.columns:
        df_itens["flag_otimizado_m6_2"] = False
    if "origem_item_m6_2" not in df_itens.columns:
        df_itens["origem_item_m6_2"] = "original_m6_1"

    df_manifestos = _recalcular_todos_manifestos(df_manifestos, df_itens)

    tentativas: List[Dict[str, Any]] = []
    movimentos_aceitos: List[Dict[str, Any]] = []

    manifestos_alvo = _selecionar_manifestos_alvo(df_manifestos, ocupacao_alvo_perc)

    for manifesto_id in manifestos_alvo:
        row_manifesto = df_manifestos.loc[df_manifestos["manifesto_id"] == manifesto_id].head(1)
        if row_manifesto.empty:
            continue

        manifesto = row_manifesto.iloc[0].to_dict()

        mesorregiao = _txt_norm(manifesto.get("mesorregiao_operacional", ""))
        itens_manifesto_atual = df_itens.loc[df_itens["manifesto_id"] == manifesto_id].copy()
        if itens_manifesto_atual.empty:
            continue

        rem_mesmo_meso = df_remanescente.loc[
            df_remanescente["mesorregiao_operacional"].astype(str).str.upper() == mesorregiao
        ].copy()

        if rem_mesmo_meso.empty:
            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "tipo_tentativa": "sem_remanescente_mesma_mesorregiao",
                    "criterio": None,
                    "aceito": False,
                    "motivo": "Não há remanescente do M5 na mesma mesorregião do manifesto.",
                }
            )
            continue

        cliente_dominante = _cliente_dominante(itens_manifesto_atual)
        cidade_dominante = _cidade_dominante(itens_manifesto_atual)

        grupos_cliente = _montar_grupos_remanescente(rem_mesmo_meso, chave="destinatario", valor_prioritario=cliente_dominante)
        grupos_cidade = _montar_grupos_remanescente(rem_mesmo_meso, chave="cidade", valor_prioritario=cidade_dominante)

        houve_movimento_neste_manifesto = False

        for criterio_nome, grupos in [
            ("mesmo_cliente", grupos_cliente),
            ("mesma_cidade", grupos_cidade),
        ]:
            if not grupos:
                tentativas.append(
                    {
                        "manifesto_id": manifesto_id,
                        "tipo_tentativa": "sem_grupos_elegiveis",
                        "criterio": criterio_nome,
                        "aceito": False,
                        "motivo": f"Não há grupos elegíveis por {criterio_nome} para este manifesto.",
                    }
                )
                continue

            for grupo in grupos:
                ids_grupo = grupo["ids"]
                itens_grupo = df_remanescente.loc[df_remanescente["id_linha_pipeline"].isin(ids_grupo)].copy()
                if itens_grupo.empty:
                    continue

                valido, motivo, comparativo = _simular_adicao_grupo(
                    manifesto=manifesto,
                    itens_manifesto=itens_manifesto_atual,
                    itens_grupo=itens_grupo,
                    ocupacao_alvo_perc=ocupacao_alvo_perc,
                )

                registro_tentativa = {
                    "manifesto_id": manifesto_id,
                    "tipo_tentativa": "adicao_grupo_remanescente_m5",
                    "criterio": criterio_nome,
                    "valor_criterio": grupo["valor"],
                    "quantidade_itens_grupo": int(len(itens_grupo)),
                    "ids_itens_grupo": [str(x) for x in ids_grupo],
                    "aceito": bool(valido),
                    "motivo": motivo,
                    **comparativo,
                }
                tentativas.append(registro_tentativa)

                if not valido:
                    continue

                houve_movimento_neste_manifesto = True

                itens_grupo_aplicar = itens_grupo.copy()
                itens_grupo_aplicar["manifesto_id"] = manifesto_id
                itens_grupo_aplicar["flag_otimizado_m6_2"] = True
                itens_grupo_aplicar["origem_item_m6_2"] = "adicionado_do_remanescente_m5"

                df_itens = pd.concat([df_itens, itens_grupo_aplicar], ignore_index=True)
                df_remanescente = df_remanescente.loc[~df_remanescente["id_linha_pipeline"].isin(ids_grupo)].copy()

                df_manifestos = _recalcular_manifesto_unico(df_manifestos, df_itens, manifesto_id)
                manifesto = df_manifestos.loc[df_manifestos["manifesto_id"] == manifesto_id].head(1).iloc[0].to_dict()
                itens_manifesto_atual = df_itens.loc[df_itens["manifesto_id"] == manifesto_id].copy()

                movimentos_aceitos.append(
                    {
                        "manifesto_id": manifesto_id,
                        "criterio": criterio_nome,
                        "valor_criterio": grupo["valor"],
                        "ids_itens_adicionados": [str(x) for x in ids_grupo],
                        "quantidade_itens_adicionados": int(len(ids_grupo)),
                        **comparativo,
                    }
                )

                ocupacao_depois = float(manifesto.get("ocupacao_dominante_perc", 0) or 0)
                if ocupacao_depois >= ocupacao_alvo_perc:
                    break

            manifesto_atualizado = df_manifestos.loc[df_manifestos["manifesto_id"] == manifesto_id].head(1)
            if not manifesto_atualizado.empty:
                ocupacao_depois = float(manifesto_atualizado.iloc[0].get("ocupacao_dominante_perc", 0) or 0)
                if ocupacao_depois >= ocupacao_alvo_perc:
                    break

        if not houve_movimento_neste_manifesto:
            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "tipo_tentativa": "sem_movimento_aceito",
                    "criterio": None,
                    "aceito": False,
                    "motivo": "Nenhum grupo do remanescente do M5 pôde ser adicionado respeitando as restrições.",
                }
            )

    df_manifestos = _recalcular_todos_manifestos(df_manifestos, df_itens)

    _validar_integridade_final(df_itens, df_remanescente, df_itens_manifestos_base_m6, df_remanescente_original)

    df_tentativas = pd.DataFrame(tentativas)
    df_movimentos_aceitos = pd.DataFrame(movimentos_aceitos)

    resumo_m6_2 = {
        "modulo": "M6.2",
        "data_base_roteirizacao": data_base_roteirizacao.isoformat(),
        "tipo_roteirizacao": tipo_roteirizacao,
        "ocupacao_alvo_perc": float(ocupacao_alvo_perc),
        "manifestos_base_total_m6_1": int(len(df_manifestos_base_m6)),
        "itens_manifestos_base_total_m6_1": int(len(df_itens_manifestos_base_m6)),
        "remanescente_m5_original_total": int(len(df_remanescente_original)),
        "manifestos_alvo_abaixo_ocupacao_alvo": int(len(manifestos_alvo)),
        "movimentos_aceitos_m6_2": int(len(df_movimentos_aceitos)),
        "tentativas_total_m6_2": int(len(df_tentativas)),
        "itens_manifestos_total_m6_2": int(len(df_itens)),
        "itens_remanescente_m6_2": int(len(df_remanescente)),
        "itens_adicionados_a_manifestos_m6_2": int(len(df_itens.loc[df_itens["flag_otimizado_m6_2"] == True])),
        "estrategia_m6_2": [
            "seleciona_manifestos_abaixo_de_85",
            "usa_apenas_remanescente_oficial_m5",
            "nunca_mistura_mesorregiao",
            "prioriza_mesmo_cliente",
            "depois_mesma_cidade",
            "nao_mexe_nos_demais_manifestos",
            "nao_gera_duplicidade",
        ],
        "caminhos_pipeline": caminhos_pipeline or {},
    }

    return {
        "outputs_m6_2": {
            "df_manifestos_m6_2": df_manifestos.reset_index(drop=True),
            "df_itens_manifestos_m6_2": df_itens.reset_index(drop=True),
            "df_remanescente_m6_2": df_remanescente.reset_index(drop=True),
            "df_remanescente_m5_original_m6_2": df_remanescente_original.reset_index(drop=True),
            "df_tentativas_m6_2": df_tentativas.reset_index(drop=True),
            "df_movimentos_aceitos_m6_2": df_movimentos_aceitos.reset_index(drop=True),
        },
        "resumo_m6_2": resumo_m6_2,
    }


def _normalizar_manifestos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _deduplicar_colunas(out)

    out["manifesto_id"] = _resolver_coluna(out, MANIFESTO_ID_ALIASES, obrigatoria=True).astype(str)
    out["mesorregiao_operacional"] = _resolver_coluna(out, MESORREGIAO_ALIASES, obrigatoria=False, default="").astype(str)
    out["perfil"] = _resolver_coluna(out, PERFIL_ALIASES, obrigatoria=False, default="").astype(str)
    out["capacidade_peso_kg"] = pd.to_numeric(_resolver_coluna(out, CAP_PESO_ALIASES, obrigatoria=False, default=np.nan), errors="coerce")
    out["capacidade_vol_m3"] = pd.to_numeric(_resolver_coluna(out, CAP_VOL_ALIASES, obrigatoria=False, default=np.nan), errors="coerce")
    out["max_entregas"] = pd.to_numeric(_resolver_coluna(out, MAX_ENTREGAS_ALIASES, obrigatoria=False, default=np.nan), errors="coerce")
    out["max_km_distancia"] = pd.to_numeric(_resolver_coluna(out, MAX_KM_ALIASES, obrigatoria=False, default=np.nan), errors="coerce")
    out["ocupacao_minima_perc"] = pd.to_numeric(_resolver_coluna(out, OCUP_MIN_ALIASES, obrigatoria=False, default=70), errors="coerce").fillna(70)
    out["ocupacao_maxima_perc"] = pd.to_numeric(_resolver_coluna(out, OCUP_MAX_ALIASES, obrigatoria=False, default=100), errors="coerce").fillna(100)

    return out.reset_index(drop=True)


def _normalizar_itens_manifestos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _deduplicar_colunas(out)

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


def _normalizar_itens_remanescente(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _deduplicar_colunas(out)

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


def _validar_entrada(
    df_manifestos: pd.DataFrame,
    df_itens: pd.DataFrame,
    df_remanescente: pd.DataFrame,
) -> None:
    if df_manifestos.empty:
        raise Exception("M6.2 recebeu df_manifestos_base_m6 vazio.")
    if df_itens.empty:
        raise Exception("M6.2 recebeu df_itens_manifestos_base_m6 vazio.")
    if "manifesto_id" not in df_manifestos.columns:
        raise Exception("M6.2 sem manifesto_id na base de manifestos.")
    if "manifesto_id" not in df_itens.columns:
        raise Exception("M6.2 sem manifesto_id na base de itens dos manifestos.")
    if "id_linha_pipeline" not in df_itens.columns:
        raise Exception("M6.2 sem id_linha_pipeline na base de itens dos manifestos.")
    if "id_linha_pipeline" not in df_remanescente.columns:
        raise Exception("M6.2 sem id_linha_pipeline no remanescente do M5.")
    if df_itens["id_linha_pipeline"].duplicated().any():
        raise Exception("M6.2 recebeu duplicidade de id_linha_pipeline nos itens dos manifestos.")
    if df_remanescente["id_linha_pipeline"].duplicated().any():
        raise Exception("M6.2 recebeu duplicidade de id_linha_pipeline no remanescente do M5.")


def _selecionar_manifestos_alvo(df_manifestos: pd.DataFrame, ocupacao_alvo_perc: float) -> List[str]:
    base = df_manifestos.copy()
    base["ocupacao_dominante_perc"] = pd.to_numeric(base.get("ocupacao_dominante_perc", np.nan), errors="coerce").fillna(0)
    base = base.loc[base["ocupacao_dominante_perc"] < ocupacao_alvo_perc].copy()
    base = base.sort_values(by=["ocupacao_dominante_perc", "qtd_entregas"], ascending=[True, True])
    return base["manifesto_id"].astype(str).tolist()


def _cliente_dominante(df_itens_manifesto: pd.DataFrame) -> str:
    if df_itens_manifesto.empty:
        return ""
    vc = df_itens_manifesto["destinatario"].astype(str).str.strip().value_counts()
    return str(vc.index[0]).strip() if len(vc) > 0 else ""


def _cidade_dominante(df_itens_manifesto: pd.DataFrame) -> str:
    if df_itens_manifesto.empty:
        return ""
    vc = df_itens_manifesto["cidade"].astype(str).str.strip().value_counts()
    return str(vc.index[0]).strip() if len(vc) > 0 else ""


def _montar_grupos_remanescente(
    df_remanescente: pd.DataFrame,
    chave: str,
    valor_prioritario: str,
) -> List[Dict[str, Any]]:
    if df_remanescente.empty or chave not in df_remanescente.columns:
        return []

    base = df_remanescente.copy()
    base[chave] = base[chave].astype(str).str.strip()

    grupos: List[Dict[str, Any]] = []
    for valor, sub in base.groupby(chave, dropna=False):
        valor = str(valor).strip()
        if valor == "":
            continue
        grupos.append(
            {
                "chave": chave,
                "valor": valor,
                "ids": sub["id_linha_pipeline"].astype(str).tolist(),
                "peso_total": float(sub["peso_calculado"].sum()),
                "vol_total": float(sub["vol_m3"].sum()),
                "qtd_itens": int(len(sub)),
                "prioritario": valor.upper() == str(valor_prioritario).strip().upper(),
            }
        )

    grupos = sorted(
        grupos,
        key=lambda x: (
            1 if x["prioritario"] else 0,
            x["qtd_itens"],
            x["peso_total"],
        ),
        reverse=True,
    )
    return grupos


def _simular_adicao_grupo(
    manifesto: Dict[str, Any],
    itens_manifesto: pd.DataFrame,
    itens_grupo: pd.DataFrame,
    ocupacao_alvo_perc: float,
) -> tuple[bool, str, Dict[str, Any]]:
    antes = _calcular_metricas_manifesto(manifesto, itens_manifesto)
    depois_df = pd.concat([itens_manifesto, itens_grupo], ignore_index=True)
    depois = _calcular_metricas_manifesto(manifesto, depois_df)

    comparativo = {
        "ocupacao_antes_perc": round(float(antes["ocupacao_dominante_perc"]), 4),
        "ocupacao_depois_perc": round(float(depois["ocupacao_dominante_perc"]), 4),
        "ganho_ocupacao_perc": round(float(depois["ocupacao_dominante_perc"] - antes["ocupacao_dominante_perc"]), 4),
        "distancia_antes_km": round(float(antes["distancia_total_km"]), 4),
        "distancia_depois_km": round(float(depois["distancia_total_km"]), 4),
        "delta_distancia_km": round(float(depois["distancia_total_km"] - antes["distancia_total_km"]), 4),
        "qtd_entregas_antes": int(antes["qtd_entregas"]),
        "qtd_entregas_depois": int(depois["qtd_entregas"]),
    }

    restricoes = _validar_restricoes_manifesto(manifesto, depois_df)
    if restricoes is not True:
        return False, str(restricoes), comparativo

    if depois["ocupacao_dominante_perc"] <= antes["ocupacao_dominante_perc"]:
        return False, "A adição do grupo não melhora a ocupação do manifesto.", comparativo

    if depois["distancia_total_km"] > antes["distancia_total_km"]:
        return False, "A adição do grupo piora a distância máxima do manifesto.", comparativo

    if depois["ocupacao_dominante_perc"] > float(manifesto.get("ocupacao_maxima_perc", 100) or 100):
        return False, "A adição do grupo ultrapassa a ocupação máxima do manifesto.", comparativo

    return True, "Grupo aceito no complemento de ocupação.", comparativo


def _calcular_metricas_manifesto(manifesto: Dict[str, Any], df_itens: pd.DataFrame) -> Dict[str, Any]:
    capacidade_peso = float(pd.to_numeric(manifesto.get("capacidade_peso_kg", np.nan), errors="coerce"))
    capacidade_vol = float(pd.to_numeric(manifesto.get("capacidade_vol_m3", np.nan), errors="coerce"))

    peso_total = float(pd.to_numeric(df_itens["peso_calculado"], errors="coerce").fillna(0).sum())
    vol_total = float(pd.to_numeric(df_itens["vol_m3"], errors="coerce").fillna(0).sum())
    distancia_total = float(pd.to_numeric(df_itens["distancia_rodoviaria_est_km"], errors="coerce").fillna(0).max())
    qtd_entregas = int(_contar_entregas(df_itens))

    ocup_peso = (peso_total / capacidade_peso * 100) if capacidade_peso > 0 else 0.0
    ocup_vol = (vol_total / capacidade_vol * 100) if capacidade_vol > 0 else 0.0
    ocup_dominante = max(ocup_peso, ocup_vol)

    return {
        "peso_total_kg": peso_total,
        "vol_total_m3": vol_total,
        "distancia_total_km": distancia_total,
        "qtd_entregas": qtd_entregas,
        "ocupacao_peso_perc": ocup_peso,
        "ocupacao_vol_perc": ocup_vol,
        "ocupacao_dominante_perc": ocup_dominante,
    }


def _validar_restricoes_manifesto(manifesto: Dict[str, Any], df_itens: pd.DataFrame) -> bool | str:
    metricas = _calcular_metricas_manifesto(manifesto, df_itens)

    capacidade_peso = float(pd.to_numeric(manifesto.get("capacidade_peso_kg", np.nan), errors="coerce"))
    capacidade_vol = float(pd.to_numeric(manifesto.get("capacidade_vol_m3", np.nan), errors="coerce"))
    max_entregas = int(pd.to_numeric(manifesto.get("max_entregas", np.nan), errors="coerce"))
    max_km = float(pd.to_numeric(manifesto.get("max_km_distancia", np.nan), errors="coerce"))
    perfil_manifesto = str(manifesto.get("perfil", "")).strip().upper()
    meso_manifesto = str(manifesto.get("mesorregiao_operacional", "")).strip().upper()

    if metricas["peso_total_kg"] > capacidade_peso:
        return "Excede capacidade de peso."
    if metricas["vol_total_m3"] > capacidade_vol:
        return "Excede capacidade de volume."
    if metricas["qtd_entregas"] > max_entregas:
        return "Excede máximo de entregas."
    if metricas["distancia_total_km"] > max_km:
        return "Excede raio/distância máxima."

    mesorregioes = set(df_itens["mesorregiao_operacional"].astype(str).str.strip().str.upper().tolist())
    mesorregioes.discard("")
    if len(mesorregioes) > 1:
        return "Mistura mais de uma mesorregião no manifesto."
    if len(mesorregioes) == 1 and list(mesorregioes)[0] != meso_manifesto:
        return "Grupo do remanescente está em mesorregião diferente do manifesto."

    restricoes = set(df_itens["restricao_veiculo"].astype(str).str.strip().str.upper().tolist())
    restricoes.discard("")
    if len(restricoes) > 1:
        return "Itens com restrições de veículo conflitantes."
    if len(restricoes) == 1 and list(restricoes)[0] != perfil_manifesto:
        return "Restrição de veículo do item é incompatível com o perfil do manifesto."

    return True


def _recalcular_manifesto_unico(
    df_manifestos: pd.DataFrame,
    df_itens: pd.DataFrame,
    manifesto_id: str,
) -> pd.DataFrame:
    out = df_manifestos.copy()
    idx = out.index[out["manifesto_id"].astype(str) == str(manifesto_id)]
    if len(idx) == 0:
        return out

    i = idx[0]
    manifesto = out.loc[i].to_dict()
    itens = df_itens.loc[df_itens["manifesto_id"].astype(str) == str(manifesto_id)].copy()
    metricas = _calcular_metricas_manifesto(manifesto, itens)

    out.loc[i, "peso_total_kg"] = metricas["peso_total_kg"]
    out.loc[i, "vol_total_m3"] = metricas["vol_total_m3"]
    out.loc[i, "distancia_total_km"] = metricas["distancia_total_km"]
    out.loc[i, "qtd_entregas"] = metricas["qtd_entregas"]
    out.loc[i, "ocupacao_peso_perc"] = metricas["ocupacao_peso_perc"]
    out.loc[i, "ocupacao_vol_perc"] = metricas["ocupacao_vol_perc"]
    out.loc[i, "ocupacao_dominante_perc"] = metricas["ocupacao_dominante_perc"]

    return out


def _recalcular_todos_manifestos(df_manifestos: pd.DataFrame, df_itens: pd.DataFrame) -> pd.DataFrame:
    out = df_manifestos.copy()
    for manifesto_id in out["manifesto_id"].astype(str).tolist():
        out = _recalcular_manifesto_unico(out, df_itens, manifesto_id)
    return out.reset_index(drop=True)


def _validar_integridade_final(
    df_itens_manifestos_final: pd.DataFrame,
    df_remanescente_final: pd.DataFrame,
    df_itens_manifestos_base_m6: pd.DataFrame,
    df_remanescente_original: pd.DataFrame,
) -> None:
    ids_base_manifestos = set(df_itens_manifestos_base_m6["id_linha_pipeline"].astype(str))
    ids_base_remanescente = set(df_remanescente_original["id_linha_pipeline"].astype(str))
    universo_base = ids_base_manifestos.union(ids_base_remanescente)

    ids_final_manifestos = set(df_itens_manifestos_final["id_linha_pipeline"].astype(str))
    ids_final_remanescente = set(df_remanescente_final["id_linha_pipeline"].astype(str))
    universo_final = ids_final_manifestos.union(ids_final_remanescente)

    if universo_base != universo_final:
        faltando = list(universo_base - universo_final)[:20]
        sobrando = list(universo_final - universo_base)[:20]
        raise Exception(
            f"M6.2 violou integridade do universo de itens. Faltando={faltando} | Sobrando={sobrando}"
        )

    if df_itens_manifestos_final["id_linha_pipeline"].duplicated().any():
        raise Exception("M6.2 gerou itens duplicados nos manifestos finais.")

    if df_remanescente_final["id_linha_pipeline"].duplicated().any():
        raise Exception("M6.2 gerou itens duplicados no remanescente final.")

    intersec = ids_final_manifestos.intersection(ids_final_remanescente)
    if intersec:
        raise Exception(f"M6.2 deixou itens ao mesmo tempo no manifesto e no remanescente: {list(intersec)[:20]}")


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
        raise Exception(f"Coluna obrigatória não encontrada. Esperado um dos aliases: {aliases}")
    return pd.Series([default] * len(df), index=df.index)


def _resolver_primeira_coluna_existente(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def _deduplicar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df.copy()
    return df.loc[:, ~df.columns.duplicated()].copy()


def _txt_norm(valor: Any) -> str:
    if pd.isna(valor):
        return ""
    return str(valor).strip().upper()


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
