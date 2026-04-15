from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
    "mesorregiao",
]

SUBREGIAO_ALIASES = [
    "subregiao_operacional",
    "subregiao",
    "sub_regiao",
    "subregiao_destino",
    "subregiao",
]

PERFIL_ALIASES = [
    "veiculo_perfil",
    "veiculo_tipo",
    "perfil_veiculo",
    "perfil",
    "tipo_veiculo",
    "veiculo",
]

CAP_PESO_ALIASES = [
    "capacidade_peso_kg_veiculo",
    "capacidade_peso_kg",
    "cap_peso_kg",
    "peso_capacidade_kg",
]

CAP_VOL_ALIASES = [
    "capacidade_vol_m3_veiculo",
    "capacidade_vol_m3",
    "cap_vol_m3",
    "volume_capacidade_m3",
]

MAX_ENTREGAS_ALIASES = [
    "max_entregas_veiculo",
    "max_entregas",
    "maximo_entregas",
    "limite_entregas",
]

MAX_KM_ALIASES = [
    "max_km_distancia_veiculo",
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

OCUP_DOMINANTE_ALIASES = [
    "ocupacao_base_antes_m6",
    "ocupacao_dominante_perc",
    "ocupacao_perc",
    "ocupacao",
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

AGENDADO_ALIASES = [
    "agendada",
    "flag_agendada",
    "agenda",
]

FOLGA_ALIASES = [
    "folga_dias",
    "folga",
    "dias_folga",
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

    manifestos_alvo = _selecionar_manifestos_alvo(df_manifestos, ocupacao_alvo_perc)

    tentativas: List[Dict[str, Any]] = []
    movimentos_aceitos: List[Dict[str, Any]] = []

    for manifesto_id in manifestos_alvo:
        row_manifesto = df_manifestos.loc[df_manifestos["manifesto_id"] == manifesto_id].head(1)
        if row_manifesto.empty:
            continue

        manifesto = row_manifesto.iloc[0].to_dict()
        itens_manifesto_atual = df_itens.loc[df_itens["manifesto_id"] == manifesto_id].copy()
        if itens_manifesto_atual.empty:
            continue

        meso_manifesto = _txt_norm(manifesto.get("mesorregiao_operacional", ""))
        rem_mesmo_meso = df_remanescente.loc[
            df_remanescente["mesorregiao_operacional"].astype(str).str.upper() == meso_manifesto
        ].copy()

        if rem_mesmo_meso.empty:
            tentativas.append(
                {
                    "manifesto_id": manifesto_id,
                    "tipo_tentativa": "sem_remanescente_mesma_mesorregiao",
                    "nivel_hierarquia": None,
                    "aceito": False,
                    "motivo": "Não há remanescente do M5 na mesma mesorregião do manifesto.",
                }
            )
            continue

        houve_movimento_neste_manifesto = False

        contexto_manifesto = {
            "cliente_dominante": _cliente_dominante(itens_manifesto_atual),
            "cidade_dominante": _cidade_dominante(itens_manifesto_atual),
            "subregiao_dominante": _subregiao_dominante(itens_manifesto_atual),
            "mesorregiao": meso_manifesto,
        }

        niveis_hierarquia = [
            ("mesmo_cliente", "destinatario"),
            ("mesma_cidade", "cidade"),
            ("mesma_subregiao", "subregiao_operacional"),
            ("mesma_mesorregiao", "mesorregiao_operacional"),
        ]

        for nome_nivel, coluna_nivel in niveis_hierarquia:
            candidatos = _selecionar_candidatos_por_hierarquia(
                rem_mesmo_meso=rem_mesmo_meso,
                itens_manifesto=itens_manifesto_atual,
                nome_nivel=nome_nivel,
                coluna_nivel=coluna_nivel,
                contexto_manifesto=contexto_manifesto,
            )

            if candidatos.empty:
                tentativas.append(
                    {
                        "manifesto_id": manifesto_id,
                        "tipo_tentativa": "sem_candidatos_nivel",
                        "nivel_hierarquia": nome_nivel,
                        "aceito": False,
                        "motivo": f"Sem candidatos no nível {nome_nivel}.",
                    }
                )
                continue

            candidatos_ordenados = _ordenar_candidatos_por_prioridade_operacional(candidatos)

            for _, item_row in candidatos_ordenados.iterrows():
                item_df = pd.DataFrame([item_row.to_dict()])

                valido, motivo, comparativo = _simular_adicao_item(
                    manifesto=manifesto,
                    itens_manifesto=itens_manifesto_atual,
                    item_candidato=item_df,
                    ocupacao_alvo_perc=ocupacao_alvo_perc,
                )

                tentativa = {
                    "manifesto_id": manifesto_id,
                    "tipo_tentativa": "adicao_item_remanescente_m5",
                    "nivel_hierarquia": nome_nivel,
                    "id_linha_pipeline": str(item_row["id_linha_pipeline"]),
                    "destinatario": str(item_row.get("destinatario", "")),
                    "cidade": str(item_row.get("cidade", "")),
                    "subregiao_operacional": str(item_row.get("subregiao_operacional", "")),
                    "mesorregiao_operacional": str(item_row.get("mesorregiao_operacional", "")),
                    "agendada": bool(item_row.get("agendada", False)),
                    "folga_dias": _to_float(item_row.get("folga_dias")),
                    "aceito": bool(valido),
                    "motivo": motivo,
                    **comparativo,
                }
                tentativas.append(tentativa)

                if not valido:
                    continue

                houve_movimento_neste_manifesto = True

                item_aplicar = item_df.copy()
                item_aplicar["manifesto_id"] = manifesto_id
                item_aplicar["flag_otimizado_m6_2"] = True
                item_aplicar["origem_item_m6_2"] = "adicionado_do_remanescente_m5"

                df_itens = pd.concat([df_itens, item_aplicar], ignore_index=True)
                df_remanescente = df_remanescente.loc[
                    df_remanescente["id_linha_pipeline"].astype(str) != str(item_row["id_linha_pipeline"])
                ].copy()

                df_manifestos = _recalcular_manifesto_unico(df_manifestos, df_itens, manifesto_id)
                manifesto = df_manifestos.loc[df_manifestos["manifesto_id"] == manifesto_id].head(1).iloc[0].to_dict()
                itens_manifesto_atual = df_itens.loc[df_itens["manifesto_id"] == manifesto_id].copy()

                movimentos_aceitos.append(
                    {
                        "manifesto_id": manifesto_id,
                        "nivel_hierarquia": nome_nivel,
                        "id_linha_pipeline": str(item_row["id_linha_pipeline"]),
                        "destinatario": str(item_row.get("destinatario", "")),
                        "cidade": str(item_row.get("cidade", "")),
                        "subregiao_operacional": str(item_row.get("subregiao_operacional", "")),
                        "mesorregiao_operacional": str(item_row.get("mesorregiao_operacional", "")),
                        "agendada": bool(item_row.get("agendada", False)),
                        "folga_dias": _to_float(item_row.get("folga_dias")),
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
                    "nivel_hierarquia": None,
                    "aceito": False,
                    "motivo": "Nenhum item do remanescente do M5 pôde ser adicionado respeitando as restrições.",
                }
            )

    df_manifestos = _recalcular_todos_manifestos(df_manifestos, df_itens)

    _validar_integridade_final(
        df_itens,
        df_remanescente,
        df_itens_manifestos_base_m6,
        df_remanescente_original,
    )

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
        "itens_adicionados_a_manifestos_m6_2": int(
            len(df_itens.loc[df_itens["flag_otimizado_m6_2"] == True])
        ),
        "estrategia_m6_2": [
            "seleciona_manifestos_abaixo_de_85_usando_ocupacao_oficial_m6_1",
            "usa_apenas_remanescente_oficial_m5",
            "hierarquia_mesmo_cliente_mesma_cidade_mesma_subregiao_mesma_mesorregiao",
            "prioriza_agendado_primeiro",
            "folga_positiva_menor_para_maior",
            "folga_negativa_por_ultimo",
            "nao_mexe_nos_demais_manifestos",
            "nao_gera_duplicidade",
            "ocupacao_calculada_por_peso_calculado_dividido_pela_capacidade_do_veiculo",
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

    out["capacidade_peso_kg"] = pd.to_numeric(
        _resolver_coluna(out, CAP_PESO_ALIASES, obrigatoria=False, default=np.nan),
        errors="coerce",
    )
    out["capacidade_vol_m3"] = pd.to_numeric(
        _resolver_coluna(out, CAP_VOL_ALIASES, obrigatoria=False, default=np.nan),
        errors="coerce",
    )
    out["max_entregas"] = pd.to_numeric(
        _resolver_coluna(out, MAX_ENTREGAS_ALIASES, obrigatoria=False, default=np.nan),
        errors="coerce",
    )
    out["max_km_distancia"] = pd.to_numeric(
        _resolver_coluna(out, MAX_KM_ALIASES, obrigatoria=False, default=np.nan),
        errors="coerce",
    )
    out["ocupacao_minima_perc"] = pd.to_numeric(
        _resolver_coluna(out, OCUP_MIN_ALIASES, obrigatoria=False, default=70),
        errors="coerce",
    ).fillna(70)
    out["ocupacao_maxima_perc"] = pd.to_numeric(
        _resolver_coluna(out, OCUP_MAX_ALIASES, obrigatoria=False, default=100),
        errors="coerce",
    ).fillna(100)
    out["ocupacao_dominante_perc"] = pd.to_numeric(
        _resolver_coluna(out, OCUP_DOMINANTE_ALIASES, obrigatoria=False, default=np.nan),
        errors="coerce",
    )

    # fallback para campos já existentes no M6.1
    if "veiculo_tipo" in out.columns:
        out["perfil"] = out["perfil"].replace("", np.nan)
        out["perfil"] = out["perfil"].fillna(out["veiculo_tipo"].astype(str))

    if "veiculo_perfil" in out.columns:
        out["perfil"] = out["perfil"].replace("", np.nan)
        out["perfil"] = out["perfil"].fillna(out["veiculo_perfil"].astype(str))

    if "qtd_entregas" not in out.columns:
        out["qtd_entregas"] = np.nan
    if "distancia_total_km" not in out.columns:
        out["distancia_total_km"] = np.nan
    if "ocupacao_peso_perc" not in out.columns:
        out["ocupacao_peso_perc"] = np.nan
    if "ocupacao_vol_perc" not in out.columns:
        out["ocupacao_vol_perc"] = np.nan
    if "peso_total_kg" not in out.columns:
        out["peso_total_kg"] = np.nan
    if "vol_total_m3" not in out.columns:
        out["vol_total_m3"] = np.nan

    return out.reset_index(drop=True)


def _normalizar_itens_manifestos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _deduplicar_colunas(out)

    out["manifesto_id"] = _resolver_coluna(out, MANIFESTO_ID_ALIASES, obrigatoria=True).astype(str)

    if "id_linha_pipeline" not in out.columns:
        out["id_linha_pipeline"] = out.index.astype(str)
    out["id_linha_pipeline"] = out["id_linha_pipeline"].astype(str)

    out["mesorregiao_operacional"] = _resolver_coluna(out, MESORREGIAO_ALIASES, obrigatoria=False, default="").astype(str)
    out["subregiao_operacional"] = _resolver_coluna(out, SUBREGIAO_ALIASES, obrigatoria=False, default="").astype(str)
    out["destinatario"] = _resolver_coluna(out, CLIENTE_ALIASES, obrigatoria=False, default="").astype(str)
    out["cidade"] = _resolver_coluna(out, CIDADE_ALIASES, obrigatoria=False, default="").astype(str)
    out["uf"] = _resolver_coluna(out, UF_ALIASES, obrigatoria=False, default="").astype(str)
    out["peso_calculado"] = pd.to_numeric(
        _resolver_coluna(out, PESO_ITEM_ALIASES, obrigatoria=True),
        errors="coerce",
    ).fillna(0)
    out["vol_m3"] = pd.to_numeric(
        _resolver_coluna(out, VOL_ITEM_ALIASES, obrigatoria=False, default=0),
        errors="coerce",
    ).fillna(0)
    out["distancia_rodoviaria_est_km"] = pd.to_numeric(
        _resolver_coluna(out, DISTANCIA_ALIASES, obrigatoria=False, default=0),
        errors="coerce",
    ).fillna(0)
    out["restricao_veiculo"] = _resolver_coluna(
        out,
        RESTRICAO_VEICULO_ALIASES,
        obrigatoria=False,
        default="",
    ).astype(str)
    out["agendada"] = _normalizar_flag_agendada(
        _resolver_coluna(out, AGENDADO_ALIASES, obrigatoria=False, default=False)
    )
    out["folga_dias"] = pd.to_numeric(
        _resolver_coluna(out, FOLGA_ALIASES, obrigatoria=False, default=np.nan),
        errors="coerce",
    )

    # fallback dos nomes originais
    if "subregiao" in out.columns:
        out["subregiao_operacional"] = out["subregiao_operacional"].replace("", np.nan)
        out["subregiao_operacional"] = out["subregiao_operacional"].fillna(out["subregiao"].astype(str))

    if "mesorregiao" in out.columns:
        out["mesorregiao_operacional"] = out["mesorregiao_operacional"].replace("", np.nan)
        out["mesorregiao_operacional"] = out["mesorregiao_operacional"].fillna(out["mesorregiao"].astype(str))

    return out.reset_index(drop=True)


def _normalizar_itens_remanescente(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _deduplicar_colunas(out)

    if "id_linha_pipeline" not in out.columns:
        out["id_linha_pipeline"] = out.index.astype(str)
    out["id_linha_pipeline"] = out["id_linha_pipeline"].astype(str)

    out["mesorregiao_operacional"] = _resolver_coluna(out, MESORREGIAO_ALIASES, obrigatoria=False, default="").astype(str)
    out["subregiao_operacional"] = _resolver_coluna(out, SUBREGIAO_ALIASES, obrigatoria=False, default="").astype(str)
    out["destinatario"] = _resolver_coluna(out, CLIENTE_ALIASES, obrigatoria=False, default="").astype(str)
    out["cidade"] = _resolver_coluna(out, CIDADE_ALIASES, obrigatoria=False, default="").astype(str)
    out["uf"] = _resolver_coluna(out, UF_ALIASES, obrigatoria=False, default="").astype(str)
    out["peso_calculado"] = pd.to_numeric(
        _resolver_coluna(out, PESO_ITEM_ALIASES, obrigatoria=True),
        errors="coerce",
    ).fillna(0)
    out["vol_m3"] = pd.to_numeric(
        _resolver_coluna(out, VOL_ITEM_ALIASES, obrigatoria=False, default=0),
        errors="coerce",
    ).fillna(0)
    out["distancia_rodoviaria_est_km"] = pd.to_numeric(
        _resolver_coluna(out, DISTANCIA_ALIASES, obrigatoria=False, default=0),
        errors="coerce",
    ).fillna(0)
    out["restricao_veiculo"] = _resolver_coluna(
        out,
        RESTRICAO_VEICULO_ALIASES,
        obrigatoria=False,
        default="",
    ).astype(str)
    out["agendada"] = _normalizar_flag_agendada(
        _resolver_coluna(out, AGENDADO_ALIASES, obrigatoria=False, default=False)
    )
    out["folga_dias"] = pd.to_numeric(
        _resolver_coluna(out, FOLGA_ALIASES, obrigatoria=False, default=np.nan),
        errors="coerce",
    )

    if "subregiao" in out.columns:
        out["subregiao_operacional"] = out["subregiao_operacional"].replace("", np.nan)
        out["subregiao_operacional"] = out["subregiao_operacional"].fillna(out["subregiao"].astype(str))

    if "mesorregiao" in out.columns:
        out["mesorregiao_operacional"] = out["mesorregiao_operacional"].replace("", np.nan)
        out["mesorregiao_operacional"] = out["mesorregiao_operacional"].fillna(out["mesorregiao"].astype(str))

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
    base["ocupacao_dominante_perc"] = pd.to_numeric(base["ocupacao_dominante_perc"], errors="coerce")
    base = base.loc[base["ocupacao_dominante_perc"].notna()].copy()
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


def _subregiao_dominante(df_itens_manifesto: pd.DataFrame) -> str:
    if df_itens_manifesto.empty or "subregiao_operacional" not in df_itens_manifesto.columns:
        return ""
    vc = df_itens_manifesto["subregiao_operacional"].astype(str).str.strip().value_counts()
    return str(vc.index[0]).strip() if len(vc) > 0 else ""


def _selecionar_candidatos_por_hierarquia(
    rem_mesmo_meso: pd.DataFrame,
    itens_manifesto: pd.DataFrame,
    nome_nivel: str,
    coluna_nivel: str,
    contexto_manifesto: Dict[str, Any],
) -> pd.DataFrame:
    base = rem_mesmo_meso.copy()
    if base.empty:
        return base

    if nome_nivel == "mesmo_cliente":
        alvo = _txt_norm(contexto_manifesto.get("cliente_dominante", ""))
        if alvo == "":
            return pd.DataFrame()
        return base.loc[base["destinatario"].astype(str).str.upper() == alvo].copy()

    if nome_nivel == "mesma_cidade":
        alvo = _txt_norm(contexto_manifesto.get("cidade_dominante", ""))
        if alvo == "":
            return pd.DataFrame()
        return base.loc[base["cidade"].astype(str).str.upper() == alvo].copy()

    if nome_nivel == "mesma_subregiao":
        alvo = _txt_norm(contexto_manifesto.get("subregiao_dominante", ""))
        if alvo == "":
            return pd.DataFrame()
        return base.loc[base["subregiao_operacional"].astype(str).str.upper() == alvo].copy()

    if nome_nivel == "mesma_mesorregiao":
        return base.copy()

    return pd.DataFrame()


def _ordenar_candidatos_por_prioridade_operacional(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["agendada"] = out["agendada"].fillna(False).astype(bool)
    out["folga_dias"] = pd.to_numeric(out["folga_dias"], errors="coerce")

    def _grupo_folga(valor: Any) -> int:
        if pd.isna(valor):
            return 2
        valor = float(valor)
        if valor >= 0:
            return 0
        return 1

    out["ord_agendada"] = np.where(out["agendada"] == True, 0, 1)
    out["ord_grupo_folga"] = out["folga_dias"].apply(_grupo_folga)
    out["ord_folga"] = out["folga_dias"].fillna(999999)

    out = out.sort_values(
        by=[
            "ord_agendada",
            "ord_grupo_folga",
            "ord_folga",
            "peso_calculado",
            "distancia_rodoviaria_est_km",
        ],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    return out


def _simular_adicao_item(
    manifesto: Dict[str, Any],
    itens_manifesto: pd.DataFrame,
    item_candidato: pd.DataFrame,
    ocupacao_alvo_perc: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    antes = _calcular_metricas_manifesto(manifesto, itens_manifesto)
    depois_df = pd.concat([itens_manifesto, item_candidato], ignore_index=True)
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
        return False, "A adição do item não melhora a ocupação do manifesto.", comparativo

    if depois["distancia_total_km"] > antes["distancia_total_km"]:
        return False, "A adição do item piora a distância máxima do manifesto.", comparativo

    if depois["ocupacao_dominante_perc"] > float(manifesto.get("ocupacao_maxima_perc", 100) or 100):
        return False, "A adição do item ultrapassa a ocupação máxima do manifesto.", comparativo

    return True, "Item aceito no complemento de ocupação.", comparativo


def _calcular_metricas_manifesto(manifesto: Dict[str, Any], df_itens: pd.DataFrame) -> Dict[str, Any]:
    capacidade_peso = _to_float(manifesto.get("capacidade_peso_kg"))
    capacidade_vol = _to_float(manifesto.get("capacidade_vol_m3"))

    peso_total = float(pd.to_numeric(df_itens["peso_calculado"], errors="coerce").fillna(0).sum())
    vol_total = float(pd.to_numeric(df_itens["vol_m3"], errors="coerce").fillna(0).sum())
    distancia_total = float(pd.to_numeric(df_itens["distancia_rodoviaria_est_km"], errors="coerce").fillna(0).max())
    qtd_entregas = int(_contar_entregas(df_itens))

    ocupacao_existente = _to_float(manifesto.get("ocupacao_dominante_perc"))

    if capacidade_peso is not None and capacidade_peso > 0:
        ocup_peso = (peso_total / capacidade_peso) * 100.0
    else:
        ocup_peso = np.nan

    if capacidade_vol is not None and capacidade_vol > 0:
        ocup_vol = (vol_total / capacidade_vol) * 100.0
    else:
        ocup_vol = np.nan

    if pd.notna(ocup_peso) and pd.notna(ocup_vol):
        ocup_dominante = max(ocup_peso, ocup_vol)
    elif pd.notna(ocup_peso):
        ocup_dominante = ocup_peso
    elif pd.notna(ocup_vol):
        ocup_dominante = ocup_vol
    elif ocupacao_existente is not None:
        ocup_dominante = ocupacao_existente
    else:
        ocup_dominante = 0.0

    return {
        "peso_total_kg": peso_total,
        "vol_total_m3": vol_total,
        "distancia_total_km": distancia_total,
        "qtd_entregas": qtd_entregas,
        "ocupacao_peso_perc": ocup_peso if pd.notna(ocup_peso) else np.nan,
        "ocupacao_vol_perc": ocup_vol if pd.notna(ocup_vol) else np.nan,
        "ocupacao_dominante_perc": float(ocup_dominante),
    }


def _validar_restricoes_manifesto(manifesto: Dict[str, Any], df_itens: pd.DataFrame) -> bool | str:
    metricas = _calcular_metricas_manifesto(manifesto, df_itens)

    capacidade_peso = _to_float(manifesto.get("capacidade_peso_kg"))
    capacidade_vol = _to_float(manifesto.get("capacidade_vol_m3"))
    max_entregas = _to_int(manifesto.get("max_entregas"))
    max_km = _to_float(manifesto.get("max_km_distancia"))
    perfil_manifesto = str(manifesto.get("perfil", "")).strip().upper()
    meso_manifesto = str(manifesto.get("mesorregiao_operacional", "")).strip().upper()

    if capacidade_peso is not None and metricas["peso_total_kg"] > capacidade_peso:
        return "Excede capacidade de peso."
    if capacidade_vol is not None and metricas["vol_total_m3"] > capacidade_vol:
        return "Excede capacidade de volume."
    if max_entregas is not None and metricas["qtd_entregas"] > max_entregas:
        return "Excede máximo de entregas."
    if max_km is not None and metricas["distancia_total_km"] > max_km:
        return "Excede raio/distância máxima."

    mesorregioes = set(df_itens["mesorregiao_operacional"].astype(str).str.strip().str.upper().tolist())
    mesorregioes.discard("")
    if len(mesorregioes) > 1:
        return "Mistura mais de uma mesorregião no manifesto."
    if len(mesorregioes) == 1 and list(mesorregioes)[0] != meso_manifesto:
        return "Item do remanescente está em mesorregião diferente do manifesto."

    restricoes = set(df_itens["restricao_veiculo"].astype(str).str.strip().str.upper().tolist())
    restricoes.discard("")
    if len(restricoes) > 1:
        return "Itens com restrições de veículo conflitantes."
    if len(restricoes) == 1 and perfil_manifesto != "" and list(restricoes)[0] != perfil_manifesto:
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

    if pd.notna(metricas["ocupacao_peso_perc"]):
        out.loc[i, "ocupacao_peso_perc"] = metricas["ocupacao_peso_perc"]
    if pd.notna(metricas["ocupacao_vol_perc"]):
        out.loc[i, "ocupacao_vol_perc"] = metricas["ocupacao_vol_perc"]

    ocupacao_anterior = _to_float(out.loc[i, "ocupacao_dominante_perc"])
    if (
        pd.isna(metricas["ocupacao_peso_perc"])
        and pd.isna(metricas["ocupacao_vol_perc"])
        and ocupacao_anterior is not None
    ):
        out.loc[i, "ocupacao_dominante_perc"] = ocupacao_anterior
    else:
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
        raise Exception(
            f"M6.2 deixou itens ao mesmo tempo no manifesto e no remanescente: {list(intersec)[:20]}"
        )


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


def _normalizar_flag_agendada(serie: pd.Series) -> pd.Series:
    def _map(valor: Any) -> bool:
        if isinstance(valor, bool):
            return valor
        if pd.isna(valor):
            return False
        txt = str(valor).strip().lower()
        return txt in {"true", "1", "sim", "s", "yes", "y", "agendada"}
    return serie.apply(_map)


def _txt_norm(valor: Any) -> str:
    if pd.isna(valor):
        return ""
    return str(valor).strip().upper()


def _to_float(valor: Any) -> Optional[float]:
    x = pd.to_numeric(valor, errors="coerce")
    if pd.isna(x):
        return None
    return float(x)


def _to_int(valor: Any) -> Optional[int]:
    x = pd.to_numeric(valor, errors="coerce")
    if pd.isna(x):
        return None
    return int(x)


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
