from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from app.pipeline.m2_enriquecimento import executar_m2_enriquecimento
from app.pipeline.m3_triagem import executar_m3_triagem
from app.schemas import RoteirizacaoRequest
from app.services.payload_service import PipelineContext, normalizar_payload_para_pipeline


def _log(
    modulo: str,
    status: str,
    mensagem: str,
    quantidade_entrada: int | None = None,
    quantidade_saida: int | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    registro = {
        "modulo": modulo,
        "status": status,
        "mensagem": mensagem,
        "quantidade_entrada": quantidade_entrada,
        "quantidade_saida": quantidade_saida,
    }
    if extra:
        registro["extra"] = extra
    return registro


def _snapshot_dataframe(df: pd.DataFrame, nome: str, max_colunas: int = 30) -> Dict[str, Any]:
    return {
        "nome": nome,
        "linhas": int(len(df)),
        "colunas": list(df.columns[:max_colunas]),
        "qtd_colunas_total": int(len(df.columns)),
    }


def _df_to_records(df: pd.DataFrame, limit: int | None = None) -> List[Dict[str, Any]]:
    df2 = df.copy()
    if limit is not None:
        df2 = df2.head(limit)

    for col in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[col]):
            df2[col] = df2[col].astype(str)

    df2 = df2.where(pd.notnull(df2), None)
    return df2.to_dict(orient="records")


def _executar_m0_adapter(contexto: PipelineContext) -> Dict[str, Any]:
    inventario = {
        "rodada_id": contexto.rodada_id,
        "upload_id": contexto.upload_id,
        "usuario_id": contexto.usuario_id,
        "filial_id": contexto.filial_id,
        "tipo_roteirizacao": contexto.tipo_roteirizacao,
        "data_execucao": contexto.data_execucao.isoformat(),
        "data_base_roteirizacao": contexto.data_base.isoformat(),
        "filial": contexto.filial,
        "inputs": {
            "carteira": _snapshot_dataframe(contexto.df_carteira_raw, "df_carteira_raw"),
            "regionalidades": _snapshot_dataframe(contexto.df_geo_raw, "df_geo_raw"),
            "parametros": _snapshot_dataframe(contexto.df_parametros_raw, "df_parametros_raw"),
            "veiculos": _snapshot_dataframe(contexto.df_veiculos_raw, "df_veiculos_raw"),
        },
        "caminhos_pipeline": contexto.caminhos_pipeline,
    }

    return {
        "inventario": inventario,
        "df_carteira_raw": contexto.df_carteira_raw,
        "df_geo_raw": contexto.df_geo_raw,
        "df_parametros_raw": contexto.df_parametros_raw,
        "df_veiculos_raw": contexto.df_veiculos_raw,
        "DATA_BASE": contexto.data_base,
        "filial_rodada": contexto.filial,
        "parametros_rodada": contexto.parametros_rodada,
        "caminhos_pipeline": contexto.caminhos_pipeline,
        "metadados_rodada": contexto.metadados_rodada,
    }


def _executar_m1_minimo(resultado_m0: Dict[str, Any]) -> pd.DataFrame:
    """
    M1 mínimo adaptado para a API.
    Padroniza somente o necessário para o M2/M3 usando o contrato do Sistema 1.
    """
    df = resultado_m0["df_carteira_raw"].copy()
    parametros_rodada = resultado_m0["parametros_rodada"]

    mapa = {
        "Filial": "filial_roteirizacao",
        "Romane": "romaneio",
        "Filial (origem)": "filial_origem",
        "Série": "serie_romaneio",
        "Nro Doc.": "nro_documento",
        "Data Des": "data_descarga",
        "Data NF": "data_emissao_nf",
        "D.L.E.": "data_leadtime",
        "Agendam.": "data_agenda",
        "Palet": "qtd_pallet",
        "Conf": "conferencia",
        "Peso": "peso_kg",
        "Vlr.Merc.": "valor_nf",
        "Qtd.": "qtd_volumes",
        "Peso C": "vol_m3",
        "Classifi": "tipo_servico",
        "Tomador": "embarcador",
        "Destinatário": "destinatario",
        "Bairro": "bairro",
        "Cida": "cidade",
        "UF": "uf",
        "NF / Serie": "nf_serie",
        "Tipo Carga": "tipo_carga",
        "Qtd.NF": "qtd_nf",
        "Região": "regiao",
        "Sub-Região": "subregiao",
        "Ocorrências NFs": "ocorrencias_nfs",
        "Remetente": "remetente",
        "Observação R": "observacao_r",
        "Ref Cliente": "ref_cliente",
        "Cidade Dest.": "cidade_destino",
        "Mesoregião": "mesorregiao",
        "Agenda": "agendada",
        "Tipo C": "tipo_veiculo_carga",
        "Última": "ultima_ocorrencia",
        "Status": "status_operacional",
        "Lat.": "latitude_destinatario",
        "Lon.": "longitude_destinatario",
    }

    df = df.rename(columns={k: v for k, v in mapa.items() if k in df.columns})

    colunas_numericas = [
        "filial_roteirizacao",
        "romaneio",
        "filial_origem",
        "serie_romaneio",
        "nro_documento",
        "qtd_pallet",
        "peso_kg",
        "valor_nf",
        "qtd_volumes",
        "vol_m3",
        "qtd_nf",
        "latitude_destinatario",
        "longitude_destinatario",
    ]
    for c in colunas_numericas:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    colunas_datas = ["data_descarga", "data_emissao_nf", "data_leadtime", "data_agenda"]
    for c in colunas_datas:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    if "agendada" in df.columns:
        df["agendada"] = (
            df["agendada"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"sim": True, "nao": False, "não": False, "true": True, "false": False})
            .fillna(False)
        )

    df["latitude_filial"] = pd.to_numeric(parametros_rodada["filial_latitude"], errors="coerce")
    df["longitude_filial"] = pd.to_numeric(parametros_rodada["filial_longitude"], errors="coerce")

    return df


def executar_pipeline(payload: RoteirizacaoRequest) -> Dict[str, Any]:
    logs: List[Dict[str, Any]] = []

    contexto = normalizar_payload_para_pipeline(payload)
    logs.append(
        _log(
            modulo="payload_service",
            status="ok",
            mensagem="Payload normalizado para o contexto interno do pipeline",
            extra={
                "rodada_id": contexto.rodada_id,
                "filial_id": contexto.filial_id,
                "data_base_roteirizacao": contexto.data_base.isoformat(),
            },
        )
    )

    resultado_m0 = _executar_m0_adapter(contexto)
    logs.append(
        _log(
            modulo="m0_adapter",
            status="ok",
            mensagem="M0 adaptado executado com sucesso",
            quantidade_entrada=int(len(contexto.df_carteira_raw)),
            quantidade_saida=int(len(contexto.df_carteira_raw)),
            extra={
                "filial": contexto.filial,
                "data_base_roteirizacao": contexto.data_base.isoformat(),
            },
        )
    )

    df_carteira_tratada = _executar_m1_minimo(resultado_m0)
    logs.append(
        _log(
            modulo="m1_minimo",
            status="ok",
            mensagem="M1 mínimo executado com sucesso para preparar o M2",
            quantidade_entrada=int(len(contexto.df_carteira_raw)),
            quantidade_saida=int(len(df_carteira_tratada)),
        )
    )

    df_geo_tratado = contexto.df_geo_raw.rename(
        columns={
            "cidade": "nome",
        }
    ).copy()

    df_parametros_tratados = contexto.df_parametros_raw.copy()

    df_carteira_enriquecida, resumo_m2 = executar_m2_enriquecimento(
        df_carteira_tratada=df_carteira_tratada,
        df_geo_tratado=df_geo_tratado,
        df_parametros_tratados=df_parametros_tratados,
        data_base_roteirizacao=contexto.data_base,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )

    logs.append(
        _log(
            modulo="m2_enriquecimento",
            status="ok",
            mensagem="M2 executado com sucesso",
            quantidade_entrada=int(len(df_carteira_tratada)),
            quantidade_saida=int(len(df_carteira_enriquecida)),
            extra=resumo_m2,
        )
    )

    df_carteira_triagem, meta_m3 = executar_m3_triagem(
        df_carteira_enriquecida=df_carteira_enriquecida,
        data_base_roteirizacao=contexto.data_base,
        caminhos_pipeline=contexto.caminhos_pipeline,
    )

    outputs_m3 = meta_m3["outputs_m3"]
    resumo_m3 = meta_m3["resumo_m3"]

    df_carteira_roteirizavel = outputs_m3["df_carteira_roteirizavel"]
    df_carteira_entrega_futura = outputs_m3["df_carteira_entrega_futura"]
    df_carteira_aguardando_agendamento = outputs_m3["df_carteira_aguardando_agendamento"]
    df_carteira_excecoes_triagem = outputs_m3["df_carteira_excecoes_triagem"]

    logs.append(
        _log(
            modulo="m3_triagem",
            status="ok",
            mensagem="M3 executado com sucesso",
            quantidade_entrada=int(len(df_carteira_enriquecida)),
            quantidade_saida=int(len(df_carteira_triagem)),
            extra=resumo_m3,
        )
    )

    return {
        "status": "ok",
        "mensagem": "Motor executou com sucesso até o M3.",
        "pipeline_real_ate": "M3",
        "resumo": {
            "total_carteira": int(len(contexto.df_carteira_raw)),
            "total_veiculos": int(len(contexto.df_veiculos_raw)),
            "total_regionalidades": int(len(contexto.df_geo_raw)),
            "total_parametros": int(len(contexto.df_parametros_raw)),
            "filial_id": contexto.filial_id,
            "tipo_roteirizacao": contexto.tipo_roteirizacao,
            "data_base_roteirizacao": contexto.data_base.isoformat(),
            "resumo_m2": resumo_m2,
            "resumo_m3": resumo_m3,
        },
        "contexto_rodada": {
            "filial": contexto.filial,
            "parametros_rodada": contexto.parametros_rodada,
        },
        "snapshots": {
            "carteira_raw": _snapshot_dataframe(contexto.df_carteira_raw, "df_carteira_raw"),
            "carteira_tratada": _snapshot_dataframe(df_carteira_tratada, "df_carteira_tratada"),
            "carteira_enriquecida": _snapshot_dataframe(df_carteira_enriquecida, "df_carteira_enriquecida"),
            "carteira_triagem": _snapshot_dataframe(df_carteira_triagem, "df_carteira_triagem"),
            "carteira_roteirizavel": _snapshot_dataframe(df_carteira_roteirizavel, "df_carteira_roteirizavel"),
            "carteira_entrega_futura": _snapshot_dataframe(df_carteira_entrega_futura, "df_carteira_entrega_futura"),
            "carteira_aguardando_agendamento": _snapshot_dataframe(
                df_carteira_aguardando_agendamento, "df_carteira_aguardando_agendamento"
            ),
            "carteira_excecoes_triagem": _snapshot_dataframe(
                df_carteira_excecoes_triagem, "df_carteira_excecoes_triagem"
            ),
            "regionalidades": _snapshot_dataframe(contexto.df_geo_raw, "df_geo_raw"),
            "parametros": _snapshot_dataframe(contexto.df_parametros_raw, "df_parametros_raw"),
            "veiculos": _snapshot_dataframe(contexto.df_veiculos_raw, "df_veiculos_raw"),
        },
        "amostras": {
            "carteira_tratada": _df_to_records(df_carteira_tratada, limit=5),
            "carteira_enriquecida": _df_to_records(df_carteira_enriquecida, limit=5),
            "carteira_roteirizavel": _df_to_records(df_carteira_roteirizavel, limit=5),
            "entrega_futura": _df_to_records(df_carteira_entrega_futura, limit=5),
            "aguardando_agendamento": _df_to_records(df_carteira_aguardando_agendamento, limit=5),
            "excecoes_triagem": _df_to_records(df_carteira_excecoes_triagem, limit=5),
        },
        "manifestos_fechados": [],
        "manifestos_compostos": [],
        "nao_roteirizados": [],
        "logs": logs,
    }
