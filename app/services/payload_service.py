from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from app.schemas import RoteirizacaoRequest


@dataclass
class PipelineContext:
    rodada_id: str
    upload_id: str
    usuario_id: str
    filial_id: str
    tipo_roteirizacao: str
    data_execucao: datetime
    data_base: datetime

    filial: Dict[str, Any]
    parametros_rodada: Dict[str, Any]
    metadados_rodada: Dict[str, Any]
    caminhos_pipeline: Dict[str, str]

    df_carteira_raw: pd.DataFrame
    df_geo_raw: pd.DataFrame
    df_parametros_raw: pd.DataFrame
    df_veiculos_raw: pd.DataFrame


def _parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _to_dataframe(items: List[Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            rows.append(item.model_dump(by_alias=True, exclude_none=False))
        elif isinstance(item, dict):
            rows.append(item)
        else:
            rows.append(dict(item))
    return pd.DataFrame(rows)


def _normalizar_parametros(payload: RoteirizacaoRequest) -> Dict[str, Any]:
    parametros = dict(payload.parametros or {})

    parametros["filial_id"] = payload.filial.id
    parametros["filial_nome"] = payload.filial.nome
    parametros["filial_cidade"] = payload.filial.cidade
    parametros["filial_uf"] = payload.filial.uf
    parametros["filial_latitude"] = float(payload.filial.latitude)
    parametros["filial_longitude"] = float(payload.filial.longitude)
    parametros["data_base_roteirizacao"] = payload.data_base_roteirizacao

    # compatibilidade temporária com módulos legados do notebook
    parametros["origem_cidade"] = payload.filial.cidade
    parametros["origem_uf"] = payload.filial.uf
    parametros["origem_latitude"] = float(payload.filial.latitude)
    parametros["origem_longitude"] = float(payload.filial.longitude)
    parametros["data_corte_referencia"] = payload.data_base_roteirizacao

    # regra operacional fechada
    parametros["velocidade_media_km_h"] = 50
    parametros["horas_direcao_dia"] = 8
    parametros["km_dia_operacional"] = 400

    return parametros


def _parametros_dict_para_dataframe(parametros: Dict[str, Any]) -> pd.DataFrame:
    rows = [{"parametro": k, "valor": v} for k, v in parametros.items()]
    return pd.DataFrame(rows)


def _montar_caminhos_pipeline(rodada_id: str) -> Dict[str, str]:
    pasta_base = Path("/tmp") / "rec_roteirizador" / rodada_id
    return {
        "pasta_saida_base": str(pasta_base),
        "rodada_id": rodada_id,
    }


def normalizar_payload_para_pipeline(payload: RoteirizacaoRequest) -> PipelineContext:
    data_base = _parse_iso_datetime(payload.data_base_roteirizacao)
    data_execucao = datetime.utcnow()

    df_carteira_raw = _to_dataframe(payload.carteira)
    df_geo_raw = _to_dataframe(payload.regionalidades)
    df_veiculos_raw = _to_dataframe(payload.veiculos)

    parametros_rodada = _normalizar_parametros(payload)
    df_parametros_raw = _parametros_dict_para_dataframe(parametros_rodada)

    filial = {
        "id": payload.filial.id,
        "nome": payload.filial.nome,
        "cidade": payload.filial.cidade,
        "uf": payload.filial.uf,
        "latitude": float(payload.filial.latitude),
        "longitude": float(payload.filial.longitude),
    }

    caminhos_pipeline = _montar_caminhos_pipeline(payload.rodada_id)

    metadados_rodada = {
        "rodada_id": payload.rodada_id,
        "upload_id": payload.upload_id,
        "usuario_id": payload.usuario_id,
        "filial_id": payload.filial_id,
        "tipo_roteirizacao": payload.tipo_roteirizacao,
        "data_base_roteirizacao": payload.data_base_roteirizacao,
        "filial": filial,
        "configuracao_frota": [
            item.model_dump(exclude_none=False) for item in payload.configuracao_frota
        ],
    }

    return PipelineContext(
        rodada_id=payload.rodada_id,
        upload_id=payload.upload_id,
        usuario_id=payload.usuario_id,
        filial_id=payload.filial_id,
        tipo_roteirizacao=payload.tipo_roteirizacao,
        data_execucao=data_execucao,
        data_base=data_base,
        filial=filial,
        parametros_rodada=parametros_rodada,
        metadados_rodada=metadados_rodada,
        caminhos_pipeline=caminhos_pipeline,
        df_carteira_raw=df_carteira_raw,
        df_geo_raw=df_geo_raw,
        df_parametros_raw=df_parametros_raw,
        df_veiculos_raw=df_veiculos_raw,
    )
