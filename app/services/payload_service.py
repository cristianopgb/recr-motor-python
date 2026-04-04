from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.schemas import RoteirizacaoRequest


@dataclass
class PipelineContext:
    """
    Contexto interno que será entregue ao pipeline.
    Equivale ao ambiente que o notebook tinha após a leitura dos arquivos.
    """
    rodada_id: str
    upload_id: str
    usuario_id: str
    filial_id: str
    tipo_roteirizacao: str
    data_execucao: datetime
    data_base: pd.Timestamp

    df_carteira_raw: pd.DataFrame
    df_veiculos_raw: pd.DataFrame
    df_geo_raw: pd.DataFrame
    df_parametros_raw: pd.DataFrame

    caminhos_pipeline: Dict[str, str]
    metadados_rodada: Dict[str, Any]


def _to_dataframe_list(items: Any) -> pd.DataFrame:
    """
    Converte lista de modelos Pydantic/dicts em DataFrame.
    """
    if items is None:
        return pd.DataFrame()

    registros = []

    for item in items:
        if hasattr(item, "model_dump"):
            # Pydantic v2
            registros.append(item.model_dump(by_alias=True))
        elif isinstance(item, dict):
            registros.append(item)
        else:
            registros.append(vars(item))

    if not registros:
        return pd.DataFrame()

    return pd.DataFrame(registros)


def _parse_data_base(parametros: Any) -> tuple[datetime, pd.Timestamp]:
    """
    Extrai data_execucao e data_base_roteirizacao do payload validado.
    """
    data_execucao = parametros.data_execucao
    data_base_roteirizacao = parametros.data_base_roteirizacao

    if isinstance(data_execucao, str):
        texto = data_execucao.strip()
        if texto.endswith("Z"):
            texto = texto[:-1] + "+00:00"
        data_execucao_dt = datetime.fromisoformat(texto)
    else:
        data_execucao_dt = data_execucao

    if isinstance(data_base_roteirizacao, str):
        texto = data_base_roteirizacao.strip()
        if texto.endswith("Z"):
            texto = texto[:-1] + "+00:00"
        data_base_dt = datetime.fromisoformat(texto)
    else:
        data_base_dt = data_base_roteirizacao

    return data_execucao_dt, pd.Timestamp(data_base_dt)


def _montar_df_parametros_raw(payload: RoteirizacaoRequest) -> pd.DataFrame:
    """
    Converte parametros do payload para formato tabular simples:
    parametro | valor

    Isso facilita adaptar o notebook existente, que lê parâmetros assim.
    """
    parametros = payload.parametros

    registros = [
        {"parametro": "usuario_id", "valor": getattr(parametros, "usuario_id", None)},
        {"parametro": "usuario_nome", "valor": getattr(parametros, "usuario_nome", None)},
        {"parametro": "filial_id", "valor": getattr(parametros, "filial_id", None)},
        {"parametro": "filial_nome", "valor": getattr(parametros, "filial_nome", None)},
        {"parametro": "upload_id", "valor": getattr(parametros, "upload_id", None)},
        {"parametro": "rodada_id", "valor": getattr(parametros, "rodada_id", None)},
        {"parametro": "data_execucao", "valor": getattr(parametros, "data_execucao", None)},
        {"parametro": "data_base_roteirizacao", "valor": getattr(parametros, "data_base_roteirizacao", None)},
        {"parametro": "origem_sistema", "valor": getattr(parametros, "origem_sistema", None)},
        {"parametro": "tipo_roteirizacao", "valor": getattr(parametros, "tipo_roteirizacao", None)},
        {"parametro": "modelo_roteirizacao", "valor": getattr(parametros, "modelo_roteirizacao", None)},
    ]

    filtros_aplicados = getattr(parametros, "filtros_aplicados", None) or {}
    for chave, valor in filtros_aplicados.items():
        registros.append(
            {"parametro": f"filtro__{chave}", "valor": valor}
        )

    configuracao_frota = getattr(parametros, "configuracao_frota", None) or []
    for i, item in enumerate(configuracao_frota, start=1):
        if hasattr(item, "model_dump"):
            item_dict = item.model_dump()
        elif isinstance(item, dict):
            item_dict = item
        else:
            item_dict = vars(item)

        registros.append(
            {"parametro": f"configuracao_frota__{i}__perfil", "valor": item_dict.get("perfil")}
        )
        registros.append(
            {"parametro": f"configuracao_frota__{i}__quantidade", "valor": item_dict.get("quantidade")}
        )

    return pd.DataFrame(registros)


def _montar_caminhos_pipeline(rodada_id: str) -> Dict[str, str]:
    """
    No notebook existia estrutura de pastas física.
    Aqui deixamos um contexto lógico.
    Se depois você quiser salvar artefatos locais, essa estrutura já está pronta.
    """
    pasta_base = Path("/tmp/rec_roteirizador") / rodada_id

    return {
        "pasta_saida_base": str(pasta_base),
        "pasta_rodada": str(pasta_base),
        "rodada_id": rodada_id,
    }


def normalizar_payload_para_pipeline(payload: RoteirizacaoRequest) -> PipelineContext:
    """
    Camada adaptadora entre a API e o pipeline.
    Recebe o contrato do Sistema 1 e entrega um contexto compatível
    com a lógica que nasceu no Jupyter.
    """
    parametros = payload.parametros

    data_execucao_dt, data_base = _parse_data_base(parametros)

    df_carteira_raw = _to_dataframe_list(payload.carteira)
    df_veiculos_raw = _to_dataframe_list(payload.veiculos)
    df_geo_raw = _to_dataframe_list(payload.regionalidades)
    df_parametros_raw = _montar_df_parametros_raw(payload)

    rodada_id = str(parametros.rodada_id)
    upload_id = str(parametros.upload_id)
    usuario_id = str(parametros.usuario_id)
    filial_id = str(parametros.filial_id)
    tipo_roteirizacao = str(parametros.tipo_roteirizacao).strip().lower()

    caminhos_pipeline = _montar_caminhos_pipeline(rodada_id)

    metadados_rodada = {
        "rodada_id": rodada_id,
        "upload_id": upload_id,
        "usuario_id": usuario_id,
        "filial_id": filial_id,
        "tipo_roteirizacao": tipo_roteirizacao,
        "data_execucao": data_execucao_dt.isoformat(),
        "data_base_roteirizacao": data_base.isoformat(),
        "qtd_carteira": int(len(df_carteira_raw)),
        "qtd_veiculos": int(len(df_veiculos_raw)),
        "qtd_regionalidades": int(len(df_geo_raw)),
        "qtd_parametros": int(len(df_parametros_raw)),
    }

    return PipelineContext(
        rodada_id=rodada_id,
        upload_id=upload_id,
        usuario_id=usuario_id,
        filial_id=filial_id,
        tipo_roteirizacao=tipo_roteirizacao,
        data_execucao=data_execucao_dt,
        data_base=data_base,
        df_carteira_raw=df_carteira_raw,
        df_veiculos_raw=df_veiculos_raw,
        df_geo_raw=df_geo_raw,
        df_parametros_raw=df_parametros_raw,
        caminhos_pipeline=caminhos_pipeline,
        metadados_rodada=metadados_rodada,
    )
