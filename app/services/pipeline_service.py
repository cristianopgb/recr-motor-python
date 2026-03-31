from typing import Any, Dict

from app.pipeline import (
    m0_leitura,
    m1_padronizacao,
    m2_enriquecimento,
    m3_triagem,
    m31_fronteira,
    m4_fechados,
    m5_compostos,
    m51_saneamento,
    m8_sobras,
    m9_consolidacao,
)
from app.services import payload_service, response_service


def executar(payload: Dict[str, Any]) -> Dict[str, Any]:
    dados = payload_service.preparar(payload)

    dados = m0_leitura.executar(dados)
    dados = m1_padronizacao.executar(dados)
    dados = m2_enriquecimento.executar(dados)
    dados = m3_triagem.executar(dados)
    dados = m31_fronteira.executar(dados)
    dados = m4_fechados.executar(dados)
    dados = m5_compostos.executar(dados)
    dados = m51_saneamento.executar(dados)
    dados = m8_sobras.executar(dados)
    dados = m9_consolidacao.executar(dados)

    return response_service.formatar(dados)
