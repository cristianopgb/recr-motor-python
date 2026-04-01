from typing import Dict, Any
from app.schemas import RoteirizacaoRequest


def executar_pipeline(payload: RoteirizacaoRequest) -> Dict[str, Any]:
    """
    Versão temporária (stub) do pipeline.
    NÃO executa lógica real.
    Apenas valida integração com Sistema 1.
    """

    total_carteira = len(payload.carteira)
    total_veiculos = len(payload.veiculos)
    total_regionalidades = len(payload.regionalidades)

    logs = [
        {
            "modulo": "stub_pipeline",
            "status": "ok",
            "mensagem": "Pipeline ainda não implementado - execução simulada",
            "quantidade_entrada": total_carteira,
            "quantidade_saida": total_carteira,
        }
    ]

    return {
        "total_carteira": total_carteira,
        "total_roteirizado": 0,
        "total_nao_roteirizado": total_carteira,
        "manifestos_fechados": [],
        "manifestos_compostos": [],
        "nao_roteirizados": [],
        "logs": logs,
        "ocupacao_media_peso": 0,
        "ocupacao_media_volume": 0,
    }
