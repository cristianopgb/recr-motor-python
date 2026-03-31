from app.schemas import RoteirizacaoRequest


def validar_payload(payload: RoteirizacaoRequest) -> None:
    """
    Valida o payload de entrada antes da execução do pipeline.
    Falha cedo e com mensagem objetiva.
    """

    # =========================
    # 1. BLOCOS OBRIGATÓRIOS
    # =========================
    if payload.carteira is None:
        raise ValueError("Bloco 'carteira' ausente no payload.")

    if payload.veiculos is None:
        raise ValueError("Bloco 'veiculos' ausente no payload.")

    if payload.regionalidades is None:
        raise ValueError("Bloco 'regionalidades' ausente no payload.")

    if payload.parametros is None:
        raise ValueError("Bloco 'parametros' ausente no payload.")

    # =========================
    # 2. TIPOS / ESTRUTURA
    # =========================
    if not isinstance(payload.carteira, list):
        raise ValueError("Bloco 'carteira' deve ser uma lista.")

    if not isinstance(payload.veiculos, list):
        raise ValueError("Bloco 'veiculos' deve ser uma lista.")

    if not isinstance(payload.regionalidades, list):
        raise ValueError("Bloco 'regionalidades' deve ser uma lista.")

    # =========================
    # 3. CONTEÚDO MÍNIMO
    # =========================
    if len(payload.carteira) == 0:
        raise ValueError("A carteira recebida está vazia.")

    if len(payload.veiculos) == 0:
        raise ValueError("Nenhum veículo foi enviado no payload.")

    if len(payload.regionalidades) == 0:
        raise ValueError("Nenhuma regionalidade foi enviada no payload.")

    # =========================
    # 4. VEÍCULOS ATIVOS
    # =========================
    veiculos_ativos = [v for v in payload.veiculos if v.ativo is True]

    if len(veiculos_ativos) == 0:
        raise ValueError("Nenhum veículo ativo disponível para roteirização.")

    # =========================
    # 5. PARÂMETROS MÍNIMOS
    # =========================
    if not payload.parametros.filial_id:
        raise ValueError("Parâmetro obrigatório ausente: 'filial_id'.")

    if not payload.parametros.data_execucao:
        raise ValueError("Parâmetro obrigatório ausente: 'data_execucao'.")

    # =========================
    # 6. CARTEIRA - CHECAGEM MÍNIMA DE LINHAS
    # =========================
    for idx, item in enumerate(payload.carteira, start=1):
        if item.Filial is None:
            raise ValueError(f"Linha {idx} da carteira sem valor em 'Filial'.")

        if item.UF is None:
            raise ValueError(f"Linha {idx} da carteira sem valor em 'UF'.")

        if item.Cida is None:
            raise ValueError(f"Linha {idx} da carteira sem valor em 'Cida'.")

        if item.Destinatário is None:
            raise ValueError(f"Linha {idx} da carteira sem valor em 'Destinatário'.")

    # Se chegou até aqui, payload válido
    return
