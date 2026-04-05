from __future__ import annotations

from datetime import datetime
from typing import Any

from app.schemas import RoteirizacaoRequest


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _is_valid_number(value: Any) -> bool:
    try:
        num = float(value)
        return num == num  # evita NaN
    except Exception:
        return False


def _validar_data_iso(value: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception as exc:
        raise ValueError(
            "Campo 'data_base_roteirizacao' inválido. "
            "Use formato ISO datetime, por exemplo: 2026-04-05T12:30:00.000Z"
        ) from exc


def validar_payload(payload: RoteirizacaoRequest) -> None:
    if _is_blank(payload.rodada_id):
        raise ValueError("Campo obrigatório ausente: rodada_id")

    if _is_blank(payload.upload_id):
        raise ValueError("Campo obrigatório ausente: upload_id")

    if _is_blank(payload.usuario_id):
        raise ValueError("Campo obrigatório ausente: usuario_id")

    if _is_blank(payload.filial_id):
        raise ValueError("Campo obrigatório ausente: filial_id")

    if _is_blank(payload.data_base_roteirizacao):
        raise ValueError("Campo obrigatório ausente: data_base_roteirizacao")

    _validar_data_iso(payload.data_base_roteirizacao)

    if payload.tipo_roteirizacao not in {"carteira", "frota"}:
        raise ValueError("tipo_roteirizacao deve ser 'carteira' ou 'frota'")

    if payload.filial is None:
        raise ValueError("Bloco obrigatório ausente: filial")

    if _is_blank(payload.filial.id):
        raise ValueError("Bloco filial inválido: id ausente")

    if _is_blank(payload.filial.nome):
        raise ValueError("Bloco filial inválido: nome ausente")

    if _is_blank(payload.filial.cidade):
        raise ValueError("Bloco filial inválido: cidade ausente")

    if _is_blank(payload.filial.uf):
        raise ValueError("Bloco filial inválido: uf ausente")

    if not _is_valid_number(payload.filial.latitude):
        raise ValueError("Bloco filial inválido: latitude ausente ou inválida")

    if not _is_valid_number(payload.filial.longitude):
        raise ValueError("Bloco filial inválido: longitude ausente ou inválida")

    lat = float(payload.filial.latitude)
    lon = float(payload.filial.longitude)

    if not (-35 <= lat <= 5):
        raise ValueError("Bloco filial inválido: latitude fora de faixa plausível")

    if not (-80 <= lon <= -30):
        raise ValueError("Bloco filial inválido: longitude fora de faixa plausível")

    if payload.tipo_roteirizacao == "frota" and len(payload.configuracao_frota) == 0:
        raise ValueError(
            "tipo_roteirizacao='frota' exige configuracao_frota com pelo menos um perfil"
        )

    if len(payload.carteira) == 0:
        raise ValueError("A carteira enviada ao motor está vazia")

    if len(payload.veiculos) == 0:
        raise ValueError("A lista de veículos enviada ao motor está vazia")

    if len(payload.regionalidades) == 0:
        raise ValueError("A lista de regionalidades enviada ao motor está vazia")
