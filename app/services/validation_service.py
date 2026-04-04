from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from app.schemas import RoteirizacaoRequest


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _parse_iso_datetime(value: Any) -> datetime | None:
    if _is_blank(value):
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        texto = value.strip()

        # suporta final "Z"
        if texto.endswith("Z"):
            texto = texto[:-1] + "+00:00"

        try:
            return datetime.fromisoformat(texto)
        except ValueError:
            return None

    return None


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _perfis_configuracao_frota(configuracao_frota: Iterable[Any]) -> set[str]:
    perfis: set[str] = set()

    for item in configuracao_frota:
        if item is None:
            continue

        perfil = getattr(item, "perfil", None)
        if isinstance(item, dict):
            perfil = item.get("perfil")

        if not _is_blank(perfil):
            perfis.add(str(perfil).strip().upper())

    return perfis


def validar_payload(payload: RoteirizacaoRequest) -> None:
    """
    Validação de contrato de entrada entre Sistema 1 e Sistema 2.

    Regras principais:
    - carteira vazia é válida
    - veiculos vazio é válido
    - regionalidades deve existir
    - parametros obrigatórios devem existir
    - tipo_roteirizacao deve ser coerente com configuracao_frota
    """

    if payload is None:
        raise ValueError("Payload ausente.")

    if payload.parametros is None:
        raise ValueError("Bloco 'parametros' ausente.")

    parametros = payload.parametros

    # ============================================================
    # CAMPOS OBRIGATÓRIOS DE PARAMETROS
    # ============================================================
    campos_obrigatorios = {
        "usuario_id": getattr(parametros, "usuario_id", None),
        "filial_id": getattr(parametros, "filial_id", None),
        "upload_id": getattr(parametros, "upload_id", None),
        "rodada_id": getattr(parametros, "rodada_id", None),
        "data_execucao": getattr(parametros, "data_execucao", None),
        "data_base_roteirizacao": getattr(parametros, "data_base_roteirizacao", None),
        "origem_sistema": getattr(parametros, "origem_sistema", None),
        "tipo_roteirizacao": getattr(parametros, "tipo_roteirizacao", None),
    }

    faltantes = [campo for campo, valor in campos_obrigatorios.items() if _is_blank(valor)]
    if faltantes:
        raise ValueError(
            "Campos obrigatórios ausentes em 'parametros': " + ", ".join(faltantes)
        )

    # ============================================================
    # VALIDAÇÃO DAS DATAS
    # ============================================================
    data_execucao = _parse_iso_datetime(parametros.data_execucao)
    if data_execucao is None:
        raise ValueError("Campo 'data_execucao' inválido. Esperado ISO 8601.")

    data_base = _parse_iso_datetime(parametros.data_base_roteirizacao)
    if data_base is None:
        raise ValueError("Campo 'data_base_roteirizacao' inválido. Esperado ISO 8601.")

    # ============================================================
    # ORIGEM DO SISTEMA
    # ============================================================
    origem_sistema = str(parametros.origem_sistema).strip()
    if origem_sistema != "sistema_1":
        raise ValueError("Campo 'origem_sistema' inválido. Esperado: 'sistema_1'.")

    # ============================================================
    # TIPO DE ROTEIRIZAÇÃO
    # ============================================================
    tipo_roteirizacao = str(parametros.tipo_roteirizacao).strip().lower()
    if tipo_roteirizacao not in {"carteira", "frota"}:
        raise ValueError(
            "Campo 'tipo_roteirizacao' inválido. Esperado: 'carteira' ou 'frota'."
        )

    configuracao_frota = _ensure_list(getattr(parametros, "configuracao_frota", []))

    if tipo_roteirizacao == "carteira":
        if len(configuracao_frota) > 0:
            raise ValueError(
                "Para tipo_roteirizacao='carteira', 'configuracao_frota' deve estar vazia."
            )

    if tipo_roteirizacao == "frota":
        if len(configuracao_frota) == 0:
            raise ValueError(
                "Para tipo_roteirizacao='frota', 'configuracao_frota' é obrigatória."
            )

        perfis = _perfis_configuracao_frota(configuracao_frota)
        if len(perfis) == 0:
            raise ValueError(
                "Para tipo_roteirizacao='frota', 'configuracao_frota' deve conter ao menos um perfil válido."
            )

        for item in configuracao_frota:
            if item is None:
                raise ValueError("Item nulo em 'configuracao_frota'.")

            perfil = getattr(item, "perfil", None)
            quantidade = getattr(item, "quantidade", None)

            if isinstance(item, dict):
                perfil = item.get("perfil")
                quantidade = item.get("quantidade")

            if _is_blank(perfil):
                raise ValueError(
                    "Todos os itens de 'configuracao_frota' devem ter 'perfil'."
                )

            if quantidade is None:
                raise ValueError(
                    "Todos os itens de 'configuracao_frota' devem ter 'quantidade'."
                )

            try:
                qtd = int(quantidade)
            except (ValueError, TypeError):
                raise ValueError(
                    "Campo 'quantidade' em 'configuracao_frota' deve ser numérico inteiro."
                )

            if qtd <= 0:
                raise ValueError(
                    "Campo 'quantidade' em 'configuracao_frota' deve ser maior que zero."
                )

    # ============================================================
    # BLOCOS PRINCIPAIS
    # ============================================================
    if payload.carteira is None:
        raise ValueError("Bloco 'carteira' ausente.")
    if payload.veiculos is None:
        raise ValueError("Bloco 'veiculos' ausente.")
    if payload.regionalidades is None:
        raise ValueError("Bloco 'regionalidades' ausente.")

    # carteira vazia é válida
    # veiculos vazio é válido
    # regionalidades vazia NÃO deve acontecer
    if len(payload.regionalidades) == 0:
        raise ValueError("Bloco 'regionalidades' vazio. É obrigatório enviar fallback geográfico.")

    # ============================================================
    # VALIDAÇÃO MÍNIMA DE REGIONALIDADES
    # ============================================================
    for idx, reg in enumerate(payload.regionalidades, start=1):
        cidade = getattr(reg, "cidade", None)
        uf = getattr(reg, "uf", None)

        if _is_blank(cidade):
            raise ValueError(f"Regionalidade #{idx} sem 'cidade'.")
        if _is_blank(uf):
            raise ValueError(f"Regionalidade #{idx} sem 'uf'.")

    # ============================================================
    # VALIDAÇÃO MÍNIMA DE VEÍCULOS
    # ============================================================
    # veiculos vazio é válido, mas se vier preenchido, ao menos o perfil deve existir
    for idx, veic in enumerate(payload.veiculos, start=1):
        perfil = getattr(veic, "perfil", None)
        if _is_blank(perfil):
            raise ValueError(f"Veículo #{idx} sem 'perfil'.")

    # ============================================================
    # COERÊNCIA EXTRA NO MODO FROTA
    # ============================================================
    if tipo_roteirizacao == "frota" and len(payload.veiculos) > 0:
        perfis_payload = {
            str(getattr(veic, "perfil", "")).strip().upper()
            for veic in payload.veiculos
            if not _is_blank(getattr(veic, "perfil", None))
        }

        perfis_config = _perfis_configuracao_frota(configuracao_frota)

        perfis_fora = perfis_payload - perfis_config
        if perfis_fora:
            raise ValueError(
                "No modo 'frota', há veículos com perfis fora de 'configuracao_frota': "
                + ", ".join(sorted(perfis_fora))
            )
