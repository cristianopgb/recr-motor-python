from typing import Any, Dict, List


def validar(payload: Dict[str, Any]) -> List[str]:
    erros: List[str] = []

    if payload is None:
        erros.append("Payload não pode ser nulo.")

    return erros
