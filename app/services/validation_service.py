from typing import Any, Dict, List


def validar(payload: Dict[str, Any]) -> List[str]:
    erros: List[str] = []

    if not payload:
        erros.append("Payload não pode ser vazio.")

    return erros
