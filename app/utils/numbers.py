from typing import Union


def arredondar(valor: float, casas: int = 2) -> float:
    return round(valor, casas)


def to_float(valor: Union[str, int, float, None], padrao: float = 0.0) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return padrao


def to_int(valor: Union[str, float, int, None], padrao: int = 0) -> int:
    try:
        return int(valor)
    except (TypeError, ValueError):
        return padrao
