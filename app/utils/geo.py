import math
from typing import Tuple


def distancia_haversine(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calcula a distância em quilômetros entre dois pontos geográficos."""
    raio_terra_km = 6371.0

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return raio_terra_km * c


def coordenadas_validas(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180
