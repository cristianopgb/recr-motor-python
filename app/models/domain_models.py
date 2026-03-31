from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Endereco:
    logradouro: str
    numero: str
    bairro: str
    cidade: str
    uf: str
    cep: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class Entrega:
    id: str
    endereco: Endereco
    peso: float = 0.0
    volume: float = 0.0
    janela_inicio: Optional[str] = None
    janela_fim: Optional[str] = None
    atributos: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rota:
    id: str
    entregas: List[Entrega] = field(default_factory=list)
    distancia_total_km: float = 0.0
    atributos: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultadoRoteirizacao:
    rotas: List[Rota] = field(default_factory=list)
    sobras: List[Entrega] = field(default_factory=list)
    atributos: Dict[str, Any] = field(default_factory=dict)
