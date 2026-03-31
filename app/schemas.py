from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class RoteirizacaoRequest(BaseModel):
    payload: Dict[str, Any]


class RoteirizacaoResponse(BaseModel):
    sucesso: bool
    resultado: Optional[Dict[str, Any]] = None
    erros: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    versao: str
