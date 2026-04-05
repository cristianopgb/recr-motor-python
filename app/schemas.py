from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class CarteiraItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    Filial: Optional[Any] = None
    Romane: Optional[Any] = None
    Filial_origem: Optional[Any] = Field(default=None, alias="Filial (origem)")
    Serie: Optional[Any] = Field(default=None, alias="Série")
    Nro_Doc: Optional[Any] = Field(default=None, alias="Nro Doc.")
    Data_Des: Optional[Any] = Field(default=None, alias="Data Des")
    Data_NF: Optional[Any] = Field(default=None, alias="Data NF")
    DLE: Optional[Any] = Field(default=None, alias="D.L.E.")
    Agendam: Optional[Any] = Field(default=None, alias="Agendam.")
    Palet: Optional[Any] = None
    Conf: Optional[Any] = None
    Peso: Optional[Any] = None
    Vlr_Merc: Optional[Any] = Field(default=None, alias="Vlr.Merc.")
    Qtd: Optional[Any] = Field(default=None, alias="Qtd.")
    Peso_C: Optional[Any] = Field(default=None, alias="Peso C")
    Classifi: Optional[Any] = None
    Tomador: Optional[Any] = None
    Destinatario: Optional[Any] = Field(default=None, alias="Destinatário")
    Bairro: Optional[Any] = None
    Cida: Optional[Any] = None
    UF: Optional[Any] = None
    NF_Serie: Optional[Any] = Field(default=None, alias="NF / Serie")
    Tipo_Carga: Optional[Any] = Field(default=None, alias="Tipo Carga")
    Qtd_NF: Optional[Any] = Field(default=None, alias="Qtd.NF")
    Regiao: Optional[Any] = Field(default=None, alias="Região")
    Sub_Regiao: Optional[Any] = Field(default=None, alias="Sub-Região")
    Ocorrencias_NFs: Optional[Any] = Field(default=None, alias="Ocorrências NFs")
    Remetente: Optional[Any] = None
    Observacao_R: Optional[Any] = Field(default=None, alias="Observação R")
    Ref_Cliente: Optional[Any] = Field(default=None, alias="Ref Cliente")
    Cidade_Dest: Optional[Any] = Field(default=None, alias="Cidade Dest.")
    Mesoregiao: Optional[Any] = Field(default=None, alias="Mesoregião")
    Agenda: Optional[Any] = None
    Tipo_C: Optional[Any] = Field(default=None, alias="Tipo C")
    Ultima: Optional[Any] = Field(default=None, alias="Última")
    Status: Optional[Any] = None
    Lat: Optional[Any] = Field(default=None, alias="Lat.")
    Lon: Optional[Any] = Field(default=None, alias="Lon.")


class Veiculo(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[Any] = None
    placa: Optional[Any] = None
    perfil: Optional[Any] = None
    qtd_eixos: Optional[Any] = None
    capacidade_peso_kg: Optional[Any] = None
    capacidade_vol_m3: Optional[Any] = None
    max_entregas: Optional[Any] = None
    max_km_distancia: Optional[Any] = None
    ocupacao_minima_perc: Optional[Any] = None
    filial_id: Optional[Any] = None
    ativo: Optional[Any] = None
    tipo_frota: Optional[Any] = None


class Regionalidade(BaseModel):
    model_config = ConfigDict(extra="allow")

    cidade: Optional[Any] = None
    uf: Optional[Any] = None
    mesorregiao: Optional[Any] = None
    microrregiao: Optional[Any] = None


class FilialRodada(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    nome: str
    cidade: str
    uf: str
    latitude: float
    longitude: float


class ConfiguracaoFrotaItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    perfil: str
    quantidade: int


class RoteirizacaoRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    rodada_id: str
    upload_id: str
    usuario_id: str
    filial_id: str
    data_base_roteirizacao: str
    tipo_roteirizacao: Literal["carteira", "frota"]

    filial: FilialRodada

    carteira: List[CarteiraItem] = Field(default_factory=list)
    veiculos: List[Veiculo] = Field(default_factory=list)
    regionalidades: List[Regionalidade] = Field(default_factory=list)

    parametros: Dict[str, Any] = Field(default_factory=dict)
    configuracao_frota: List[ConfiguracaoFrotaItem] = Field(default_factory=list)
