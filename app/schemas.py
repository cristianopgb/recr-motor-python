from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# =========================
# CARTEIRA (38 COLUNAS EXATAS)
# =========================
class CarteiraItem(BaseModel):
    Filial: Optional[str]
    Romane: Optional[str]
    Filial_origem: Optional[str] = Field(alias="Filial (origem)")
    Série: Optional[str]
    Nro_Doc: Optional[str] = Field(alias="Nro Doc.")
    Data_Des: Optional[str] = Field(alias="Data Des")
    Data_NF: Optional[str] = Field(alias="Data NF")
    DLE: Optional[str] = Field(alias="D.L.E.")
    Agendam: Optional[str] = Field(alias="Agendam.")
    Palet: Optional[str]
    Conf: Optional[str]
    Peso: Optional[str]
    Vlr_Merc: Optional[str] = Field(alias="Vlr.Merc.")
    Qtd: Optional[str] = Field(alias="Qtd.")
    Peso_C: Optional[str] = Field(alias="Peso C")
    Classifi: Optional[str]
    Tomador: Optional[str]
    Destinatário: Optional[str]
    Bairro: Optional[str]
    Cida: Optional[str]
    UF: Optional[str]
    NF_Serie: Optional[str] = Field(alias="NF / Serie")
    Tipo_Carga: Optional[str] = Field(alias="Tipo Carga")
    Qtd_NF: Optional[str] = Field(alias="Qtd.NF")
    Região: Optional[str]
    Sub_Região: Optional[str] = Field(alias="Sub-Região")
    Ocorrencias_NFs: Optional[str] = Field(alias="Ocorrências NFs")
    Remetente: Optional[str]
    Observacao_R: Optional[str] = Field(alias="Observação R")
    Ref_Cliente: Optional[str] = Field(alias="Ref Cliente")
    Cidade_Dest: Optional[str] = Field(alias="Cidade Dest.")
    Mesoregiao: Optional[str] = Field(alias="Mesoregião")
    Agenda: Optional[str]
    Tipo_C: Optional[str] = Field(alias="Tipo C")
    Ultima: Optional[str] = Field(alias="Última")
    Status: Optional[str]
    Lat: Optional[str] = Field(alias="Lat.")
    Lon: Optional[str] = Field(alias="Lon.")


# =========================
# VEÍCULOS
# =========================
class Veiculo(BaseModel):
    id: str
    placa: Optional[str]
    perfil: Optional[str]
    tipo_veiculo: Optional[str]
    capacidade_peso_kg: Optional[float]
    capacidade_vol_m3: Optional[float]
    qtd_eixos: Optional[int]
    max_entregas: Optional[int]
    max_km_distancia: Optional[float]
    ocupacao_minima_perc: Optional[float]
    dedicado: Optional[bool]
    tipo_frota: Optional[str]
    filial_id: Optional[str]
    ativo: Optional[bool]


# =========================
# REGIONALIDADE
# =========================
class Regionalidade(BaseModel):
    cidade: str
    uf: str
    mesorregiao: Optional[str]
    microrregiao: Optional[str]


# =========================
# PARÂMETROS
# =========================
class Parametros(BaseModel):
    usuario_id: Optional[str]
    filial_id: Optional[str]
    data_execucao: Optional[str]
    modelo_roteirizacao: Optional[str]
    filtros_aplicados: Optional[Dict[str, Any]]


# =========================
# REQUEST COMPLETO
# =========================
class RoteirizacaoRequest(BaseModel):
    carteira: List[CarteiraItem]
    veiculos: List[Veiculo]
    regionalidades: List[Regionalidade]
    parametros: Parametros


# =========================
# LOG
# =========================
class LogItem(BaseModel):
    modulo: str
    status: str
    mensagem: str
    quantidade_entrada: Optional[int]
    quantidade_saida: Optional[int]


# =========================
# RESPOSTA FINAL
# =========================
class RoteirizacaoResponse(BaseModel):
    status: str
    mensagem: str
    resumo: Dict[str, Any]
    manifestos_fechados: List[Dict[str, Any]]
    manifestos_compostos: List[Dict[str, Any]]
    nao_roteirizados: List[Dict[str, Any]]
    logs: List[LogItem]
