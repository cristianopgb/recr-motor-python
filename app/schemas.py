from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# CARTEIRA REC - CONTRATO OFICIAL (38 COLUNAS)
# IMPORTANTE:
# - aceitar exatamente o shape vindo do Sistema 1
# - permitir campos extras sem quebrar
# - não inventar renomeações fora do contrato
# ============================================================

class CarteiraItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    Filial: Optional[Any] = None
    Romane: Optional[Any] = None
    Filial_origem: Optional[Any] = Field(default=None, alias="Filial (origem)")
    Série: Optional[Any] = None
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
    Destinatário: Optional[Any] = None
    Bairro: Optional[Any] = None
    Cida: Optional[Any] = None
    UF: Optional[Any] = None
    NF_Serie: Optional[Any] = Field(default=None, alias="NF / Serie")
    Tipo_Carga: Optional[Any] = Field(default=None, alias="Tipo Carga")
    Qtd_NF: Optional[Any] = Field(default=None, alias="Qtd.NF")
    Região: Optional[Any] = None
    Sub_Região: Optional[Any] = Field(default=None, alias="Sub-Região")
    Ocorrencias_NFs: Optional[Any] = Field(default=None, alias="Ocorrências NFs")
    Remetente: Optional[Any] = None
    Observacao_R: Optional[Any] = Field(default=None, alias="Observação R")
    Ref_Cliente: Optional[Any] = Field(default=None, alias="Ref Cliente")
    Cidade_Dest: Optional[Any] = Field(default=None, alias="Cidade Dest.")
    Mesoregião: Optional[Any] = None
    Agenda: Optional[Any] = None
    Tipo_C: Optional[Any] = Field(default=None, alias="Tipo C")
    Última: Optional[Any] = None
    Status: Optional[Any] = None
    Lat: Optional[Any] = Field(default=None, alias="Lat.")
    Lon: Optional[Any] = Field(default=None, alias="Lon.")


# ============================================================
# VEÍCULOS - CONTRATO OFICIAL DO PAYLOAD
# IMPORTANTE:
# - removido qualquer conceito antigo como "dedicado"
# - aceitar somente o shape operacional do Sistema 1
# ============================================================

class Veiculo(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

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
    tipo_frota: Optional[Any] = None
    ativo: Optional[Any] = None


# ============================================================
# REGIONALIDADES - CONTRATO ENXUTO
# ============================================================

class Regionalidade(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    cidade: Optional[Any] = None
    uf: Optional[Any] = None
    mesorregiao: Optional[Any] = None
    microrregiao: Optional[Any] = None


# ============================================================
# CONFIGURAÇÃO DE FROTA
# ============================================================

class ConfiguracaoFrotaItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    perfil: Optional[Any] = None
    quantidade: Optional[Any] = None


# ============================================================
# PARÂMETROS - CONTRATO OFICIAL
# IMPORTANTE:
# - manter exatamente os campos acordados com o Sistema 1
# - permitir extras sem quebrar
# ============================================================

class Parametros(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    usuario_id: Optional[Any] = None
    usuario_nome: Optional[Any] = None
    filial_id: Optional[Any] = None
    filial_nome: Optional[Any] = None
    upload_id: Optional[Any] = None
    rodada_id: Optional[Any] = None
    data_execucao: Optional[Any] = None
    data_base_roteirizacao: Optional[Any] = None
    origem_sistema: Optional[Any] = None
    tipo_roteirizacao: Optional[Any] = None
    modelo_roteirizacao: Optional[Any] = None
    filtros_aplicados: Optional[Dict[str, Any]] = None
    configuracao_frota: Optional[List[ConfiguracaoFrotaItem]] = None


# ============================================================
# REQUEST COMPLETO
# ============================================================

class RoteirizacaoRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    carteira: List[CarteiraItem]
    veiculos: List[Veiculo]
    regionalidades: List[Regionalidade]
    parametros: Parametros


# ============================================================
# LOG PADRÃO
# ============================================================

class LogItem(BaseModel):
    modulo: str
    status: str
    mensagem: str
    quantidade_entrada: Optional[int] = None
    quantidade_saida: Optional[int] = None


# ============================================================
# RESPONSE PADRÃO
# ============================================================

class RoteirizacaoResponse(BaseModel):
    status: str
    mensagem: str
    resumo: Dict[str, Any]
    manifestos_fechados: List[Dict[str, Any]]
    manifestos_compostos: List[Dict[str, Any]]
    nao_roteirizados: List[Dict[str, Any]]
    logs: List[LogItem]
