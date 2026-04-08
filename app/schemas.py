from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class CarteiraItem(BaseModel):
    """
    Schema bruto da carteira recebida do Sistema 1.

    Regra deste schema:
    - aceitar o layout novo do cliente como principal
    - manter aliases de compatibilidade para nomes antigos relevantes
    - não fazer regra de negócio aqui
    - deixar o M1 responsável pela padronização interna do pipeline
    """
    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
    )

    # ============================================================
    # IDENTIFICAÇÃO / DOCUMENTO
    # ============================================================
    Filial_R: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Filial R", "Filial"),
    )
    Romane: Optional[Any] = None
    Filial_D: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Filial D", "Filial (origem)"),
    )
    Serie: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Série", "Serie"),
    )
    Nro_Doc: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Nro Doc."),
    )

    # ============================================================
    # DATAS
    # ============================================================
    Data_Des: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Data Des", "Data"),
    )
    Data_NF: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Data NF"),
    )
    DLE: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("D.L.E."),
    )
    Agendam: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Agendam."),
    )

    # ============================================================
    # CARGA / PESO / VALOR
    # ============================================================
    Palet: Optional[Any] = None
    Conf: Optional[Any] = None
    Peso: Optional[Any] = None
    Vlr_Merc: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Vlr.Merc."),
    )
    Qtd: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Qtd."),
    )
    Peso_Cub: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Peso Cub.", "Peso C"),
    )
    Peso_Calculo: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Peso Calculo", "Peso Calculado"),
    )

    # ============================================================
    # CLASSIFICAÇÃO / CLIENTES
    # ============================================================
    Classif: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Classif", "Classifi"),
    )
    Tomad: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Tomad", "Tomador"),
    )
    Destin: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Destin", "Destinatário"),
    )
    Bairro: Optional[Any] = None
    Cidad: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Cidad", "Cida"),
    )
    UF: Optional[Any] = None
    NF_Serie: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("NF / Serie"),
    )
    Tipo_Ca: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Tipo Ca"),
    )
    Tipo_Carga: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Tipo Carga", "Tipo C"),
    )
    Qtd_NF: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Qtd.NF"),
    )

    # ============================================================
    # REGIONALIDADE / OBSERVAÇÕES
    # ============================================================
    Mesoregiao: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Mesoregião"),
    )
    Sub_Regiao: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Sub-Região"),
    )
    Ocorrencias_NF: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Ocorrências NF", "Ocorrências NFs"),
    )
    Remetente: Optional[Any] = None
    Observacao: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Observação", "Observação R"),
    )
    Ref_Cliente: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Ref Cliente"),
    )
    Cidade_Dest: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Cidade Dest."),
    )
    Agenda: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Agenda"),
    )
    Ultima_Ocorrencia: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Última Ocorrência", "Última"),
    )
    Status_R: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Status R", "Status"),
    )

    # ============================================================
    # GEO
    # ============================================================
    Latitude: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Latitude", "Lat."),
    )
    Longitude: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Longitude", "Lon."),
    )

    # ============================================================
    # NOVOS CAMPOS OPERACIONAIS V2
    # ============================================================
    Prioridade: Optional[Any] = None
    Restricao_Veiculo: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Restrição Veículo"),
    )
    Carro_Dedicado: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Carro Dedicado", "Veiculo Exclusivo"),
    )
    Inicio_Ent: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Inicio Ent."),
    )
    Fim_En: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Fim En", "Fim Ent.", "Fim Ent"),
    )


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
    dedicado: Optional[Any] = None


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
    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
    )

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
