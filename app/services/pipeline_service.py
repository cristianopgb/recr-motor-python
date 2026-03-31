from typing import Dict, Any
from app.schemas import RoteirizacaoRequest

# IMPORT DOS MÓDULOS DO PIPELINE
from app.pipeline.m0_leitura import executar_m0
from app.pipeline.m1_padronizacao import executar_m1
from app.pipeline.m2_enriquecimento import executar_m2
from app.pipeline.m3_triagem import executar_m3
from app.pipeline.m31_fronteira import executar_m31
from app.pipeline.m4_fechados import executar_m4
from app.pipeline.m5_compostos import executar_m5
from app.pipeline.m51_saneamento import executar_m51
from app.pipeline.m8_sobras import executar_m8
from app.pipeline.m9_consolidacao import executar_m9


def executar_pipeline(payload: RoteirizacaoRequest) -> Dict[str, Any]:
    """
    Executa o pipeline completo respeitando a linhagem oficial:
    M0 → M1 → M2 → M3 → M3.1 → M4 → M5 → M5.1 → M8 → M9
    """

    logs = []

    # =========================
    # M0 — LEITURA
    # =========================
    m0 = executar_m0(payload)
    logs.append({
        "modulo": "M0",
        "status": "ok",
        "mensagem": "Leitura do payload realizada",
        "quantidade_entrada": len(payload.carteira),
        "quantidade_saida": len(m0["df_carteira_raw"])
    })

    # =========================
    # M1 — PADRONIZAÇÃO
    # =========================
    m1 = executar_m1(m0)
    logs.append({
        "modulo": "M1",
        "status": "ok",
        "mensagem": "Base padronizada",
        "quantidade_entrada": len(m0["df_carteira_raw"]),
        "quantidade_saida": len(m1["df_carteira_tratada"])
    })

    # =========================
    # M2 — ENRIQUECIMENTO
    # =========================
    m2 = executar_m2(m1)
    logs.append({
        "modulo": "M2",
        "status": "ok",
        "mensagem": "Base enriquecida",
        "quantidade_entrada": len(m1["df_carteira_tratada"]),
        "quantidade_saida": len(m2["df_carteira_enriquecida"])
    })

    # =========================
    # M3 — TRIAGEM
    # =========================
    m3 = executar_m3(m2)
    logs.append({
        "modulo": "M3",
        "status": "ok",
        "mensagem": "Triagem operacional concluída",
        "quantidade_entrada": len(m2["df_carteira_enriquecida"]),
        "quantidade_saida": len(m3["df_roteirizavel"])
    })

    # =========================
    # M3.1 — FRONTEIRA
    # =========================
    m31 = executar_m31(m3)
    logs.append({
        "modulo": "M3.1",
        "status": "ok",
        "mensagem": "Validação de fronteira aprovada",
        "quantidade_entrada": len(m3["df_roteirizavel"]),
        "quantidade_saida": len(m31["df_roteirizavel_validado"])
    })

    # =========================
    # M4 — FECHADOS
    # =========================
    m4 = executar_m4(m31)
    logs.append({
        "modulo": "M4",
        "status": "ok",
        "mensagem": "Manifestos fechados gerados",
        "quantidade_entrada": len(m31["df_roteirizavel_validado"]),
        "quantidade_saida": len(m4["df_manifestos_fechados"])
    })

    # =========================
    # M5 — COMPOSTOS
    # =========================
    m5 = executar_m5(m4)
    logs.append({
        "modulo": "M5",
        "status": "ok",
        "mensagem": "Manifestos compostos gerados",
        "quantidade_entrada": len(m4["df_remanescente"]),
        "quantidade_saida": len(m5["df_manifestos_compostos"])
    })

    # =========================
    # M5.1 — SANEAMENTO
    # =========================
    m51 = executar_m51(m5)
    logs.append({
        "modulo": "M5.1",
        "status": "ok",
        "mensagem": "Saneamento concluído",
        "quantidade_entrada": len(m5["df_manifestos_compostos"]),
        "quantidade_saida": len(m51["df_manifestos_compostos_validos"])
    })

    # =========================
    # M8 — SOBRAS
    # =========================
    m8 = executar_m8(m51)
    logs.append({
        "modulo": "M8",
        "status": "ok",
        "mensagem": "Não roteirizados classificados",
        "quantidade_entrada": len(m51["df_remanescente_final"]),
        "quantidade_saida": len(m8["df_nao_roteirizados"])
    })

    # =========================
    # M9 — CONSOLIDAÇÃO FINAL
    # =========================
    m9 = executar_m9(m4, m51, m8, logs)

    return m9
