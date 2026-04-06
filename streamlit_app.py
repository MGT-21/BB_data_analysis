import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ==========================================
# Configurações Iniciais do Streamlit
# ==========================================
st.set_page_config(
    page_title="Dashboard de Risco e Operações",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo global dos gráficos
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
})

# ==========================================
# Carregamento e Processamento de Dados
# ==========================================
@st.cache_data
def load_data():
    """Carrega os dados e processa as regras de fraude para uso em todo o app."""
    try:
        from config import SILVER_DATASET
        df = pd.read_csv(SILVER_DATASET)
    except ImportError:
        # Fallback caso não encontre o arquivo config.py no mesmo diretório
        st.warning("⚠️ Arquivo config.py não encontrado. Certifique-se de que o caminho do dataset está correto. Tentando ler 'silver_dataset.csv' localmente.")
        df = pd.read_csv("silver_dataset.csv") # Substitua pelo caminho real se necessário

    # ---- Criação das features de Fraude baseadas no notebook ----
    if "fraude_score" not in df.columns:
        df["fraude_idioma_estrangeiro"] = (df.get("text_language", "PT") != "PT").astype(int)
        df["fraude_sem_garantia"]       = (df.get("collateral_type", "") == "SEM_GARANTIA").astype(int)
        df["fraude_match_baixo"]        = (df.get("match_score", 1.0) < 0.4).astype(int)
        df["fraude_pii_idioma"]         = ((df.get("text_language", "PT") != "PT") & (df.get("pii_detected", 1) == 0)).astype(int)
        df["fraude_muitas_violacoes"]   = (df.get("rule_violations", 0) > 2).astype(int)
        df["fraude_duplicado"]          = (df.get("is_duplicate", 0) == 1).astype(int)
        df["fraude_ocr_baixo"]          = (df.get("ocr_confidence", 1.0) < 0.5).astype(int)
        df["fraude_compliance_review"]  = (df.get("compliance_status", "") == "REVIEW").astype(int)

        sinais = [
            "fraude_idioma_estrangeiro", "fraude_sem_garantia", "fraude_match_baixo",
            "fraude_pii_idioma", "fraude_muitas_violacoes", "fraude_duplicado",
            "fraude_ocr_baixo", "fraude_compliance_review"
        ]

        # Filtra apenas colunas que realmente existem no DF para evitar erros
        sinais_existentes = [s for s in sinais if s in df.columns]
        df["fraude_score"] = df[sinais_existentes].sum(axis=1)
        df["fraude_risco"] = pd.cut(
            df["fraude_score"],
            bins=[-1, 1, 3, 8],
            labels=["BAIXO", "MEDIO", "ALTO"]
        )
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()

# ==========================================
# Menu Lateral (Sidebar)
# ==========================================
st.sidebar.title("📊 Navegação")
menu = st.sidebar.radio(
    "Selecione a Visão:",
    [
        "1. Visão por Segmentos",
        "2. Performance do Sistema (OCR)",
        "3. Inadimplência",
        "4. Risco de Fraude"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Total de Registros:** {len(df):,}")

# ==========================================
# PÁGINA 1: VISÃO POR SEGMENTOS
# ==========================================
if menu == "1. Visão por Segmentos":
    st.title("👥 Visão por Segmentos de Cliente")

    resumo = (
        df.groupby("customer_segment", observed=True)
        .agg(
            total               = ("default_12m", "count"),
            valor_medio         = ("credit_requested_value", "mean"),
            valor_total         = ("credit_requested_value", "sum"),
            renda_media         = ("income_declared", "mean"),
            ltv_medio           = ("ltv", "mean"),
            pd_score_medio      = ("pd_model_score", "mean"),
            default_rate        = ("default_12m", "mean"),
            pct_aprovado        = ("final_decision", lambda x: (x == "APPROVE").mean()),
        )
        .round(4)
        .reset_index()
        .sort_values("valor_total", ascending=False)
    )

    # Métricas Globais
    col1, col2, col3 = st.columns(3)
    col1.metric("Segmentos Únicos", df["customer_segment"].nunique())
    col2.metric("Valor Total Solicitado", f"R$ {resumo['valor_total'].sum():,.0f}")
    col3.metric("Ticket Médio Geral", f"R$ {df['credit_requested_value'].mean():,.0f}")

    st.markdown("---")

    # Volume e Valor
    st.subheader("Volume e Valor de Crédito")
    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5))

    # Número de solicitações
    sns.barplot(data=resumo, x="customer_segment", y="total", hue="customer_segment", palette="Blues_d", legend=False, ax=axes1[0])
    for bar, val in zip(axes1[0].patches, resumo["total"]):
        axes1[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (val*0.02), f"{int(val):,}", ha="center", fontsize=9, fontweight="bold")
    axes1[0].set_title("Número de Solicitações")
    axes1[0].set_ylabel("Quantidade")
    axes1[0].set_xlabel("")
    axes1[0].tick_params(axis='x', rotation=30)

    # Valor Médio
    resumo_sorted_vm = resumo.sort_values("valor_medio", ascending=False)
    sns.barplot(data=resumo_sorted_vm, x="customer_segment", y="valor_medio", hue="customer_segment", palette="Oranges_d", legend=False, ax=axes1[1])
    for bar, val in zip(axes1[1].patches, resumo_sorted_vm["valor_medio"]):
        axes1[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (val*0.02), f"R${val:,.0f}", ha="center", fontsize=9, fontweight="bold")
    axes1[1].set_title("Valor Médio Solicitado (R$)")
    axes1[1].set_ylabel("Valor (R$)")
    axes1[1].set_xlabel("")
    axes1[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    st.pyplot(fig1)

    # Taxas de Aprovação e Default
    st.subheader("Taxas de Aprovação e Inadimplência")
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))

    ordem_apr = resumo.sort_values("pct_aprovado", ascending=False)["customer_segment"]
    dados_apr = resumo.set_index("customer_segment").loc[ordem_apr, "pct_aprovado"] * 100
    bars = axes2[0].bar(dados_apr.index, dados_apr.values, color="#42a5f5", edgecolor="white")
    for bar, val in zip(bars, dados_apr.values):
        axes2[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes2[0].set_title("Taxa de Aprovação (%)")
    axes2[0].set_ylim(0, 115)
    axes2[0].tick_params(axis='x', rotation=30)

    ordem_def = resumo.sort_values("default_rate", ascending=False)["customer_segment"]
    dados_def = resumo.set_index("customer_segment").loc[ordem_def, "default_rate"] * 100
    cores_def = ["#f44336" if v >= dados_def.mean() else "#ef9a9a" for v in dados_def.values]
    bars = axes2[1].bar(dados_def.index, dados_def.values, color=cores_def, edgecolor="white")
    for bar, val in zip(bars, dados_def.values):
        axes2[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes2[1].axhline(y=dados_def.mean(), color="gray", linestyle="--", label=f"Média: {dados_def.mean():.1f}%")
    axes2[1].set_title("Taxa de Inadimplência (%)")
    axes2[1].set_ylim(0, dados_def.max() + 5)
    axes2[1].tick_params(axis='x', rotation=30)
    axes2[1].legend()

    plt.tight_layout()
    st.pyplot(fig2)

    # Perfil Financeiro
    st.subheader("Perfil Financeiro por Segmento")
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    ordem_renda = resumo.sort_values("valor_medio", ascending=True)["customer_segment"]
    dados_renda = resumo.set_index("customer_segment").loc[ordem_renda]

    # Renda
    cores_renda = sns.color_palette("Blues_d", len(dados_renda))
    axes3[0].barh(dados_renda.index, dados_renda["renda_media"], color=cores_renda, edgecolor="white")
    axes3[0].set_title("Renda Média Declarada")
    axes3[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))

    # Valor Solicitado
    cores_credito = sns.color_palette("Oranges_d", len(dados_renda))
    axes3[1].barh(dados_renda.index, dados_renda["valor_medio"], color=cores_credito, edgecolor="white")
    axes3[1].set_title("Crédito Médio Solicitado")
    axes3[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))

    # Ratio
    dados3 = resumo.copy()
    dados3["ratio"] = dados3["valor_medio"] / dados3["renda_media"]
    dados3 = dados3.sort_values("ratio", ascending=True)
    cores_ratio = ["#f44336" if v > 0.5 else "#ff9800" if v > 0.4 else "#4caf50" for v in dados3["ratio"]]
    axes3[2].barh(dados3["customer_segment"], dados3["ratio"], color=cores_ratio, edgecolor="white")
    axes3[2].axvline(x=0.5, color="gray", linestyle="--", label="Limite (0.5x)")
    axes3[2].set_title("Relação Crédito / Renda")
    axes3[2].legend()

    plt.tight_layout()
    st.pyplot(fig3)

# ==========================================
# PÁGINA 2: PERFORMANCE DO SISTEMA
# ==========================================
elif menu == "2. Performance do Sistema (OCR)":
    st.title("⚙️ Performance do Sistema e Motores OCR")

    # Matriz Acerto / Erro
    aprovados = df[df["final_decision"] == "APPROVE"]
    em_revisao = df[df["final_decision"] == "REVIEW"]

    acerto_aprovado = (aprovados["default_12m"] == 0).sum()
    erro_aprovado   = (aprovados["default_12m"] == 1).sum()
    acerto_revisao  = (em_revisao["default_12m"] == 1).sum()
    falso_alarme    = (em_revisao["default_12m"] == 0).sum()
    total           = len(df)

    st.subheader("Resumo de Decisões")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Aprovados", f"{len(aprovados):,}")
    col2.metric("Total em Revisão", f"{len(em_revisao):,}")
    col3.metric("Taxa de Erro Geral", f"{erro_aprovado/total*100:.1f}%", help="Aprovações que viraram inadimplência sobre o total.")

    # Gráfico de Acerto/Erro
    fig_sys, axes_sys = plt.subplots(1, 3, figsize=(16, 6))

    # Approve
    val_apr = [acerto_aprovado/len(aprovados)*100, erro_aprovado/len(aprovados)*100]
    axes_sys[0].bar(["APPROVE"], [val_apr[0]], color="#4caf50", width=0.5)
    axes_sys[0].bar(["APPROVE"], [val_apr[1]], bottom=[val_apr[0]], color="#f44336", width=0.5)
    axes_sys[0].set_title(f"APPROVE ({len(aprovados):,})")
    axes_sys[0].set_ylabel("% dos casos")
    axes_sys[0].legend(["Acertou (pagou)", "Errou (inadimpliu)"], loc="upper right", fontsize=9)

    # Review
    val_rev = [acerto_revisao/len(em_revisao)*100, falso_alarme/len(em_revisao)*100]
    axes_sys[1].bar(["REVIEW"], [val_rev[0]], color="#4caf50", width=0.5)
    axes_sys[1].bar(["REVIEW"], [val_rev[1]], bottom=[val_rev[0]], color="#ff9800", width=0.5)
    axes_sys[1].set_title(f"REVIEW ({len(em_revisao):,})")
    axes_sys[1].legend(["Acertou (risco real)", "Falso alarme (pagou)"], loc="upper right", fontsize=9)

    # Geral Pie
    categorias = ["Aprovado/Pagou", "Aprovado/Inadimpliu", "Revisão/Risco Real", "Revisão/Falso Alarme"]
    valores_pizza = [acerto_aprovado, erro_aprovado, acerto_revisao, falso_alarme]
    cores_pizza   = ["#4caf50", "#f44336", "#81c784", "#ff9800"]
    axes_sys[2].pie(valores_pizza, labels=categorias, colors=cores_pizza, autopct="%1.1f%%", explode=[0, 0.05, 0, 0.05])
    axes_sys[2].set_title("Visão Geral - Todas as Decisões")

    plt.tight_layout()
    st.pyplot(fig_sys)

    st.markdown("---")

    # Análise de OCR
    st.subheader("Performance dos Motores OCR")

    c1, c2 = st.columns(2)
    with c1:
        # Erros e Confiança OCR
        erros_motor = (
            df.groupby("ocr_engine", observed=True)
            .agg(erros_medio=("ocr_error_count", "mean"), confianca_media=("ocr_confidence", "mean"))
            .reset_index().sort_values("erros_medio", ascending=False)
        )
        fig_ocr, axes_ocr = plt.subplots(1, 2, figsize=(10, 4))
        sns.barplot(data=erros_motor, x="ocr_engine", y="erros_medio", hue="ocr_engine", palette="Blues_r", legend=False, ax=axes_ocr[0])
        axes_ocr[0].set_title("Erros por Documento")
        axes_ocr[0].tick_params(axis='x', rotation=30)

        ordem_conf = erros_motor.sort_values("confianca_media", ascending=False)
        sns.barplot(data=ordem_conf, x="ocr_engine", y="confianca_media", hue="ocr_engine", palette="Greens_r", legend=False, ax=axes_ocr[1])
        axes_ocr[1].set_title("Confiança Média (0-1)")
        axes_ocr[1].set_ylim(0, 1.0)
        axes_ocr[1].tick_params(axis='x', rotation=30)

        plt.tight_layout()
        st.pyplot(fig_ocr)

    with c2:
        # Heatmap Confiança
        if "doc_type" in df.columns:
            pivot_conf = (df.groupby(["ocr_engine", "doc_type"], observed=True)["ocr_confidence"].mean().unstack() * 100)
            fig_hm, ax_hm = plt.subplots(figsize=(8, 4))
            sns.heatmap(pivot_conf, annot=True, fmt=".1f", cmap="RdYlGn", vmin=50, vmax=100, linewidths=0.5, ax=ax_hm)
            ax_hm.set_title("Confiança do OCR (%) por Tipo de Doc")
            st.pyplot(fig_hm)

# ==========================================
# PÁGINA 3: INADIMPLÊNCIA
# ==========================================
elif menu == "3. Inadimplência":
    st.title("💸 Análise de Inadimplência (Default 12m)")

    taxa_geral = df["default_12m"].mean() * 100

    col1, col2 = st.columns(2)
    col1.metric("Total de Inadimplentes", f"{df['default_12m'].sum():,}")
    col2.metric("Taxa de Inadimplência Geral", f"{taxa_geral:.1f}%")

    st.markdown("---")

    # Visão Geral (Pie e Região)
    fig_def, axes_def = plt.subplots(1, 2, figsize=(13, 5))

    contagem = df["default_12m"].value_counts().sort_index()
    axes_def[0].pie(contagem.values, labels=["Adimplente", "Inadimplente"], colors=["#4caf50", "#f44336"], autopct="%1.1f%%")
    axes_def[0].set_title("Proporção Geral")

    if "regiao" in df.columns:
        def_regiao = df.groupby("regiao")["default_12m"].mean() * 100
        def_regiao = def_regiao.sort_values(ascending=False)
        cores_bar = ["#f44336" if v >= taxa_geral else "#ef9a9a" for v in def_regiao.values]
        axes_def[1].bar(def_regiao.index, def_regiao.values, color=cores_bar)
        axes_def[1].axhline(y=taxa_geral, color="gray", linestyle="--", label=f"Média: {taxa_geral:.1f}%")
        axes_def[1].set_title("Inadimplência por Região (%)")
        axes_def[1].legend()

    plt.tight_layout()
    st.pyplot(fig_def)

    # LTV
    st.subheader("Inadimplência por Faixa de LTV")
    df["ltv_faixa"] = pd.cut(df["ltv"], bins=[0, 0.5, 1.0, 1.5, 2.0, 100], labels=["0–50%", "50–100%", "100–150%", "150–200%", ">200%"])
    ltv_default = df.groupby("ltv_faixa", observed=True).agg(default_rate=("default_12m", "mean"), total=("default_12m", "count")).reset_index()

    fig_ltv, ax_ltv = plt.subplots(figsize=(10, 4))
    sns.barplot(data=ltv_default, x="ltv_faixa", y=ltv_default["default_rate"]*100, hue="ltv_faixa", palette="RdYlGn_r", legend=False, ax=ax_ltv)
    ax_ltv.axhline(y=taxa_geral, color="gray", linestyle="--", label="Média Geral")
    ax_ltv.set_title("Taxa de Inadimplência por LTV (Loan-to-Value)")
    ax_ltv.set_ylabel("% Inadimplência")
    ax_ltv.legend()
    st.pyplot(fig_ltv)

    # Heatmap Segmento x Setor
    if "industry_sector" in df.columns:
        st.subheader("Inadimplência: Segmento x Setor")
        pivot_setor = (df.groupby(["customer_segment", "industry_sector"], observed=True)["default_12m"].mean().unstack() * 100)
        fig_hs, ax_hs = plt.subplots(figsize=(12, 5))
        sns.heatmap(pivot_setor, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=0.5, ax=ax_hs)
        st.pyplot(fig_hs)

# ==========================================
# PÁGINA 4: RISCO DE FRAUDE
# ==========================================
elif menu == "4. Risco de Fraude":
    st.title("🚨 Análise e Scoring de Risco de Fraude")

    # Cores de Risco
    COR_BAIXO, COR_MEDIO, COR_ALTO = "#4caf50", "#ff9800", "#f44336"
    CORES_RISCO = [COR_BAIXO, COR_MEDIO, COR_ALTO]

    st.markdown("""
    **Metodologia do Score:** O sistema calcula o risco somando violações operacionais, anomalias de OCR, documentação em idiomas estrangeiros sem dados sensíveis associados e outros indicadores suspeitos (0 a 8 pontos).
    """)

    # Distribuição
    contagem = df["fraude_risco"].value_counts().reindex(["BAIXO", "MEDIO", "ALTO"])

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Risco Baixo", f"{contagem['BAIXO']:,}")
    col2.metric("🟠 Risco Médio", f"{contagem['MEDIO']:,}")
    col3.metric("🔴 Risco Alto", f"{contagem['ALTO']:,}")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Frequência dos sinais
        sinais = ["fraude_idioma_estrangeiro", "fraude_sem_garantia", "fraude_match_baixo", "fraude_pii_idioma", "fraude_muitas_violacoes", "fraude_duplicado", "fraude_ocr_baixo", "fraude_compliance_review"]
        sinais_validos = [s for s in sinais if s in df.columns]

        freq = df[sinais_validos].mean().sort_values(ascending=True) * 100
        fig_sinais, ax_sin = plt.subplots(figsize=(8, 5))
        axes = ax_sin.barh(freq.index, freq.values, color="#e57373")
        ax_sin.set_title("Frequência de Sinais de Alerta (%)")
        st.pyplot(fig_sinais)

    with col_chart2:
        # Score médio por segmento
        score_segmento = df.groupby("customer_segment", observed=True)["fraude_score"].mean().sort_values(ascending=False)
        fig_seg, ax_seg = plt.subplots(figsize=(8, 5))
        cores_seg = [COR_ALTO if v >= 2.5 else COR_MEDIO if v >= 2.0 else COR_BAIXO for v in score_segmento.values]
        ax_seg.bar(score_segmento.index, score_segmento.values, color=cores_seg)
        ax_seg.set_title("Score Médio de Fraude por Segmento")
        ax_seg.tick_params(axis='x', rotation=30)
        ax_seg.axhline(y=score_segmento.mean(), color="gray", linestyle="--")
        st.pyplot(fig_seg)

    # Tabela de Alto Risco para investigação
    st.subheader("📋 Amostra de Casos de Alto Risco para Investigação")
    casos_alto_risco = df[df["fraude_risco"] == "ALTO"].sort_values("fraude_score", ascending=False).head(50)

    colunas_exibir = ["customer_segment", "final_decision", "fraude_score", "ocr_engine", "text_language", "rule_violations"]
    colunas_validas = [c for c in colunas_exibir if c in casos_alto_risco.columns]

    st.dataframe(casos_alto_risco[colunas_validas], use_container_width=True)
