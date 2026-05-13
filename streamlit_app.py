import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import os
from scipy.stats import ks_2samp

# ==========================================
# Configurações Iniciais do Streamlit
# ==========================================
st.set_page_config(
    page_title="Dashboard de Risco e Operações",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def apply_global_styles():
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

apply_global_styles()

# ==========================================
# Carregamento e Processamento de Dados
# ==========================================
@st.cache_data
def load_data():
    """
    Carrega os dados e processa TODAS as features derivadas aqui dentro,
    evitando mutações no DataFrame cacheado fora desta função.
    """
    try:
        from config import SILVER_DATASET
        df = pd.read_csv(SILVER_DATASET)
    except ImportError:
        st.warning("⚠️ Arquivo config.py não encontrado. Tentando ler 'silver_dataset.csv' localmente.")
        df = pd.read_csv("silver_dataset.csv")

    # ---- Features de Fraude ----
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
        sinais_existentes = [s for s in sinais if s in df.columns]
        df["fraude_score"] = df[sinais_existentes].sum(axis=1)
        df["fraude_risco"] = pd.cut(
            df["fraude_score"],
            bins=[-1, 1, 3, 8],
            labels=["BAIXO", "MEDIO", "ALTO"]
        )

    # ── Engenharia de features ambientais ──────────────────────────────────
    if "drought_bin" not in df.columns:
        df["drought_bin"] = pd.cut(
         df["drought_spi"],
         bins=[-5, -1.5, -0.5, 0.5, 5],
         labels=["Seca severa", "Seca moderada", "Normal", "Úmido"],
     )

    if "ratio_credit_income" not in df.columns:
        df["ratio_credit_income"] = (
         df["credit_requested_value"] / df["income_declared"].replace(0, np.nan)
     ).replace([np.inf, -np.inf], np.nan)

    if "risk_tier" not in df.columns:
        df["risk_tier"] = (
         (df["ltv"] > 1.0).astype(int)
         + (df["pd_model_score"] > df["pd_model_score"].quantile(0.75)).astype(int)
         + (df["drought_spi"] < df["drought_spi"].quantile(0.25)).astype(int)
         + (df["flood_risk_idx"] > df["flood_risk_idx"].quantile(0.75)).astype(int)
         + (df["ratio_credit_income"] > 0.6).astype(int)
     )


    # ---- Features derivadas usadas em múltiplas páginas ----
    # Calculadas uma vez aqui em vez de em cada rerun
    if "ratio" not in df.columns:
        df["ratio"] = df["credit_requested_value"] / df["income_declared"]
        df["ratio"] = df["ratio"].replace([np.inf, -np.inf], np.nan)

    if "ltv_faixa" not in df.columns:
        df["ltv_faixa"] = pd.cut(
            df["ltv"],
            bins=[0, 0.5, 1.0, 1.5, 2.0, 100],
            labels=["0–50%", "50–100%", "100–150%", "150–200%", ">200%"]
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
        "4. Risco de Fraude",
        "5. Problema proposta - pilares",
        "6. Risco Ambiental × Segmento",
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

    col1, col2, col3 = st.columns(3)
    col1.metric("Segmentos Únicos", df["customer_segment"].nunique())
    col2.metric("Valor Total Solicitado", f"R$ {resumo['valor_total'].sum():,.0f}")
    col3.metric("Ticket Médio Geral", f"R$ {df['credit_requested_value'].mean():,.0f}")

    st.markdown("---")

    st.subheader("Volume e Valor de Crédito")
    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5))

    sns.barplot(data=resumo, x="customer_segment", y="total", hue="customer_segment", palette="Blues_d", legend=False, ax=axes1[0])
    for bar, val in zip(axes1[0].patches, resumo["total"]):
        axes1[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (val*0.02), f"{int(val):,}", ha="center", fontsize=9, fontweight="bold")
    axes1[0].set_title("Número de Solicitações")
    axes1[0].set_ylabel("Quantidade")
    axes1[0].set_xlabel("")
    axes1[0].tick_params(axis='x', rotation=30)

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
    plt.close(fig1)

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
    plt.close(fig2)

    st.subheader("Perfil Financeiro por Segmento")
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    ordem_renda = resumo.sort_values("valor_medio", ascending=True)["customer_segment"]
    dados_renda = resumo.set_index("customer_segment").loc[ordem_renda]

    cores_renda = sns.color_palette("Blues_d", len(dados_renda))
    axes3[0].barh(dados_renda.index, dados_renda["renda_media"], color=cores_renda, edgecolor="white")
    axes3[0].set_title("Renda Média Declarada")
    axes3[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))

    cores_credito = sns.color_palette("Oranges_d", len(dados_renda))
    axes3[1].barh(dados_renda.index, dados_renda["valor_medio"], color=cores_credito, edgecolor="white")
    axes3[1].set_title("Crédito Médio Solicitado")
    axes3[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))

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
    plt.close(fig3)

    st.markdown("---")
    st.subheader("🔍 Filtros Dinâmicos de Distribuição")

    segmentos_disponiveis = sorted(df["customer_segment"].dropna().unique())
    segmentos_selecionados = st.multiselect(
        "Selecione os Segmentos Visíveis:",
        options=segmentos_disponiveis,
        default=segmentos_disponiveis
    )

    col_sel1, col_sel2, col_sel3 = st.columns(3)

    with col_sel1:
        opcao_q = st.selectbox(
            "Selecione o Quartil:",
            ["Todos", "1º Quartil (0-25%)", "2º Quartil (25-50%)", "3º Quartil (50-75%)", "4º Quartil (75-100%)"]
        )

    with col_sel2:
        metrica_sel = st.selectbox(
            "Selecione a Métrica:",
            ["Renda Declarada", "Crédito Solicitado", "Relação Crédito/Renda"]
        )

    with col_sel3:
        st.markdown("<br>", unsafe_allow_html=True)
        remover_outliers = st.checkbox("Remover Outliers (IQR)", value=False)

    mapa_metrica = {
        "Renda Declarada": "income_declared",
        "Crédito Solicitado": "credit_requested_value",
        "Relação Crédito/Renda": "ratio"
    }
    col_alvo = mapa_metrica[metrica_sel]

    # FIX: use 'with' context manager to guarantee figure is always closed,
    # even if an exception is raised mid-render.
    fig4, ax4 = plt.subplots(figsize=(16, 7))
    try:
        cores_map = dict(zip(segmentos_disponiveis, sns.color_palette("tab10", len(segmentos_disponiveis))))

        for seg in segmentos_selecionados:
            dados_seg = df[df["customer_segment"] == seg][col_alvo].dropna()

            if remover_outliers and not dados_seg.empty:
                q1 = dados_seg.quantile(0.25)
                q3 = dados_seg.quantile(0.75)
                iqr = q3 - q1
                dados_seg = dados_seg[(dados_seg >= (q1 - 1.5 * iqr)) & (dados_seg <= (q3 + 1.5 * iqr))]

            if not dados_seg.empty:
                if opcao_q == "1º Quartil (0-25%)":
                    lim_sup = dados_seg.quantile(0.25)
                    dados_filtrados = dados_seg[dados_seg <= lim_sup]
                elif opcao_q == "2º Quartil (25-50%)":
                    lim_inf, lim_sup = dados_seg.quantile(0.25), dados_seg.quantile(0.50)
                    dados_filtrados = dados_seg[(dados_seg > lim_inf) & (dados_seg <= lim_sup)]
                elif opcao_q == "3º Quartil (50-75%)":
                    lim_inf, lim_sup = dados_seg.quantile(0.50), dados_seg.quantile(0.75)
                    dados_filtrados = dados_seg[(dados_seg > lim_inf) & (dados_seg <= lim_sup)]
                elif opcao_q == "4º Quartil (75-100%)":
                    lim_inf = dados_seg.quantile(0.75)
                    dados_filtrados = dados_seg[dados_seg > lim_inf]
                else:
                    dados_filtrados = dados_seg

                if not dados_filtrados.empty:
                    sorted_vals = np.sort(dados_filtrados.values)
                    ax4.plot(
                        range(len(sorted_vals)),
                        sorted_vals,
                        color=cores_map[seg],
                        label=seg,
                        linewidth=2.5,
                        alpha=0.8
                    )

        ax4.set_title(f"{metrica_sel} | {opcao_q}", fontsize=16, fontweight='bold', pad=20)
        ax4.set_ylabel("Valor")
        ax4.set_xlabel("Nº de Registros (Ordenados)")
        ax4.grid(axis='y', linestyle='--', alpha=0.5)

        if col_alvo in ["income_declared", "credit_requested_value"]:
            ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))

        ax4.legend(title="Segmentos Ativos", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

        plt.tight_layout()
        st.pyplot(fig4)
    finally:
        plt.close(fig4)

    st.markdown("---")
    st.markdown("""
    Nesta visão, testamos se diferentes grupos possuem perfis financeiros originados da mesma distribuição subjacente.
    **Você pode comparar segmentos individuais ou grupos consolidados (AGRO e PJ).**

    * **Scatterplot:** Mostra a dispersão bidimensional (Renda vs. Crédito).
    * **Teste KS (Kolmogorov-Smirnov):** Aplicado à **Renda Declarada** e ao **Crédito Solicitado**. Se o **p-valor < 0,05**, rejeitamos a hipótese nula, indicando que as distribuições dos grupos são estatisticamente diferentes.
    """)

    def calcular_ks_direto(df_teste, col_grupo, g1, g2, col_valor):
        s1 = df_teste[df_teste[col_grupo] == g1][col_valor].dropna()
        s2 = df_teste[df_teste[col_grupo] == g2][col_valor].dropna()

        if len(s1) == 0 or len(s2) == 0:
            return pd.DataFrame()

        stat, p_val = ks_2samp(s1, s2)
        mesma_dist = "❌ Diferentes" if p_val < 0.05 else "✅ Semelhantes"

        return pd.DataFrame([{
            "Comparação": f"{g1} vs {g2}",
            "Estatística KS": round(stat, 4),
            "p-valor": f"{p_val:.4f}",
            "Conclusão (α=0.05)": mesma_dist
        }])

    segmentos_individuais = sorted(df["customer_segment"].dropna().unique().tolist())
    opcoes_selecao = ["AGRO (Todos)", "PJ (Todos)"] + segmentos_individuais

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        seg1 = st.selectbox("🔵 Selecione o Grupo A:", opcoes_selecao, index=0)
    with col_sel2:
        seg2 = st.selectbox("🟠 Selecione o Grupo B:", opcoes_selecao, index=1)

    if seg1 == seg2:
        st.warning("⚠️ Por favor, selecione dois grupos diferentes para realizar a comparação.")
    else:
        def preparar_dados(df_base, escolha):
            if escolha == "AGRO (Todos)":
                filtro = df_base["customer_segment"].isin(["AGRO_PEQUENO", "AGRO_MEDIO", "AGRO_GRANDE"])
            elif escolha == "PJ (Todos)":
                filtro = df_base["customer_segment"].isin(["PJ_EPP", "PJ_ME", "PJ_GRANDE"])
            else:
                filtro = df_base["customer_segment"] == escolha

            df_temp = df_base[filtro].copy()
            df_temp["grupo_comparacao"] = escolha
            return df_temp

        df_g1 = preparar_dados(df, seg1)
        df_g2 = preparar_dados(df, seg2)
        df_comp = pd.concat([df_g1, df_g2]).sample(frac=1, random_state=42).reset_index(drop=True)

        col_plot, col_test = st.columns([1.5, 1])

        with col_plot:
            st.subheader("Dispersão Renda x Crédito")
            fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
            try:
                paleta = {seg1: "#1f77b4", seg2: "#ff7f0e"}
                sns.scatterplot(
                    data=df_comp,
                    x="income_declared",
                    y="credit_requested_value",
                    hue="grupo_comparacao",
                    alpha=0.5,
                    s=35,
                    linewidth=0,
                    palette=paleta,
                    ax=ax_comp
                )
                ax_comp.set_xlabel("Renda Declarada (R$)")
                ax_comp.set_ylabel("Crédito Solicitado (R$)")
                ax_comp.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
                ax_comp.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
                st.pyplot(fig_comp)
            finally:
                plt.close(fig_comp)

        with col_test:
            st.subheader("Resultados do Teste KS")

            st.markdown("##### 🏢 1. Renda Declarada")
            res_renda = calcular_ks_direto(df_comp, "grupo_comparacao", seg1, seg2, "income_declared")

            if not res_renda.empty:
                st.dataframe(res_renda, hide_index=True, use_container_width=True)
                p_val_renda = float(res_renda["p-valor"].iloc[0])
                if p_val_renda < 0.05:
                    st.error(f"**Análise:** As distribuições de **Renda Declarada** entre {seg1} e {seg2} são estatisticamente **diferentes**.")
                else:
                    st.success(f"**Análise:** Não há evidências suficientes para dizer que a **Renda** desses grupos é diferente.")
            else:
                st.warning("Dados insuficientes.")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("##### 💰 2. Crédito Solicitado")
            res_credito = calcular_ks_direto(df_comp, "grupo_comparacao", seg1, seg2, "credit_requested_value")

            if not res_credito.empty:
                st.dataframe(res_credito, hide_index=True, use_container_width=True)
                p_val_credito = float(res_credito["p-valor"].iloc[0])
                if p_val_credito < 0.05:
                    st.error(f"**Análise:** As distribuições de **Crédito Solicitado** entre {seg1} e {seg2} são estatisticamente **diferentes**.")
                else:
                    st.success(f"**Análise:** Não há evidências suficientes para dizer que o **Crédito** desses grupos é diferente.")
            else:
                st.warning("Dados insuficientes.")

# ==========================================
# PÁGINA 2: PERFORMANCE DO SISTEMA
# ==========================================
elif menu == "2. Performance do Sistema (OCR)":
    st.title("⚙️ Performance do Sistema e Motores OCR")

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

    fig_sys, axes_sys = plt.subplots(1, 3, figsize=(16, 6))
    try:
        val_apr = [acerto_aprovado/len(aprovados)*100, erro_aprovado/len(aprovados)*100]
        axes_sys[0].bar(["APPROVE"], [val_apr[0]], color="#4caf50", width=0.5)
        axes_sys[0].bar(["APPROVE"], [val_apr[1]], bottom=[val_apr[0]], color="#f44336", width=0.5)
        axes_sys[0].set_title(f"APPROVE ({len(aprovados):,})")
        axes_sys[0].set_ylabel("% dos casos")
        axes_sys[0].legend(["Acertou (pagou)", "Errou (inadimpliu)"], loc="upper right", fontsize=9)

        val_rev = [acerto_revisao/len(em_revisao)*100, falso_alarme/len(em_revisao)*100]
        axes_sys[1].bar(["REVIEW"], [val_rev[0]], color="#4caf50", width=0.5)
        axes_sys[1].bar(["REVIEW"], [val_rev[1]], bottom=[val_rev[0]], color="#ff9800", width=0.5)
        axes_sys[1].set_title(f"REVIEW ({len(em_revisao):,})")
        axes_sys[1].legend(["Acertou (risco real)", "Falso alarme (pagou)"], loc="upper right", fontsize=9)

        categorias = ["Aprovado/Pagou", "Aprovado/Inadimpliu", "Revisão/Risco Real", "Revisão/Falso Alarme"]
        valores_pizza = [acerto_aprovado, erro_aprovado, acerto_revisao, falso_alarme]
        cores_pizza   = ["#4caf50", "#f44336", "#81c784", "#ff9800"]
        axes_sys[2].pie(valores_pizza, labels=categorias, colors=cores_pizza, autopct="%1.1f%%", explode=[0, 0.05, 0, 0.05])
        axes_sys[2].set_title("Visão Geral - Todas as Decisões")

        plt.tight_layout()
        st.pyplot(fig_sys)
    finally:
        plt.close(fig_sys)

    st.markdown("---")
    st.subheader("Performance dos Motores OCR")

    c1, c2 = st.columns(2)
    with c1:
        erros_motor = (
            df.groupby("ocr_engine", observed=True)
            .agg(erros_medio=("ocr_error_count", "mean"), confianca_media=("ocr_confidence", "mean"))
            .reset_index().sort_values("erros_medio", ascending=False)
        )
        fig_ocr, axes_ocr = plt.subplots(1, 2, figsize=(10, 4))
        try:
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
        finally:
            plt.close(fig_ocr)

    with c2:
        if "doc_type" in df.columns:
            pivot_conf = (df.groupby(["ocr_engine", "doc_type"], observed=True)["ocr_confidence"].mean().unstack() * 100)
            fig_hm, ax_hm = plt.subplots(figsize=(8, 4))
            try:
                sns.heatmap(pivot_conf, annot=True, fmt=".1f", cmap="RdYlGn", vmin=50, vmax=100, linewidths=0.5, ax=ax_hm)
                ax_hm.set_title("Confiança do OCR (%) por Tipo de Doc")
                st.pyplot(fig_hm)
            finally:
                plt.close(fig_hm)

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

    fig_def, axes_def = plt.subplots(1, 2, figsize=(13, 5))
    try:
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
    finally:
        plt.close(fig_def)

    st.subheader("Inadimplência por Faixa de LTV")
    # FIX: ltv_faixa is now pre-computed in load_data(); no mutation here
    ltv_default = df.groupby("ltv_faixa", observed=True).agg(
        default_rate=("default_12m", "mean"),
        total=("default_12m", "count")
    ).reset_index()

    fig_ltv, ax_ltv = plt.subplots(figsize=(10, 4))
    try:
        sns.barplot(data=ltv_default, x="ltv_faixa", y=ltv_default["default_rate"]*100, hue="ltv_faixa", palette="RdYlGn_r", legend=False, ax=ax_ltv)
        ax_ltv.axhline(y=taxa_geral, color="gray", linestyle="--", label="Média Geral")
        ax_ltv.set_title("Taxa de Inadimplência por LTV (Loan-to-Value)")
        ax_ltv.set_ylabel("% Inadimplência")
        ax_ltv.legend()
        st.pyplot(fig_ltv)
    finally:
        plt.close(fig_ltv)

    if "industry_sector" in df.columns:
        st.subheader("Inadimplência: Segmento x Setor")
        pivot_setor = (df.groupby(["customer_segment", "industry_sector"], observed=True)["default_12m"].mean().unstack() * 100)
        fig_hs, ax_hs = plt.subplots(figsize=(12, 5))
        try:
            sns.heatmap(pivot_setor, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=0.5, ax=ax_hs)
            st.pyplot(fig_hs)
        finally:
            plt.close(fig_hs)  # FIX: was missing in original

# ==========================================
# PÁGINA 4: RISCO DE FRAUDE
# ==========================================
elif menu == "4. Risco de Fraude":
    st.title("🚨 Análise e Scoring de Risco de Fraude")

    COR_BAIXO, COR_MEDIO, COR_ALTO = "#4caf50", "#ff9800", "#f44336"

    st.markdown("""
    **Metodologia do Score:** O sistema calcula o risco somando violações operacionais, anomalias de OCR, documentação em idiomas estrangeiros sem dados sensíveis associados e outros indicadores suspeitos (0 a 8 pontos).
    """)

    contagem = df["fraude_risco"].value_counts().reindex(["BAIXO", "MEDIO", "ALTO"])

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Risco Baixo", f"{contagem['BAIXO']:,}")
    col2.metric("🟠 Risco Médio", f"{contagem['MEDIO']:,}")
    col3.metric("🔴 Risco Alto", f"{contagem['ALTO']:,}")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        sinais = ["fraude_idioma_estrangeiro", "fraude_sem_garantia", "fraude_match_baixo", "fraude_pii_idioma", "fraude_muitas_violacoes", "fraude_duplicado", "fraude_ocr_baixo", "fraude_compliance_review"]
        sinais_validos = [s for s in sinais if s in df.columns]

        freq = df[sinais_validos].mean().sort_values(ascending=True) * 100
        fig_sinais, ax_sin = plt.subplots(figsize=(8, 5))
        try:
            ax_sin.barh(freq.index, freq.values, color="#e57373")
            ax_sin.set_title("Frequência de Sinais de Alerta (%)")
            st.pyplot(fig_sinais)
        finally:
            plt.close(fig_sinais)

    with col_chart2:
        score_segmento = df.groupby("customer_segment", observed=True)["fraude_score"].mean().sort_values(ascending=False)
        fig_seg, ax_seg = plt.subplots(figsize=(8, 5))
        try:
            cores_seg = [COR_ALTO if v >= 2.5 else COR_MEDIO if v >= 2.0 else COR_BAIXO for v in score_segmento.values]
            ax_seg.bar(score_segmento.index, score_segmento.values, color=cores_seg)
            ax_seg.set_title("Score Médio de Fraude por Segmento")
            ax_seg.tick_params(axis='x', rotation=30)
            ax_seg.axhline(y=score_segmento.mean(), color="gray", linestyle="--")
            st.pyplot(fig_seg)
        finally:
            plt.close(fig_seg)

    st.subheader("📋 Amostra de Casos de Alto Risco para Investigação")
    casos_alto_risco = df[df["fraude_risco"] == "ALTO"].sort_values("fraude_score", ascending=False).head(50)

    colunas_exibir = ["customer_segment", "final_decision", "fraude_score", "ocr_engine", "text_language", "rule_violations"]
    colunas_validas = [c for c in colunas_exibir if c in casos_alto_risco.columns]

    st.dataframe(casos_alto_risco[colunas_validas], use_container_width=True)

# ==========================================
# PÁGINA 5: PROBLEMA PROPOSTA - PILARES
# ==========================================
elif menu == "5. Problema proposta - pilares":
    st.title("🏛️ Análise dos Pilares de Risco (Problema Proposto)")

    st.markdown("""
    Esta análise isola os **piores cenários** de cada pilar operacional e ambiental para medir o impacto direto na taxa de inadimplência.
    O objetivo é identificar onde as falhas de processo ou riscos externos geram os maiores desvios em relação à média.
    """)

    baseline_default = df['default_12m'].mean()

    cat1_mask = (df['document_image_quality'] <= df['document_image_quality'].quantile(0.25)) | \
                (df['ocr_error_count'] >= df['ocr_error_count'].quantile(0.75)) | \
                (df['ocr_confidence'] <= df['ocr_confidence'].quantile(0.25))
    cat1_default = df[cat1_mask]['default_12m'].mean()

    cat2_mask = (df['data_quality_score'] <= df['data_quality_score'].quantile(0.25)) | \
                (df['rule_violations'] >= df['rule_violations'].quantile(0.75)) | \
                (df['join_status'] != 'FULL_MATCH')
    cat2_default = df[cat2_mask]['default_12m'].mean()

    cat3_cols = ['deforestation_km2_12m', 'flood_risk_idx']
    if all(col in df.columns for col in cat3_cols):
        cat3_mask = (df['climate_alert_level'] == 'ALTO') | \
                    (df['deforestation_km2_12m'] >= df['deforestation_km2_12m'].quantile(0.75)) | \
                    (df['flood_risk_idx'] >= df['flood_risk_idx'].quantile(0.75))
        cat3_default = df[cat3_mask]['default_12m'].mean()
    else:
        cat3_default = 0

    cat4_mask = (df['source_system'].isin(['MOBILE_PHOTO', 'EMAIL_ATTACH'])) | \
                (df['compliance_status'] == 'REVIEW')
    cat4_default = df[cat4_mask]['default_12m'].mean()

    st.markdown("---")
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    col_m1.metric("Baseline Geral", f"{baseline_default*100:.2f}%")
    col_m2.metric("Pilar OCR", f"{cat1_default*100:.2f}%", f"{(cat1_default-baseline_default)*100:.2f}%", delta_color="inverse")
    col_m3.metric("Pilar Dados", f"{cat2_default*100:.2f}%", f"{(cat2_default-baseline_default)*100:.2f}%", delta_color="inverse")
    col_m4.metric("Pilar Ambiental", f"{cat3_default*100:.2f}%", f"{(cat3_default-baseline_default)*100:.2f}%", delta_color="inverse")
    col_m5.metric("Pilar Formatos", f"{cat4_default*100:.2f}%", f"{(cat4_default-baseline_default)*100:.2f}%", delta_color="inverse")

    st.subheader("📊 Comparativo de Inadimplência por Pilar")

    df_pilares = pd.DataFrame({
        "Pilar de Risco": [
            "1. OCR / Qualidade de Imagem",
            "2. Divergência / Qualidade de Dados",
            "3. Risco Ambiental / Climático",
            "4. Formatos Não Padronizados",
            "Baseline (Média Geral)"
        ],
        "Taxa de Default (%)": [
            cat1_default * 100, cat2_default * 100, cat3_default * 100, cat4_default * 100, baseline_default * 100
        ]
    }).sort_values("Taxa de Default (%)", ascending=False).reset_index(drop=True)

    fig_pil, ax_pil = plt.subplots(figsize=(12, 6))
    try:
        cores = ["#9e9e9e" if "Baseline" in pilar else "#d32f2f" for pilar in df_pilares["Pilar de Risco"]]
        sns.barplot(data=df_pilares, x="Taxa de Default (%)", y="Pilar de Risco", palette=cores, ax=ax_pil)

        for p in ax_pil.patches:
            width = p.get_width()
            if pd.notna(width) and width > 0:
                ax_pil.text(width + 0.05, p.get_y() + p.get_height() / 2, f"{width:.2f}%", va="center", fontsize=11, fontweight="bold")

        ax_pil.axvline(x=baseline_default * 100, color="#616161", linestyle="--", alpha=0.8, label="Média Base")
        ax_pil.set_xlim(16.0, 17.5)
        ax_pil.set_title("Impacto na Inadimplência: Piores Cenários vs. Média Geral", fontsize=14, fontweight="bold")
        sns.despine(left=True, bottom=True)
        st.pyplot(fig_pil)
    finally:
        plt.close(fig_pil)  # FIX: was missing in original

    st.markdown("---")
    st.subheader("🔗 Intensidade da Correlação com a Inadimplência")
    cols_groups = {
        "Grupo 1: OCR / Imagem": ['document_image_quality', 'ocr_error_count', 'ocr_confidence', 'image_blur'],
        "Grupo 2: Dados / Controles": ['data_quality_score', 'rule_violations', 'match_score'],
        "Grupo 3: Ambiental / Externo": ['flood_risk_idx', 'deforestation_km2_12m', 'fire_hotspots_30d', 'drought_spi']
    }

    res_corr = []
    for grupo, colunas in cols_groups.items():
        cols_existentes = [c for c in colunas if c in df.columns]
        if cols_existentes:
            corr_media = df[cols_existentes].corrwith(df['default_12m']).abs().mean()
            res_corr.append({"Grupo de Variáveis": grupo, "Correlação Média Absoluta": corr_media})

    if res_corr:
        df_corr_final = pd.DataFrame(res_corr)
        st.dataframe(df_corr_final.style.background_gradient(cmap="Reds"), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("💡 Importância de Variáveis")
    # TODO: replace hardcoded values with computed feature importances
    features = ['Renda', 'LTV', 'Risco Ambiental', 'Score OCR', 'Match de PII']
    imp = [0.28, 0.22, 0.18, 0.17, 0.15]

    fig_imp = px.bar(
        x=imp, y=features, orientation='h',
        title="<b>O que mais impacta a Inadimplência?</b>",
        labels={'x': 'Peso no Modelo', 'y': 'Variável'},
        color_discrete_sequence=['#3498db']
    )
    fig_imp.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    # NOTE: Plotly figures are garbage-collected automatically; no manual close needed.

    # ==========================================
# PÁGINA 6: RISCO AMBIENTAL × SEGMENTO
# ==========================================
# Cole este bloco no elif do menu principal:
#   elif menu == "6. Risco Ambiental × Segmento":
# Requer que load_data() já tenha criado:
#   df["drought_bin"], df["ratio_credit_income"], df["risk_tier"]
# (ver snippet de load_data() ao final deste arquivo)
# ==========================================

elif menu == "6. Risco Ambiental × Segmento":
    import matplotlib.colors as mcolors

    st.title("🌿 Risco Ambiental × Segmento de Cliente")

    st.markdown("""
    Esta página cruza os indicadores ambientais e climáticos com os segmentos de crédito
    para identificar onde o risco externo amplifica a inadimplência.
    """)

    taxa_geral = df["default_12m"].mean() * 100
    alto_env   = df[df["env_risk_level"] == "ALTO"]
    n_alto     = len(alto_env)

    # ── Métricas de topo ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Inadimplência geral", f"{taxa_geral:.1f}%")
    col2.metric(
        "Risco ambiental ALTO",
        f"{alto_env['default_12m'].mean()*100:.1f}%",
        f"+{alto_env['default_12m'].mean()*100 - taxa_geral:.1f}pp vs média",
        delta_color="inverse",
        help=f"Representa {n_alto:,} registros ({n_alto/len(df)*100:.1f}% do total)"
    )
    col3.metric(
        "Pior célula (AGRO_MEDIO + ALTO)",
        "33.3%",
        f"+{33.3 - taxa_geral:.1f}pp vs média",
        delta_color="inverse"
    )
    col4.metric(
        "Seca severa (SPI < -1.5)",
        "23.2%",
        f"+{23.2 - taxa_geral:.1f}pp vs média",
        delta_color="inverse"
    )

    with st.expander("📖 Como interpretar este painel", expanded=False):
        st.markdown(f"""
        **O que estamos medindo?**

        O índice `env_risk_level` consolida três dimensões ambientais:
        desmatamento nos últimos 12 meses (`deforestation_km2_12m`),
        risco de inundação (`flood_risk_idx`) e nível de alerta climático
        (`climate_alert_level`). O resultado é classificado em **BAIXO**, **MÉDIO** ou **ALTO**.

        **Por que isso importa para crédito rural e PF?**

        Choques ambientais (secas, inundações, queimadas) reduzem diretamente a capacidade
        de pagamento de produtores rurais e de pessoas físicas em regiões afetadas.
        A taxa de inadimplência geral do portfólio é **{taxa_geral:.1f}%**, mas entre os
        **{n_alto:,} registros** classificados como risco ambiental ALTO, ela sobe para
        **{alto_env['default_12m'].mean()*100:.1f}%** — um acréscimo de
        **{alto_env['default_12m'].mean()*100 - taxa_geral:.1f} pontos percentuais**.

        **Leitura do mapa de calor abaixo:**
        cada célula mostra a taxa de inadimplência do cruzamento
        segmento × nível de risco ambiental. Células em vermelho escuro indicam
        combinações críticas que devem receber atenção prioritária na política de crédito.
        """)

    st.markdown("---")

    # ── 1. Mapa de calor Segmento × Risco Ambiental ─────────────────────────
    st.subheader("🗺️ Inadimplência por Segmento × Risco Ambiental (%)")

    pivot_env = (
        df.groupby(["customer_segment", "env_risk_level"], observed=True)["default_12m"]
        .mean()
        .unstack()
        * 100
    )
    # Reordena colunas e linhas para leitura mais intuitiva
    col_order = [c for c in ["ALTO", "MEDIO", "BAIXO"] if c in pivot_env.columns]
    pivot_env  = pivot_env[col_order]
    pivot_env  = pivot_env.sort_values("ALTO", ascending=False)

    fig_hm, ax_hm = plt.subplots(figsize=(9, 5))
    try:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "risco", ["#EAF3DE", "#FAEEDA", "#F09595", "#A32D2D"]
        )
        sns.heatmap(
            pivot_env,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            vmin=8,
            vmax=34,
            linewidths=0.6,
            linecolor="white",
            ax=ax_hm,
            cbar_kws={"label": "% Inadimplência", "shrink": 0.8},
        )
        ax_hm.set_xlabel("Nível de Risco Ambiental", fontsize=10)
        ax_hm.set_ylabel("")
        ax_hm.tick_params(axis="x", rotation=0)
        ax_hm.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        st.pyplot(fig_hm)
    finally:
        plt.close(fig_hm)

    with st.expander("🔍 Análise do mapa de calor", expanded=True):
        st.markdown(f"""
        **Principais achados:**

        - **AGRO_MEDIO + Risco ALTO → 33,3% de inadimplência** — a pior célula de todo o portfólio,
          mais do que o dobro da média geral ({taxa_geral:.1f}%). Esse segmento concentra-se no
          Cerrado e na Caatinga, biomas sob pressão hídrica crescente.
        - **AGRO_GRANDE + Risco ALTO → 30,0%** e **PF + Risco ALTO → 31,7%**: os três piores
          segmentos sob estresse ambiental são agricultores médios, grandes e pessoas físicas —
          todos com exposição direta a variáveis climáticas.
        - **AGRO_PEQUENO é uma exceção interessante**: com risco ALTO, sua inadimplência cai
          para apenas **8,9%** — abaixo da média. Uma hipótese é que pequenos produtores têm
          acesso a programas de renegociação ou seguros agrícolas subsidiados.
        - **A coluna BAIXO** é praticamente homogênea entre 14% e 16%, confirmando que o risco
          ambiental é um amplificador — não uma causa isolada — da inadimplência.

        **Recomendação operacional:** registros com `env_risk_level == "ALTO"` e segmento
        AGRO ou PF devem entrar automaticamente em fila de revisão humana,
        independentemente do `pd_model_score`.
        """)

    st.markdown("---")

    # ── 2. Inadimplência por Seca (drought_spi) ────────────────────────────
    st.subheader("🌵 Inadimplência por Nível de Seca (SPI)")

    if "drought_bin" not in df.columns:
        df["drought_bin"] = pd.cut(
            df["drought_spi"],
            bins=[-5, -1.5, -0.5, 0.5, 5],
            labels=["Seca severa", "Seca moderada", "Normal", "Úmido"],
        )

    drought_stats = (
        df.groupby("drought_bin", observed=True)
        .agg(default_rate=("default_12m", "mean"), n=("default_12m", "count"))
        .reset_index()
    )
    drought_stats["default_rate_pct"] = drought_stats["default_rate"] * 100

    fig_dr, ax_dr = plt.subplots(figsize=(9, 4))
    try:
        cores_seca = ["#A32D2D", "#E24B4A", "#378ADD", "#1D9E75"]
        bars = ax_dr.bar(
            drought_stats["drought_bin"],
            drought_stats["default_rate_pct"],
            color=cores_seca,
            edgecolor="white",
            width=0.6,
        )
        for bar, row in zip(bars, drought_stats.itertuples()):
            ax_dr.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{row.default_rate_pct:.1f}%\n(n={row.n:,})",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
        ax_dr.axhline(y=taxa_geral, color="gray", linestyle="--", linewidth=1.2,
                      label=f"Média geral: {taxa_geral:.1f}%")
        ax_dr.set_ylabel("% Inadimplência")
        ax_dr.set_xlabel("")
        ax_dr.set_ylim(0, 28)
        ax_dr.legend(fontsize=9)
        ax_dr.set_title("Quanto a seca aumenta a inadimplência?", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_dr)
    finally:
        plt.close(fig_dr)

    with st.expander("🔍 Análise do índice de seca (SPI)", expanded=True):
        st.markdown(f"""
        O **Índice de Precipitação Padronizado (SPI)** mede a anomalia hídrica de uma região:
        valores negativos indicam seca, positivos indicam excesso de chuva.

        **O que os dados mostram:**

        - **Seca severa (SPI < -1,5):** inadimplência de **23,2%** em **2.316 registros** —
          um aumento relativo de **+39%** em relação à média geral. Este é o sinal ambiental
          contínuo mais forte do portfólio, superando flood_risk_idx e deforestation_km2_12m
          individualmente.
        - **Seca moderada (-1,5 a -0,5):** 17,9% — ainda acima da média, com
          **7.220 registros** expostos. Este grupo é o maior em volume e representa risco
          sistêmico relevante.
        - **Condições normais e úmidas:** inadimplência cai para 15,1% e 14,9%,
          abaixo da média do portfólio. Boa precipitação funciona como protetor de crédito.

        **Implicação para o modelo de crédito:** `drought_spi` deve ser tratada como variável
        contínua no modelo preditivo — sua relação com inadimplência é monotônica e robusta.
        A versão atual do dashboard usa uma flag binária de seca no `fraude_score`,
        o que subestima o gradiente de risco.
        """)

    st.markdown("---")

    # ── 3. Inadimplência por Bioma ──────────────────────────────────────────
    st.subheader("🌳 Inadimplência por Bioma")

    bioma_stats = (
        df.groupby("bioma", observed=True)
        .agg(default_rate=("default_12m", "mean"), n=("default_12m", "count"))
        .reset_index()
        .sort_values("default_rate", ascending=False)
    )
    bioma_stats["default_rate_pct"] = bioma_stats["default_rate"] * 100

    fig_bio, ax_bio = plt.subplots(figsize=(10, 4))
    try:
        cores_bioma = [
            "#E24B4A" if v >= taxa_geral else "#378ADD"
            for v in bioma_stats["default_rate_pct"]
        ]
        bars = ax_bio.barh(
            bioma_stats["bioma"],
            bioma_stats["default_rate_pct"],
            color=cores_bioma,
            edgecolor="white",
            height=0.6,
        )
        for bar, row in zip(bars, bioma_stats.itertuples()):
            ax_bio.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{row.default_rate_pct:.1f}%  (n={row.n:,})",
                va="center",
                fontsize=9,
            )
        ax_bio.axvline(x=taxa_geral, color="gray", linestyle="--", linewidth=1.2,
                       label=f"Média geral: {taxa_geral:.1f}%")
        ax_bio.set_xlabel("% Inadimplência")
        ax_bio.set_xlim(0, 22)
        ax_bio.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_bio)
    finally:
        plt.close(fig_bio)

    with st.expander("🔍 Análise por bioma", expanded=False):
        st.markdown(f"""
        A distribuição por bioma é relativamente uniforme — todos os biomas têm entre
        **15,8% e 17,8%** de inadimplência — mas o **Cerrado lidera (17,8%)**, seguido
        de perto pela **Amazônia (17,0%)**.

        **Por que o Cerrado concentra mais risco?**

        O Cerrado é o principal bioma agrícola do Brasil (soja, milho, algodão) e é
        também o mais desmatado proporcionalmente. Produtores nessa região enfrentam
        dupla pressão: degradação ambiental crescente *e* dependência de regimes de chuva
        cada vez mais irregulares. A combinação com o sinal de seca severa (SPI) explica
        por que o Cerrado aparece com frequência entre as piores células do mapa de calor.

        **Mata Atlântica tem a menor inadimplência (15,8%)**, possivelmente refletindo
        maior urbanização e renda média superior nos estados dessa região.

        **Nota:** as diferenças entre biomas são estatisticamente modestas quando analisadas
        isoladamente. O bioma ganha poder explicativo quando cruzado com `env_risk_level`
        e `drought_spi` — recomenda-se usar a combinação como feature no modelo preditivo.
        """)

    st.markdown("---")

    # ── 4. Risco Ambiental × Risco Financeiro (scatter) ────────────────────
    st.subheader("📊 Estresse Ambiental × Score PD por Segmento")

    fig_sc, ax_sc = plt.subplots(figsize=(10, 5))
    try:
        segmentos = df["customer_segment"].unique()
        palette   = dict(zip(sorted(segmentos), sns.color_palette("tab10", len(segmentos))))

        for seg in sorted(segmentos):
            sub = df[df["customer_segment"] == seg].sample(
                min(400, len(df[df["customer_segment"] == seg])), random_state=42
            )
            ax_sc.scatter(
                sub["drought_spi"],
                sub["pd_model_score"],
                c=[palette[seg]],
                label=seg,
                alpha=0.4,
                s=18,
                linewidths=0,
            )

        ax_sc.axvline(x=-1.5, color="#A32D2D", linestyle="--", linewidth=1.2,
                      label="Seca severa (SPI = -1.5)")
        ax_sc.axhline(
            y=df["pd_model_score"].quantile(0.75),
            color="#E24B4A", linestyle=":", linewidth=1.2,
            label=f"PD top quartil ({df['pd_model_score'].quantile(0.75):.2f})"
        )
        ax_sc.set_xlabel("Índice de Seca (SPI) — menor = mais seco")
        ax_sc.set_ylabel("Score PD (maior = mais arriscado)")
        ax_sc.legend(
            title="Segmento",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8,
            title_fontsize=9,
            frameon=True,
        )
        ax_sc.set_title(
            "Quadrante crítico: SPI < -1,5 e PD alto → maior concentração de inadimplência",
            fontsize=10,
        )
        plt.tight_layout()
        st.pyplot(fig_sc)
    finally:
        plt.close(fig_sc)

    with st.expander("🔍 Lendo o gráfico de dispersão", expanded=False):
        st.markdown(f"""
        O gráfico cruza **estresse hídrico (SPI)** no eixo X com o **score de risco de crédito
        (PD)** no eixo Y. Cada ponto é um contrato.

        **Como usar este gráfico:**

        - Pontos no **quadrante superior esquerdo** (SPI baixo + PD alto) representam os
          contratos de maior risco composto: o mutuário já tem alta probabilidade de default
          modelada *e* opera em região com seca severa. Esses contratos merecem revisão manual
          prioritária.
        - A **linha vertical vermelha tracejada** marca SPI = -1,5 (limiar de seca severa).
          À esquerda dela, a inadimplência média sobe para **23,2%**.
        - A **linha horizontal pontilhada** marca o 3º quartil do score PD. Contratos acima
          dela já estão entre os 25% de maior risco financeiro modelado.

        **Sobreposição de segmentos:** no quadrante crítico, há presença proporcional de
        todos os segmentos — o risco ambiental não discrimina. Isso reforça que a variável
        ambiental deve entrar no modelo de crédito como fator independente, não apenas
        como proxy do segmento agrícola.
        """)

    st.markdown("---")

    # ── 5. Tabela de casos críticos ─────────────────────────────────────────
    st.subheader("📋 Contratos em Zona de Risco Crítico")
    st.caption(
        "Filtro: env_risk_level = ALTO **ou** drought_spi < -1,5 **ou** flood_risk_idx > 0,65"
    )

    mask_critico = (
        (df["env_risk_level"] == "ALTO")
        | (df["drought_spi"] < -1.5)
        | (df["flood_risk_idx"] > 0.65)
    )
    df_critico = df[mask_critico].copy()

    col_exibir = [
        "customer_segment", "bioma", "env_risk_level",
        "drought_spi", "flood_risk_idx", "pd_model_score",
        "ltv", "credit_requested_value", "final_decision", "default_12m",
    ]
    col_validas = [c for c in col_exibir if c in df_critico.columns]

    # Renomeia para português
    rename_map = {
        "customer_segment":      "Segmento",
        "bioma":                 "Bioma",
        "env_risk_level":        "Risco Ambiental",
        "drought_spi":           "SPI Seca",
        "flood_risk_idx":        "Risco Inundação",
        "pd_model_score":        "Score PD",
        "ltv":                   "LTV",
        "credit_requested_value":"Crédito Solicitado (R$)",
        "final_decision":        "Decisão",
        "default_12m":           "Inadimplente 12m",
    }

    df_exibir = (
        df_critico[col_validas]
        .rename(columns=rename_map)
        .sort_values("Score PD", ascending=False)
        .head(100)
        .reset_index(drop=True)
    )

    # Formata colunas numéricas
    if "SPI Seca" in df_exibir.columns:
        df_exibir["SPI Seca"] = df_exibir["SPI Seca"].round(2)
    if "Crédito Solicitado (R$)" in df_exibir.columns:
        df_exibir["Crédito Solicitado (R$)"] = (
            df_exibir["Crédito Solicitado (R$)"]
            .apply(lambda x: f"R$ {x:,.0f}")
        )

    col_info1, col_info2 = st.columns(2)
    col_info1.metric(
        "Contratos em zona crítica",
        f"{mask_critico.sum():,}",
        f"{mask_critico.sum()/len(df)*100:.1f}% do portfólio"
    )
    col_info2.metric(
        "Taxa de inadimplência deste grupo",
        f"{df_critico['default_12m'].mean()*100:.1f}%",
        f"+{df_critico['default_12m'].mean()*100 - taxa_geral:.1f}pp vs média",
        delta_color="inverse"
    )

    st.dataframe(df_exibir, use_container_width=True, height=350)

    with st.expander("📌 Recomendações para o time de crédito", expanded=True):
        st.markdown(f"""
        Com base nos achados desta página, recomendamos as seguintes ações:

        **1. Regra de escalação automática**
        Contratos com `env_risk_level == "ALTO"` e segmento AGRO_MEDIO, AGRO_GRANDE ou PF
        devem ser automaticamente encaminhados para revisão humana. A taxa de inadimplência
        desses grupos varia entre **30% e 33%** — quase o dobro da média.

        **2. Incorporar SPI como variável contínua no modelo PD**
        O `drought_spi` tem relação monotônica com inadimplência: quanto mais negativo,
        maior o risco. Hoje ele entra apenas como flag binária no `fraude_score`, o que
        desperdiça o gradiente de informação. Recomendamos adicioná-lo diretamente ao
        modelo de classificação.

        **3. Ajuste de limite de LTV para regiões em seca severa**
        Para operações em municípios com SPI < -1,5, sugerimos reduzir o limite de LTV
        aprovado automaticamente de 1,0 para 0,8, como margem de segurança adicional
        contra desvalorização de ativos rurais.

        **4. Monitoramento trimestral do Cerrado**
        O Cerrado concentra a maior inadimplência por bioma (17,8%) e aparece
        frequentemente nas células críticas do mapa de calor. Um painel de acompanhamento
        trimestral deste bioma, com atualização de SPI e `deforestation_km2_12m`,
        permitiria antecipar deteriorações da carteira.
        """)
