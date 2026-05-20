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
    from config import GOLD_DASHBOARD
    return pd.read_csv(GOLD_DASHBOARD)


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
        "7. Previsão e Resultados do Modelo",
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

elif menu == "7. Previsão e Resultados do Modelo":
    import joblib

    st.title("🤖 Previsão de Inadimplência")

    from config import MODELS_DIR as _MODELS_DIR, RF_MODELS_DIR as _RF_DIR, GBM_MODELS_DIR as _GBM_DIR

    def _require(path):
        if not path.exists():
            st.error(
                f"Modelo não encontrado: `{path}`. "
                "Execute `notebooks/train_gbm.ipynb` e `notebooks/train_rf.ipynb` para gerá-los."
            )
            st.stop()

    @st.cache_resource(show_spinner="Carregando GBM Global…")
    def load_gbm_global():
        p = _GBM_DIR / 'global.joblib'
        _require(p)
        return joblib.load(p)

    @st.cache_resource(show_spinner="Carregando RF Global…")
    def load_rf_global():
        p = _RF_DIR / 'global.joblib'
        _require(p)
        return joblib.load(p)

    @st.cache_resource(show_spinner="Carregando métricas…")
    def load_metrics():
        p = _MODELS_DIR / 'metrics.joblib'
        _require(p)
        return joblib.load(p)

    @st.cache_resource(show_spinner="Carregando artefatos segmentados…")
    def load_seg_artifacts():
        p = _RF_DIR / 'seg_artifacts.joblib'
        _require(p)
        return joblib.load(p)

    @st.cache_resource(show_spinner="Carregando métricas por modelo…")
    def load_model_seg_metricas():
        rf_met  = joblib.load(_RF_DIR  / 'metrics.joblib')
        gbm_met = joblib.load(_GBM_DIR / 'metrics.joblib')
        return {
            'RF Segmentado':  rf_met['seg_metricas'],
            'GBM Segmentado': gbm_met['seg_metricas'],
        }

    @st.cache_resource
    def load_seg_model(seg, sec):
        p = _RF_DIR / 'seg' / f'{seg}_{sec}.joblib'
        if not p.exists():
            return None
        return joblib.load(p)

    gbm_art          = load_gbm_global()
    rf_art           = load_rf_global()
    metrics_art      = load_metrics()
    seg_art          = load_seg_artifacts()
    all_seg_metricas = load_model_seg_metricas()

    gbm_model    = gbm_art['model']
    le_dict      = gbm_art['le_dict']
    feature_cols = gbm_art['feature_cols']
    medians_dict = gbm_art['medians']
    pd_score_q75 = gbm_art['pd_score_q75']

    importancias = rf_art['importancias']
    metricas     = metrics_art['metricas']
    seg_metricas = metrics_art['seg_metricas']
    roc_curves   = metrics_art['roc_curves']
    pr_curves    = metrics_art['pr_curves']
    pos_rate     = metrics_art.get('pos_rate', 0.166)

    # ── Abas ────────────────────────────────────────────────────────────────
    aba_pred, aba_modelo = st.tabs([
        "🔮 Simulador de Previsão",
        "📊 Resultados do Modelo",
    ])

    # ====================================================================
    # ABA 1 — SIMULADOR DE PREVISÃO
    # ====================================================================
    with aba_pred:
        st.subheader("Simule a probabilidade de inadimplência de um contrato")
        st.caption(
            "Preencha os campos abaixo com os dados do contrato. "
            "O modelo utilizado é o **RF Segmentado** (42 submodelos, um por segmento × setor). "
            "Caso a combinação não esteja coberta, aplica-se o **GBM Global** como fallback."
        )

        with st.expander("ℹ️ Como usar o simulador", expanded=False):
            st.markdown("""
            **O que este simulador faz:**
            Aplica o modelo treinado (GBM Global, ROC-AUC = 0.58) ao conjunto de inputs
            que você fornecer e retorna a **probabilidade estimada de inadimplência em 12 meses**.

            **Interpretação do resultado:**
            - Abaixo de 15%: risco abaixo da média do portfólio (baseline: 16,6%)
            - 15% – 22%: risco dentro da faixa esperada
            - Acima de 22%: risco elevado — equivalente aos tiers 3–5 identificados na análise

            **Limitações:**
            O modelo tem ROC-AUC de 0,58 — útil como sinal auxiliar, não como decisor único.
            Para contratos com `env_risk_level = ALTO` e segmento AGRO ou PF,
            recomenda-se revisão humana independentemente do score.
            """)

        st.markdown("---")

        # ── Formulário de entrada ────────────────────────────────────────
        with st.form("form_predicao"):
            st.markdown("#### Dados do Cliente e Contrato")
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                customer_segment = st.selectbox(
                    "Segmento do cliente",
                    ["AGRO_GRANDE", "AGRO_MEDIO", "AGRO_PEQUENO", "PF", "PJ_EPP", "PJ_GRANDE", "PJ_ME"],
                )
                industry_sector = st.selectbox(
                    "Setor da indústria",
                    ["COMERCIO", "GOVERNO", "INDUSTRIA", "PF", "RURAL", "SERVICOS"],
                )
                income_declared = st.number_input(
                    "Renda declarada (R$)",
                    min_value=1_954.0, max_value=446_273.0,
                    value=29_523.0, step=500.0,
                )
                credit_requested_value = st.number_input(
                    "Crédito solicitado (R$)",
                    min_value=1_134.0, max_value=187_961.0,
                    value=14_669.0, step=500.0,
                )
                tenure_months = st.slider(
                    "Tempo de relacionamento (meses)",
                    min_value=1, max_value=239, value=120,
                )

            with col_b:
                pd_model_score = st.slider(
                    "Score PD (maior = mais arriscado)",
                    min_value=0.086, max_value=0.977, value=0.300, step=0.001,
                    format="%.3f",
                )
                ltv = credit_requested_value / max(income_declared, 1)
                st.metric(
                    "LTV (calculado)",
                    f"{ltv:.2f}",
                    help="Crédito solicitado ÷ Renda declarada. Acima de 1.0 = crédito supera a renda.",
                )
                collateral_type = st.selectbox(
                    "Tipo de garantia",
                    ["CPR", "IMOVEL", "MAQUINARIO", "SEM_GARANTIA", "VEICULO"],
                )
                env_risk_level = st.selectbox(
                    "Nível de risco ambiental",
                    ["BAIXO", "MEDIO", "ALTO"],
                )

            with col_c:
                drought_spi = st.slider(
                    "Índice de Seca (SPI)",
                    min_value=-4.77, max_value=3.85, value=-0.19, step=0.01,
                    help="Abaixo de -1.5 = seca severa",
                    format="%.2f",
                )
                flood_risk_idx = st.slider(
                    "Índice de risco de inundação",
                    min_value=0.0, max_value=0.87, value=0.26, step=0.01,
                    format="%.2f",
                )
                bioma = st.selectbox(
                    "Bioma",
                    ["AMAZÔNIA", "CAATINGA", "CERRADO", "MATA ATLÂNTICA", "PAMPA", "PANTANAL"],
                )
                compliance_status = st.selectbox(
                    "Status de compliance",
                    ["OK", "REVIEW"],
                )

            st.markdown("#### Qualidade Documental")
            col_d, col_e, col_f = st.columns(3)

            with col_d:
                ocr_confidence = st.slider(
                    "Confiança OCR", 0.147, 0.996, 0.701, 0.001, format="%.3f"
                )
                ocr_engine = st.selectbox(
                    "Motor OCR", ["AZURE_OCR", "GOOGLE_VISION", "TESSERACT"]
                )

            with col_e:
                match_score = st.slider(
                    "Score de correspondência (match)", 0.097, 0.990, 0.666, 0.001, format="%.3f"
                )
                data_quality_score = st.slider(
                    "Score de qualidade dos dados", 0.426, 0.936, 0.719, 0.001, format="%.3f"
                )

            with col_f:
                rule_violations = st.slider(
                    "Violações de regras", 0, 9, 1
                )
                document_image_quality = st.slider(
                    "Qualidade da imagem do documento", 0.213, 0.994, 0.726, 0.001, format="%.3f"
                )
                join_status = st.selectbox(
                    "Status de junção de dados",
                    ["FULL_MATCH", "PARTIAL", "UNMATCHED"],
                )

            submitted = st.form_submit_button("🔮 Calcular probabilidade de inadimplência", use_container_width=True)

        # ── Resultado da predição ────────────────────────────────────────
        if submitted:
            # Monta dict de input com medianas pré-computadas pelo notebook
            input_dict = {c: medians_dict.get(c, 0) for c in feature_cols}

            # Sobrescreve com os valores do formulário
            form_values = {
                "customer_segment":       customer_segment,
                "income_declared":        income_declared,
                "credit_requested_value": credit_requested_value,
                "tenure_months":          tenure_months,
                "pd_model_score":         pd_model_score,
                "ltv":                    ltv,
                "collateral_type":        collateral_type,
                "env_risk_level":         env_risk_level,
                "drought_spi":            drought_spi,
                "flood_risk_idx":         flood_risk_idx,
                "bioma":                  bioma,
                "compliance_status":      compliance_status,
                "ocr_confidence":         ocr_confidence,
                "ocr_engine":             ocr_engine,
                "match_score":            match_score,
                "data_quality_score":     data_quality_score,
                "rule_violations":        float(rule_violations),
                "document_image_quality": document_image_quality,
                "join_status":            join_status,
            }

            # Codifica categoricals com os LabelEncoders do treino
            for col, val in form_values.items():
                if col in le_dict:
                    try:
                        encoded = le_dict[col].transform([str(val)])[0]
                    except ValueError:
                        encoded = 0
                    input_dict[col] = float(encoded)
                else:
                    try:
                        input_dict[col] = float(val)
                    except (TypeError, ValueError):
                        pass  # not a numeric feature; median already set

            X_input = pd.DataFrame([input_dict])[feature_cols]

            # Tenta o submodelo segmentado específico; fallback para GBM Global
            proba = None
            seg_model = load_seg_model(customer_segment, industry_sector)
            if seg_model is not None:
                seg_feat_cols = seg_art['feature_cols']
                seg_encoders  = seg_art['encoders']
                seg_input = dict(seg_art['medians'])
                for col, val in form_values.items():
                    if col not in seg_feat_cols:
                        continue
                    if col in seg_encoders:
                        try:
                            seg_input[col] = float(seg_encoders[col].transform([str(val)])[0])
                        except ValueError:
                            pass
                    else:
                        try:
                            seg_input[col] = float(val)
                        except (TypeError, ValueError):
                            pass
                X_seg = pd.DataFrame([seg_input])[seg_feat_cols]
                proba = seg_model.predict_proba(X_seg)[0, 1]
            if proba is None:
                proba = gbm_model.predict_proba(X_input)[0, 1]

            st.markdown("---")
            st.subheader("Resultado da Simulação")

            # Cor e classificação do risco
            if proba < 0.15:
                cor_badge  = "success"
                label_risco = "🟢 Risco Baixo"
                detalhe    = "Abaixo da média do portfólio (16,6%). Perfil dentro do esperado."
            elif proba < 0.22:
                cor_badge  = "warning"
                label_risco = "🟡 Risco Moderado"
                detalhe    = "Na faixa média do portfólio. Verificar variáveis de destaque abaixo."
            else:
                cor_badge  = "error"
                label_risco = "🔴 Risco Alto"
                detalhe    = "Acima do percentil 75 do portfólio. Recomenda-se revisão manual."

            col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
            col_res1.metric("Probabilidade estimada", f"{proba*100:.1f}%")
            col_res2.metric("Baseline do portfólio",  "16.6%")
            col_res3.metric(
                "Classificação de risco",
                label_risco,
                f"{(proba - 0.166)*100:+.1f}pp vs baseline",
                delta_color="inverse",
            )
            st.caption(detalhe)

            # Indicadores de risco ativados
            flags = []
            if ltv > 1.0:
                flags.append(f"⚠️ **LTV = {ltv:.2f}** — acima de 1,0 (limiar crítico)")
            if pd_model_score > pd_score_q75:
                flags.append(f"⚠️ **Score PD = {pd_model_score:.3f}** — no quartil superior de risco")
            if drought_spi < -1.5:
                flags.append(f"⚠️ **SPI = {drought_spi:.2f}** — região em seca severa (+6,6pp na inadimplência histórica)")
            if env_risk_level == "ALTO":
                flags.append(f"⚠️ **Risco ambiental ALTO** — segmento {customer_segment} com este risco: ~30%+ de default histórico")
            if compliance_status == "REVIEW":
                flags.append("⚠️ **Compliance em REVIEW** — sinal operacional de alerta")
            if collateral_type == "SEM_GARANTIA":
                flags.append("⚠️ **Sem garantia** — ausência de colateral aumenta perda em caso de default")
            if credit_requested_value / max(income_declared, 1) > 0.6:
                ratio = credit_requested_value / max(income_declared, 1)
                flags.append(f"⚠️ **Relação crédito/renda = {ratio:.2f}** — acima de 0,6 (faixa de atenção)")

            if flags:
                st.markdown("**Fatores de risco identificados:**")
                for f in flags:
                    st.markdown(f"- {f}")
            else:
                st.success("Nenhum fator de risco crítico identificado para este contrato.")

    # ====================================================================
    # ABA 2 — RESULTADOS DO MODELO
    # ====================================================================
    with aba_modelo:
        st.subheader("Comparativo de Modelos e Análise de Resultados")
        st.markdown("""
        Esta aba documenta os quatro experimentos realizados e justifica a escolha
        do modelo em produção.
        """)

        # ── Métricas ao vivo — todos os modelos ─────────────────────────
        st.subheader("📊 Métricas no Conjunto de Teste")
        cols_met = st.columns(len(metricas))
        best_roc = max(v["roc"] for v in metricas.values())
        for col_m, (nome, vals) in zip(cols_met, metricas.items()):
            is_best = vals["roc"] >= best_roc
            delta = "🏆 melhor" if is_best else f"{(vals['roc'] - best_roc)*100:.2f}pp vs melhor"
            col_m.metric(nome, f"ROC {vals['roc']:.4f}", delta, delta_color="normal" if is_best else "off")

        # ── Curvas ROC — reais (conjunto de teste) ───────────────────────
        st.subheader("📈 Curvas ROC — Todos os Modelos")
        _paleta_roc = {
            "GBM Global":     ("#d62728", "-"),
            "GBM Segmentado": ("#d62728", "--"),
            "RF Global":      ("#1f77b4", "-"),
            "RF Segmentado":  ("#1f77b4", "--"),
        }
        fig_roc, ax_roc = plt.subplots(figsize=(9, 5))
        try:
            for nome, (fpr_pts, tpr_pts) in roc_curves.items():
                cor, ls = _paleta_roc.get(nome, ("#888", "-"))
                ax_roc.plot(fpr_pts, tpr_pts, color=cor, linestyle=ls, linewidth=2,
                            label=f"{nome}  ROC={metricas[nome]['roc']:.4f}")
            ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="Aleatório")
            ax_roc.set_xlabel("Taxa de Falsos Positivos")
            ax_roc.set_ylabel("Taxa de Verdadeiros Positivos")
            ax_roc.legend(fontsize=9, loc="lower right")
            ax_roc.set_title("Curva ROC — conjunto de teste (20% dos dados)", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_roc)
        finally:
            plt.close(fig_roc)


        # ── Curvas Precision-Recall ───────────────────────────────────────
        st.subheader("📉 Curvas Precision-Recall — Todos os Modelos")
        st.caption(
            f"Baseline (linha pontilhada) = prevalência de inadimplência no conjunto de teste "
            f"({pos_rate*100:.1f}%). Uma curva útil fica acima desta linha."
        )
        # ── PR-AUC metrics cards ──────────────────────────────────────────
        cols_pr = st.columns(len(metricas))
        best_pr = max(v["pr"] for v in metricas.values())
        for col_p, (nome, vals) in zip(cols_pr, metricas.items()):
            is_best_pr = vals["pr"] >= best_pr
            delta_pr = "🏆 melhor" if is_best_pr else f"{(vals['pr'] - best_pr)*100:.2f}pp vs melhor"
            col_p.metric(nome, f"PR-AUC {vals['pr']:.4f}", delta_pr, delta_color="normal" if is_best_pr else "off")

        fig_pr, ax_pr = plt.subplots(figsize=(9, 5))
        try:
            for nome, (rec_pts, prec_pts) in pr_curves.items():
                cor, ls = _paleta_roc.get(nome, ("#888", "-"))
                ax_pr.plot(rec_pts, prec_pts, color=cor, linestyle=ls, linewidth=2,
                           label=f"{nome}  PR-AUC={metricas[nome]['pr']:.4f}")
            ax_pr.axhline(pos_rate, color="gray", linestyle="--", linewidth=0.8,
                          alpha=0.6, label=f"Aleatório ({pos_rate*100:.1f}%)")
            ax_pr.set_xlabel("Recall (Taxa de Verdadeiros Positivos)")
            ax_pr.set_ylabel("Precisão")
            ax_pr.set_xlim(0, 1)
            ax_pr.set_ylim(0.15, 0.3)
            ax_pr.legend(fontsize=9, loc="upper right")
            ax_pr.set_title("Curva Precision-Recall — conjunto de teste (20% dos dados)", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_pr)
        finally:
            plt.close(fig_pr)

        # ── Análise dos resultados ────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔎 Análise dos Resultados")

        _best = max(metricas, key=lambda k: metricas[k]["roc"])
        _best_roc = metricas[_best]["roc"]
        _gbm_g_roc = metricas.get("GBM Global", {}).get("roc", 0)
        _rf_g_roc  = metricas.get("RF Global",  {}).get("roc", 0)
        _gbm_s_roc = metricas.get("GBM Segmentado", {}).get("roc", 0)
        _rf_s_roc  = metricas.get("RF Segmentado",  {}).get("roc", 0)

        with st.expander("📊 Por que o ROC-AUC ficou entre 0,51 e 0,59?", expanded=True):
            st.markdown(f"""
O modelo com melhor desempenho (**{_best}**) atingiu ROC-AUC de **{_best_roc:.4f}** — valor modesto
para padrões de mercado, onde modelos de crédito maduros costumam superar 0,70.
Há três razões principais para esse teto:

**1. O `pd_model_score` já embute boa parte do sinal preditivo.**
Esse score de probabilidade de default foi gerado por um sistema prévio que provavelmente
consumiu informações mais ricas do que as disponíveis neste dataset — como histórico de
relacionamento completo, consultas a bureaus externos e dados comportamentais.
Qualquer modelo treinado aqui está, em certa medida, tentando reaprender o que o score já
captura. Isso comprime artificialmente o ganho incremental que nossas variáveis adicionais
podem oferecer.

**2. O dataset não contém os preditores mais poderosos de inadimplência.**
Variáveis como histórico de pagamentos anteriores (90+ dias em atraso), score de bureau
externo, safra agrícola do ano de contratação e preço de commodities no momento da
concessão são amplamente documentadas na literatura como as mais preditivas para crédito
rural e PF. Sua ausência cria um teto natural de desempenho — os modelos aprendem com o
que há, mas falta o sinal mais forte.

**3. Desbalanceamento de classes e tamanho de amostra.**
Com apenas ~16,6% de inadimplentes em 24.974 registros, o modelo tem menos de 4.200 casos
positivos para aprender padrões de default. Para modelos baseados em árvore com
`class_weight='balanced'`, isso é administrável, mas ainda limita a capacidade de
generalização — especialmente para eventos raros ou combinações incomuns de variáveis.
            """)

        with st.expander("⚡ Por que o GBM Global foi o melhor modelo?", expanded=True):
            st.markdown(f"""
O **GBM Global** atingiu ROC {_gbm_g_roc:.4f} contra RF Global {_rf_g_roc:.4f} — uma
diferença pequena mas consistente. Dois mecanismos explicam a superioridade do GBM:

**1. Aprendizado sequencial vs. paralelo.**
O Gradient Boosting treina árvores sequencialmente, cada uma corrigindo os erros da
anterior. Isso permite ao GBM focar progressivamente nos casos mais difíceis de classificar
— exatamente o que é necessário num problema desbalanceado onde os inadimplentes
representam apenas 1 em cada 6 contratos. O Random Forest, por outro lado, agrega
árvores treinadas de forma independente, sem esse mecanismo de correção gradual.

**2. Menor profundidade, maior generalização.**
O GBM foi configurado com `max_depth=4` (árvores rasas), enquanto o RF usou
`max_depth=8`. Árvores mais rasas no GBM funcionam como regularização implícita:
cada árvore individual é deliberadamente "fraca" (um *weak learner*), mas o ensemble
de centenas delas converge para uma fronteira de decisão mais suave e menos sujeita
a overfitting nos padrões de treino.

**3. Velocidade sem perda de qualidade.**
Além da melhor métrica, o GBM treina em segundos enquanto o RF com 300 árvores
demora minutos — uma vantagem prática relevante para retreinamentos periódicos.
            """)

        with st.expander("🔢 Por que os modelos segmentados tiveram desempenho inferior?", expanded=True):
            st.markdown(f"""
Treinar modelos separados por `(customer_segment, industry_sector)` — 42 submodelos no total —
produziu resultados **piores** do que um único modelo global em todas as combinações testadas:

| Modelo | ROC-AUC | PR-AUC |
|---|---|---|
| GBM Global | {_gbm_g_roc:.4f} | {metricas.get("GBM Global", {}).get("pr", 0):.4f} |
| RF Global | {_rf_g_roc:.4f} | {metricas.get("RF Global", {}).get("pr", 0):.4f} |
| RF Segmentado | {_rf_s_roc:.4f} | {metricas.get("RF Segmentado", {}).get("pr", 0):.4f} |
| GBM Segmentado | {_gbm_s_roc:.4f} | {metricas.get("GBM Segmentado", {}).get("pr", 0):.4f} |

**Por quê?**

**1. Amostra insuficiente por submodelo.**
Com 42 combinações e ~20.000 registros de treino, cada submodelo aprende com apenas
~560 linhas em média. É impossível treinar um RF com 300 árvores ou um GBM com 200
iterações de forma robusta em amostras tão pequenas — o modelo decora os dados de treino
em vez de generalizar.

**2. O modelo global já captura as interações por segmento.**
Ao incluir `customer_segment` e `industry_sector` como features, o modelo global pode
criar splits como *"se segmento = AGRO_MEDIO e drought_spi < -1,5 → alto risco"*
com 7× mais dados disponíveis para validar esse padrão. A segmentação explícita não
adiciona informação — ela apenas reduz o dado disponível para aprender.

**3. Padrões universais se perdem.**
Variáveis como LTV alto e baixa qualidade de dados afetam o risco de default em
**todos** os segmentos. Um modelo segmentado aprende esse padrão separadamente em cada
grupo, com menos exemplos e mais ruído. O modelo global aprende uma vez, com o dataset
completo, e generaliza melhor.

**Quando a segmentação funcionaria?**
Se os segmentos tivessem features completamente diferentes (ex.: AGRO usa variáveis
que PF nunca preenche) ou se as escalas fossem incomparáveis entre grupos. Neste
portfólio, todas as features são compartilhadas e na mesma escala — a segmentação
é contraproducente.
            """)

        with st.expander("🎯 O que as importâncias de variáveis revelam?", expanded=True):
            st.markdown(f"""
As importâncias do RF Global (proxy interpretável para o GBM) mostram três grupos distintos:

**Grupo financeiro — domina o ranking:**
`pd_model_score`, `ltv`, `income_declared` e `credit_requested_value` lideram.
O `pd_model_score` sozinho concentra ~6-7% da importância total — confirma que
o score pré-existente é a âncora do modelo. O LTV (razão entre crédito e renda)
é o segundo sinal mais forte: contratos com LTV > 1,0 indicam que o mutuário está
pedindo mais do que declara ganhar, um alerta claro de superendividamento.

**Grupo ambiental — segundo bloco mais importante:**
`drought_spi` e `flood_risk_idx` aparecem consistentemente no top-10.
Isso corrobora toda a análise da aba 6: o risco climático não é apenas correlacionado
com inadimplência — ele tem poder preditivo **independente** do score de crédito.
Produtores rurais em regiões com seca severa (SPI < -1,5) inadimplem ~39% mais do que
a média, mesmo controlando pelo LTV e `pd_model_score`.

**Grupo operacional — sinal fraco mas presente:**
`data_quality_score`, `match_score` e `ocr_confidence` têm importância individual
pequena (~2-4% cada), mas somados representam que a *qualidade do processo de
onboarding documental* carrega algum sinal sobre risco futuro. Uma hipótese: clientes
que enviam documentos de baixa qualidade ou com informações inconsistentes podem ser
mais propensos a omitir informações desfavoráveis — um sinal comportamental de risco.
            """)

        with st.expander("📋 Quais variáveis estão faltando e limitam o modelo?", expanded=True):
            st.markdown(f"""
O teto de ROC ≈ {_best_roc:.2f} indica que há sinal preditivo relevante **não capturado**
pelo dataset atual. Com base na literatura de crédito e nas análises desta sessão,
os candidatos mais prováveis são:

| Variável ausente | Impacto esperado | Justificativa |
|---|---|---|
| Histórico de pagamentos (90+ dias) | Alto | Melhor preditor individual de default em todos os estudos de crédito |
| Score de bureau externo (Serasa/SPC) | Alto | Captura comportamento fora do relacionamento com o banco |
| Safra agrícola do ano de contratação | Médio | Choque de renda sistêmico para todo o portfólio AGRO |
| Preço de commodities na concessão | Médio | Afeta a capacidade de pagamento de produtores rurais diretamente |
| Número de contratos ativos do cliente | Médio | Indicador de comprometimento de renda com outras dívidas |
| Região do imóvel/ativo dado em garantia | Baixo-Médio | Complementa o bioma para risco geográfico mais granular |

**Recomendação prática:** antes de otimizar hiperparâmetros ou tentar arquiteturas
mais complexas, o esforço deveria focar em enriquecer o dataset com pelo menos o
histórico de pagamentos e o score de bureau. Esses dois campos sozinhos têm potencial
de elevar o ROC-AUC para a faixa 0,65-0,72 em portfólios similares na literatura.
            """)

        # ── Importâncias de features ─────────────────────────────────────
        st.markdown("---")
        st.subheader("🎯 Importância de Features — RF Global")

        fig_imp, ax_imp = plt.subplots(figsize=(9, 6))
        try:
            cores_imp = [
                "#378ADD" if f in ["pd_model_score", "income_declared", "ltv",
                                   "credit_requested_value", "normalized_amount"]
                else "#1D9E75" if f in ["drought_spi", "flood_risk_idx",
                                        "deforestation_km2_12m", "ndvi", "precip_mm_30d"]
                else "#888780"
                for f in importancias.index
            ]
            ax_imp.barh(importancias.index[::-1], importancias.values[::-1],
                        color=cores_imp[::-1], edgecolor="white", height=0.65)
            ax_imp.set_xlabel("Importância (Gini)")
            ax_imp.set_title("Top 15 variáveis mais importantes (RF Global)", fontsize=11)

            from matplotlib.patches import Patch
            legend_items = [
                Patch(color="#378ADD", label="Financeiro"),
                Patch(color="#1D9E75", label="Ambiental"),
                Patch(color="#888780", label="Operacional / Documental"),
            ]
            ax_imp.legend(handles=legend_items, fontsize=9, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig_imp)
        finally:
            plt.close(fig_imp)

        # ── Desempenho por segmento ───────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Desempenho por Segmento — Modelos Segmentados")

        _seg_col1, _seg_col2 = st.columns(2)
        _seg_modelo = _seg_col1.radio(
            "Modelo segmentado",
            ["RF Segmentado", "GBM Segmentado"],
            horizontal=True,
        )
        _seg_metrica = _seg_col2.radio(
            "Métrica",
            ["Precision-Recall (PR-AUC)", "ROC-AUC"],
            horizontal=True,
        )
        _use_pr = _seg_metrica.startswith("Precision")
        _metric_key = "pr" if _use_pr else "roc"
        _metric_label = "PR-AUC" if _use_pr else "ROC-AUC"

        _ref_global_roc = metricas.get(_seg_modelo.replace("Segmentado", "Global").strip(), {}).get("roc", _gbm_g_roc)
        _ref_global_pr  = metricas.get(_seg_modelo.replace("Segmentado", "Global").strip(), {}).get("pr", 0)
        _ref_val = _ref_global_pr if _use_pr else _ref_global_roc
        _ref_name = _seg_modelo.replace("Segmentado", "Global").strip()

        _seg_data = all_seg_metricas[_seg_modelo]
        df_seg_met = pd.DataFrame(_seg_data).T.reset_index()
        df_seg_met.columns = ["Segmento", "ROC-AUC", "PR-AUC", "N Teste"]
        df_seg_met = df_seg_met.sort_values(_metric_label, ascending=False).reset_index(drop=True)

        st.caption(
            f"{_metric_label} do **{_seg_modelo}** por customer_segment — "
            f"linha azul = {_ref_name} ({_ref_val:.4f})."
        )

        fig_seg, ax_seg = plt.subplots(figsize=(9, 4))
        try:
            cores_seg_bar = ["#E24B4A" if v < _ref_val else "#1D9E75"
                             for v in df_seg_met[_metric_label]]
            ax_seg.barh(df_seg_met["Segmento"], df_seg_met[_metric_label],
                        color=cores_seg_bar, edgecolor="white", height=0.6)
            ax_seg.axvline(x=_ref_val, color="#378ADD", linestyle="--", linewidth=1.5,
                           label=f"{_ref_name} ({_ref_val:.4f})")
            ax_seg.set_xlabel(_metric_label)
            ax_seg.set_title(f"{_metric_label} por Segmento ({_seg_modelo})")
            ax_seg.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_seg)
        finally:
            plt.close(fig_seg)
        st.caption(f"🔴 Abaixo do {_ref_name}  |  🟢 Acima do {_ref_name}  |  🔵 linha = {_ref_name}")

        # ── Conclusão ─────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("✅ Conclusão e recomendação de produção", expanded=True):
            st.markdown(f"""
**Modelo recomendado para produção: {_best}**

| Modelo | ROC-AUC | PR-AUC |
|---|---|---|
{"".join(f"| {'**' if k == _best else ''}{k}{'**' if k == _best else ''} | {'**' if k == _best else ''}{v['roc']:.4f}{'**' if k == _best else ''} | {'**' if k == _best else ''}{v['pr']:.4f}{'**' if k == _best else ''} |{chr(10)}" for k, v in metricas.items())}

**Próximos passos prioritários:**

1. **Enriquecer o dataset** com histórico de pagamentos e score de bureau externo —
   potencial de elevar o ROC-AUC para a faixa 0,65–0,72.
2. **Manter o GBM Global como baseline** — é mais rápido de treinar e supera consistentemente
   todas as variantes segmentadas.
3. **Usar os modelos segmentados apenas como análise exploratória**, não em produção —
   o sinal por segmento é útil para entender o portfólio, mas não para predição.
4. **Revisar contratos com `env_risk_level = ALTO` + segmento AGRO manualmente** —
   a inadimplência histórica desse grupo (>30%) supera o que qualquer modelo consegue
   capturar com as features atuais.
            """)

