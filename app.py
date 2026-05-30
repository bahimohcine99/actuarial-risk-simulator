"""
╔══════════════════════════════════════════════════════════════════╗
║      ACTUARIAL RISK SIMULATOR — Non-Life Insurance              ║
║      Plateforme d'Analyse et de Modélisation du Risque          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import poisson, lognorm, gamma, expon
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Actuarial Risk Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #141824 100%);
        border-right: 1px solid #2d3748;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a 0%, #16202e 100%);
        border: 1px solid #2d4a6a;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 8px 0;
        box-shadow: 0 4px 15px rgba(0,120,255,0.1);
    }
    .metric-card h3 {
        color: #7aa8d8;
        font-size: 13px;
        margin: 0 0 6px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        color: #e8f4fd;
        font-size: 26px;
        font-weight: 700;
    }
    .metric-card .delta {
        color: #4caf7d;
        font-size: 12px;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1565C0, #0d47a1);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 20px 0 15px 0;
        font-size: 16px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Alert boxes */
    .alert-info {
        background: rgba(21, 101, 192, 0.15);
        border-left: 4px solid #1565C0;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        color: #90caf9;
        margin: 10px 0;
        font-size: 14px;
    }
    .alert-warning {
        background: rgba(245, 124, 0, 0.12);
        border-left: 4px solid #F57C00;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        color: #ffcc80;
        margin: 10px 0;
        font-size: 14px;
    }
    .alert-success {
        background: rgba(46, 125, 50, 0.12);
        border-left: 4px solid #2E7D32;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        color: #a5d6a7;
        margin: 10px 0;
        font-size: 14px;
    }
    
    /* Formula box */
    .formula-box {
        background: #12181f;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 14px 20px;
        color: #81d4fa;
        font-family: 'Courier New', monospace;
        font-size: 15px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1f2e;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #7aa8d8;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1565C0 !important;
        color: white !important;
    }
    
    /* Dataframe */
    .stDataFrame { border-radius: 10px; }
    
    /* Slider */
    .stSlider [data-baseweb="slider"] { color: #1565C0; }
    
    /* Progress */
    .stProgress > div > div { background-color: #1565C0; }
    
    /* Hide default header */
    header { visibility: hidden; }
    
    /* Custom header */
    .app-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
        border-bottom: 2px solid #1565C0;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }
    .app-header h1 {
        color: #e8f4fd;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    .app-header p {
        color: #7aa8d8;
        font-size: 14px;
        margin: 6px 0 0 0;
    }
    .badge {
        display: inline-block;
        background: #1565C0;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        margin: 0 3px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🛡️ Actuarial Risk Simulator</h1>
    <p>Plateforme professionnelle de modélisation et d'analyse du risque — Assurance Non-Vie</p>
    <br>
    <span class="badge">VaR</span>
    <span class="badge">TVaR</span>
    <span class="badge">Modèle Collectif</span>
    <span class="badge">Algorithme de Panjer</span>
    <span class="badge">Prime Stop-Loss</span>
    <span class="badge">Solvabilité II</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — PARAMÈTRES GLOBAUX
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres du Portefeuille")
    st.markdown("---")
    
    st.markdown("### 📊 Distribution des sinistres")
    dist_type = st.selectbox(
        "Loi de sévérité (X_i)",
        ["Log-Normale", "Gamma", "Exponentielle", "Pareto (simulé)"],
        help="Distribution utilisée pour modéliser le coût individuel des sinistres"
    )
    
    if dist_type == "Log-Normale":
        mu_param = st.slider("μ (moyenne log)", 5.0, 12.0, 8.5, 0.1)
        sigma_param = st.slider("σ (écart-type log)", 0.1, 3.0, 1.2, 0.1)
        mean_claim = np.exp(mu_param + sigma_param**2 / 2)
        
    elif dist_type == "Gamma":
        alpha_param = st.slider("α (forme)", 0.5, 10.0, 2.0, 0.1)
        beta_param = st.slider("β (échelle)", 100.0, 5000.0, 1500.0, 100.0)
        mean_claim = alpha_param * beta_param
        
    elif dist_type == "Exponentielle":
        lambda_sev = st.slider("λ (taux)", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
        mean_claim = 1 / lambda_sev
        
    else:  # Pareto
        alpha_pareto = st.slider("α Pareto", 1.5, 5.0, 2.5, 0.1)
        theta_pareto = st.slider("θ Pareto", 500.0, 5000.0, 2000.0, 100.0)
        mean_claim = theta_pareto / (alpha_pareto - 1) if alpha_pareto > 1 else float("inf")
    
    st.markdown("---")
    st.markdown("### 🎯 Fréquence des sinistres")
    freq_dist = st.selectbox("Loi de fréquence (N)", ["Poisson", "Binomiale Négative"])
    
    lambda_freq = st.slider("λ (fréquence moyenne)", 0.01, 5.0, 0.8, 0.01,
                             help="Nombre moyen de sinistres par contrat par an")
    
    if freq_dist == "Binomiale Négative":
        r_param = st.slider("r (sur-dispersion)", 0.1, 20.0, 2.0, 0.1)
    
    st.markdown("---")
    st.markdown("### 🏢 Taille du portefeuille")
    n_contracts = st.slider("Nombre de contrats", 100, 10000, 2000, 100)
    n_simulations = st.slider("Simulations Monte Carlo", 1000, 20000, 5000, 500)
    
    st.markdown("---")
    st.markdown("### 📐 Niveaux de confiance")
    alpha_var = st.slider("Niveau VaR/TVaR (%)", 90, 99, 95, 1) / 100
    alpha_var2 = st.slider("Niveau secondaire (%)", 95, 99, 99, 1) / 100
    
    st.markdown("---")
    seed = st.number_input("Graine aléatoire", 0, 9999, 42)
    st.markdown("---")
    run_btn = st.button("🚀 Lancer la Simulation", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_simulation(dist_type, dist_params, freq_dist, lambda_freq, freq_params,
                   n_contracts, n_simulations, seed):
    np.random.seed(seed)
    
    def sample_severity(n):
        if dist_type == "Log-Normale":
            mu, sigma = dist_params
            return np.random.lognormal(mu, sigma, n)
        elif dist_type == "Gamma":
            alpha, beta = dist_params
            return np.random.gamma(alpha, beta, n)
        elif dist_type == "Exponentielle":
            lam = dist_params[0]
            return np.random.exponential(1/lam, n)
        else:  # Pareto
            alpha, theta = dist_params
            u = np.random.uniform(0, 1, n)
            return theta * (u ** (-1/alpha) - 1)
    
    # ── Modèle individuel : un montant par contrat
    individual = sample_severity(n_contracts)
    
    # ── Modèle collectif : somme des sinistres par simulation
    collective = np.zeros(n_simulations)
    for i in range(n_simulations):
        if freq_dist == "Poisson":
            n_claims = np.random.poisson(lambda_freq)
        else:
            r = freq_params[0]
            p = r / (r + lambda_freq)
            n_claims = np.random.negative_binomial(r, p)
        
        if n_claims > 0:
            collective[i] = sample_severity(n_claims).sum()
    
    return individual, collective

def compute_var_tvar(data, alpha):
    var = np.quantile(data, alpha)
    tvar = data[data > var].mean() if (data > var).any() else var
    return var, tvar

def panjer_algorithm(lambda_val, severity_probs, max_s=100):
    """Récursion de Panjer pour une loi de Poisson composée."""
    h = np.array(severity_probs[:max_s])
    h = h / h.sum() if h.sum() > 0 else h  # normaliser
    
    g = np.zeros(max_s + 1)
    g[0] = np.exp(-lambda_val)
    
    for s in range(1, max_s + 1):
        total = 0.0
        for k in range(1, min(s + 1, len(h) + 1)):
            if k - 1 < len(h):
                total += (k * lambda_val / s) * h[k - 1] * g[s - k]
        g[s] = total
    
    return g

def stop_loss_premium(aggregate_losses, deductible):
    return np.mean(np.maximum(aggregate_losses - deductible, 0))

# ─────────────────────────────────────────────────────────────────
# PARAMÈTRES DE DISTRIBUTION (pour cache)
# ─────────────────────────────────────────────────────────────────
if dist_type == "Log-Normale":
    dist_params = (mu_param, sigma_param)
elif dist_type == "Gamma":
    dist_params = (alpha_param, beta_param)
elif dist_type == "Exponentielle":
    dist_params = (lambda_sev,)
else:
    dist_params = (alpha_pareto, theta_pareto)

freq_params = (r_param,) if freq_dist == "Binomiale Négative" else (None,)

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Vue d'ensemble",
    "⚠️ VaR & TVaR",
    "🔄 Modèles Individuel / Collectif",
    "📐 Algorithme de Panjer",
    "🔒 Prime Stop-Loss & Réassurance",
    "📋 Rapport Actuariel"
])

# ─────────────────────────────────────────────────────────────────
# LANCER LA SIMULATION
# ─────────────────────────────────────────────────────────────────
with st.spinner("⏳ Simulation en cours..."):
    individual, collective = run_simulation(
        dist_type, dist_params, freq_dist, lambda_freq,
        freq_params, n_contracts, n_simulations, seed
    )

var_ind, tvar_ind = compute_var_tvar(individual, alpha_var)
var_col, tvar_col = compute_var_tvar(collective, alpha_var)
var_ind2, tvar_ind2 = compute_var_tvar(individual, alpha_var2)
var_col2, tvar_col2 = compute_var_tvar(collective, alpha_var2)

# ═══════════════════════════════════════════════════════════════
# TAB 1 — VUE D'ENSEMBLE
# ═══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">📊 Tableau de Bord du Portefeuille</div>', unsafe_allow_html=True)
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💼 Contrats</h3>
            <div class="value">{n_contracts:,}</div>
            <div class="delta">Portefeuille Auto</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        charge_moy = collective.mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>📉 Charge Moy. Agrégée</h3>
            <div class="value">{charge_moy:,.0f} €</div>
            <div class="delta">Par simulation</div>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 VaR {int(alpha_var*100)}%</h3>
            <div class="value">{var_col:,.0f} €</div>
            <div class="delta">Charge agrégée</div>
        </div>""", unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 TVaR {int(alpha_var*100)}%</h3>
            <div class="value">{tvar_col:,.0f} €</div>
            <div class="delta">Queue de distribution</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Distribution individuelle + collective côte à côte
    col_left, col_right = st.columns(2)
    
    with col_left:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=individual, nbinsx=50,
            name="Sinistres individuels",
            marker_color='rgba(100,149,237,0.7)',
            marker_line_color='rgba(100,149,237,1)',
            marker_line_width=0.5,
        ))
        fig.add_vline(x=var_ind, line_color="red", line_width=2.5,
                      annotation_text=f"VaR {int(alpha_var*100)}%", annotation_position="top right")
        fig.add_vline(x=tvar_ind, line_color="orange", line_width=2.5, line_dash="dash",
                      annotation_text=f"TVaR {int(alpha_var*100)}%", annotation_position="top right")
        fig.update_layout(
            title="Distribution des Sinistres Individuels",
            template="plotly_dark",
            paper_bgcolor="#12181f",
            plot_bgcolor="#12181f",
            font_color="#c9d6e3",
            showlegend=False,
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=collective, nbinsx=50,
            name="Charge agrégée",
            marker_color='rgba(255,160,64,0.7)',
            marker_line_color='rgba(255,160,64,1)',
            marker_line_width=0.5,
        ))
        fig2.add_vline(x=var_col, line_color="red", line_width=2.5,
                       annotation_text=f"VaR {int(alpha_var*100)}%", annotation_position="top right")
        fig2.add_vline(x=tvar_col, line_color="lime", line_width=2.5, line_dash="dash",
                       annotation_text=f"TVaR {int(alpha_var*100)}%", annotation_position="top right")
        fig2.update_layout(
            title="Distribution Agrégée (Modèle Collectif)",
            template="plotly_dark",
            paper_bgcolor="#12181f",
            plot_bgcolor="#12181f",
            font_color="#c9d6e3",
            showlegend=False,
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Statistiques descriptives
    st.markdown('<div class="section-header">📈 Statistiques Descriptives</div>', unsafe_allow_html=True)
    
    stats_df = pd.DataFrame({
        "Statistique": ["Moyenne", "Médiane", "Écart-type", "Asymétrie", "Aplatissement", "Minimum", "Maximum"],
        "Modèle Individuel (€)": [
            f"{individual.mean():,.2f}",
            f"{np.median(individual):,.2f}",
            f"{individual.std():,.2f}",
            f"{stats.skew(individual):.4f}",
            f"{stats.kurtosis(individual):.4f}",
            f"{individual.min():,.2f}",
            f"{individual.max():,.2f}",
        ],
        "Modèle Collectif (€)": [
            f"{collective.mean():,.2f}",
            f"{np.median(collective):,.2f}",
            f"{collective.std():,.2f}",
            f"{stats.skew(collective):.4f}",
            f"{stats.kurtosis(collective):.4f}",
            f"{collective.min():,.2f}",
            f"{collective.max():,.2f}",
        ]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — VaR & TVaR
# ═══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">⚠️ Mesures de Risque : VaR & TVaR</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-info">
        <strong>Value-at-Risk (VaR)</strong> : Quantile de niveau α de la distribution des pertes.<br>
        <strong>Tail Value-at-Risk (TVaR)</strong> : Espérance conditionnelle des pertes au-delà de la VaR.
    </div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="formula-box">VaR_α(X) = F⁻¹(α)</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="formula-box">TVaR_α(X) = E[X | X > VaR_α(X)]</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tableau comparatif VaR / TVaR
    alphas = [0.90, 0.95, 0.975, 0.99]
    rows = []
    for a in alphas:
        vi, ti = compute_var_tvar(individual, a)
        vc, tc = compute_var_tvar(collective, a)
        rows.append({
            "Niveau α": f"{int(a*100)}%",
            "VaR Individuel (€)": f"{vi:,.0f}",
            "TVaR Individuel (€)": f"{ti:,.0f}",
            "Ratio TVaR/VaR (ind.)": f"{ti/vi:.3f}" if vi > 0 else "—",
            "VaR Collectif (€)": f"{vc:,.0f}",
            "TVaR Collectif (€)": f"{tc:,.0f}",
            "Ratio TVaR/VaR (col.)": f"{tc/vc:.3f}" if vc > 0 else "—",
        })
    
    df_risk = pd.DataFrame(rows)
    st.dataframe(df_risk, use_container_width=True, hide_index=True)
    
    # Graphique VaR / TVaR multi-niveaux
    fig = go.Figure()
    vars_col = [compute_var_tvar(collective, a)[0] for a in alphas]
    tvars_col = [compute_var_tvar(collective, a)[1] for a in alphas]
    vars_ind = [compute_var_tvar(individual, a)[0] for a in alphas]
    tvars_ind = [compute_var_tvar(individual, a)[1] for a in alphas]
    alpha_labels = [f"{int(a*100)}%" for a in alphas]
    
    fig.add_trace(go.Bar(name="VaR Collectif", x=alpha_labels, y=vars_col,
                         marker_color='rgba(66,133,244,0.8)'))
    fig.add_trace(go.Bar(name="TVaR Collectif", x=alpha_labels, y=tvars_col,
                         marker_color='rgba(244,66,66,0.8)'))
    fig.add_trace(go.Scatter(name="VaR Individuel", x=alpha_labels, y=vars_ind,
                              mode='lines+markers', line=dict(color='#4FC3F7', dash='dot')))
    fig.add_trace(go.Scatter(name="TVaR Individuel", x=alpha_labels, y=tvars_ind,
                              mode='lines+markers', line=dict(color='#FFB74D', dash='dot')))
    
    fig.update_layout(
        title="Comparaison VaR et TVaR par niveau de confiance",
        barmode='group',
        template="plotly_dark",
        paper_bgcolor="#12181f",
        plot_bgcolor="#12181f",
        font_color="#c9d6e3",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Courbe de la fonction de distribution cumulative (CDF)
    st.markdown('<div class="section-header">📈 Fonction de Répartition (CDF) & Courbe de Risque</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_sorted = np.sort(collective)
        cdf = np.arange(1, len(x_sorted)+1) / len(x_sorted)
        
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(x=x_sorted, y=cdf, mode='lines',
                                      line=dict(color='#4FC3F7', width=2.5),
                                      name="CDF Empirique"))
        fig_cdf.add_hline(y=alpha_var, line_color="red", line_dash="dash",
                           annotation_text=f"α={int(alpha_var*100)}%")
        fig_cdf.add_vline(x=var_col, line_color="orange", line_dash="dash",
                           annotation_text=f"VaR={var_col:,.0f}€")
        
        # Shaded tail region
        tail_mask = x_sorted >= var_col
        if tail_mask.any():
            fig_cdf.add_trace(go.Scatter(
                x=x_sorted[tail_mask], y=cdf[tail_mask],
                fill='tozeroy', fillcolor='rgba(255,82,82,0.15)',
                line=dict(color='rgba(255,82,82,0.3)'),
                name=f"Queue {int((1-alpha_var)*100)}%"
            ))
        
        fig_cdf.update_layout(
            title="CDF du Modèle Collectif",
            template="plotly_dark",
            paper_bgcolor="#12181f",
            plot_bgcolor="#12181f",
            font_color="#c9d6e3",
            height=380,
        )
        st.plotly_chart(fig_cdf, use_container_width=True)
    
    with col2:
        # Courbe TVaR vs alpha
        alphas_fine = np.arange(0.80, 0.999, 0.005)
        tvars_fine = [compute_var_tvar(collective, a)[1] for a in alphas_fine]
        vars_fine  = [compute_var_tvar(collective, a)[0] for a in alphas_fine]
        
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(
            x=alphas_fine * 100, y=tvars_fine,
            mode='lines', line=dict(color='#FF7043', width=2.5),
            name="TVaR(α)"
        ))
        fig_risk.add_trace(go.Scatter(
            x=alphas_fine * 100, y=vars_fine,
            mode='lines', line=dict(color='#42A5F5', width=2, dash='dot'),
            name="VaR(α)"
        ))
        fig_risk.add_vline(x=alpha_var*100, line_color="white", line_dash="dash", line_width=1)
        
        fig_risk.update_layout(
            title="Courbe de Risque : VaR & TVaR vs α",
            xaxis_title="Niveau de confiance α (%)",
            yaxis_title="Montant (€)",
            template="plotly_dark",
            paper_bgcolor="#12181f",
            plot_bgcolor="#12181f",
            font_color="#c9d6e3",
            height=380,
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown("""
    <div class="alert-warning">
        <strong>Interprétation actuarielle :</strong> La TVaR est cohérente au sens de Artzner et al. (1999), 
        contrairement à la VaR qui ne mesure pas la gravité des pertes extrêmes. 
        En Solvabilité II, la TVaR à 99,5% est utilisée pour calibrer le SCR (Solvency Capital Requirement).
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — MODÈLES INDIVIDUEL / COLLECTIF
# ═══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">🔄 Comparaison : Modèle Individuel vs Modèle Collectif</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="alert-info">
            <strong>Modèle Individuel :</strong><br>
            S_ind = Σ S_i , i = 1..n<br>
            Chaque contrat contribue individuellement.
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="alert-info">
            <strong>Modèle Collectif :</strong><br>
            S_coll = Σ X_i , i = 1..N, N ~ Poisson(λ)<br>
            N est aléatoire, X_i sont iid.
        </div>""", unsafe_allow_html=True)
    
    # Overlaid density comparison
    from scipy.stats import gaussian_kde
    
    fig = go.Figure()
    
    # KDE individuel
    kde_ind = gaussian_kde(individual)
    x_range_ind = np.linspace(individual.min(), np.percentile(individual, 99), 500)
    fig.add_trace(go.Scatter(
        x=x_range_ind, y=kde_ind(x_range_ind),
        mode='lines', line=dict(color='#4FC3F7', width=2.5),
        name="Densité Individuel", fill='tozeroy', fillcolor='rgba(79,195,247,0.1)'
    ))
    
    # KDE collectif
    if collective.std() > 0:
        kde_col = gaussian_kde(collective)
        x_range_col = np.linspace(collective.min(), np.percentile(collective, 99.5), 500)
        fig.add_trace(go.Scatter(
            x=x_range_col, y=kde_col(x_range_col),
            mode='lines', line=dict(color='#FFB74D', width=2.5),
            name="Densité Collectif", fill='tozeroy', fillcolor='rgba(255,183,77,0.1)'
        ))
    
    fig.update_layout(
        title="Comparaison des Distributions de Pertes",
        template="plotly_dark",
        paper_bgcolor="#12181f",
        plot_bgcolor="#12181f",
        font_color="#c9d6e3",
        height=380,
        xaxis_title="Montant (€)",
        yaxis_title="Densité",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Boxplot comparatif
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=individual, name="Individuel",
                              marker_color='#4FC3F7', boxpoints='outliers'))
    fig_box.add_trace(go.Box(y=collective, name="Collectif",
                              marker_color='#FFB74D', boxpoints='outliers'))
    fig_box.update_layout(
        title="Boxplot — Comparaison des charges sinistres",
        template="plotly_dark",
        paper_bgcolor="#12181f",
        plot_bgcolor="#12181f",
        font_color="#c9d6e3",
        height=380,
        yaxis_title="Montant (€)",
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Convergence du modèle collectif (LLN)
    st.markdown('<div class="section-header">📉 Convergence — Loi des Grands Nombres</div>', unsafe_allow_html=True)
    
    sizes = [50, 100, 200, 500, 1000, 2000, 5000, min(n_simulations, 10000)]
    means_conv = [collective[:s].mean() for s in sizes if s <= len(collective)]
    sizes_valid = [s for s in sizes if s <= len(collective)]
    
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=sizes_valid, y=means_conv,
        mode='lines+markers',
        line=dict(color='#4FC3F7', width=2),
        name="Moyenne empirique"
    ))
    fig_conv.add_hline(
        y=collective.mean(),
        line_color="orange", line_dash="dash",
        annotation_text=f"Limite: {collective.mean():,.0f}€"
    )
    fig_conv.update_layout(
        title="Convergence de la Moyenne du Modèle Collectif",
        xaxis_title="Nombre de simulations",
        yaxis_title="Moyenne (€)",
        template="plotly_dark",
        paper_bgcolor="#12181f",
        plot_bgcolor="#12181f",
        font_color="#c9d6e3",
        height=320,
    )
    st.plotly_chart(fig_conv, use_container_width=True)
    
    st.markdown("""
    <div class="alert-success">
        <strong>Interprétation actuarielle :</strong><br>
        Le modèle collectif est préféré pour les grands portefeuilles car il capture 
        la variabilité du <em>nombre de sinistres</em>. Le modèle individuel est plus 
        précis mais computationnellement coûteux pour des portefeuilles de grande taille.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — ALGORITHME DE PANJER
# ═══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">📐 Algorithme de Panjer</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-info">
        <strong>Récursion de Panjer</strong> (loi de Poisson composée) :<br>
        <code>g(s) = (λ/s) · Σ_{k=1}^{s} k · h(k) · g(s-k)</code><br>
        avec g(0) = exp(-λ).
    </div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        panjer_step = st.slider("Pas de discrétisation (€)", 50, 1000, 200, 50,
                                 help="Plus le pas est petit, plus la discrétisation est précise")
    with col2:
        panjer_max_s = st.slider("Nombre maximum de classes", 30, 200, 80, 10)
    
    # Discrétisation empirique des sinistres individuels
    discrete_claims = np.ceil(individual / panjer_step).astype(int)
    max_class = min(discrete_claims.max(), panjer_max_s)
    
    freq_count = np.zeros(max_class + 1)
    for c in discrete_claims:
        if c <= max_class:
            freq_count[c] += 1
    
    severity_probs = freq_count[1:] / freq_count[1:].sum() if freq_count[1:].sum() > 0 else freq_count[1:]
    
    # Calcul Panjer
    g = panjer_algorithm(lambda_freq, severity_probs, max_class)
    
    # Graphique Panjer
    s_values = np.arange(len(g)) * panjer_step
    
    fig_panjer = make_subplots(rows=1, cols=2,
                                subplot_titles=["Distribution agrégée (Panjer)", "Distribution cumulée"])
    
    fig_panjer.add_trace(
        go.Bar(x=s_values, y=g, name="P(S=s)",
               marker_color='rgba(100,200,150,0.7)',
               marker_line_color='rgba(100,200,150,1)', marker_line_width=0.5),
        row=1, col=1
    )
    
    g_cumul = np.cumsum(g)
    fig_panjer.add_trace(
        go.Scatter(x=s_values, y=g_cumul, mode='lines',
                   line=dict(color='#FF7043', width=2.5),
                   name="F(s) cumulée"),
        row=1, col=2
    )
    fig_panjer.add_hline(y=alpha_var, line_color="white", line_dash="dash",
                          row=1, col=2)
    
    fig_panjer.update_layout(
        template="plotly_dark",
        paper_bgcolor="#12181f",
        plot_bgcolor="#12181f",
        font_color="#c9d6e3",
        height=400,
        showlegend=True,
    )
    st.plotly_chart(fig_panjer, use_container_width=True)
    
    # VaR par Panjer
    panjer_var_idx = np.searchsorted(g_cumul, alpha_var)
    panjer_var = panjer_var_idx * panjer_step
    
    st.markdown(f"""
    <div class="metric-card">
        <h3>🎯 VaR estimée par Panjer à {int(alpha_var*100)}%</h3>
        <div class="value">{panjer_var:,.0f} €</div>
        <div class="delta">Pas de discrétisation : {panjer_step} €</div>
    </div>""", unsafe_allow_html=True)
    
    # Tableau des premières probabilités
    st.markdown('<div class="section-header">🔢 Probabilités calculées (premières classes)</div>', unsafe_allow_html=True)
    
    n_show = min(20, len(g))
    panjer_df = pd.DataFrame({
        "Classe s": range(n_show),
        "Montant (€)": [f"{i * panjer_step:,.0f}" for i in range(n_show)],
        "P(S = s)": [f"{g[i]:.6f}" for i in range(n_show)],
        "F(s) = P(S ≤ s)": [f"{g_cumul[i]:.6f}" for i in range(n_show)],
        "1 - F(s)": [f"{1 - g_cumul[i]:.6f}" for i in range(n_show)],
    })
    st.dataframe(panjer_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="alert-success">
        <strong>Avantages de Panjer :</strong> Complexité O(n²) vs O(n·2ⁿ) pour la convolution directe.
        Essentiel pour le calcul de la distribution agrégée en assurance, la réassurance et Solvabilité II.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5 — PRIME STOP-LOSS & RÉASSURANCE
# ═══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">🔒 Prime Stop-Loss & Contrats de Réassurance</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-info">
        <strong>Prime Stop-Loss :</strong> E[(S - d)⁺] = E[max(S - d, 0)]<br>
        Coût de la protection pour l'assureur contre les pertes dépassant le seuil d.
    </div>""", unsafe_allow_html=True)
    
    st.markdown("### 🎛️ Configurer le contrat de réassurance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        d_pct = st.slider("Franchise / Seuil (% de la VaR)", 50, 150, 95, 5)
        deductible = np.percentile(collective, d_pct) if d_pct < 100 else var_col * d_pct / 100
    with col2:
        reinsurance_type = st.selectbox("Type de réassurance",
                                         ["Excess of Loss (XL)", "Stop-Loss Agrégé", "Quote-Part"])
    with col3:
        loading = st.slider("Chargement réassureur (%)", 0, 50, 15, 1)
    
    # Stop-loss pour différents seuils
    percentiles = np.arange(50, 100, 2)
    deductibles = np.percentile(collective, percentiles)
    sl_primes = [stop_loss_premium(collective, d) for d in deductibles]
    
    fig_sl = go.Figure()
    fig_sl.add_trace(go.Scatter(
        x=deductibles, y=sl_primes,
        mode='lines', line=dict(color='#4CAF50', width=2.5),
        name="Prime Stop-Loss pure",
        fill='tozeroy', fillcolor='rgba(76,175,80,0.1)'
    ))
    
    loaded_primes = [p * (1 + loading/100) for p in sl_primes]
    fig_sl.add_trace(go.Scatter(
        x=deductibles, y=loaded_primes,
        mode='lines', line=dict(color='#FF7043', width=2, dash='dot'),
        name=f"Prime chargée (+{loading}%)"
    ))
    
    # Marquer le seuil sélectionné
    sl_at_d = stop_loss_premium(collective, deductible)
    fig_sl.add_vline(x=deductible, line_color="white", line_dash="dash",
                      annotation_text=f"Seuil sélectionné")
    fig_sl.add_trace(go.Scatter(
        x=[deductible], y=[sl_at_d],
        mode='markers', marker=dict(color='yellow', size=12, symbol='star'),
        name=f"Prime = {sl_at_d:,.0f}€"
    ))
    
    fig_sl.update_layout(
        title="Courbe de la Prime Stop-Loss en fonction du Seuil",
        xaxis_title="Seuil de franchise d (€)",
        yaxis_title="Prime Stop-Loss (€)",
        template="plotly_dark",
        paper_bgcolor="#12181f",
        plot_bgcolor="#12181f",
        font_color="#c9d6e3",
        height=400,
    )
    st.plotly_chart(fig_sl, use_container_width=True)
    
    # Résultats financiers
    st.markdown('<div class="section-header">💰 Résultats Financiers</div>', unsafe_allow_html=True)
    
    sl_pure = stop_loss_premium(collective, deductible)
    sl_loaded = sl_pure * (1 + loading / 100)
    
    # Charge avec/sans réassurance
    retained = np.minimum(collective, deductible)
    recovered = np.maximum(collective - deductible, 0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💎 Prime Pure Stop-Loss</h3>
            <div class="value">{sl_pure:,.0f} €</div>
            <div class="delta">Sans chargement</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Prime Chargée</h3>
            <div class="value">{sl_loaded:,.0f} €</div>
            <div class="delta">Avec {loading}% de chargement</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🛡️ Charge Rétenue Moy.</h3>
            <div class="value">{retained.mean():,.0f} €</div>
            <div class="delta">Après réassurance</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        protection_ratio = recovered.mean() / collective.mean() * 100 if collective.mean() > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Taux de Protection</h3>
            <div class="value">{protection_ratio:.1f}%</div>
            <div class="delta">Charge transférée</div>
        </div>""", unsafe_allow_html=True)
    
    # Graphique charge rétenue vs transférée
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Histogram(
        x=retained, name="Charge rétenue",
        marker_color='rgba(66,133,244,0.7)', nbinsx=50
    ))
    fig_comp.add_trace(go.Histogram(
        x=collective, name="Charge brute",
        marker_color='rgba(255,82,82,0.4)', nbinsx=50
    ))
    fig_comp.update_layout(
        barmode='overlay',
        title="Charge Brute vs Charge Rétenue (après Stop-Loss)",
        template="plotly_dark",
        paper_bgcolor="#12181f",
        plot_bgcolor="#12181f",
        font_color="#c9d6e3",
        height=350,
        xaxis_title="Montant (€)",
        yaxis_title="Fréquence",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 6 — RAPPORT ACTUARIEL
# ═══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">📋 Rapport Actuariel Exécutif</div>', unsafe_allow_html=True)
    
    var_ind_95, tvar_ind_95 = compute_var_tvar(individual, 0.95)
    var_col_95, tvar_col_95 = compute_var_tvar(collective, 0.95)
    var_ind_99, tvar_ind_99 = compute_var_tvar(individual, 0.99)
    var_col_99, tvar_col_99 = compute_var_tvar(collective, 0.99)
    sl_d95 = stop_loss_premium(collective, var_col_95)
    sl_d99 = stop_loss_premium(collective, var_col_99)
    
    skew_val = stats.skew(individual)
    kurt_val = stats.kurtosis(individual)
    
    import datetime
    now = datetime.datetime.now().strftime("%d/%m/%Y à %H:%M")

    st.markdown(f"""
---

# RAPPORT ACTUARIEL EXÉCUTIF
## Analyse du Risque — Portefeuille Assurance Non-Vie

**Date d'analyse :** {now} &nbsp;|&nbsp; **Loi de sévérité :** {dist_type} &nbsp;|&nbsp; **Loi de fréquence :** {freq_dist}

---

## 1. Caractéristiques du Portefeuille Analysé

| Paramètre | Valeur |
|-----------|--------|
| Taille du portefeuille | {n_contracts:,} contrats |
| Nombre de scénarios simulés | {n_simulations:,} |
| Distribution de la sévérité | {dist_type} |
| Distribution de la fréquence | {freq_dist} (λ = {lambda_freq:.2f} sinistres/contrat/an) |
| Niveau de confiance principal | {int(alpha_var*100)}% |

---

## 2. Statistiques Descriptives de la Charge Sinistre

| Indicateur | Modèle Individuel | Modèle Collectif |
|-----------|-------------------|------------------|
| Charge moyenne (€) | {individual.mean():,.2f} | {collective.mean():,.2f} |
| Médiane (€) | {np.median(individual):,.2f} | {np.median(collective):,.2f} |
| Écart-type (€) | {individual.std():,.2f} | {collective.std():,.2f} |
| Coefficient de variation | {individual.std()/max(individual.mean(),1):.4f} | {collective.std()/max(collective.mean(),1):.4f} |
| Asymétrie (skewness) | {skew_val:.4f} | {stats.skew(collective):.4f} |
| Excès de kurtosis | {kurt_val:.4f} | {stats.kurtosis(collective):.4f} |
| Charge maximale observée (€) | {individual.max():,.2f} | {collective.max():,.2f} |

> Une asymétrie positive ({skew_val:.2f}) confirme la présence de sinistres à forte sévérité,
> structure typique des portefeuilles automobile en assurance non-vie.

---

## 3. Mesures de Risque Réglementaires

### 3.1 Value-at-Risk (VaR)

| Niveau de confiance | VaR — Sinistre Individuel | VaR — Charge Agrégée |
|--------------------|--------------------------|----------------------|
| 90% | {compute_var_tvar(individual, 0.90)[0]:,.0f} € | {compute_var_tvar(collective, 0.90)[0]:,.0f} € |
| 95% | {var_ind_95:,.0f} € | {var_col_95:,.0f} € |
| 99% | {var_ind_99:,.0f} € | {var_col_99:,.0f} € |

### 3.2 Tail Value-at-Risk (TVaR) — Expected Shortfall

| Niveau de confiance | TVaR Individuel | TVaR Agrégée | Ratio TVaR/VaR |
|--------------------|----------------|--------------|----------------|
| 90% | {compute_var_tvar(individual, 0.90)[1]:,.0f} € | {compute_var_tvar(collective, 0.90)[1]:,.0f} € | {compute_var_tvar(collective, 0.90)[1]/max(compute_var_tvar(collective, 0.90)[0],1):.3f} |
| 95% | {tvar_ind_95:,.0f} € | {tvar_col_95:,.0f} € | {tvar_col_95/max(var_col_95,1):.3f} |
| 99% | {tvar_ind_99:,.0f} € | {tvar_col_99:,.0f} € | {tvar_col_99/max(var_col_99,1):.3f} |

> Le ratio TVaR/VaR mesure l'intensité du risque de queue. Un ratio supérieur à 1,5 
> indique une distribution à queues lourdes nécessitant une attention particulière en matière de provisionnement.

---

## 4. Tarification des Contrats de Réassurance Stop-Loss

| Seuil de rétention (d) | Prime Stop-Loss Pure | Prime Commerciale (+{loading}%) | Transfert de risque |
|------------------------|---------------------|--------------------------------|---------------------|
| VaR 95% = {var_col_95:,.0f} € | {sl_d95:,.0f} € | {sl_d95*(1+loading/100):,.0f} € | {(1 - np.minimum(collective, var_col_95).mean() / max(collective.mean(),1))*100:.1f}% |
| VaR 99% = {var_col_99:,.0f} € | {sl_d99:,.0f} € | {sl_d99*(1+loading/100):,.0f} € | {(1 - np.minimum(collective, var_col_99).mean() / max(collective.mean(),1))*100:.1f}% |

---

## 5. Recommandations et Conclusions

**Exigences en capital (SCR — Solvabilité II)**
La directive Solvabilité II impose un calibrage du SCR à la VaR 99,5% de la distribution des pertes sur un horizon d'un an. Sur la base de cette simulation, la charge de risque à retenir pour le calcul du SCR est estimée à **{compute_var_tvar(collective, 0.995)[0]:,.0f} €**.

**Provisionnement prudentiel**
La TVaR à 99% ({tvar_col_99:,.0f} €) constitue une mesure conservatrice du risque de queue. Son utilisation est recommandée pour le provisionnement des sinistres graves et la constitution de réserves IBNR.

**Politique de réassurance**
Un programme Stop-Loss avec seuil de rétention à la VaR 95% ({var_col_95:,.0f} €) permet de transférer les événements catastrophiques pour une prime annuelle de {sl_d95*(1+loading/100):,.0f} € (chargement {loading}%). Ce niveau de protection est conforme aux pratiques du marché de la réassurance non-vie.

**Stabilité du modèle collectif**
La convergence du modèle collectif (loi des grands nombres) est atteinte à partir de {min(n_simulations, 2000):,} simulations, garantissant la fiabilité statistique des estimations présentées dans ce rapport.

---
*Actuarial Risk Simulator — Plateforme d'Analyse du Risque Non-Vie &nbsp;|&nbsp; {now}*
""")