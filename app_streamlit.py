import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.special import gamma
import warnings
import io
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE PÁGINA
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Analizador de Confiabilidad Industrial",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta de colores ────────────────────────────────────────
BG_DARK   = "#0D1117"
BG_CARD   = "#161B22"
BG_INPUT  = "#21262D"
ACCENT    = "#58A6FF"
ACCENT2   = "#3FB950"
ACCENT3   = "#F78166"
ACCENT4   = "#D2A8FF"
TEXT_MAIN = "#E6EDF3"
TEXT_DIM  = "#8B949E"
BORDER    = "#30363D"
WEIBULL_C = "#58A6FF"
NORMAL_C  = "#3FB950"
EXP_C     = "#F78166"
LOGN_C    = "#D2A8FF"

DIST_COLORS = {
    "Weibull":     WEIBULL_C,
    "Normal":      NORMAL_C,
    "Exponencial": EXP_C,
    "Lognormal":   LOGN_C,
}

# ── CSS personalizado ────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

  html, body, [class*="css"] {{
      font-family: 'JetBrains Mono', monospace;
      background-color: {BG_DARK};
      color: {TEXT_MAIN};
  }}
  .stApp {{ background-color: {BG_DARK}; }}

  /* Header */
  .main-header {{
      background: {BG_CARD};
      border-bottom: 1px solid {BORDER};
      padding: 18px 28px;
      margin: -1rem -1rem 1.5rem -1rem;
      display: flex;
      align-items: center;
      gap: 16px;
  }}
  .main-header h1 {{
      color: {ACCENT};
      font-size: 1.3rem;
      font-weight: 700;
      margin: 0;
      letter-spacing: 0.05em;
  }}
  .main-header span {{
      color: {TEXT_DIM};
      font-size: 0.78rem;
  }}

  /* KPI Cards */
  .kpi-card {{
      background: {BG_CARD};
      border: 1px solid {BORDER};
      border-radius: 8px;
      padding: 16px 18px;
      text-align: left;
  }}
  .kpi-label {{
      color: {TEXT_DIM};
      font-size: 0.68rem;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
  }}
  .kpi-value {{
      font-size: 1.6rem;
      font-weight: 700;
      margin: 0;
      line-height: 1.1;
  }}
  .kpi-unit {{
      color: {TEXT_DIM};
      font-size: 0.68rem;
      margin-top: 2px;
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
      background-color: {BG_CARD};
      border-right: 1px solid {BORDER};
  }}
  section[data-testid="stSidebar"] * {{
      font-family: 'JetBrains Mono', monospace !important;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
      background-color: {BG_CARD};
      gap: 4px;
      padding: 4px;
      border-radius: 8px;
  }}
  .stTabs [data-baseweb="tab"] {{
      background-color: transparent;
      color: {TEXT_DIM};
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
      border-radius: 6px;
      padding: 8px 16px;
  }}
  .stTabs [aria-selected="true"] {{
      background-color: {BG_DARK} !important;
      color: {ACCENT} !important;
  }}

  /* Dataframe */
  .stDataFrame {{ border: 1px solid {BORDER}; border-radius: 8px; }}

  /* Buttons */
  .stButton > button {{
      background-color: {ACCENT2};
      color: {BG_DARK};
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      border: none;
      border-radius: 6px;
      padding: 8px 20px;
  }}
  .stButton > button:hover {{
      background-color: #2ea043;
      color: {BG_DARK};
  }}

  /* Inputs */
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input {{
      background-color: {BG_INPUT};
      color: {TEXT_MAIN};
      border: 1px solid {BORDER};
      border-radius: 6px;
      font-family: 'JetBrains Mono', monospace;
  }}

  /* Info boxes */
  .best-dist-box {{
      background: linear-gradient(135deg, #1a3a1a, #162316);
      border: 1px solid {ACCENT2};
      border-radius: 8px;
      padding: 12px 16px;
      color: {ACCENT2};
      font-weight: 700;
      font-size: 0.9rem;
  }}

  /* Divider */
  hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  MOTOR DE CÁLCULO (sin cambios)
# ════════════════════════════════════════════════════════════
class ReliabilityEngine:
    def __init__(self, ttf_data: np.ndarray):
        self.data = np.sort(ttf_data[ttf_data > 0])
        self.n    = len(self.data)
        self.fits = {}
        self._fit_all()

    def _fit_all(self):
        data = self.data
        results = {}
        try:
            shape, loc, scale = stats.weibull_min.fit(data, floc=0)
            ll = np.sum(stats.weibull_min.logpdf(data, shape, loc, scale))
            ks = stats.kstest(data, "weibull_min", args=(shape, loc, scale)).statistic
            results["Weibull"] = dict(
                dist=stats.weibull_min, params=(shape, loc, scale),
                beta=shape, eta=scale, mean=scale * gamma(1 + 1/shape),
                ll=ll, ks=ks, label=f"β={shape:.3f}  η={scale:.2f}"
            )
        except Exception: pass
        try:
            mu, sigma = stats.norm.fit(data)
            ll = np.sum(stats.norm.logpdf(data, mu, sigma))
            ks = stats.kstest(data, "norm", args=(mu, sigma)).statistic
            results["Normal"] = dict(
                dist=stats.norm, params=(mu, sigma), mean=mu, std=sigma,
                ll=ll, ks=ks, label=f"μ={mu:.2f}  σ={sigma:.2f}"
            )
        except Exception: pass
        try:
            loc_e, scale_e = stats.expon.fit(data, floc=0)
            lam = 1.0 / scale_e
            ll = np.sum(stats.expon.logpdf(data, loc_e, scale_e))
            ks = stats.kstest(data, "expon", args=(loc_e, scale_e)).statistic
            results["Exponencial"] = dict(
                dist=stats.expon, params=(loc_e, scale_e), lam=lam, mean=scale_e,
                ll=ll, ks=ks, label=f"λ={lam:.4f}  MTBF={scale_e:.2f}"
            )
        except Exception: pass
        try:
            sigma_ln, loc_ln, scale_ln = stats.lognorm.fit(data, floc=0)
            mu_ln = np.log(scale_ln)
            ll = np.sum(stats.lognorm.logpdf(data, sigma_ln, loc_ln, scale_ln))
            ks = stats.kstest(data, "lognorm", args=(sigma_ln, loc_ln, scale_ln)).statistic
            results["Lognormal"] = dict(
                dist=stats.lognorm, params=(sigma_ln, loc_ln, scale_ln),
                mu_ln=mu_ln, sigma_ln=sigma_ln,
                mean=np.exp(mu_ln + sigma_ln**2 / 2),
                ll=ll, ks=ks, label=f"μ_ln={mu_ln:.3f}  σ_ln={sigma_ln:.3f}"
            )
        except Exception: pass
        self.fits = results

    def kpis(self, dist_name: str) -> dict:
        if dist_name not in self.fits: return {}
        f  = self.fits[dist_name]
        d, p, mn = f["dist"], f["params"], f["mean"]
        mtbf  = mn
        pdf_v = d.pdf(mn, *p); sf_v = d.sf(mn, *p)
        hz    = pdf_v / sf_v if sf_v > 0 else 0.0
        r50   = float(d.sf(0.5 * mn, *p))
        r90   = float(d.sf(0.9 * mn, *p))
        b10   = float(d.ppf(0.10, *p))
        b50   = float(d.ppf(0.50, *p))
        mttr  = 8.0
        avail = mtbf / (mtbf + mttr) if (mtbf + mttr) > 0 else 0.0
        k     = len(p) - 1
        aic   = 2*k - 2*f["ll"]
        bic   = k*np.log(self.n) - 2*f["ll"]
        return dict(
            MTBF=mtbf, MTTF=mtbf,
            Lambda=1.0/mtbf if mtbf > 0 else 0,
            Hazard_media=hz, R_50pct=r50, R_90pct=r90,
            B10=b10, B50=b50, Disponibilidad=avail,
            AIC=aic, BIC=bic, KS=f["ks"], Params=f["label"],
        )

    def curves(self, dist_name: str, n_points=300):
        if dist_name not in self.fits: return None
        f  = self.fits[dist_name]
        d, p = f["dist"], f["params"]
        t = np.linspace(1e-3, float(d.ppf(0.999, *p)), n_points)
        return dict(
            t=t, pdf=d.pdf(t, *p), cdf=d.cdf(t, *p), sf=d.sf(t, *p),
            hz=d.pdf(t,*p) / np.where(d.sf(t,*p)>1e-10, d.sf(t,*p), 1e-10),
        )

    def best_fit(self) -> str:
        if not self.fits: return ""
        return min(self.fits, key=lambda k: self.fits[k]["ks"])


# ════════════════════════════════════════════════════════════
#  ESTILOS MATPLOTLIB
# ════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": BG_CARD, "axes.facecolor": BG_DARK,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT_MAIN,
    "axes.titlecolor": TEXT_MAIN, "xtick.color": TEXT_DIM,
    "ytick.color": TEXT_DIM, "grid.color": BORDER, "grid.alpha": 0.5,
    "text.color": TEXT_MAIN, "legend.facecolor": BG_CARD,
    "legend.edgecolor": BORDER, "font.family": "monospace",
})

# ════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════
if "data_rows" not in st.session_state:
    st.session_state.data_rows = []
if "engine" not in st.session_state:
    st.session_state.engine = None

# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <div>
    <h1>⚙ ANALIZADOR DE CONFIABILIDAD INDUSTRIAL</h1>
    <span>Weibull · Normal · Exponencial · Lognormal</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  SIDEBAR – INGRESO DE DATOS
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<p style='color:{ACCENT};font-weight:700;font-size:0.9rem;letter-spacing:0.08em;'>📋 REGISTRAR FALLA</p>", unsafe_allow_html=True)

    fecha  = st.text_input("Fecha (YYYY-MM-DD)", value="2024-01-15")
    equipo = st.text_input("Equipo / Activo",    value="Compresor-01")
    modo   = st.text_input("Modo de Fallo",       value="Falla mecánica")
    ttf    = st.number_input("TTF (horas)",        value=720.0, min_value=0.01, step=10.0)
    causa  = st.text_input("Causa Raíz",          value="Desgaste rodamiento")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Agregar", use_container_width=True):
            n = len(st.session_state.data_rows) + 1
            st.session_state.data_rows.append(
                (n, fecha, equipo, modo, ttf, causa)
            )
            st.success("Falla agregada ✓")

    with col2:
        if st.button("🗑 Limpiar", use_container_width=True):
            st.session_state.data_rows = []
            st.session_state.engine = None
            st.rerun()

    st.markdown("---")
    st.markdown(f"<p style='color:{ACCENT4};font-weight:700;font-size:0.82rem;'>IMPORTAR / EXPORTAR</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("📂 Cargar CSV", type=["csv"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            req = {"Fecha","Equipo","Modo_Fallo","TTF_h","Causa"}
            if req.issubset(df_up.columns):
                st.session_state.data_rows = []
                for i, row in df_up.iterrows():
                    st.session_state.data_rows.append(
                        (i+1, row["Fecha"], row["Equipo"],
                         row["Modo_Fallo"], float(row["TTF_h"]), row["Causa"])
                    )
                st.success(f"{len(df_up)} registros cargados ✓")
            else:
                st.error(f"El CSV necesita: {', '.join(req)}")
        except Exception as ex:
            st.error(str(ex))

    if st.session_state.data_rows:
        df_exp = pd.DataFrame(
            st.session_state.data_rows,
            columns=["N","Fecha","Equipo","Modo_Fallo","TTF_h","Causa"]
        )
        csv_bytes = df_exp.to_csv(index=False).encode()
        st.download_button("💾 Exportar CSV", data=csv_bytes,
                           file_name="fallas_confiabilidad.csv",
                           mime="text/csv", use_container_width=True)

    st.markdown("---")
    if st.button("🔬 Cargar Datos Demo", use_container_width=True):
        st.session_state.data_rows = []
        np.random.seed(42)
        n_demo = 20; beta_d = 2.2; eta_d = 1200
        ttfs_d = eta_d * np.random.weibull(beta_d, n_demo)
        bases  = pd.date_range("2023-01-01", periods=n_demo, freq="30D")
        modos  = ["Desgaste rodamiento","Falla sello","Vibración excesiva",
                  "Sobrecalentamiento","Fatiga superficial"]
        causas = ["Lubricación deficiente","Contaminación","Sobrecarga",
                  "Mal alineamiento","Corrosión"]
        for i, (t, d) in enumerate(zip(ttfs_d, bases)):
            st.session_state.data_rows.append(
                (i+1, d.strftime("%Y-%m-%d"), "Compresor-01",
                 np.random.choice(modos), round(t, 1), np.random.choice(causas))
            )
        st.success("20 fallas demo cargadas ✓")
        st.rerun()

    # Stats rápidas
    if st.session_state.data_rows:
        ttfs_s = np.array([r[4] for r in st.session_state.data_rows])
        st.markdown(f"""
        <div style='background:{BG_INPUT};border-radius:8px;padding:12px;margin-top:8px;font-size:0.75rem;color:{TEXT_DIM};'>
        N fallas : <b style='color:{TEXT_MAIN}'>{len(ttfs_s)}</b><br>
        TTF min  : <b style='color:{TEXT_MAIN}'>{ttfs_s.min():.1f} h</b><br>
        TTF max  : <b style='color:{TEXT_MAIN}'>{ttfs_s.max():.1f} h</b><br>
        TTF mean : <b style='color:{TEXT_MAIN}'>{ttfs_s.mean():.1f} h</b><br>
        TTF std  : <b style='color:{TEXT_MAIN}'>{ttfs_s.std():.1f} h</b>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TABLA DE DATOS
# ════════════════════════════════════════════════════════════
if st.session_state.data_rows:
    df_show = pd.DataFrame(
        st.session_state.data_rows,
        columns=["N°","Fecha","Equipo","Modo de Fallo","TTF (h)","Causa Raíz"]
    )
    st.dataframe(df_show, use_container_width=True, height=220,
                 hide_index=True)

    # Botón analizar
    if st.button("▶  ANALIZAR DATOS", use_container_width=True):
        ttfs_arr = np.array([r[4] for r in st.session_state.data_rows], dtype=float)
        if len(ttfs_arr) < 3:
            st.warning("Se necesitan al menos 3 registros para ajustar distribuciones.")
        else:
            with st.spinner("Ajustando distribuciones..."):
                st.session_state.engine = ReliabilityEngine(ttfs_arr)
            st.success(f"✅ Análisis completado con {len(ttfs_arr)} datos.")
else:
    st.info("👈 Agrega fallas desde el panel izquierdo o carga datos demo para comenzar.")

# ════════════════════════════════════════════════════════════
#  RESULTADOS
# ════════════════════════════════════════════════════════════
if st.session_state.engine:
    eng  = st.session_state.engine
    best = eng.best_fit()

    st.markdown("---")
    st.markdown(f"""
    <div class="best-dist-box">
        ✅ Mejor ajuste (menor KS): &nbsp;<span style='font-size:1.1rem'>{best}</span>
        &nbsp;—&nbsp; {eng.fits[best]['label']}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs([
        "  📊  KPIs & Resultados  ",
        "  📈  Gráficas  ",
        "  🔵  Papel Weibull  ",
        "  📋  Comparación  "
    ])

    # ── TAB 1: KPIs ──────────────────────────────────────────
    with tab1:
        dist_sel = st.radio(
            "Distribución",
            list(DIST_COLORS.keys()),
            index=list(DIST_COLORS.keys()).index(best),
            horizontal=True,
        )
        kpis = eng.kpis(dist_sel)

        def kpi_card(label, value, unit, color):
            return f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}">{value}</div>
                <div class="kpi-unit">{unit}</div>
            </div>"""

        kpi_defs = [
            ("MTBF",          "Tiempo Medio Entre Fallas", "h",   ACCENT,  f"{kpis['MTBF']:,.2f}"),
            ("MTTF",          "Tiempo Medio Para Fallar",  "h",   ACCENT,  f"{kpis['MTTF']:,.2f}"),
            ("Lambda",        "Tasa de Fallas λ",          "1/h", ACCENT3, f"{kpis['Lambda']:.6f}"),
            ("Hazard_media",  "Tasa de Riesgo h(t̄)",       "1/h", ACCENT4, f"{kpis['Hazard_media']:.6f}"),
            ("R_50pct",       "Confiabilidad R(0.5·MTBF)", "%",   ACCENT2, f"{kpis['R_50pct']*100:.2f}%"),
            ("R_90pct",       "Confiabilidad R(0.9·MTBF)", "%",   ACCENT2, f"{kpis['R_90pct']*100:.2f}%"),
            ("B10",           "Vida B10 (10% fallos)",     "h",   ACCENT3, f"{kpis['B10']:,.2f}"),
            ("B50",           "Vida B50 (50% fallos)",     "h",   ACCENT3, f"{kpis['B50']:,.2f}"),
            ("Disponibilidad","Disponibilidad (MTTR=8h)",  "%",   ACCENT2, f"{kpis['Disponibilidad']*100:.2f}%"),
            ("KS",            "Test K-S (bondad ajuste)",  "",    TEXT_DIM,f"{kpis['KS']:.4f}"),
            ("AIC",           "Criterio AIC",              "",    TEXT_DIM,f"{kpis['AIC']:.2f}"),
            ("BIC",           "Criterio BIC",              "",    TEXT_DIM,f"{kpis['BIC']:.2f}"),
        ]

        cols = st.columns(4)
        for i, (key, label, unit, color, val) in enumerate(kpi_defs):
            with cols[i % 4]:
                st.markdown(kpi_card(label, val, unit, color), unsafe_allow_html=True)
                st.markdown("")

    # ── TAB 2: GRÁFICAS ──────────────────────────────────────
    with tab2:
        dists_vis = st.multiselect(
            "Distribuciones visibles",
            list(DIST_COLORS.keys()),
            default=list(DIST_COLORS.keys()),
        )

        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor(BG_CARD)
        gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[0,2])
        ax4 = fig.add_subplot(gs[1,0])
        ax5 = fig.add_subplot(gs[1,1])
        ax6 = fig.add_subplot(gs[1,2])

        data = eng.data
        for dn, color in DIST_COLORS.items():
            if dn not in dists_vis: continue
            cv = eng.curves(dn)
            if not cv: continue
            t, pdf, cdf, sf, hz = cv["t"], cv["pdf"], cv["cdf"], cv["sf"], cv["hz"]
            ax1.plot(t, pdf, color=color, lw=2, label=dn)
            ax2.plot(t, cdf, color=color, lw=2, label=dn)
            ax3.plot(t, sf,  color=color, lw=2, label=dn)
            ax4.plot(t, hz,  color=color, lw=2, label=dn)

        ax5.hist(data, bins=min(15, max(5, len(data)//3)),
                 color=ACCENT, alpha=0.3, density=True,
                 edgecolor=BORDER, label="Datos")
        for dn, color in DIST_COLORS.items():
            if dn not in dists_vis: continue
            cv = eng.curves(dn)
            if cv: ax5.plot(cv["t"], cv["pdf"], color=color, lw=2, label=dn)

        n_d = len(data)
        chf = np.cumsum(1.0 / np.arange(n_d, 0, -1))
        ax6.step(data, chf, color=ACCENT, lw=2, label="Nelson-Aalen")
        for dn, color in DIST_COLORS.items():
            if dn not in dists_vis: continue
            cv = eng.curves(dn)
            if cv:
                f2 = eng.fits[dn]; d2 = f2["dist"]; p2 = f2["params"]
                nchf = -np.log(np.where(d2.sf(cv["t"],*p2)>1e-10, d2.sf(cv["t"],*p2), 1e-10))
                ax6.plot(cv["t"], nchf, color=color, lw=1.5, linestyle="--", label=dn)

        titles  = ["Función de Densidad f(t)", "Función Acumulada F(t)",
                   "Confiabilidad R(t)", "Tasa de Riesgo h(t)",
                   "Histograma + PDF", "Riesgo Acumulado H(t)"]
        ylabels = ["f(t)", "F(t)", "R(t)", "h(t)", "Densidad", "H(t)"]
        for ax, title, yl in zip([ax1,ax2,ax3,ax4,ax5,ax6], titles, ylabels):
            ax.set_title(title, fontsize=9, pad=6)
            ax.set_xlabel("Tiempo (h)", fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, framealpha=0.7)

        st.pyplot(fig, use_container_width=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        st.download_button("💾 Descargar Figura", data=buf.getvalue(),
                           file_name="graficas_confiabilidad.png",
                           mime="image/png")

    # ── TAB 3: PAPEL WEIBULL ─────────────────────────────────
    with tab3:
        st.caption("Los puntos deben alinearse sobre la recta para confirmar el ajuste Weibull.")
        data_wb = eng.data
        n_wb    = len(data_wb)
        ranks   = (np.arange(1, n_wb+1) - 0.3) / (n_wb + 0.4)
        x_wb    = np.log(data_wb)
        y_wb    = np.log(-np.log(1 - ranks))

        fig_wb, ax_wb = plt.subplots(figsize=(10, 5))
        fig_wb.patch.set_facecolor(BG_CARD)
        ax_wb.scatter(x_wb, y_wb, color=ACCENT, s=50, zorder=5, label="Datos observados")

        m, b_lin = np.polyfit(x_wb, y_wb, 1)
        xf = np.linspace(x_wb.min(), x_wb.max(), 100)
        ax_wb.plot(xf, m*xf + b_lin, color=ACCENT3, lw=2, label=f"Ajuste lineal  β̂≈{m:.2f}")

        if "Weibull" in eng.fits:
            fw   = eng.fits["Weibull"]
            beta = fw["beta"]; eta = fw["eta"]
            t_l  = np.linspace(data_wb.min()*0.5, data_wb.max()*1.5, 200)
            cdf_l = np.clip(stats.weibull_min.cdf(t_l, beta, 0, eta), 1e-9, 1-1e-9)
            y_l   = np.log(-np.log(1 - cdf_l))
            ax_wb.plot(np.log(t_l), y_l, color=WEIBULL_C, lw=1.5,
                       linestyle="--", label=f"Weibull β={beta:.2f} η={eta:.1f}")

        ax_wb.set_xlabel("ln(t)", fontsize=9)
        ax_wb.set_ylabel("ln(−ln(1−F))", fontsize=9)
        ax_wb.set_title("Papel de Probabilidad de Weibull", fontsize=11)
        ax_wb.grid(True, alpha=0.3)
        ax_wb.legend(fontsize=9)
        st.pyplot(fig_wb, use_container_width=True)

    # ── TAB 4: COMPARACIÓN ───────────────────────────────────
    with tab4:
        rows_comp = []
        for d_name in ("Weibull", "Normal", "Exponencial", "Lognormal"):
            k = eng.kpis(d_name)
            if not k: continue
            rows_comp.append({
                "Distribución": d_name,
                "Parámetros":   eng.fits[d_name]["label"],
                "MTBF (h)":     f"{k['MTBF']:,.1f}",
                "λ (1/h)":      f"{k['Lambda']:.6f}",
                "B10 (h)":      f"{k['B10']:,.1f}",
                "KS stat":      f"{k['KS']:.4f}",
                "AIC":          f"{k['AIC']:.1f}",
                "BIC":          f"{k['BIC']:.1f}",
                "Mejor ajuste": "✅" if d_name == best else "",
            })
        df_comp = pd.DataFrame(rows_comp)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        csv_comp = df_comp.to_csv(index=False).encode()
        st.download_button("💾 Exportar tabla comparativa", data=csv_comp,
                           file_name="comparacion_distribuciones.csv",
                           mime="text/csv")
