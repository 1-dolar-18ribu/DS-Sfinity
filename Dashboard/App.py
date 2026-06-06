import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import io
 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
 
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Finansial Mahasiswa 🎓",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
 
/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.metric-card .label {
    font-size: 12px;
    color: #94a3b8;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 16px;
    font-weight: 800;
    color: #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 12px;
    color: #64748b;
    margin-top: 4px;
}
 
/* Status badge */
.badge-bahaya    { background:#E63946; color:#fff; border-radius:8px; padding:2px 10px; font-size:12px; font-weight:700; }
.badge-waspada   { background:#F4A261; color:#fff; border-radius:8px; padding:2px 10px; font-size:12px; font-weight:700; }
.badge-stabil    { background:#E9C46A; color:#1e293b; border-radius:8px; padding:2px 10px; font-size:12px; font-weight:700; }
.badge-sehat     { background:#2A9D8F; color:#fff; border-radius:8px; padding:2px 10px; font-size:12px; font-weight:700; }
 
/* Section header */
.section-header {
    font-size: 22px;
    font-weight: 800;
    color: #f1f5f9;
    margin-bottom: 4px;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(99,102,241,0.4);
}
.section-sub {
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 20px;
}
 
/* Insight box */
.insight-box {
    background: rgba(99,102,241,0.08);
    border-left: 4px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    font-size: 14px;
    color: #c7d2fe;
    margin-top: 10px;
    margin-bottom: 10px;
}
 
/* Cluster summary card */
.cluster-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 14px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 12px;
}
 
/* Streamlit overrides */
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size:26px !important; }
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 15px; }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
KURS_USD_IDR = 16_000
RASIO = {
    'pendidikan':     0.20,
    'tempat_tinggal': 0.30,
    'makanan':        0.35,
    'transportasi':   0.30,
    'buku':           0.40,
    'hiburan':        0.30,
    'perawatan':      0.35,
    'teknologi':      0.45,
    'kesehatan':      0.35,
    'lainnya':        0.35,
}
STATUS_ORDER  = ["Bahaya", "Waspada", "Stabil", "Sangat Sehat"]
STATUS_COLORS = {"Bahaya": "#E63946", "Waspada": "#F4A261",
                 "Stabil": "#E9C46A", "Sangat Sehat": "#2A9D8F"}
CLUSTER_COLORS = ['#E63946', '#2A9D8F', '#E9C46A', '#457B9D', '#F4A261', '#A8DADC']
 
# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fmt_rupiah(val, short=True):
    if pd.isna(val):
        return "N/A"
    if short:
        if abs(val) >= 1_000_000:
            return f"Rp {val/1_000_000:.1f} jt"
        elif abs(val) >= 1_000:
            return f"Rp {val/1_000:.0f} rb"
        return f"Rp {val:.0f}"
    return f"Rp {val:,.0f}"
 
def get_status_badge(status):
    cls_map = {"Bahaya": "bahaya", "Waspada": "waspada",
               "Stabil": "stabil", "Sangat Sehat": "sehat"}
    cls = cls_map.get(status, "stabil")
    return f'<span class="badge-{cls}">{status}</span>'
 
@st.cache_data(show_spinner=False)
def load_and_process_data(filepath_or_buffer):
    """Load CSV hasil EDA & clustering, compute derived features."""
    df_raw = pd.read_csv("Data/student_spending (1).csv")
 
    df = df_raw.rename(columns={
        'Unnamed: 0'     : 'id',
        'age'            : 'usia',
        'monthly_income' : 'pendapatan',
        'financial_aid'  : 'bantuan',
        'tuition'        : 'pendidikan',
        'housing'        : 'tempat_tinggal',
        'food'           : 'makanan',
        'transportation' : 'transportasi',
        'books_supplies' : 'buku',
        'entertainment'  : 'hiburan',
        'personal_care'  : 'perawatan',
        'technology'     : 'teknologi',
        'health_wellness': 'kesehatan',
        'miscellaneous'  : 'lainnya'
    })
 
    for col in ['pendapatan', 'bantuan']:
        if col in df.columns:
            df[col] = (df[col] * KURS_USD_IDR).round(0)
 
    for col, rasio_val in RASIO.items():
        if col in df.columns:
            if col == 'pendidikan':
                df[col] = (df[col] / 6 * KURS_USD_IDR * rasio_val).round(0)
            else:
                df[col] = (df[col] * KURS_USD_IDR * rasio_val).round(0)
 
    df['total_pemasukan']   = df['pendapatan'] + df['bantuan']
    df['total_pengeluaran'] = df[list(RASIO.keys())].sum(axis=1)
    df['sisa_uang']         = df['total_pemasukan'] - df['total_pengeluaran']
    df['rasio_pengeluaran'] = df['total_pengeluaran'] / df['total_pemasukan'].replace(0, np.nan)
 
    # Financial score
    def hitung_score(row):
        ti = row['total_pemasukan']
        tp = row['total_pengeluaran']
        rasio_tab = max(0, min(1, row['sisa_uang'] / ti)) if ti > 0 else 0
        skor_a = rasio_tab * 100
        esensial = row['tempat_tinggal'] + row['makanan'] + row['transportasi']
        skor_b = (esensial / tp * 100) if tp > 0 else 50
        buffer = max(0, min(1, row['sisa_uang'] / tp)) if tp > 0 else 0
        skor_c = buffer * 100
        return round(0.40*skor_a + 0.30*skor_b + 0.30*skor_c, 2)
 
    df['financial_score'] = df.apply(hitung_score, axis=1)
 
    _p25 = df['financial_score'].quantile(0.25)
    _p50 = df['financial_score'].quantile(0.50)
    _p75 = df['financial_score'].quantile(0.75)
 
    def get_status(s):
        if s >= _p75: return "Sangat Sehat"
        elif s >= _p50: return "Stabil"
        elif s >= _p25: return "Waspada"
        return "Bahaya"
 
    df['status_finansial'] = df['financial_score'].apply(get_status)
 
    # Clustering
    categorical_cols = [c for c in df.select_dtypes('object').columns
                        if c not in ['status_finansial', 'nama_cluster']]
    df_enc = df.copy()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_enc[col+'_enc'] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
 
    fitur = ['total_pemasukan','total_pengeluaran','sisa_uang','rasio_pengeluaran',
             'pendidikan','tempat_tinggal','makanan','transportasi','hiburan','teknologi',
             'financial_score'] + [c+'_enc' for c in categorical_cols if c+'_enc' in df_enc.columns]
    fitur = [c for c in fitur if c in df_enc.columns]
 
    X = df_enc[fitur].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    # Best k via silhouette
    sil_scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(X_scaled)
        sil_scores[k] = silhouette_score(X_scaled, km.labels_)
    best_k = max(sil_scores, key=sil_scores.get)
 
    kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=20, random_state=42)
    kmeans.fit(X_scaled)
 
    df.loc[X.index, 'cluster'] = kmeans.labels_
    df['cluster'] = df['cluster'].astype('Int64')
 
    sil = silhouette_score(X_scaled, kmeans.labels_)
    db  = davies_bouldin_score(X_scaled, kmeans.labels_)
 
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100
 
    # Cluster naming
    cluster_list = sorted(df['cluster'].dropna().unique())
    kolom_profil = ['total_pemasukan','total_pengeluaran','sisa_uang',
                    'rasio_pengeluaran','pendidikan','tempat_tinggal',
                    'makanan','hiburan','teknologi','financial_score']
    kolom_profil = [c for c in kolom_profil if c in df.columns]
    profil = df.groupby('cluster')[kolom_profil].mean().round(0)
 
    sisa_rata = profil['sisa_uang']
    rasio_rata = profil['rasio_pengeluaran']
    pend_rata  = profil['total_pemasukan']
    n_cl = len(cluster_list)
 
    nama_cluster = {}
    rank_sisa  = sisa_rata.rank(ascending=False)
    rank_rasio = rasio_rata.rank(ascending=True)
    rank_pend  = pend_rata.rank(ascending=False)
 
    for cl in cluster_list:
        r_s = rank_sisa[cl]; r_r = rank_rasio[cl]; r_p = rank_pend[cl]
        if r_s <= 1 and r_p <= 1:
            label = "Mahasiswa Mapan & Hemat"
        elif r_s <= 1:
            label = "Mahasiswa Hemat"
        elif r_p <= 1 and r_r >= n_cl:
            label = "Mahasiswa Berpenghasilan Tinggi & Boros"
        elif r_r >= n_cl:
            label = "Mahasiswa Boros / Defisit"
        elif r_p >= n_cl:
            label = "Mahasiswa Berpenghasilan Rendah"
        else:
            label = "Mahasiswa Rata-rata"
        nama_cluster[cl] = label
 
    df['nama_cluster'] = df['cluster'].map(nama_cluster)
 
    meta = dict(
        best_k=best_k, sil=sil, db=db,
        profil=profil, cluster_list=cluster_list,
        nama_cluster=nama_cluster, kmeans=kmeans,
        X=X, X_scaled=X_scaled, X_pca=X_pca,
        var_exp=var_exp, _p25=_p25, _p50=_p50, _p75=_p75,
        get_status=get_status, fitur=fitur,
        categorical_cols=categorical_cols, encoders=encoders,
        sil_scores=sil_scores
    )
    return df, meta
 
def fig_to_streamlit(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)
 
def style_fig(fig):
    fig.patch.set_facecolor('#0f172a')
    for ax in fig.get_axes():
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#f1f5f9')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
 
# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Financial Dashboard")
    st.markdown("**Clustering Finansial Mahasiswa Indonesia**")
    st.divider()

    st.markdown("### 📂 Sumber Data")
    st.markdown("""
    Data yang digunakan adalah hasil pipeline **EDA + K-Means Clustering** yang sudah diproses sebelumnya.

    **📋 File data:**
    - `hasil_clustering_finansial_IDR.csv`
    """)
    st.divider()

    # Optional: override dengan file lain
    st.markdown("### 🔄 Ganti Dataset *(opsional)*")
    uploaded_file = st.file_uploader(
        "Upload CSV lain (opsional)",
        type=["csv"],
        help="Kosongkan untuk menggunakan data default hasil EDA & clustering"
    )
    st.divider()
    st.caption("Dashboard by Claude · Data: Kaggle Student Spending")
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# MAIN — LOAD DATA
# ─────────────────────────────────────────────
st.markdown('<h1 style="font-family:Plus Jakarta Sans;font-size:36px;font-weight:800;color:#f1f5f9;margin-bottom:4px;">💰 Dashboard Finansial Mahasiswa Indonesia</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8;font-size:16px;margin-bottom:24px;">Analisis Clustering K-Means · Konversi USD → IDR · Financial Health Score</p>', unsafe_allow_html=True)

# ─── Tentukan sumber data ───
DEFAULT_DATA_PATH = "hasil_clustering_finansial_IDR.csv"

if uploaded_file is not None:
    data_source = uploaded_file
    st.toast("✅ Menggunakan file yang di-upload", icon="📂")
elif os.path.exists(DEFAULT_DATA_PATH):
    data_source = DEFAULT_DATA_PATH
else:
    st.error(
        f"⚠️ File data default **`{DEFAULT_DATA_PATH}`** tidak ditemukan. "
        "Pastikan file CSV hasil EDA & clustering berada di folder yang sama dengan `App.py`, "
        "atau upload manual melalui sidebar."
    )
    st.stop()

# ─── LOAD DATA ───
with st.spinner("⚙️ Memuat data hasil EDA & clustering..."):
    df, meta = load_and_process_data(data_source)
 
profil        = meta['profil']
cluster_list  = meta['cluster_list']
nama_cluster  = meta['nama_cluster']
get_status    = meta['get_status']
best_k        = meta['best_k']
 
pengeluaran_cols = [c for c in list(RASIO.keys()) if c in df.columns]
categorical_cols = meta['categorical_cols']
 
# ─────────────────────────────────────────────
# KPI BAR
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
kpi_data = [
    ("Total Mahasiswa", f"{len(df):,}", "orang"),
    ("Cluster", str(best_k), "K-Means optimal"),
    ("Silhouette", f"{meta['sil']:.3f}", "kualitas cluster"),
    ("Avg Income", fmt_rupiah(df['total_pemasukan'].mean()), "per bulan"),
    ("Avg Score", f"{df['financial_score'].mean():.1f}", "financial health"),
]
for col, (label, value, sub) in zip([c1, c2, c3, c4, c5], kpi_data):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔍 EDA",
    "🤖 Clustering",
    "🏷️ Profil Cluster",
    "📥 Export"
])
 
# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📊 Ringkasan Kesehatan Finansial</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Gambaran besar kondisi keuangan seluruh mahasiswa dalam dataset</div>', unsafe_allow_html=True)
 
    # Status distribution
    dist_status = df['status_finansial'].value_counts().reindex(STATUS_ORDER, fill_value=0)
    dist_pct    = (dist_status / len(df) * 100).round(1)
 
    col_left, col_right = st.columns([1, 1])
 
    with col_left:
        st.markdown("#### 🩺 Distribusi Status Finansial")
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig)
        colors_pie = [STATUS_COLORS[s] for s in STATUS_ORDER]
        wedges, texts, autotexts = ax.pie(
            dist_pct.values, labels=STATUS_ORDER, colors=colors_pie,
            autopct='%1.1f%%', startangle=140,
            wedgeprops=dict(edgecolor='#0f172a', linewidth=2),
            pctdistance=0.78
        )
        for t in texts:   t.set_color('#cbd5e1'); t.set_fontsize(10)
        for t in autotexts: t.set_fontsize(10); t.set_fontweight('bold'); t.set_color('white')
        ax.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0f172a')
        fig_to_streamlit(fig)
 
    with col_right:
        st.markdown("#### 📈 Jumlah per Status")
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig)
        bars = ax.barh(
            [s for s in STATUS_ORDER[::-1]],
            [dist_status[s] for s in STATUS_ORDER[::-1]],
            color=[STATUS_COLORS[s] for s in STATUS_ORDER[::-1]],
            edgecolor='#0f172a', linewidth=1.5
        )
        for bar, stat in zip(bars, STATUS_ORDER[::-1]):
            v = dist_status[stat]
            pct = dist_pct[stat]
            ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                    f' {v} ({pct}%)', va='center', fontsize=10,
                    fontweight='bold', color='#f1f5f9')
        ax.set_xlim(0, dist_status.max()*1.35)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False)
        fig_to_streamlit(fig)
 
    # Financial score distribution
    st.markdown("#### 🎯 Distribusi Financial Health Score")
    fig, ax = plt.subplots(figsize=(12, 4))
    style_fig(fig)
    sns.histplot(df['financial_score'], bins=30, kde=True, ax=ax,
                 color='#6366f1', alpha=0.7, edgecolor='#0f172a', linewidth=0.5)
    ax.axvline(df['financial_score'].mean(), color='#f59e0b', linestyle='--', lw=2,
               label=f"Rata-rata: {df['financial_score'].mean():.1f}")
    ax.axvline(df['financial_score'].median(), color='#10b981', linestyle='-', lw=2,
               label=f"Median: {df['financial_score'].median():.1f}")
    for threshold, (color, lbl) in zip(
        [meta['_p25'], meta['_p50'], meta['_p75']],
        [('#E63946','Bahaya↔Waspada'), ('#F4A261','Waspada↔Stabil'), ('#2A9D8F','Stabil↔Sehat')]
    ):
        ax.axvline(threshold, color=color, linestyle=':', lw=1.5, alpha=0.8, label=f'{lbl}: {threshold:.1f}')
    ax.set_xlabel('Financial Health Score (0–100)')
    ax.set_ylabel('Frekuensi')
    ax.legend(fontsize=9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')
    fig_to_streamlit(fig)
 
    st.markdown(f"""
    <div class="insight-box">
    💡 <b>Interpretasi Score:</b> Financial Health Score dihitung dari 3 komponen:
    40% rasio tabungan, 30% proporsi pengeluaran esensial, 30% buffer likuiditas.
    Score rata-rata <b>{df['financial_score'].mean():.1f}/100</b> menunjukkan kondisi finansial mahasiswa secara keseluruhan.
    </div>
    """, unsafe_allow_html=True)
 
# ══════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Distribusi, korelasi, dan pola pengeluaran mahasiswa</div>', unsafe_allow_html=True)
 
    eda_sub = st.selectbox("Pilih analisis:", [
        "Distribusi Numerik",
        "Boxplot Pengeluaran",
        "Komposisi Pengeluaran",
        "Matriks Korelasi",
        "Distribusi Kategorikal"
    ])
 
    if eda_sub == "Distribusi Numerik":
        numerik_cols = ['usia','pendapatan','bantuan','total_pemasukan',
                        'total_pengeluaran','sisa_uang','rasio_pengeluaran','financial_score']
        numerik_cols = [c for c in numerik_cols if c in df.columns]
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        style_fig(fig)
        axes = axes.flatten()
        for i, col in enumerate(numerik_cols):
            ax = axes[i]
            data_col = df[col].dropna()
            ax.hist(data_col, bins=25, color='#6366f1', alpha=0.7, edgecolor='#0f172a')
            ax2 = ax.twinx()
            ax2.set_yticks([])
            ax2.set_facecolor('#1e293b')
            if data_col.nunique() > 1:
                try: data_col.plot.kde(ax=ax2, color='#f59e0b', linewidth=2)
                except: pass
            mv = data_col.mean(); mdv = data_col.median()
            ax.axvline(mv, color='#ef4444', linestyle='--', lw=1.5, label=f'Mean: {fmt_rupiah(mv) if col not in ["usia","rasio_pengeluaran","financial_score"] else f"{mv:.2f}"}')
            ax.axvline(mdv, color='#10b981', linestyle='-',  lw=1.5, label=f'Median: {fmt_rupiah(mdv) if col not in ["usia","rasio_pengeluaran","financial_score"] else f"{mdv:.2f}"}')
            ax.set_title(col, fontsize=10, fontweight='bold', color='#f1f5f9')
            ax.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')
            ax.spines[['top','right']].set_visible(False)
        for j in range(len(numerik_cols), len(axes)): axes[j].set_visible(False)
        fig.suptitle('Distribusi Kolom Numerik | Merah=Mean · Hijau=Median', color='#f1f5f9', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig_to_streamlit(fig)
 
    elif eda_sub == "Boxplot Pengeluaran":
        fig, ax = plt.subplots(figsize=(14, 6))
        style_fig(fig)
        bp = ax.boxplot(
            [df[c].dropna() for c in pengeluaran_cols],
            labels=pengeluaran_cols,
            patch_artist=True,
            medianprops=dict(color='#ef4444', linewidth=2),
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(color='#94a3b8', linewidth=1.2),
            capprops=dict(color='#94a3b8'),
            flierprops=dict(marker='o', markersize=3, alpha=0.3, color='#94a3b8')
        )
        palette = sns.color_palette("husl", len(pengeluaran_cols))
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_title('Boxplot Pengeluaran per Kategori (IDR)', fontsize=12, fontweight='bold', color='#f1f5f9')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rp{x/1_000:.0f}rb"))
        ax.tick_params(axis='x', rotation=30, colors='#94a3b8')
        plt.tight_layout()
        fig_to_streamlit(fig)
 
    elif eda_sub == "Komposisi Pengeluaran":
        rata = df[pengeluaran_cols].mean().sort_values(ascending=False)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        style_fig(fig)
        colors_p = sns.color_palette("husl", len(rata))
        ax1.pie(rata, labels=rata.index, autopct='%1.1f%%', colors=colors_p,
                startangle=140, wedgeprops=dict(edgecolor='#0f172a', linewidth=1.5),
                pctdistance=0.78)
        for t in ax1.texts: t.set_color('#cbd5e1')
        ax1.set_facecolor('#0f172a')
        ax1.set_title('Komposisi Pengeluaran (%)', color='#f1f5f9', fontweight='bold')
        bars = ax2.barh(rata.index[::-1], rata.values[::-1], color=colors_p[::-1], edgecolor='#0f172a')
        for bar, val in zip(bars, rata.values[::-1]):
            ax2.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                     fmt_rupiah(val), va='center', fontsize=9, fontweight='bold', color='#f1f5f9')
        ax2.set_title('Rata-rata Pengeluaran per Kategori', color='#f1f5f9', fontweight='bold')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rp{x/1_000:.0f}rb"))
        ax2.set_xlim(0, rata.max()*1.35)
        ax2.spines[['top','right']].set_visible(False)
        fig.suptitle('💸 Kemana Uang Mahasiswa Indonesia Pergi?', color='#f1f5f9', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_to_streamlit(fig)
 
    elif eda_sub == "Matriks Korelasi":
        num_cols = df.drop(columns=['id'], errors='ignore').select_dtypes(include='number').columns
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(14, 11))
        style_fig(fig)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='RdYlGn', fmt='.2f',
                    vmin=-1, vmax=1, linewidths=0.5, linecolor='#0f172a',
                    annot_kws={"size": 7, "color": "#f1f5f9"}, ax=ax)
        ax.set_title('Matriks Korelasi Antar Variabel', fontsize=12, fontweight='bold', color='#f1f5f9')
        ax.tick_params(colors='#94a3b8')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fig_to_streamlit(fig)
 
    else:  # Distribusi Kategorikal
        if len(categorical_cols) > 0:
            fig, axes = plt.subplots(1, len(categorical_cols), figsize=(6*len(categorical_cols), 5))
            style_fig(fig)
            if len(categorical_cols) == 1: axes = [axes]
            colors_cat = ['#6366f1','#f59e0b','#10b981','#ef4444','#8b5cf6','#06b6d4']
            for ax, col in zip(axes, categorical_cols):
                counts = df[col].value_counts()
                bars = ax.bar(counts.index, counts.values,
                              color=colors_cat[:len(counts)], edgecolor='#0f172a')
                for bar, val in zip(bars, counts.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                            str(val), ha='center', fontsize=9, fontweight='bold', color='#f1f5f9')
                ax.set_title(col.upper(), fontsize=12, fontweight='bold', color='#f1f5f9')
                ax.spines[['top','right']].set_visible(False)
                ax.tick_params(axis='x', rotation=30)
            fig.suptitle('Distribusi Data Kategorikal', color='#f1f5f9', fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig_to_streamlit(fig)
        else:
            st.info("Tidak ada kolom kategorikal yang terdeteksi.")
 
# ══════════════════════════════════════════════
# TAB 3 — CLUSTERING
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🤖 Proses Clustering K-Means</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Seleksi jumlah cluster optimal, visualisasi hasil, dan metrik evaluasi</div>', unsafe_allow_html=True)
 
    # Elbow + Silhouette
    col_a, col_b = st.columns(2)
    k_range = sorted(meta['sil_scores'].keys())
 
    with col_a:
        st.markdown("#### 📐 Silhouette Score per K")
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig)
        sil_vals = [meta['sil_scores'][k] for k in k_range]
        ax.plot(k_range, sil_vals, 'o-', color='#10b981', linewidth=2.5, markersize=9, markerfacecolor='white', markeredgewidth=2.5)
        ax.fill_between(k_range, sil_vals, alpha=0.15, color='#10b981')
        ax.axvline(best_k, color='#ef4444', linestyle='--', lw=2, label=f'Optimal k={best_k}')
        ax.set_title(f'Silhouette Score (Best k={best_k})', color='#f1f5f9', fontweight='bold')
        ax.set_xlabel('Jumlah Cluster (k)'); ax.set_ylabel('Silhouette Score')
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')
        ax.set_xticks(list(k_range))
        ax.spines[['top','right']].set_visible(False)
        fig_to_streamlit(fig)
 
    with col_b:
        st.markdown("#### 📊 Evaluasi Model")
        ev_data = {
            "Metrik": ["Jumlah Cluster (k)", "Silhouette Score", "Davies-Bouldin Index", "Total Data", "Fitur Clustering"],
            "Nilai": [str(best_k), f"{meta['sil']:.4f}", f"{meta['db']:.4f}", str(len(meta['X'])), str(len(meta['fitur']))],
            "Interpretasi": ["Jumlah segmen mahasiswa", "Semakin → 1 makin baik", "Semakin → 0 makin baik", "Data valid (non-null)", "Dimensi input"]
        }
        st.dataframe(pd.DataFrame(ev_data), use_container_width=True, hide_index=True)
 
    # PCA scatter
    st.markdown("#### 🔵 Visualisasi Cluster (PCA 2D)")
    fig, ax = plt.subplots(figsize=(10, 6))
    style_fig(fig)
    X_pca = meta['X_pca']
    km    = meta['kmeans']
    for i, cl in enumerate(cluster_list):
        idx = km.labels_ == int(cl)
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                   c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                   label=f'Cluster {int(cl)} — {nama_cluster[cl]}',
                   alpha=0.6, s=55, edgecolors='#0f172a', linewidth=0.4)
 
    pca_obj = PCA(n_components=2, random_state=42)
    pca_obj.fit(meta['X_scaled'])
    centroids_pca = pca_obj.transform(km.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c='white', marker='X', s=250, zorder=5, label='Pusat Cluster', edgecolors='#0f172a')
 
    var_exp = meta['var_exp']
    ax.set_title(f'PCA 2D — PC1: {var_exp[0]:.1f}% · PC2: {var_exp[1]:.1f}% variasi',
                 fontsize=12, fontweight='bold', color='#f1f5f9')
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)'); ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)')
    ax.legend(fontsize=9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1',
              loc='upper right')
    ax.spines[['top','right']].set_visible(False)
    fig_to_streamlit(fig)
 
    st.markdown(f"""
    <div class="insight-box">
    💡 <b>Tentang PCA:</b> Dua komponen utama (PC1+PC2) menjelaskan
    <b>{var_exp[0]+var_exp[1]:.1f}%</b> dari total variasi data.
    Setiap titik = 1 mahasiswa. Cluster yang terpisah jauh = karakteristik finansial yang sangat berbeda.
    </div>
    """, unsafe_allow_html=True)
 
# ══════════════════════════════════════════════
# TAB 4 — PROFIL CLUSTER
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🏷️ Profil & Karakteristik Cluster</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ringkasan mendalam setiap segmen mahasiswa</div>', unsafe_allow_html=True)
 
    # Cluster summary cards
    cols_cards = st.columns(len(cluster_list))
    for col, cl in zip(cols_cards, cluster_list):
        subset  = df[df['cluster'] == cl]
        n_mhs   = len(subset)
        avg_sc  = subset['financial_score'].mean()
        status  = get_status(avg_sc)
        color   = CLUSTER_COLORS[int(cl) % len(CLUSTER_COLORS)]
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{color}22,{color}11);
                        border:1px solid {color}55;border-radius:14px;padding:16px;text-align:center;">
                <div style="font-size:11px;color:#94a3b8;font-weight:700;letter-spacing:.08em;text-transform:uppercase;">Cluster {int(cl)}</div>
                <div style="font-size:17px;font-weight:800;color:{color};margin:6px 0;">{nama_cluster[cl]}</div>
                <div style="font-size:13px;color:#94a3b8;">{n_mhs} mahasiswa</div>
                <div style="margin-top:8px;">{get_status_badge(status)}</div>
                <div style="font-size:12px;color:#64748b;margin-top:4px;">Score: {avg_sc:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Heatmap profil
    st.markdown("#### 🗺️ Heatmap Profil Rata-rata per Cluster")
    kolom_hm = [c for c in profil.columns if c in ['total_pemasukan','total_pengeluaran','sisa_uang',
               'rasio_pengeluaran','pendidikan','tempat_tinggal','makanan','hiburan','teknologi','financial_score']]
    profil_hm = profil[kolom_hm]
    profil_norm = (profil_hm - profil_hm.mean()) / profil_hm.std()
 
    fig, ax = plt.subplots(figsize=(13, 5))
    style_fig(fig)
    sns.heatmap(profil_norm.T, annot=profil_hm.T, fmt='.0f',
                cmap='RdYlGn', linewidths=0.5, linecolor='#0f172a',
                annot_kws={'size': 8, 'color': '#0f172a'}, ax=ax)
    ax.set_xticklabels([f'Cluster {int(c)}\n{nama_cluster[c]}' for c in profil.index],
                        rotation=0, ha='center', color='#f1f5f9', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color='#f1f5f9')
    ax.set_title('Profil Rata-rata per Cluster (Hijau=Tinggi · Merah=Rendah)',
                 color='#f1f5f9', fontweight='bold', fontsize=12)
    plt.tight_layout()
    fig_to_streamlit(fig)
 
    # Dashboard bar comparison
    st.markdown("#### 📊 Perbandingan KPI Antar Cluster")
    cols_plot = ['total_pemasukan','total_pengeluaran','sisa_uang','rasio_pengeluaran',
                 'hiburan','teknologi','financial_score']
    cols_plot = [c for c in cols_plot if c in df.columns]
    label_map = {
        'total_pemasukan':'Rata-rata Pemasukan','total_pengeluaran':'Rata-rata Pengeluaran',
        'sisa_uang':'Rata-rata Sisa Uang','rasio_pengeluaran':'Rasio Pengeluaran',
        'hiburan':'Pengeluaran Hiburan','teknologi':'Pengeluaran Teknologi',
        'financial_score':'Financial Health Score'
    }
 
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    style_fig(fig)
    axes = axes.flatten()
    for ax, col in zip(axes, cols_plot):
        vals = df.groupby('cluster')[col].mean()
        colors_bar = [CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)] for c in vals.index]
        bars = ax.bar([f'C{int(c)}' for c in vals.index], vals.values,
                      color=colors_bar, edgecolor='#0f172a', linewidth=1)
        for bar, val in zip(bars, vals.values):
            lbl = f'{val:.2f}' if col in ['rasio_pengeluaran','financial_score'] else fmt_rupiah(val)
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                    lbl, ha='center', fontsize=8, fontweight='bold', color='#f1f5f9')
        ax.set_title(label_map.get(col, col), fontsize=10, fontweight='bold', color='#f1f5f9')
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x', rotation=0)
    for j in range(len(cols_plot), len(axes)): axes[j].set_visible(False)
    fig.suptitle('Dashboard KPI Finansial per Cluster', color='#f1f5f9', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_to_streamlit(fig)
 
    # Status distribution per cluster — stacked bar
    st.markdown("#### 📊 Distribusi Status Finansial per Cluster")
    status_counts = df.groupby(['cluster','status_finansial']).size().unstack(fill_value=0)
    status_counts = status_counts.reindex(columns=STATUS_ORDER, fill_value=0)
    status_pct    = status_counts.div(status_counts.sum(axis=1), axis=0) * 100
 
    fig, ax = plt.subplots(figsize=(10, 5))
    style_fig(fig)
    bottom = np.zeros(len(status_pct))
    xticks = [f'Cluster {int(c)}\n{nama_cluster[c]}' for c in status_pct.index]
    for status in STATUS_ORDER:
        vals = status_pct[status].values
        bars = ax.bar(xticks, vals, bottom=bottom,
                      color=STATUS_COLORS[status], label=status, edgecolor='#0f172a', linewidth=1)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val > 5:
                ax.text(bar.get_x()+bar.get_width()/2, bottom[i]+val/2,
                        f'{val:.0f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
        bottom += vals
    ax.set_ylim(0, 108)
    ax.set_ylabel('Persentase (%)', color='#94a3b8')
    ax.set_title('Proporsi Status Finansial per Cluster', color='#f1f5f9', fontweight='bold', fontsize=12)
    ax.legend(title='Status', facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1',
              loc='upper right', fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    fig_to_streamlit(fig)
 
    # Violin plot financial score per cluster
    st.markdown("#### 🎻 Distribusi Financial Score per Cluster")
    fig, ax = plt.subplots(figsize=(10, 5))
    style_fig(fig)
    df_plot = df.dropna(subset=['cluster','financial_score'])
    violin_data = [df_plot[df_plot['cluster']==cl]['financial_score'].values for cl in cluster_list]
    parts = ax.violinplot(violin_data, positions=range(len(cluster_list)), showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(CLUSTER_COLORS[i % len(CLUSTER_COLORS)]); pc.set_alpha(0.75)
    parts['cmedians'].set_color('white'); parts['cmedians'].set_linewidth(2.5)
    for th_val, th_col, th_lbl in [
        (meta['_p25'],'#E63946','Bahaya→Waspada'),
        (meta['_p50'],'#F4A261','Waspada→Stabil'),
        (meta['_p75'],'#2A9D8F','Stabil→Sehat')
    ]:
        ax.axhline(th_val, linestyle='--', lw=1.5, color=th_col, alpha=0.8,
                   label=f'{th_lbl} ({th_val:.1f})')
    ax.set_xticks(range(len(cluster_list)))
    ax.set_xticklabels([f'Cluster {int(c)}\n{nama_cluster[c]}' for c in cluster_list], fontsize=9)
    ax.set_ylabel('Financial Health Score (0–100)'); ax.set_title('Violin Plot Financial Score per Cluster', color='#f1f5f9', fontweight='bold')
    ax.legend(fontsize=9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    fig_to_streamlit(fig)
 
# ══════════════════════════════════════════════
# TAB 5 — EXPORT
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📥 Export & Ringkasan Akhir</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Download hasil analisis lengkap</div>', unsafe_allow_html=True)
 
    # Summary table
    summary_rows = []
    for cl in cluster_list:
        subset   = df[df['cluster'] == cl]
        avg_sc   = subset['financial_score'].mean()
        dist_pct = subset['status_finansial'].value_counts(normalize=True).reindex(STATUS_ORDER, fill_value=0) * 100
        summary_rows.append({
            "Cluster": f"Cluster {int(cl)}",
            "Nama Segmen": nama_cluster[cl],
            "Jumlah": len(subset),
            "Avg Income": fmt_rupiah(subset['total_pemasukan'].mean()),
            "Avg Expense": fmt_rupiah(subset['total_pengeluaran'].mean()),
            "Avg Saving": fmt_rupiah(subset['sisa_uang'].mean()),
            "Financial Score": f"{avg_sc:.1f}",
            "Status Dominan": get_status(avg_sc),
            "% Bahaya": f"{dist_pct['Bahaya']:.1f}%",
            "% Waspada": f"{dist_pct['Waspada']:.1f}%",
            "% Stabil": f"{dist_pct['Stabil']:.1f}%",
            "% Sangat Sehat": f"{dist_pct['Sangat Sehat']:.1f}%",
        })
    df_summary = pd.DataFrame(summary_rows)
    st.markdown("#### 📋 Ringkasan Cluster")
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
 
    # Download buttons
    st.markdown("#### 💾 Download Data")
    col1, col2 = st.columns(2)
    with col1:
        csv_full = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Dataset Lengkap (CSV)",
                           data=csv_full,
                           file_name="hasil_clustering_finansial_IDR.csv",
                           mime="text/csv",
                           use_container_width=True)
    with col2:
        csv_sum = df_summary.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Ringkasan Cluster (CSV)",
                           data=csv_sum,
                           file_name="ringkasan_cluster_finansial.csv",
                           mime="text/csv",
                           use_container_width=True)
 
    st.markdown(f"""
    <div class="insight-box">
    ✅ <b>Ringkasan Analisis:</b><br>
    • Dataset: <b>{len(df):,} mahasiswa</b> · Kurs: 1 USD = Rp {KURS_USD_IDR:,}<br>
    • Algoritma: KMeans (k={best_k}) · Silhouette={meta['sil']:.4f} · Davies-Bouldin={meta['db']:.4f}<br>
    • Fitur: {len(meta['fitur'])} variabel finansial + kategorikal terenkode<br>
    • Kolom output: <code>cluster</code>, <code>nama_cluster</code>, <code>financial_score</code>, <code>status_finansial</code>
    </div>
    """, unsafe_allow_html=True)
