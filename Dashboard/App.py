import os
import requests
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Dashboard Finansial Mahasiswa 🎓",
    page_icon="💰", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
body { background: #f1f5f9 !important; }
[data-testid="stAppViewContainer"] { background: #f1f5f9; }

.metric-card {
    background: white; border-radius: 14px; padding: 18px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
    min-height: 115px; display: flex; flex-direction: column; justify-content: center;
}
.metric-card .label {
    font-size: 0.68rem; letter-spacing: .07em; color: #64748b;
    text-transform: uppercase; margin-bottom: 6px; line-height: 1.4;
}
.metric-card .value {
    font-size: 1.2rem; font-weight: 800; color: #0f172a; line-height: 1.2;
    word-break: break-word; white-space: normal;
}
.metric-card .sub { font-size: 0.73rem; color: #94a3b8; margin-top: 5px; }

.section-card {
    background: white; border-radius: 14px; padding: 22px 26px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07); margin-bottom: 1rem;
}
.section-title { font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 3px; }
.section-sub   { font-size: 0.82rem; color: #94a3b8; margin-bottom: 16px; }

.insight-box {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0; padding: 12px 16px;
    font-size: 0.88rem; color: #1e40af; margin-top: 12px; line-height: 1.7;
}
.badge-bahaya  { background:#ef4444; color:#fff; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:700; }
.badge-waspada { background:#f59e0b; color:#fff; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:700; }
.badge-stabil  { background:#3b82f6; color:#fff; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:700; }
.badge-sehat   { background:#22c55e; color:#fff; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:700; }
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS — SAMA PERSIS DENGAN NOTEBOOK ────────────────────────────────────


def get_live_kurs():
    try:
        r = requests.get(
            "https://api.frankfurter.app/latest?from=USD&to=IDR",
            timeout=5
        )
        return int(r.json()["rates"]["IDR"])
    except Exception:
        return 18_000  # fallback

KURS_USD_IDR = get_live_kurs()
# Rasio penyesuaian konteks Indonesia — SAMA PERSIS NOTEBOOK
RASIO = {
    'pendidikan'     : 0.20,
    'tempat_tinggal' : 0.30,
    'makanan'        : 0.35,
    'transportasi'   : 0.30,
    'buku'           : 0.40,
    'hiburan'        : 0.30,
    'perawatan'      : 0.35,
    'teknologi'      : 0.45,
    'kesehatan'      : 0.35,
    'lainnya'        : 0.35,
}

SPENDING_COLS = [
    'pendidikan', 'tempat_tinggal', 'makanan', 'transportasi',
    'buku', 'hiburan', 'perawatan', 'teknologi', 'kesehatan', 'lainnya'
]

STATUS_ORDER  = ["Bahaya", "Waspada", "Stabil", "Sangat Sehat"]
STATUS_COLORS = {
    "Bahaya":"#ef4444", "Waspada":"#f59e0b",
    "Stabil":"#3b82f6", "Sangat Sehat":"#22c55e"
}
CLUSTER_PALETTE = ["#E63946","#2A9D8F","#E9C46A","#457B9D","#F4A261","#A8DADC"]
CLUSTER_NAME_COLORS = {
    "Mahasiswa Mapan & Hemat"                : "#22c55e",
    "Mahasiswa Hemat"                        : "#10b981",
    "Mahasiswa Rata-rata"                    : "#3b82f6",
    "Mahasiswa Berpenghasilan Rendah"        : "#f59e0b",
    "Mahasiswa Boros / Defisit"              : "#ef4444",
    "Mahasiswa Berpenghasilan Tinggi & Boros": "#f97316",
}

LABEL_KOLOM = {
    'total_pemasukan'   : 'Total Pemasukan (Rp)',
    'total_pengeluaran' : 'Total Pengeluaran (Rp)',
    'sisa_uang'         : 'Sisa Uang (Rp)',
    'rasio_pengeluaran' : 'Rasio Pengeluaran (0–1)',
    'pendidikan'        : 'Pendidikan (Rp)',
    'tempat_tinggal'    : 'Tempat Tinggal (Rp)',
    'makanan'           : 'Makanan (Rp)',
    'transportasi'      : 'Transportasi (Rp)',
    'buku'              : 'Buku & Alat Tulis (Rp)',
    'hiburan'           : 'Hiburan (Rp)',
    'perawatan'         : 'Perawatan Diri (Rp)',
    'teknologi'         : 'Teknologi (Rp)',
    'kesehatan'         : 'Kesehatan (Rp)',
    'lainnya'           : 'Lainnya (Rp)',
    'financial_score'   : 'Skor Keuangan (0–100)',
    'usia'              : 'Usia (tahun)',
    'pendapatan'        : 'Pendapatan (Rp)',
    'bantuan'           : 'Bantuan Keuangan (Rp)',
}

# ── HELPERS ────────────────────────────────────────────────────────────────────
def kolom_label(col):
    return LABEL_KOLOM.get(col, col.replace('_', ' ').title())

def make_status(score, p25, p50, p75):
    if score >= p75: return "Sangat Sehat"
    if score >= p50: return "Stabil"
    if score >= p25: return "Waspada"
    return "Bahaya"

def fmt_rupiah(val, short=True):
    if pd.isna(val): return "N/A"
    if short:
        if abs(val) >= 1_000_000: return f"Rp {val/1_000_000:.1f} Jt"
        if abs(val) >= 1_000:     return f"Rp {val/1_000:.0f} rb"
        return f"Rp {val:.0f}"
    return f"Rp {val:,.0f}"

def get_status_badge(status):
    cls = {"Bahaya":"bahaya","Waspada":"waspada","Stabil":"stabil","Sangat Sehat":"sehat"}
    return f'<span class="badge-{cls.get(status,"stabil")}">{status}</span>'

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#334155"),
)
M  = dict(t=45, b=35, l=15, r=15)
ML = dict(t=45, b=20, l=170, r=15)

def jt_ticks(series, n=6):
    mn, mx = float(series.min()), float(series.max())
    if mn == mx:
        return [mn], [fmt_rupiah(mn)]
    vals  = np.linspace(mn, mx, n)
    texts = [f"Rp{v/1_000_000:.1f}Jt" if abs(v) >= 1_000_000 else f"Rp{v/1_000:.0f}rb" for v in vals]
    return vals.tolist(), texts

# ── DATA LOADING — LOGIKA KONVERSI SAMA PERSIS NOTEBOOK ───────────────────────
@st.cache_data(show_spinner=False)
def load_and_process_data(file_key: str):
    path = "Data/student_spending (1).csv" if file_key == "__default__" else file_key
    df_raw = pd.read_csv(path)

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
        'miscellaneous'  : 'lainnya',
    })

    # ── Konversi income (kali kurs saja, tanpa rasio) — SAMA NOTEBOOK
    for col in ['pendapatan', 'bantuan']:
        if col in df.columns:
            df[col] = (df[col] * KURS_USD_IDR).round(0)

    # ── Konversi spending pakai RASIO — SAMA NOTEBOOK
    for col, rasio_val in RASIO.items():
        if col in df.columns:
            if col == 'pendidikan':
                # Bagi 6 (semester → bulanan) lalu kali kurs & rasio — SAMA NOTEBOOK
                df[col] = (df[col] / 6 * KURS_USD_IDR * rasio_val).round(0)
            else:
                df[col] = (df[col] * KURS_USD_IDR * rasio_val).round(0)

    # ── Derived columns — SAMA NOTEBOOK
    df['total_pemasukan']   = df['pendapatan'] + df['bantuan']
    df['total_pengeluaran'] = df[SPENDING_COLS].sum(axis=1)
    df['sisa_uang']         = df['total_pemasukan'] - df['total_pengeluaran']
    df['rasio_pengeluaran'] = (
        df['total_pengeluaran'] / df['total_pemasukan'].replace(0, np.nan)
    )

    # ── Financial Health Score — SAMA PERSIS NOTEBOOK
    def hitung_financial_score(row):
        total_peng = row['total_pengeluaran']
        total_in   = row['total_pemasukan']

        if total_in > 0:
            rasio_tabungan = max(0, min(1, row['sisa_uang'] / total_in))
        else:
            rasio_tabungan = 0
        skor_a = rasio_tabungan * 100

        esensial = row['tempat_tinggal'] + row['makanan'] + row['transportasi']
        if total_peng > 0:
            rasio_esensial = esensial / total_peng
            skor_b = rasio_esensial * 100
        else:
            skor_b = 50

        if total_peng > 0:
            buffer_ratio = max(0, min(1, row['sisa_uang'] / total_peng))
        else:
            buffer_ratio = 0
        skor_c = buffer_ratio * 100

        return round(0.40 * skor_a + 0.30 * skor_b + 0.30 * skor_c, 2)

    df['financial_score'] = df.apply(hitung_financial_score, axis=1)

    p25 = float(df['financial_score'].quantile(0.25))
    p50 = float(df['financial_score'].quantile(0.50))
    p75 = float(df['financial_score'].quantile(0.75))

    def get_status(score):
        if score >= p75: return "Sangat Sehat"
        if score >= p50: return "Stabil"
        if score >= p25: return "Waspada"
        return "Bahaya"

    df['status_finansial'] = df['financial_score'].apply(get_status)

    # ── Clustering — SAMA NOTEBOOK
    cat_cols = [c for c in df.select_dtypes('object').columns
                if c not in ['status_finansial', 'nama_cluster']]
    df_enc = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col + '_enc'] = le.fit_transform(df_enc[col].astype(str))

    fitur = (
        ['total_pemasukan', 'total_pengeluaran', 'sisa_uang', 'rasio_pengeluaran',
         'pendidikan', 'tempat_tinggal', 'makanan', 'transportasi', 'hiburan',
         'teknologi', 'financial_score']
        + [c + '_enc' for c in cat_cols if c + '_enc' in df_enc.columns]
    )
    fitur = [c for c in fitur if c in df_enc.columns]
    X = df_enc[fitur].dropna()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sil_scores = {}
    for k in range(2, 11):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(X_scaled)
        sil_scores[k] = float(silhouette_score(X_scaled, km.labels_))
    best_k = max(sil_scores, key=sil_scores.get)

    kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=20, random_state=42)
    kmeans.fit(X_scaled)
    sil_val = float(silhouette_score(X_scaled, kmeans.labels_))
    db_val  = float(davies_bouldin_score(X_scaled, kmeans.labels_))

    df.loc[X.index, 'cluster'] = kmeans.labels_.astype(int)
    df['cluster'] = df['cluster'].astype('Int64')

    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = [float(v * 100) for v in pca.explained_variance_ratio_]

    cluster_list = sorted(df['cluster'].dropna().unique().tolist())
    n_clusters   = len(cluster_list)

    kolom_profil = [c for c in ['total_pemasukan', 'total_pengeluaran', 'sisa_uang',
                                  'rasio_pengeluaran', 'pendidikan', 'tempat_tinggal',
                                  'makanan', 'hiburan', 'teknologi', 'financial_score']
                    if c in df.columns]
    profil = df.groupby('cluster')[kolom_profil].mean().round(0)

    # Penamaan cluster — SAMA NOTEBOOK
    sisa_rank  = profil['sisa_uang'].rank(ascending=False)
    rasio_rank = profil['rasio_pengeluaran'].rank(ascending=True)
    pend_rank  = profil['total_pemasukan'].rank(ascending=False)

    nama_cluster = {}
    for cl in cluster_list:
        r_sisa = sisa_rank[cl]
        r_rasio = rasio_rank[cl]
        r_pend  = pend_rank[cl]
        if r_sisa <= 1 and r_pend <= 1:
            label = "Mahasiswa Mapan & Hemat"
        elif r_sisa <= 1:
            label = "Mahasiswa Hemat"
        elif r_pend <= 1 and r_rasio >= n_clusters:
            label = "Mahasiswa Berpenghasilan Tinggi & Boros"
        elif r_rasio >= n_clusters:
            label = "Mahasiswa Boros / Defisit"
        elif r_pend >= n_clusters:
            label = "Mahasiswa Berpenghasilan Rendah"
        else:
            label = "Mahasiswa Rata-rata"
        nama_cluster[int(cl)] = label

    df['nama_cluster'] = df['cluster'].map(nama_cluster)

    pca_df = pd.DataFrame(X_pca, index=X.index, columns=['PC1', 'PC2'])
    df = df.join(pca_df, how='left')

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    centroid_df   = pd.DataFrame(centroids_pca, columns=['PC1', 'PC2'])
    centroid_df['cluster'] = list(range(best_k))
    profil.index = profil.index.astype(int)

    meta = dict(
        best_k=int(best_k), sil=float(sil_val), db=float(db_val),
        profil=profil, cluster_list=[int(c) for c in cluster_list],
        nama_cluster={int(k): str(v) for k, v in nama_cluster.items()},
        var_exp=var_exp, p25=p25, p50=p50, p75=p75,
        fitur=fitur, sil_scores={int(k): float(v) for k, v in sil_scores.items()},
        centroid_df=centroid_df,
        kolom_profil=kolom_profil,
    )
    return df, meta

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Financial Dashboard")
    st.markdown("**Clustering Finansial Mahasiswa**")
    st.divider()
    st.caption(f"Dashboard · Data: Kaggle Student Spending · Kurs Rp{KURS_USD_IDR:,}/USD")
DEFAULT_DATA_PATH = "Data/student_spending (1).csv"
uploaded_file = None  # tidak ada uploader lagi
# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="font-size:1.8rem;font-weight:800;color:#0f172a;margin-bottom:4px;">'
    '💰 Dashboard Finansial Mahasiswa Indonesia</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#64748b;font-size:0.95rem;margin-bottom:20px;">'
    f'Analisis K-Means Clustering · Konversi USD → IDR (kurs Rp{KURS_USD_IDR:,} + rasio biaya hidup) · Financial Health Score</p>',
    unsafe_allow_html=True)

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    uploaded_file.seek(0)
    tmp_path = f"/tmp/{uploaded_file.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())
    file_key = tmp_path
elif os.path.exists(DEFAULT_DATA_PATH):
    file_key = "__default__"
else:
    st.error(f"⚠️ File `{DEFAULT_DATA_PATH}` tidak ditemukan. Upload manual via sidebar.")
    st.stop()

with st.spinner("⚙️ Memuat & memproses data..."):
    df, meta = load_and_process_data(file_key)

profil       = meta['profil']
cluster_list = meta['cluster_list']
nama_cluster = meta['nama_cluster']
best_k       = meta['best_k']

with st.sidebar:
    st.markdown("### 🔍 Filter Data")
    cluster_filter = st.multiselect(
        "Cluster", options=df['nama_cluster'].dropna().unique().tolist(),
        default=df['nama_cluster'].dropna().unique().tolist())
    gender_opts = df['gender'].dropna().unique().tolist() if 'gender' in df.columns else []
    gender_filter = st.multiselect("Gender", options=gender_opts, default=gender_opts)
    year_opts = df['year_in_school'].dropna().unique().tolist() if 'year_in_school' in df.columns else []
    year_filter = st.multiselect("Tahun Kuliah", options=year_opts, default=year_opts)

fdf = df[df['nama_cluster'].isin(cluster_filter)]
if gender_opts: fdf = fdf[fdf['gender'].isin(gender_filter)]
if year_opts:   fdf = fdf[fdf['year_in_school'].isin(year_filter)]

# ── KPI CARDS ──────────────────────────────────────────────────────────────────
kpi_data = [
    ("Total Mahasiswa",     f"{len(fdf):,}",                                   "responden"),
    ("Cluster Optimal",     str(best_k),                                        "K-Means"),
    ("Silhouette Score",    f"{meta['sil']:.3f}",                               "kualitas cluster"),
    ("Rata-rata Pemasukan", fmt_rupiah(fdf['total_pemasukan'].mean()),           "per bulan"),
    ("Rata-rata Sisa",      fmt_rupiah(fdf['sisa_uang'].mean()),                 "per bulan"),
    ("Skor Keuangan",       f"{fdf['financial_score'].mean():.1f}",             "rata-rata / 100"),
]
cols = st.columns(len(kpi_data))
for col, (label, value, sub) in zip(cols, kpi_data):
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview","🔍 EDA","🤖 Clustering","🏷️ Profil Cluster","📥 Export"
])

IDR_COLS = ['pendapatan','bantuan','total_pemasukan','total_pengeluaran','sisa_uang'] + SPENDING_COLS

# ══════════════ TAB 1 — OVERVIEW ═══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-card"><div class="section-title">📊 Ringkasan Kesehatan Finansial</div>'
                '<div class="section-sub">Gambaran besar kondisi keuangan seluruh mahasiswa yang difilter</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        dist_status = fdf['status_finansial'].value_counts().reindex(STATUS_ORDER, fill_value=0)
        fig = px.pie(values=dist_status.values, names=dist_status.index,
                     color=dist_status.index, color_discrete_map=STATUS_COLORS,
                     hole=0.45, title="🩺 Distribusi Status Finansial")
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update_layout(**PL, height=320, showlegend=False, margin=M)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        dist_pct = (dist_status / len(fdf) * 100).round(1)
        fig = px.bar(
            x=dist_status.values, y=dist_status.index, orientation='h',
            color=dist_status.index, color_discrete_map=STATUS_COLORS,
            text=[f"{v} mhs ({dist_pct[s]:.1f}%)" for s, v in zip(dist_status.index, dist_status.values)],
            title="📈 Jumlah Mahasiswa per Status Finansial")
        fig.update_traces(textposition='outside')
        fig.update_layout(**PL, height=320, showlegend=False, margin=M,
                          xaxis_title="Jumlah Mahasiswa", yaxis_title="Status Finansial")
        st.plotly_chart(fig, use_container_width=True)

    # Histogram financial score
    fig = px.histogram(fdf, x='financial_score', nbins=30,
                       color_discrete_sequence=['#6366f1'],
                       title="🎯 Distribusi Financial Health Score",
                       labels={'financial_score':'Financial Health Score (0–100)','count':'Jumlah Mahasiswa'},
                       marginal='rug')
    mean_val = fdf['financial_score'].mean()
    med_val  = fdf['financial_score'].median()
    for xval, col_line, ann_lbl in [
        (mean_val,    '#f59e0b', f'Mean: {mean_val:.1f}'),
        (med_val,     '#10b981', f'Median: {med_val:.1f}'),
        (meta['p25'], '#ef4444', f'P25: {meta["p25"]:.1f}'),
        (meta['p75'], '#22c55e', f'P75: {meta["p75"]:.1f}'),
    ]:
        fig.add_vline(x=xval, line_dash='dash', line_color=col_line,
                      annotation_text=ann_lbl, annotation_position='top right',
                      annotation_font_color=col_line)
    fig.update_layout(**PL, height=340, margin=M,
                      xaxis_title="Financial Health Score (0–100)",
                      yaxis_title="Jumlah Mahasiswa")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div class="insight-box">
    💡 <b>Interpretasi Score:</b> Financial Health Score dihitung dari 3 komponen:
    <b>40%</b> rasio tabungan terhadap pemasukan, <b>30%</b> proporsi pengeluaran esensial
    (tempat tinggal + makanan + transportasi) terhadap total pengeluaran,
    <b>30%</b> buffer likuiditas (sisa / total pengeluaran).<br>
    Score rata-rata saat ini <b>{mean_val:.1f}/100</b> — threshold otomatis dari data:
    Bahaya &lt;{meta['p25']:.1f} · Waspada {meta['p25']:.1f}–{meta['p50']:.1f} ·
    Stabil {meta['p50']:.1f}–{meta['p75']:.1f} · Sangat Sehat ≥{meta['p75']:.1f}.
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════ TAB 2 — EDA ════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-card"><div class="section-title">🔍 Exploratory Data Analysis</div>'
                '<div class="section-sub">Distribusi, korelasi, dan pola pengeluaran mahasiswa</div>',
                unsafe_allow_html=True)

    eda_sub = st.selectbox("Pilih analisis:", [
        "Distribusi Numerik", "Boxplot Pengeluaran per Kategori",
        "Komposisi Pengeluaran", "Matriks Korelasi", "Distribusi Kategorikal"
    ])

    if eda_sub == "Distribusi Numerik":
        numerik_cols = [c for c in ['usia', 'pendapatan', 'bantuan', 'total_pemasukan',
                        'total_pengeluaran', 'sisa_uang', 'rasio_pengeluaran', 'financial_score']
                        if c in fdf.columns]
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=[kolom_label(c) for c in numerik_cols],
            vertical_spacing=0.18, horizontal_spacing=0.08,
        )
        for i, col_n in enumerate(numerik_cols):
            row_n, col_idx = divmod(i, 4)
            data_col = fdf[col_n].dropna()
            is_idr   = col_n in IDR_COLS

            fig.add_trace(go.Histogram(
                x=data_col, nbinsx=25, name=kolom_label(col_n),
                marker_color='#6366f1', opacity=0.75, showlegend=False,
            ), row=row_n+1, col=col_idx+1)

            fig.add_vline(x=float(data_col.mean()),   line_dash='dash',  line_color='#ef4444', row=row_n+1, col=col_idx+1)
            fig.add_vline(x=float(data_col.median()), line_dash='solid', line_color='#10b981', row=row_n+1, col=col_idx+1)

            if is_idr and len(data_col) > 0:
                tvs, tts = jt_ticks(data_col)
                axis_key = f"xaxis{'' if (row_n*4+col_idx)==0 else (row_n*4+col_idx+1)}"
                fig.update_layout(**{axis_key: dict(
                    tickvals=tvs, ticktext=tts, tickangle=30, tickfont=dict(size=9)
                )})

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#334155"),
            height=560, title_text="Distribusi Numerik  🔴 Mean  🟢 Median",
            showlegend=False, margin=dict(t=50, b=40, l=15, r=15),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif eda_sub == "Boxplot Pengeluaran per Kategori":
        pengeluaran_cols = [c for c in SPENDING_COLS if c in fdf.columns]
        fig = go.Figure()
        for i, col in enumerate(pengeluaran_cols):
            data = fdf[col].dropna()
            fig.add_trace(go.Box(
                y=data, name=kolom_label(col).replace(" (Rp)", ""),
                marker_color=px.colors.qualitative.Pastel[i % 10],
                boxmean=True,
            ))
        all_vals = fdf[pengeluaran_cols].values.flatten()
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals) > 0:
            tvs, tts = jt_ticks(pd.Series(all_vals))
        else:
            tvs, tts = [0], ["Rp0"]
        fig.update_layout(**PL, height=450, margin=M,
                          title="Distribusi Pengeluaran per Kategori",
                          xaxis_title="Kategori Pengeluaran",
                          yaxis=dict(title="Jumlah Pengeluaran per Bulan", tickvals=tvs, ticktext=tts),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif eda_sub == "Komposisi Pengeluaran":
        pengeluaran_cols = [c for c in SPENDING_COLS if c in fdf.columns]
        rata = fdf[pengeluaran_cols].mean().sort_values(ascending=False)
        rata_pct = (rata / rata.sum() * 100).round(1)
        labels_short = [kolom_label(c).replace(" (Rp)", "") for c in rata.index]

        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.pie(values=rata.values, names=labels_short,
                         title="Komposisi Rata-rata Pengeluaran (%)", hole=0.38)
            fig.update_traces(textposition='outside', textinfo='percent+label')
            fig.update_layout(**PL, height=400, showlegend=False, margin=dict(t=45,b=45,l=15,r=15))
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            tvs, tts = jt_ticks(rata)
            fig = px.bar(
                x=rata.values[::-1], y=labels_short[::-1], orientation='h',
                color=rata.values[::-1], color_continuous_scale='Blues',
                text=[fmt_rupiah(v) for v in rata.values[::-1]],
                title="Rata-rata Pengeluaran per Kategori per Bulan",
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(**PL, height=400, margin=M,
                              coloraxis_showscale=False, showlegend=False,
                              xaxis=dict(title="Jumlah (Rp/bulan)", tickvals=tvs, ticktext=tts),
                              yaxis_title="Kategori")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""<div class="insight-box">
        💡 Pengeluaran terbesar: <b>{labels_short[0]}</b> ({fmt_rupiah(rata.values[0])}/bln, {rata_pct.values[0]:.1f}%).
        Pengeluaran terkecil: <b>{labels_short[-1]}</b> ({fmt_rupiah(rata.values[-1])}/bln, {rata_pct.values[-1]:.1f}%).
        </div>""", unsafe_allow_html=True)

    elif eda_sub == "Matriks Korelasi":
        num_cols = fdf.drop(columns=['id'], errors='ignore').select_dtypes('number').columns
        corr = fdf[num_cols].corr().round(2)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_masked = corr.where(~mask)
        corr_display = corr_masked.copy()
        corr_display.columns = [LABEL_KOLOM.get(c, c) for c in corr_display.columns]
        corr_display.index   = [LABEL_KOLOM.get(c, c) for c in corr_display.index]
        fig = px.imshow(corr_display, text_auto=".2f", color_continuous_scale='RdYlGn',
                        zmin=-1, zmax=1, title="Matriks Korelasi Antar Variabel", aspect="auto")
        fig.update_layout(**PL, height=580, margin=dict(t=45,b=15,l=15,r=15))
        st.plotly_chart(fig, use_container_width=True)

    else:  # Kategorikal
        cat_cols_show = [c for c in fdf.select_dtypes('object').columns
                         if c not in ['status_finansial', 'nama_cluster']]
        if cat_cols_show:
            selected_cat = st.selectbox("Pilih kolom kategori:", cat_cols_show)
            counts = fdf[selected_cat].value_counts().reset_index()
            counts.columns = [selected_cat, 'Jumlah']
            fig = px.bar(counts, x=selected_cat, y='Jumlah',
                         color='Jumlah', color_continuous_scale='Blues',
                         text='Jumlah', title=f"Distribusi {selected_cat}")
            fig.update_traces(textposition='outside')
            fig.update_layout(**PL, height=380, margin=M,
                              coloraxis_showscale=False,
                              xaxis_title=selected_cat, yaxis_title="Jumlah Mahasiswa")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada kolom kategorikal.")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════ TAB 3 — CLUSTERING ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-card"><div class="section-title">🤖 Proses Clustering K-Means</div>'
                '<div class="section-sub">Seleksi cluster optimal, visualisasi PCA 2D, dan metrik evaluasi</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        k_range  = sorted(meta['sil_scores'].keys())
        sil_vals = [meta['sil_scores'][k] for k in k_range]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_range, y=sil_vals, mode='lines+markers',
            line=dict(color='#10b981', width=2.5),
            marker=dict(size=9, color='white', line=dict(color='#10b981', width=2.5)),
            fill='tozeroy', fillcolor='rgba(16,185,129,0.12)', name='Silhouette Score',
        ))
        fig.add_vline(x=best_k, line_dash='dash', line_color='#ef4444',
                      annotation_text=f'Optimal k={best_k}', annotation_font_color='#ef4444')
        fig.update_layout(**PL, height=300, margin=M,
                          title=f"Silhouette Score per K (Best k={best_k})",
                          xaxis=dict(tickmode='array', tickvals=list(k_range),
                                     gridcolor='#e2e8f0', showgrid=True, title_text='Jumlah Cluster (k)'),
                          yaxis=dict(gridcolor='#e2e8f0', showgrid=True, title_text='Silhouette Score'))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        ev_data = {
            "Metrik":     ["Jumlah Cluster (k)", "Silhouette Score", "Davies-Bouldin Index", "Total Data", "Fitur dipakai"],
            "Nilai":      [str(best_k), f"{meta['sil']:.4f}", f"{meta['db']:.4f}", str(len(df)), str(len(meta['fitur']))],
            "Keterangan": ["Segmen mahasiswa", "Mendekati 1 = lebih baik", "Mendekati 0 = lebih baik",
                           "Observasi valid", "Dimensi input ke model"],
        }
        st.markdown("#### 📊 Evaluasi Model Clustering")
        st.dataframe(pd.DataFrame(ev_data), use_container_width=True, hide_index=True)

    pca_df_plot = fdf.dropna(subset=['PC1', 'PC2', 'nama_cluster'])
    fig = px.scatter(
        pca_df_plot, x='PC1', y='PC2', color='nama_cluster',
        color_discrete_sequence=CLUSTER_PALETTE, opacity=0.65,
        hover_data={'financial_score': ':.1f', 'status_finansial': True, 'PC1': False, 'PC2': False},
        title=f"Visualisasi Cluster dalam Ruang PCA 2D  "
              f"(PC1={meta['var_exp'][0]:.1f}% · PC2={meta['var_exp'][1]:.1f}% variasi)",
        labels={'PC1': f"PC1 — {meta['var_exp'][0]:.1f}% variasi",
                'PC2': f"PC2 — {meta['var_exp'][1]:.1f}% variasi",
                'nama_cluster': 'Segmen Cluster'},
    )
    c_df = meta['centroid_df'].copy()
    c_df['label'] = c_df['cluster'].map(nama_cluster)
    fig.add_trace(go.Scatter(
        x=c_df['PC1'], y=c_df['PC2'], mode='markers+text',
        marker=dict(symbol='x', size=18, color='black', line=dict(width=2.5, color='white')),
        text=c_df['label'],
        textposition='top center', textfont=dict(size=10, color='#0f172a'),
        name='Pusat Cluster', hovertext=c_df['label'], hoverinfo='text',
    ))
    fig.update_layout(**PL, height=460, margin=dict(t=45,b=90,l=15,r=15),
                      legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div class="insight-box">
    💡 <b>Cara membaca PCA:</b> Setiap titik adalah 1 mahasiswa. Jarak antar titik = kemiripan profil keuangan.
    PC1 + PC2 menjelaskan <b>{meta['var_exp'][0]+meta['var_exp'][1]:.1f}%</b> total variasi data.
    Tanda ✕ = pusat (centroid) tiap cluster.
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════ TAB 4 — PROFIL CLUSTER ════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-card"><div class="section-title">🏷️ Profil & Karakteristik Cluster</div>'
                '<div class="section-sub">Ringkasan mendalam setiap segmen mahasiswa berdasarkan hasil clustering</div>',
                unsafe_allow_html=True)

    cols_cards = st.columns(len(cluster_list))
    for col, cl in zip(cols_cards, cluster_list):
        subset = fdf[fdf['cluster'] == cl]
        avg_sc = subset['financial_score'].mean() if len(subset) > 0 else 0
        status = make_status(avg_sc, meta['p25'], meta['p50'], meta['p75'])
        color  = CLUSTER_PALETTE[int(cl) % len(CLUSTER_PALETTE)]
for col, cl in zip(cols_cards, cluster_list):
    subset = fdf[fdf['cluster'] == cl]
    avg_sc = subset['financial_score'].mean() if len(subset) > 0 else 0
    status = make_status(avg_sc, meta['p25'], meta['p50'], meta['p75'])
    color  = CLUSTER_NAME_COLORS.get(nama_cluster[int(cl)], "#3b82f6")  # ← ini yang diubah
    with col:
        st.markdown(f"""
        <div style="background:white;border-top:4px solid {color};border-radius:12px;
                    padding:16px;text-align:center;box-shadow:0 1px 6px rgba(0,0,0,.08);">
            <div style="font-size:11px;color:#94a3b8;font-weight:700;letter-spacing:.08em;
                        text-transform:uppercase;">Cluster {int(cl)}</div>
            <div style="font-size:13px;font-weight:800;color:{color};margin:6px 0;line-height:1.4;">
                {nama_cluster[int(cl)]}</div>
            <div style="font-size:12px;color:#64748b;">{len(subset):,} mahasiswa</div>
            <div style="margin-top:8px;">{get_status_badge(status)}</div>
            <div style="font-size:12px;color:#94a3b8;margin-top:5px;">
                Score: {avg_sc:.1f} · Pemasukan: {fmt_rupiah(subset['total_pemasukan'].mean())}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Heatmap profil
    kolom_hm = meta['kolom_profil']
    profil_hm   = profil[kolom_hm]
    profil_norm = ((profil_hm - profil_hm.mean()) / profil_hm.std()).round(2)
    x_labels    = [f"Cluster {int(c)}: {nama_cluster.get(int(c), '')}" for c in profil_hm.index]
    annot_raw   = profil_hm.values.astype(float)

    def _fmt_cell(v, col):
        if col == 'rasio_pengeluaran': return f"{v:.2f}x"
        if col == 'financial_score':   return f"{v:.1f}"
        return fmt_rupiah(v)

    fig = go.Figure(go.Heatmap(
        z=profil_norm.values.T, x=x_labels,
        y=[kolom_label(c) for c in kolom_hm],
        text=[[_fmt_cell(v, col) for v in row] for col, row in zip(kolom_hm, annot_raw.T)],
        texttemplate="%{text}", colorscale='RdYlGn', zmin=-2, zmax=2,
        colorbar=dict(title="Z-score<br>(merah=rendah,<br>hijau=tinggi)"),
        hoverongaps=False,
    ))
    fig.update_layout(**PL, height=440, margin=ML,
                      title="Heatmap Profil Rata-rata per Cluster (Z-score, nilai asli di tiap sel)",
                      xaxis=dict(title="Segmen Cluster", tickangle=0, tickfont=dict(size=11)),
                      yaxis=dict(title="Variabel Keuangan", tickfont=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True, key="t4_heatmap")

    # Perbandingan KPI
    st.markdown("#### 📊 Perbandingan KPI Antar Cluster")
    kpi_cols = [c for c in ['total_pemasukan', 'total_pengeluaran', 'sisa_uang',
                             'rasio_pengeluaran', 'hiburan', 'teknologi', 'financial_score']
                if c in fdf.columns]
    selected_kpi = st.selectbox("Pilih KPI:", kpi_cols, format_func=kolom_label, key="kpi_select")

    kpi_vals = fdf.groupby('nama_cluster')[selected_kpi].mean().reset_index()
    kpi_vals.columns = ['Cluster', 'Nilai']
    is_ratio = selected_kpi == 'rasio_pengeluaran'
    is_score = selected_kpi == 'financial_score'
    is_idr   = not is_ratio and not is_score

    kpi_vals['Label'] = kpi_vals['Nilai'].apply(
        lambda v: f"{v:.2f}x" if is_ratio else (f"{v:.1f}/100" if is_score else fmt_rupiah(v))
    )
    fig = px.bar(kpi_vals, x='Cluster', y='Nilai', color='Cluster',
                 color_discrete_sequence=CLUSTER_PALETTE, text='Label',
                 title=f"Rata-rata {kolom_label(selected_kpi)} per Cluster",
                 labels={'Nilai': kolom_label(selected_kpi), 'Cluster': 'Segmen Cluster'})
    fig.update_traces(textposition='outside')
    if is_idr and len(kpi_vals) > 0:
        tvs, tts = jt_ticks(kpi_vals['Nilai'])
        fig.update_layout(yaxis=dict(tickvals=tvs, ticktext=tts, title=kolom_label(selected_kpi)))
    fig.update_layout(**PL, height=380, showlegend=False, margin=M,
                      xaxis=dict(tickangle=0, gridcolor='#e2e8f0', title='Segmen Cluster'))
    st.plotly_chart(fig, use_container_width=True)

    # Distribusi status per cluster
    st.markdown("#### 📊 Distribusi Status Finansial per Cluster")
    status_data = fdf.groupby(['nama_cluster', 'status_finansial']).size().reset_index(name='Jumlah')
    total_per_cl = status_data.groupby('nama_cluster')['Jumlah'].transform('sum')
    status_data['Persen'] = (status_data['Jumlah'] / total_per_cl * 100).round(1)
    fig = px.bar(status_data, x='nama_cluster', y='Persen', color='status_finansial',
                 color_discrete_map=STATUS_COLORS,
                 text=status_data['Persen'].apply(lambda x: f"{x:.0f}%"),
                 barmode='stack', title="Proporsi Status Finansial per Cluster",
                 labels={'nama_cluster': 'Segmen Cluster', 'Persen': 'Persentase (%)', 'status_finansial': 'Status'},
                 category_orders={'status_finansial': STATUS_ORDER})
    fig.update_traces(textposition='inside', insidetextanchor='middle')
    fig.update_layout(**PL, height=400, margin=dict(t=45,b=90,l=15,r=15),
                      xaxis=dict(tickangle=0, gridcolor='#e2e8f0', title='Segmen Cluster'),
                      yaxis_title="Persentase Mahasiswa (%)",
                      legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5))
    st.plotly_chart(fig, use_container_width=True)

    # Violin plot
    st.markdown("#### 🎻 Sebaran Financial Score per Cluster")
    fig = px.violin(fdf.dropna(subset=['cluster', 'financial_score']),
                    x='nama_cluster', y='financial_score', color='nama_cluster',
                    color_discrete_sequence=CLUSTER_PALETTE, box=True, points='outliers',
                    labels={'nama_cluster': 'Segmen Cluster', 'financial_score': 'Financial Health Score (0–100)'},
                    title="Distribusi Financial Health Score per Cluster")
    for th_val, th_col, th_ann in [
        (meta['p25'], '#ef4444', f'Batas Bahaya/Waspada: {meta["p25"]:.1f}'),
        (meta['p50'], '#f59e0b', f'Batas Waspada/Stabil: {meta["p50"]:.1f}'),
        (meta['p75'], '#22c55e', f'Batas Stabil/Sehat: {meta["p75"]:.1f}'),
    ]:
        fig.add_hline(y=th_val, line_dash='dash', line_color=th_col,
                      annotation_text=th_ann, annotation_position='right',
                      annotation_font_color=th_col)
    fig.update_layout(**PL, height=420, showlegend=False, margin=M,
                      xaxis=dict(tickangle=0, gridcolor='#e2e8f0', title='Segmen Cluster'),
                      yaxis_title="Financial Health Score (0–100)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════ TAB 5 — EXPORT ════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-card"><div class="section-title">📥 Export & Ringkasan Akhir</div>'
                '<div class="section-sub">Download hasil analisis lengkap dalam format CSV</div>',
                unsafe_allow_html=True)

    summary_rows = []
    for cl in cluster_list:
        subset = fdf[fdf['cluster'] == cl]
        avg_sc = subset['financial_score'].mean() if len(subset) > 0 else 0
        dist_pct = (subset['status_finansial'].value_counts(normalize=True)
                    .reindex(STATUS_ORDER, fill_value=0) * 100)
        summary_rows.append({
            "Cluster":           nama_cluster.get(int(cl), f"Cluster {int(cl)}"),
            "Jumlah":            len(subset),
            "Avg Pemasukan":     fmt_rupiah(subset['total_pemasukan'].mean()),
            "Avg Pengeluaran":   fmt_rupiah(subset['total_pengeluaran'].mean()),
            "Avg Sisa":          fmt_rupiah(subset['sisa_uang'].mean()),
            "Rasio Pengeluaran": f"{subset['rasio_pengeluaran'].mean():.2f}x",
            "Fin. Score":        f"{avg_sc:.1f}/100",
            "Status Dominan":    make_status(avg_sc, meta['p25'], meta['p50'], meta['p75']),
            "% Bahaya":          f"{dist_pct['Bahaya']:.1f}%",
            "% Waspada":         f"{dist_pct['Waspada']:.1f}%",
            "% Stabil":          f"{dist_pct['Stabil']:.1f}%",
            "% Sangat Sehat":    f"{dist_pct['Sangat Sehat']:.1f}%",
        })
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Dataset Lengkap (CSV)",
                           data=fdf.drop(columns=['PC1', 'PC2'], errors='ignore').to_csv(index=False).encode('utf-8'),
                           file_name="hasil_clustering_finansial_IDR.csv", mime="text/csv",
                           use_container_width=True)
    with col2:
        st.download_button("📥 Download Ringkasan Cluster (CSV)",
                           data=df_summary.to_csv(index=False).encode('utf-8'),
                           file_name="ringkasan_cluster_finansial.csv", mime="text/csv",
                           use_container_width=True)

    st.markdown(f"""<div class="insight-box">
    ✅ <b>Ringkasan Analisis:</b><br>
    • Dataset: <b>{len(df):,} mahasiswa</b> · Kurs: 1 USD = Rp {KURS_USD_IDR:,} + rasio biaya hidup Indonesia<br>
    • Algoritma: K-Means (k optimal={best_k}) · Silhouette={meta['sil']:.4f} · Davies-Bouldin={meta['db']:.4f}<br>
    • Fitur clustering: {len(meta['fitur'])} variabel · Data terfilter: {len(fdf):,} mahasiswa<br>
    • Kolom tambahan: <code>cluster</code>, <code>nama_cluster</code>,
      <code>financial_score</code>, <code>status_finansial</code>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    f'<div style="text-align:center;color:#94a3b8;font-size:.78rem;margin-top:1.5rem;">'
    f'Financial Clustering Dashboard · K-Means (k={best_k}) · '
    f'Kurs 1 USD = Rp {KURS_USD_IDR:,} · Data: Kaggle Student Spending</div>',
    unsafe_allow_html=True)
