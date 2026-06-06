import os
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
# CUSTOM CSS  (light bg seperti versi 400 baris)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

body { background: #f1f5f9 !important; }
[data-testid="stAppViewContainer"] { background: #f1f5f9; }

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    text-align: center;
}
.metric-card .label {
    font-size: 0.72rem;
    letter-spacing: .08em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 4px;
}

.section-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
}
.section-sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 16px;
}

.insight-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #1e40af;
    margin-top: 10px;
}

.badge-bahaya  { background:#ef4444; color:#fff; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:700; }
.badge-waspada { background:#f59e0b; color:#fff; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:700; }
.badge-stabil  { background:#3b82f6; color:#fff; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:700; }
.badge-sehat   { background:#22c55e; color:#fff; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:700; }

.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 14px; }
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
STATUS_COLORS = {
    "Bahaya":      "#ef4444",
    "Waspada":     "#f59e0b",
    "Stabil":      "#3b82f6",
    "Sangat Sehat":"#22c55e",
}
CLUSTER_PALETTE = [
    "#3b82f6","#f59e0b","#22c55e","#ef4444","#8b5cf6","#06b6d4"
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def make_status(score: float, p25: float, p50: float, p75: float) -> str:
    """Fungsi global – tidak disimpan di cache, selalu serializable."""
    if score >= p75: return "Sangat Sehat"
    if score >= p50: return "Stabil"
    if score >= p25: return "Waspada"
    return "Bahaya"

def fmt_rupiah(val, short=True):
    """Format nilai IDR. Nilai mahasiswa tipikalnya ratusan ribu – belasan juta."""
    if pd.isna(val):
        return "N/A"
    if short:
        if abs(val) >= 1_000_000_000:          # >= 1 Miliar
            return f"Rp {val/1_000_000_000:.1f}M"
        elif abs(val) >= 1_000_000:            # >= 1 Juta
            return f"Rp {val/1_000_000:.2f}Jt"
        elif abs(val) >= 1_000:               # >= 1 Ribu
            return f"Rp {val/1_000:.1f}rb"
        return f"Rp {val:.0f}"
    return f"Rp {val:,.0f}"

def get_status_badge(status):
    cls_map = {
        "Bahaya": "bahaya", "Waspada": "waspada",
        "Stabil": "stabil", "Sangat Sehat": "sehat"
    }
    return f'<span class="badge-{cls_map.get(status, "stabil")}">{status}</span>'

# Plotly layout defaults (light theme, responsive)
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#334155"),
    margin=dict(t=30, b=30, l=10, r=10),
)
# Helper: default axis style — merge manual di setiap update_layout
_AX = dict(gridcolor="#e2e8f0", showgrid=True, zeroline=False)
_AX_NOGRID = dict(showgrid=False, zeroline=False)

# ─────────────────────────────────────────────
# DATA LOADING — hanya cache DataFrame + angka,
# TIDAK cache model sklearn (biang kelambatan)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process_data(file_key: str):
    """
    FIX utama: kembalikan hanya plain Python types + DataFrame.
    Hindari menyimpan objek sklearn ke cache.
    """
    # ── baca ──
    if file_key == "__default__":
        df_raw = pd.read_csv("Data/student_spending (1).csv")
    else:
        df_raw = pd.read_csv(file_key)

    # ── rename ──
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

    # ── konversi USD → IDR ──
    for col in ['pendapatan', 'bantuan']:
        if col in df.columns:
            df[col] = (df[col] * KURS_USD_IDR).round(0)

    for col, r in RASIO.items():
        if col in df.columns:
            if col == 'pendidikan':
                df[col] = (df[col] / 6 * KURS_USD_IDR * r).round(0)
            else:
                df[col] = (df[col] * KURS_USD_IDR * r).round(0)

    # ── derived ──
    df['total_pemasukan']   = df['pendapatan'] + df['bantuan']
    df['total_pengeluaran'] = df[list(RASIO.keys())].sum(axis=1)
    df['sisa_uang']         = df['total_pemasukan'] - df['total_pengeluaran']
    df['rasio_pengeluaran'] = (
        df['total_pengeluaran'] / df['total_pemasukan'].replace(0, np.nan)
    )

    # ── financial score ──
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

    p25 = float(df['financial_score'].quantile(0.25))
    p50 = float(df['financial_score'].quantile(0.50))
    p75 = float(df['financial_score'].quantile(0.75))

    # Gunakan fungsi global (tidak disimpan di meta agar cache bisa serialize)
    df['status_finansial'] = df['financial_score'].apply(
        lambda s: (
            "Sangat Sehat" if s >= p75 else
            "Stabil"       if s >= p50 else
            "Waspada"      if s >= p25 else
            "Bahaya"
        )
    )

    # ── clustering ──
    cat_cols = [c for c in df.select_dtypes('object').columns
                if c not in ['status_finansial', 'nama_cluster']]
    df_enc = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col+'_enc'] = le.fit_transform(df_enc[col].astype(str))

    fitur = (
        ['total_pemasukan','total_pengeluaran','sisa_uang','rasio_pengeluaran',
         'pendidikan','tempat_tinggal','makanan','transportasi','hiburan',
         'teknologi','financial_score']
        + [c+'_enc' for c in cat_cols if c+'_enc' in df_enc.columns]
    )
    fitur = [c for c in fitur if c in df_enc.columns]

    X = df_enc[fitur].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pilih k optimal (silhouette)
    sil_scores = {}
    for k in range(2, 7):
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

    # PCA 2D — simpan sebagai list biasa (bukan numpy array)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = [float(v*100) for v in pca.explained_variance_ratio_]

    # Nama cluster
    cluster_list = sorted(df['cluster'].dropna().unique().tolist())
    kol = ['total_pemasukan','total_pengeluaran','sisa_uang',
           'rasio_pengeluaran','financial_score']
    kol = [c for c in kol if c in df.columns]
    profil = df.groupby('cluster')[kol].mean().round(0)

    sisa_rank  = profil['sisa_uang'].rank(ascending=False)
    rasio_rank = profil['rasio_pengeluaran'].rank(ascending=True)
    pend_rank  = profil['total_pemasukan'].rank(ascending=False)
    n_cl = len(cluster_list)

    nama_cluster = {}
    for cl in cluster_list:
        rs = sisa_rank[cl]; rr = rasio_rank[cl]; rp = pend_rank[cl]
        if rs <= 1 and rp <= 1:
            label = "Mapan & Hemat"
        elif rs <= 1:
            label = "Hemat"
        elif rp <= 1 and rr >= n_cl:
            label = "Penghasilan Tinggi & Boros"
        elif rr >= n_cl:
            label = "Boros / Defisit"
        elif rp >= n_cl:
            label = "Penghasilan Rendah"
        else:
            label = "Rata-rata"
        nama_cluster[int(cl)] = f"Cluster {int(cl)}: {label}"

    df['nama_cluster'] = df['cluster'].map(nama_cluster)

    # Tambahkan kolom PCA ke df subset untuk scatter
    pca_df = pd.DataFrame(
        X_pca, index=X.index, columns=['PC1', 'PC2']
    )
    df = df.join(pca_df, how='left')

    # Centroid PCA
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    centroid_df = pd.DataFrame(centroids_pca, columns=['PC1','PC2'])
    centroid_df['cluster'] = list(range(best_k))

    # Reset profil index ke int biasa agar serializable
    profil.index = profil.index.astype(int)

    meta = dict(
        best_k=int(best_k),
        sil=float(sil_val),
        db=float(db_val),
        profil=profil,
        cluster_list=[int(c) for c in cluster_list],
        nama_cluster={int(k): str(v) for k, v in nama_cluster.items()},
        var_exp=[float(v) for v in var_exp],
        p25=float(p25), p50=float(p50), p75=float(p75),
        fitur=list(fitur),
        sil_scores={int(k): float(v) for k, v in sil_scores.items()},
        centroid_df=centroid_df,
        # get_status TIDAK disimpan — pakai fungsi global make_status()
    )
    return df, meta

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Financial Dashboard")
    st.markdown("**Clustering Finansial Mahasiswa**")
    st.divider()

    st.markdown("### 📂 Sumber Data")
    DEFAULT_DATA_PATH = "Data/student_spending (1).csv"
    uploaded_file = st.file_uploader("Upload CSV lain *(opsional)*", type=["csv"])
    st.divider()
    st.caption("Dashboard · Data: Kaggle Student Spending")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
st.markdown(
    '<h1 style="font-size:1.8rem;font-weight:800;color:#0f172a;margin-bottom:4px;">'
    '💰 Dashboard Finansial Mahasiswa Indonesia</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="color:#64748b;font-size:0.95rem;margin-bottom:20px;">'
    'Analisis K-Means Clustering · Konversi USD → IDR · Financial Health Score</p>',
    unsafe_allow_html=True
)

if uploaded_file is not None:
    uploaded_file.seek(0)
    tmp_path = f"/tmp/{uploaded_file.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())
    file_key = tmp_path
elif os.path.exists(DEFAULT_DATA_PATH):
    file_key = "__default__"
else:
    st.error(
        f"⚠️ File **`{DEFAULT_DATA_PATH}`** tidak ditemukan. "
        "Upload manual via sidebar."
    )
    st.stop()

with st.spinner("⚙️ Memuat & memproses data..."):
    df, meta = load_and_process_data(file_key)

profil       = meta['profil']
cluster_list = meta['cluster_list']
nama_cluster = meta['nama_cluster']
# get_status diambil dari fungsi global make_status()
best_k       = meta['best_k']
pengeluaran_cols = [c for c in list(RASIO.keys()) if c in df.columns]

# ── Sidebar filters (setelah data loaded) ──
with st.sidebar:
    st.markdown("### 🔍 Filter Data")
    cluster_filter = st.multiselect(
        "Cluster", options=df['nama_cluster'].dropna().unique().tolist(),
        default=df['nama_cluster'].dropna().unique().tolist()
    )
    gender_opts = df['gender'].dropna().unique().tolist() if 'gender' in df.columns else []
    gender_filter = st.multiselect("Gender", options=gender_opts, default=gender_opts)

    year_opts = df['year_in_school'].dropna().unique().tolist() if 'year_in_school' in df.columns else []
    year_filter = st.multiselect("Tahun Kuliah", options=year_opts, default=year_opts)

fdf = df[df['nama_cluster'].isin(cluster_filter)]
if gender_opts:
    fdf = fdf[fdf['gender'].isin(gender_filter)]
if year_opts:
    fdf = fdf[fdf['year_in_school'].isin(year_filter)]

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
kpi_data = [
    ("Total Mahasiswa", f"{len(fdf):,}", "responden"),
    ("Cluster Optimal", str(best_k), "K-Means"),
    ("Silhouette Score", f"{meta['sil']:.3f}", "kualitas cluster"),
    ("Avg Pemasukan", fmt_rupiah(fdf['total_pemasukan'].mean()), "per bulan"),
    ("Avg Sisa Uang", fmt_rupiah(fdf['sisa_uang'].mean()), "per bulan"),
    ("Avg Fin. Score", f"{fdf['financial_score'].mean():.1f}", "dari 100"),
]
cols = st.columns(len(kpi_data))
for col, (label, value, sub) in zip(cols, kpi_data):
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
    "📊 Overview", "🔍 EDA", "🤖 Clustering",
    "🏷️ Profil Cluster", "📥 Export"
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-card"><div class="section-title">📊 Ringkasan Kesehatan Finansial</div>'
                '<div class="section-sub">Gambaran besar kondisi keuangan seluruh mahasiswa</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        dist_status = fdf['status_finansial'].value_counts().reindex(STATUS_ORDER, fill_value=0)
        fig = px.pie(
            values=dist_status.values,
            names=dist_status.index,
            color=dist_status.index,
            color_discrete_map=STATUS_COLORS,
            hole=0.45,
            title="🩺 Distribusi Status Finansial",
        )
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        dist_pct = (dist_status / len(fdf) * 100).round(1)
        fig = px.bar(
            x=dist_status.values,
            y=dist_status.index,
            orientation='h',
            color=dist_status.index,
            color_discrete_map=STATUS_COLORS,
            text=[f"{v} ({dist_pct[s]:.1f}%)" for s, v in zip(dist_status.index, dist_status.values)],
            title="📈 Jumlah per Status",
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Histogram financial score
    fig = px.histogram(
        fdf, x='financial_score', nbins=30,
        color_discrete_sequence=['#6366f1'],
        title="🎯 Distribusi Financial Health Score",
        labels={'financial_score': 'Financial Health Score (0–100)', 'count': 'Frekuensi'},
        marginal='rug',
    )
    mean_val = fdf['financial_score'].mean()
    med_val  = fdf['financial_score'].median()
    for xval, col_line, lbl in [
        (mean_val, '#f59e0b', f'Mean: {mean_val:.1f}'),
        (med_val,  '#10b981', f'Median: {med_val:.1f}'),
        (meta['p25'], '#ef4444', f'P25: {meta["p25"]:.1f}'),
        (meta['p75'], '#22c55e', f'P75: {meta["p75"]:.1f}'),
    ]:
        fig.add_vline(x=xval, line_dash='dash', line_color=col_line,
                      annotation_text=lbl, annotation_position='top right',
                      annotation_font_color=col_line)
    fig.update_layout(**PLOTLY_LAYOUT, height=340)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
    💡 <b>Interpretasi Score:</b> Financial Health Score dihitung dari 3 komponen:
    40% rasio tabungan, 30% proporsi pengeluaran esensial, 30% buffer likuiditas.
    Score rata-rata <b>{fdf['financial_score'].mean():.1f}/100</b>.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-card"><div class="section-title">🔍 Exploratory Data Analysis</div>'
                '<div class="section-sub">Distribusi, korelasi, dan pola pengeluaran</div>', unsafe_allow_html=True)

    eda_sub = st.selectbox("Pilih analisis:", [
        "Distribusi Numerik", "Boxplot Pengeluaran",
        "Komposisi Pengeluaran", "Matriks Korelasi",
        "Distribusi Kategorikal"
    ])

    if eda_sub == "Distribusi Numerik":
        numerik_cols = ['usia','pendapatan','bantuan','total_pemasukan',
                        'total_pengeluaran','sisa_uang','rasio_pengeluaran','financial_score']
        numerik_cols = [c for c in numerik_cols if c in fdf.columns]

        # Plotly: 2 baris × 4 kolom, responsif otomatis
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=numerik_cols,
            vertical_spacing=0.15,
            horizontal_spacing=0.08,
        )
        for i, col in enumerate(numerik_cols):
            r, c = divmod(i, 4)
            data_col = fdf[col].dropna()
            fig.add_trace(
                go.Histogram(x=data_col, nbinsx=25, name=col,
                             marker_color='#6366f1', opacity=0.75,
                             showlegend=False),
                row=r+1, col=c+1,
            )
            fig.add_vline(
                x=float(data_col.mean()),
                line_dash='dash', line_color='#ef4444',
                row=r+1, col=c+1,
            )
            fig.add_vline(
                x=float(data_col.median()),
                line_dash='solid', line_color='#10b981',
                row=r+1, col=c+1,
            )
        fig.update_layout(height=520, title_text="Distribusi Numerik  🔴 Mean  🟢 Median",
                          **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ['xaxis','yaxis']},
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif eda_sub == "Boxplot Pengeluaran":
        fig = go.Figure()
        for i, col in enumerate(pengeluaran_cols):
            fig.add_trace(go.Box(
                y=fdf[col].dropna(), name=col,
                marker_color=px.colors.qualitative.Pastel[i % 10],
                boxmean=True,
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=420,
            title="Boxplot Pengeluaran per Kategori (IDR)",
            yaxis_title="IDR",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif eda_sub == "Komposisi Pengeluaran":
        rata = fdf[pengeluaran_cols].mean().sort_values(ascending=False)
        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.pie(values=rata.values, names=rata.index,
                         title="Komposisi Pengeluaran (%)", hole=0.35)
            fig.update_traces(textposition='outside', textinfo='percent+label')
            fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            fig = px.bar(
                x=rata.values[::-1], y=rata.index[::-1],
                orientation='h',
                color=rata.values[::-1],
                color_continuous_scale='Blues',
                text=[fmt_rupiah(v) for v in rata.values[::-1]],
                title="Rata-rata Pengeluaran per Kategori",
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(**PLOTLY_LAYOUT, height=380,
                              coloraxis_showscale=False, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    elif eda_sub == "Matriks Korelasi":
        num_cols = fdf.drop(columns=['id'], errors='ignore').select_dtypes('number').columns
        corr = fdf[num_cols].corr().round(2)
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_masked = corr.where(~mask)
        fig = px.imshow(
            corr_masked,
            text_auto=".2f",
            color_continuous_scale='RdYlGn',
            zmin=-1, zmax=1,
            title="Matriks Korelasi Antar Variabel",
            aspect="auto",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=560)
        st.plotly_chart(fig, use_container_width=True)

    else:  # Kategorikal
        cat_cols_show = [c for c in fdf.select_dtypes('object').columns
                         if c not in ['status_finansial', 'nama_cluster']]
        if cat_cols_show:
            selected_cat = st.selectbox("Pilih kolom:", cat_cols_show)
            counts = fdf[selected_cat].value_counts().reset_index()
            counts.columns = [selected_cat, 'Jumlah']
            fig = px.bar(counts, x=selected_cat, y='Jumlah',
                         color='Jumlah', color_continuous_scale='Blues',
                         text='Jumlah',
                         title=f"Distribusi {selected_cat}")
            fig.update_traces(textposition='outside')
            fig.update_layout(**PLOTLY_LAYOUT, height=360,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada kolom kategorikal terdeteksi.")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — CLUSTERING
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-card"><div class="section-title">🤖 Proses Clustering K-Means</div>'
                '<div class="section-sub">Seleksi cluster optimal, visualisasi PCA, dan metrik evaluasi</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        k_range = sorted(meta['sil_scores'].keys())
        sil_vals = [meta['sil_scores'][k] for k in k_range]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_range, y=sil_vals,
            mode='lines+markers',
            line=dict(color='#10b981', width=2.5),
            marker=dict(size=9, color='white', line=dict(color='#10b981', width=2.5)),
            fill='tozeroy', fillcolor='rgba(16,185,129,0.12)',
            name='Silhouette Score',
        ))
        fig.add_vline(x=best_k, line_dash='dash', line_color='#ef4444',
                      annotation_text=f'Optimal k={best_k}',
                      annotation_font_color='#ef4444')
        fig.update_layout(
            **PLOTLY_LAYOUT, height=300,
            title=f"📐 Silhouette Score per K (Best k={best_k})",
            xaxis=dict(tickmode='array', tickvals=list(k_range),
                       gridcolor='#e2e8f0', showgrid=True, title_text='Jumlah Cluster (k)'),
            yaxis=dict(gridcolor='#e2e8f0', showgrid=True, title_text='Silhouette Score'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        ev_data = {
            "Metrik": ["Jumlah Cluster (k)", "Silhouette Score",
                       "Davies-Bouldin Index", "Total Data", "Fitur"],
            "Nilai":  [str(best_k), f"{meta['sil']:.4f}",
                       f"{meta['db']:.4f}", str(len(df)), str(len(meta['fitur']))],
            "Keterangan": ["Segmen mahasiswa", "→1 makin baik",
                           "→0 makin baik", "Data valid (non-null)", "Dimensi input"],
        }
        st.markdown("#### 📊 Evaluasi Model")
        st.dataframe(pd.DataFrame(ev_data), use_container_width=True, hide_index=True)

    # PCA scatter — pakai kolom yang sudah ada di df
    pca_df_plot = fdf.dropna(subset=['PC1','PC2','nama_cluster'])
    fig = px.scatter(
        pca_df_plot,
        x='PC1', y='PC2',
        color='nama_cluster',
        color_discrete_sequence=CLUSTER_PALETTE,
        opacity=0.65,
        hover_data={
            'financial_score': ':.1f',
            'status_finansial': True,
            'PC1': False, 'PC2': False,
        },
        title=f"🔵 PCA 2D — PC1: {meta['var_exp'][0]:.1f}%  ·  PC2: {meta['var_exp'][1]:.1f}% variasi",
        labels={'PC1': f"PC1 ({meta['var_exp'][0]:.1f}%)",
                'PC2': f"PC2 ({meta['var_exp'][1]:.1f}%)",
                'nama_cluster': 'Cluster'},
    )
    # Tambah centroid
    c_df = meta['centroid_df']
    c_df_labeled = c_df.copy()
    c_df_labeled['label'] = c_df_labeled['cluster'].map(nama_cluster)
    fig.add_trace(go.Scatter(
        x=c_df['PC1'], y=c_df['PC2'],
        mode='markers',
        marker=dict(symbol='x', size=16, color='black',
                    line=dict(width=2, color='white')),
        name='Pusat Cluster',
        hovertext=c_df_labeled['label'],
        hoverinfo='text',
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=440,
                      legend=dict(orientation='h', yanchor='bottom',
                                  y=-0.25, xanchor='center', x=0.5))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
    💡 <b>Tentang PCA:</b> PC1 + PC2 menjelaskan
    <b>{meta['var_exp'][0]+meta['var_exp'][1]:.1f}%</b> total variasi.
    Titik = 1 mahasiswa. Hover untuk detail. ✕ = pusat cluster.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — PROFIL CLUSTER
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-card"><div class="section-title">🏷️ Profil & Karakteristik Cluster</div>'
                '<div class="section-sub">Ringkasan mendalam setiap segmen mahasiswa</div>',
                unsafe_allow_html=True)

    # Summary cards
    cols_cards = st.columns(len(cluster_list))
    for col, cl in zip(cols_cards, cluster_list):
        subset  = fdf[fdf['cluster'] == cl]
        avg_sc  = subset['financial_score'].mean() if len(subset) > 0 else 0
        status  = make_status(avg_sc, meta['p25'], meta['p50'], meta['p75'])
        color   = CLUSTER_PALETTE[int(cl) % len(CLUSTER_PALETTE)]
        with col:
            st.markdown(f"""
            <div style="background:white;border-top:4px solid {color};
                        border-radius:10px;padding:16px;text-align:center;
                        box-shadow:0 1px 4px rgba(0,0,0,.08);">
                <div style="font-size:11px;color:#94a3b8;font-weight:700;
                            letter-spacing:.08em;text-transform:uppercase;">
                    Cluster {int(cl)}</div>
                <div style="font-size:14px;font-weight:800;color:{color};
                            margin:6px 0;line-height:1.3;">
                    {nama_cluster[int(cl)].replace(f"Cluster {int(cl)}: ","")}</div>
                <div style="font-size:12px;color:#94a3b8;">{len(subset)} mahasiswa</div>
                <div style="margin-top:8px;">{get_status_badge(status)}</div>
                <div style="font-size:12px;color:#64748b;margin-top:4px;">Score: {avg_sc:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Heatmap profil (Plotly — interaktif, teks tidak terpotong)
    kolom_hm = [c for c in ['total_pemasukan','total_pengeluaran','sisa_uang',
                             'rasio_pengeluaran','pendidikan','tempat_tinggal',
                             'makanan','hiburan','teknologi','financial_score']
                if c in profil.columns]
    profil_hm   = profil[kolom_hm]
    profil_norm = ((profil_hm - profil_hm.mean()) / profil_hm.std()).round(2)

    x_labels = [nama_cluster.get(int(c), f"Cluster {int(c)}") for c in profil_hm.index]
    annot = profil_hm.values.astype(float).round(0)

    fig = go.Figure(go.Heatmap(
        z=profil_norm.values.T,
        x=x_labels,
        y=kolom_hm,
        text=[[
               f"{v:.3f}" if col == 'rasio_pengeluaran' else
               f"{v:.1f}" if col == 'financial_score'   else
               fmt_rupiah(v)
               for v in row
           ]
           for col, row in zip(kolom_hm, annot.T)],
        texttemplate="%{text}",
        colorscale='RdYlGn',
        zmin=-2, zmax=2,
        colorbar=dict(title="Z-score"),
        hoverongaps=False,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=400,
        title="🗺️ Heatmap Profil Rata-rata per Cluster",
        xaxis=dict(tickangle=0, tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Perbandingan KPI — bar chart interaktif
    st.markdown("#### 📊 Perbandingan KPI Antar Cluster")
    kpi_cols = ['total_pemasukan','total_pengeluaran','sisa_uang',
                'rasio_pengeluaran','hiburan','teknologi','financial_score']
    kpi_cols = [c for c in kpi_cols if c in fdf.columns]
    selected_kpi = st.selectbox("Pilih KPI:", kpi_cols, key="kpi_select")

    kpi_vals = fdf.groupby('nama_cluster')[selected_kpi].mean().reset_index()
    kpi_vals.columns = ['Cluster', 'Nilai']
    fig = px.bar(
        kpi_vals, x='Cluster', y='Nilai',
        color='Cluster',
        color_discrete_sequence=CLUSTER_PALETTE,
        text=kpi_vals['Nilai'].apply(
            lambda v: f"{v:.2f}" if selected_kpi in ['rasio_pengeluaran','financial_score']
            else fmt_rupiah(v)
        ),
        title=f"Rata-rata {selected_kpi} per Cluster",
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=False,
                      xaxis=dict(tickangle=-10, gridcolor='#e2e8f0'))
    st.plotly_chart(fig, use_container_width=True)

    # Status distribusi per cluster
    st.markdown("#### 📊 Distribusi Status Finansial per Cluster")
    status_data = (
        fdf.groupby(['nama_cluster','status_finansial'])
        .size().reset_index(name='Jumlah')
    )
    total_per_cl = status_data.groupby('nama_cluster')['Jumlah'].transform('sum')
    status_data['Persen'] = (status_data['Jumlah'] / total_per_cl * 100).round(1)

    fig = px.bar(
        status_data, x='nama_cluster', y='Persen',
        color='status_finansial',
        color_discrete_map=STATUS_COLORS,
        text=status_data['Persen'].apply(lambda x: f"{x:.0f}%"),
        barmode='stack',
        labels={'nama_cluster': 'Cluster', 'Persen': 'Persentase (%)'},
        category_orders={'status_finansial': STATUS_ORDER},
        title="Proporsi Status Finansial per Cluster",
    )
    fig.update_traces(textposition='inside', insidetextanchor='middle')
    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                      xaxis=dict(tickangle=-10, gridcolor='#e2e8f0'),
                      legend=dict(orientation='h', yanchor='bottom',
                                  y=-0.3, xanchor='center', x=0.5))
    st.plotly_chart(fig, use_container_width=True)

    # Violin plot
    st.markdown("#### 🎻 Distribusi Financial Score per Cluster")
    fig = px.violin(
        fdf.dropna(subset=['cluster','financial_score']),
        x='nama_cluster', y='financial_score',
        color='nama_cluster',
        color_discrete_sequence=CLUSTER_PALETTE,
        box=True, points='outliers',
        labels={'nama_cluster': 'Cluster', 'financial_score': 'Financial Health Score'},
        title="Violin Plot Financial Score per Cluster",
    )
    for th_val, th_col, th_lbl in [
        (meta['p25'], '#ef4444', f'Bahaya↔Waspada ({meta["p25"]:.1f})'),
        (meta['p50'], '#f59e0b', f'Waspada↔Stabil ({meta["p50"]:.1f})'),
        (meta['p75'], '#22c55e', f'Stabil↔Sehat ({meta["p75"]:.1f})'),
    ]:
        fig.add_hline(y=th_val, line_dash='dash', line_color=th_col,
                      annotation_text=th_lbl, annotation_position='right',
                      annotation_font_color=th_col)
    fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False,
                      xaxis=dict(tickangle=-10, gridcolor='#e2e8f0'))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 5 — EXPORT
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-card"><div class="section-title">📥 Export & Ringkasan Akhir</div>'
                '<div class="section-sub">Download hasil analisis lengkap</div>', unsafe_allow_html=True)

    summary_rows = []
    for cl in cluster_list:
        subset  = fdf[fdf['cluster'] == cl]
        avg_sc  = subset['financial_score'].mean() if len(subset) > 0 else 0
        dist_pct = (
            subset['status_finansial']
            .value_counts(normalize=True)
            .reindex(STATUS_ORDER, fill_value=0) * 100
        )
        summary_rows.append({
            "Cluster":       nama_cluster.get(int(cl), f"Cluster {int(cl)}"),
            "Jumlah":        len(subset),
            "Avg Income":    fmt_rupiah(subset['total_pemasukan'].mean()),
            "Avg Expense":   fmt_rupiah(subset['total_pengeluaran'].mean()),
            "Avg Saving":    fmt_rupiah(subset['sisa_uang'].mean()),
            "Fin. Score":    f"{avg_sc:.1f}",
            "Status Dominan": make_status(avg_sc, meta['p25'], meta['p50'], meta['p75']),
            "% Bahaya":      f"{dist_pct['Bahaya']:.1f}%",
            "% Waspada":     f"{dist_pct['Waspada']:.1f}%",
            "% Stabil":      f"{dist_pct['Stabil']:.1f}%",
            "% Sangat Sehat":f"{dist_pct['Sangat Sehat']:.1f}%",
        })
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Dataset Lengkap (CSV)",
            data=fdf.drop(columns=['PC1','PC2'], errors='ignore').to_csv(index=False).encode('utf-8'),
            file_name="hasil_clustering_finansial_IDR.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "📥 Download Ringkasan Cluster (CSV)",
            data=df_summary.to_csv(index=False).encode('utf-8'),
            file_name="ringkasan_cluster_finansial.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown(f"""
    <div class="insight-box">
    ✅ <b>Ringkasan Analisis:</b><br>
    • Dataset: <b>{len(df):,} mahasiswa</b> · Kurs: 1 USD = Rp {KURS_USD_IDR:,}<br>
    • Algoritma: KMeans (k={best_k}) · Silhouette={meta['sil']:.4f} · Davies-Bouldin={meta['db']:.4f}<br>
    • Fitur: {len(meta['fitur'])} variabel · Data difilter: {len(fdf):,} mahasiswa<br>
    • Kolom output: <code>cluster</code>, <code>nama_cluster</code>,
      <code>financial_score</code>, <code>status_finansial</code>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────
st.markdown(
    '<div style="text-align:center;color:#94a3b8;font-size:.78rem;margin-top:1.5rem;">'
    f'Financial Clustering Dashboard · K-Means (k={best_k}) · Data Mahasiswa IDR'
    '</div>',
    unsafe_allow_html=True
)
