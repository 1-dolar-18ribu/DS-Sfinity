import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Financial Clustering Dashboard",
    page_icon="💰",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark navy header */
.main-header {
    background: #0f172a;
    color: white;
    padding: 16px 28px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.main-header h1 { font-size: 1.5rem; margin: 0; font-weight: 700; }
.main-header span { font-size: 0.85rem; color: #94a3b8; }

/* Metric cards */
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
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* Section cards */
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
    margin-bottom: 16px;
}

body { background: #f1f5f9 !important; }
[data-testid="stAppViewContainer"] { background: #f1f5f9; }
[data-testid="block-container"] { padding-top: 0; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = df = pd.read_csv("https://github.com/BinQ-07/dashboard-capstone/blob/main/data/hasil_clustering_finansial_IDR.csv?raw=true")
    return df

df = load_data()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div>
        <h1>💰 Financial Clustering Dashboard</h1>
        <span>Analisis Keuangan Mahasiswa · K-Means Clustering · IDR</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filter Data")
    cluster_filter = st.multiselect(
        "Cluster",
        options=df["nama_cluster"].unique(),
        default=df["nama_cluster"].unique(),
    )
    gender_filter = st.multiselect(
        "Gender",
        options=df["gender"].unique(),
        default=df["gender"].unique(),
    )
    year_filter = st.multiselect(
        "Tahun Kuliah",
        options=df["year_in_school"].unique(),
        default=df["year_in_school"].unique(),
    )

fdf = df[
    df["nama_cluster"].isin(cluster_filter) &
    df["gender"].isin(gender_filter) &
    df["year_in_school"].isin(year_filter)
]

# ── KPI Cards ────────────────────────────────────────────────────────────────
total_students  = len(fdf)
avg_income      = fdf["total_pemasukan"].mean()
avg_expense     = fdf["total_pengeluaran"].mean()
avg_savings     = fdf["sisa_uang"].mean()
avg_fin_score   = fdf["financial_score"].mean()
avg_ratio       = fdf["rasio_pengeluaran"].mean()

def fmt_idr(val):
    if val >= 1_000_000:
        return f"Rp {val/1_000_000:.1f}Jt"
    return f"Rp {val:,.0f}"

col1, col2, col3, col4, col5, col6 = st.columns(6)
metrics = [
    (col1, "Total Mahasiswa",    f"{total_students:,}",       "responden"),
    (col2, "Avg Pemasukan",      fmt_idr(avg_income),         "per bulan"),
    (col3, "Avg Pengeluaran",    fmt_idr(avg_expense),        "per bulan"),
    (col4, "Avg Sisa Uang",      fmt_idr(avg_savings),        "per bulan"),
    (col5, "Avg Financial Score",f"{avg_fin_score:.1f}",      "dari 100"),
    (col6, "Avg Rasio Belanja",  f"{avg_ratio*100:.1f}%",     "pengeluaran/pemasukan"),
]
for col, label, value, sub in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: Cluster Distribution + Status Finansial ───────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-card"><div class="section-title">📊 Distribusi Cluster</div>', unsafe_allow_html=True)
    cluster_counts = fdf["nama_cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Jumlah"]
    fig_cluster = px.bar(
        cluster_counts, x="Cluster", y="Jumlah",
        color="Cluster",
        color_discrete_sequence=["#3b82f6", "#f59e0b"],
        text="Jumlah",
    )
    fig_cluster.update_traces(textposition="outside")
    fig_cluster.update_layout(
        showlegend=False, height=300,
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
    )
    st.plotly_chart(fig_cluster, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="section-card"><div class="section-title">🏦 Status Finansial</div>', unsafe_allow_html=True)
    status_counts = fdf["status_finansial"].value_counts().reset_index()
    status_counts.columns = ["Status", "Jumlah"]
    color_map = {
        "Sangat Sehat": "#22c55e",
        "Stabil": "#3b82f6",
        "Waspada": "#f59e0b",
        "Bahaya": "#ef4444",
    }
    fig_status = px.pie(
        status_counts, values="Jumlah", names="Status",
        color="Status", color_discrete_map=color_map,
        hole=0.45,
    )
    fig_status.update_traces(textposition="outside", textinfo="percent+label")
    fig_status.update_layout(
        showlegend=False, height=300,
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_status, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Row 2: Spending Categories + Gender Distribution ─────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.markdown('<div class="section-card"><div class="section-title">💸 Rata-rata Pengeluaran per Kategori</div>', unsafe_allow_html=True)
    cats = {
        "Tempat Tinggal": "tempat_tinggal",
        "Pendidikan":     "pendidikan",
        "Makanan":        "makanan",
        "Teknologi":      "teknologi",
        "Buku":           "buku",
        "Kesehatan":      "kesehatan",
        "Transportasi":   "transportasi",
        "Lainnya":        "lainnya",
        "Hiburan":        "hiburan",
        "Perawatan":      "perawatan",
    }
    cat_means = {k: fdf[v].mean() for k, v in cats.items()}
    cat_df = pd.DataFrame({"Kategori": list(cat_means.keys()), "Rata-rata (Rp)": list(cat_means.values())})
    cat_df = cat_df.sort_values("Rata-rata (Rp)", ascending=True)
    fig_cat = px.bar(
        cat_df, x="Rata-rata (Rp)", y="Kategori",
        orientation="h",
        color="Rata-rata (Rp)",
        color_continuous_scale="Blues",
        text=cat_df["Rata-rata (Rp)"].apply(lambda x: f"Rp{x/1e6:.2f}Jt"),
    )
    fig_cat.update_traces(textposition="outside")
    fig_cat.update_layout(
        showlegend=False, height=340,
        margin=dict(t=10, b=10, l=0, r=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_d:
    st.markdown('<div class="section-card"><div class="section-title">👥 Distribusi Gender per Cluster</div>', unsafe_allow_html=True)
    gender_cluster = fdf.groupby(["nama_cluster", "gender"]).size().reset_index(name="Jumlah")
    fig_gender = px.bar(
        gender_cluster, x="nama_cluster", y="Jumlah", color="gender",
        barmode="group",
        color_discrete_sequence=["#3b82f6", "#ec4899", "#a855f7"],
        labels={"nama_cluster": "Cluster", "gender": "Gender"},
    )
    fig_gender.update_layout(
        height=340,
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_gender, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Row 3: Payment Method + Financial Score Distribution ─────────────────────
col_e, col_f = st.columns(2)

with col_e:
    st.markdown('<div class="section-card"><div class="section-title">💳 Metode Pembayaran</div>', unsafe_allow_html=True)
    pay_counts = fdf["preferred_payment_method"].value_counts().reset_index()
    pay_counts.columns = ["Metode", "Jumlah"]
    fig_pay = px.pie(
        pay_counts, values="Jumlah", names="Metode",
        color_discrete_sequence=["#06b6d4", "#6366f1", "#f59e0b"],
        hole=0.4,
    )
    fig_pay.update_traces(textposition="outside", textinfo="percent+label")
    fig_pay.update_layout(
        height=300,
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_pay, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_f:
    st.markdown('<div class="section-card"><div class="section-title">📈 Distribusi Financial Score per Cluster</div>', unsafe_allow_html=True)
    fig_box = px.box(
        fdf, x="nama_cluster", y="financial_score",
        color="nama_cluster",
        color_discrete_sequence=["#3b82f6", "#f59e0b"],
        labels={"nama_cluster": "Cluster", "financial_score": "Financial Score"},
        points="outliers",
    )
    fig_box.update_layout(
        showlegend=False, height=300,
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Row 4: Year in School + Income vs Expense Scatter ────────────────────────
col_g, col_h = st.columns(2)

with col_g:
    st.markdown('<div class="section-card"><div class="section-title">🎓 Distribusi Tahun Kuliah per Cluster</div>', unsafe_allow_html=True)
    year_order = ["Freshman", "Sophomore", "Junior", "Senior"]
    year_cluster = fdf.groupby(["year_in_school", "nama_cluster"]).size().reset_index(name="Jumlah")
    fig_year = px.bar(
        year_cluster, x="year_in_school", y="Jumlah", color="nama_cluster",
        barmode="stack",
        color_discrete_sequence=["#3b82f6", "#f59e0b"],
        category_orders={"year_in_school": year_order},
        labels={"year_in_school": "Tahun Kuliah", "nama_cluster": "Cluster"},
    )
    fig_year.update_layout(
        height=300,
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_year, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_h:
    st.markdown('<div class="section-card"><div class="section-title">🔵 Pemasukan vs Pengeluaran</div>', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        fdf.sample(min(400, len(fdf)), random_state=42),
        x="total_pemasukan", y="total_pengeluaran",
        color="nama_cluster",
        color_discrete_sequence=["#3b82f6", "#f59e0b"],
        opacity=0.6,
        labels={
            "total_pemasukan": "Total Pemasukan (Rp)",
            "total_pengeluaran": "Total Pengeluaran (Rp)",
            "nama_cluster": "Cluster",
        },
    )
    # 45-degree reference line
    max_val = fdf[["total_pemasukan", "total_pengeluaran"]].max().max()
    fig_scatter.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="#ef4444", dash="dash", width=1.5),
    )
    fig_scatter.update_layout(
        height=300,
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Row 5: Comparison Table ───────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">📋 Ringkasan Per Cluster</div>', unsafe_allow_html=True)
summary = fdf.groupby("nama_cluster").agg(
    Jumlah=("id", "count"),
    Avg_Pemasukan=("total_pemasukan", "mean"),
    Avg_Pengeluaran=("total_pengeluaran", "mean"),
    Avg_Sisa=("sisa_uang", "mean"),
    Avg_Score=("financial_score", "mean"),
    Avg_Rasio=("rasio_pengeluaran", "mean"),
).reset_index()
summary.columns = ["Cluster", "Jumlah", "Avg Pemasukan", "Avg Pengeluaran", "Avg Sisa Uang", "Avg Score", "Avg Rasio"]
summary["Avg Pemasukan"]    = summary["Avg Pemasukan"].apply(lambda x: f"Rp {x:,.0f}")
summary["Avg Pengeluaran"]  = summary["Avg Pengeluaran"].apply(lambda x: f"Rp {x:,.0f}")
summary["Avg Sisa Uang"]    = summary["Avg Sisa Uang"].apply(lambda x: f"Rp {x:,.0f}")
summary["Avg Score"]        = summary["Avg Score"].apply(lambda x: f"{x:.1f}")
summary["Avg Rasio"]        = summary["Avg Rasio"].apply(lambda x: f"{x*100:.1f}%")
st.dataframe(summary, use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.78rem; margin-top:1rem;">
    Financial Clustering Dashboard · K-Means (n=2) · Data Mahasiswa IDR
</div>
""", unsafe_allow_html=True)
