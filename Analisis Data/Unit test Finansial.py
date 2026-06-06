# -*- coding: utf-8 -*-
"""
test_clustering_finansial.py
=============================
Unit test untuk skrip Clustering Finansial Mahasiswa (IDR).

Cakupan test:
  - Konversi USD → IDR
  - Kalkulasi total pemasukan / pengeluaran / sisa uang
  - Rasio pengeluaran
  - Financial Health Score & kategorisasi status
  - Feature Engineering (rasio kategori, flag defisit, log transform, dll.)
  - KMeans clustering (smoke test)
  - Encoding & scaling
"""

import unittest
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Helper: buat DataFrame sampel kecil agar test tidak butuh Kaggle
# ──────────────────────────────────────────────────────────────────────────────
KURS_USD_IDR = 16_000
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


def buat_df_sampel(n=50, seed=42):
    """Buat DataFrame sintetis yang menyimulasikan pipeline konversi IDR."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        'id'              : range(n),
        'usia'            : rng.integers(18, 30, n),
        'pendapatan'      : rng.integers(500, 2000, n).astype(float) * KURS_USD_IDR,
        'bantuan'         : rng.integers(0, 500, n).astype(float) * KURS_USD_IDR,
        'pendidikan'      : rng.integers(100, 500, n).astype(float) / 6 * KURS_USD_IDR * RASIO['pendidikan'],
        'tempat_tinggal'  : rng.integers(200, 800, n).astype(float) * KURS_USD_IDR * RASIO['tempat_tinggal'],
        'makanan'         : rng.integers(100, 400, n).astype(float) * KURS_USD_IDR * RASIO['makanan'],
        'transportasi'    : rng.integers(50, 200, n).astype(float) * KURS_USD_IDR * RASIO['transportasi'],
        'buku'            : rng.integers(20, 100, n).astype(float) * KURS_USD_IDR * RASIO['buku'],
        'hiburan'         : rng.integers(30, 200, n).astype(float) * KURS_USD_IDR * RASIO['hiburan'],
        'perawatan'       : rng.integers(20, 100, n).astype(float) * KURS_USD_IDR * RASIO['perawatan'],
        'teknologi'       : rng.integers(30, 150, n).astype(float) * KURS_USD_IDR * RASIO['teknologi'],
        'kesehatan'       : rng.integers(20, 100, n).astype(float) * KURS_USD_IDR * RASIO['kesehatan'],
        'lainnya'         : rng.integers(10, 80, n).astype(float) * KURS_USD_IDR * RASIO['lainnya'],
        'gender'          : rng.choice(['Laki-laki', 'Perempuan'], n),
        'major'           : rng.choice(['Teknik', 'Ekonomi', 'Sains', 'Sosial'], n),
    })

    pengeluaran_cols = ['pendidikan','tempat_tinggal','makanan','transportasi',
                        'buku','hiburan','perawatan','teknologi','kesehatan','lainnya']

    df['total_pemasukan']   = df['pendapatan'] + df['bantuan']
    df['total_pengeluaran'] = df[pengeluaran_cols].sum(axis=1)
    df['sisa_uang']         = df['total_pemasukan'] - df['total_pengeluaran']
    df['rasio_pengeluaran'] = df['total_pengeluaran'] / df['total_pemasukan'].replace(0, np.nan)
    return df


def hitung_financial_score(row):
    """Salinan fungsi dari skrip utama."""
    if row['total_pemasukan'] > 0:
        rasio_tabungan = max(0, min(1, row['sisa_uang'] / row['total_pemasukan']))
    else:
        rasio_tabungan = 0
    skor_a = rasio_tabungan * 100

    esensial   = row['tempat_tinggal'] + row['makanan'] + row['transportasi']
    total_peng = row['total_pengeluaran']
    if total_peng > 0:
        rasio_non_esensial = 1 - esensial / total_peng
        skor_b = max(0, (1 - rasio_non_esensial * 2)) * 100
    else:
        skor_b = 50

    if total_peng > 0:
        buffer_ratio = max(0, min(1, row['sisa_uang'] / total_peng))
    else:
        buffer_ratio = 0
    skor_c = buffer_ratio * 100

    return round(0.40 * skor_a + 0.30 * skor_b + 0.30 * skor_c, 2)


def get_status(score):
    if score >= 80: return "Sangat Sehat"
    elif score >= 60: return "Stabil"
    elif score >= 40: return "Waspada"
    else: return "Bahaya"


def terapkan_feature_engineering(df):
    """Salinan blok Feature Engineering dari skrip utama."""
    pengeluaran_cols_fe = ['pendidikan','tempat_tinggal','makanan','transportasi',
                           'buku','hiburan','perawatan','teknologi','kesehatan','lainnya']
    pengeluaran_cols_fe = [c for c in pengeluaran_cols_fe if c in df.columns]

    for col in pengeluaran_cols_fe:
        df[f'rasio_{col}'] = df[col] / df['total_pengeluaran'].replace(0, np.nan)

    esensial_cols     = [c for c in ['tempat_tinggal','makanan','transportasi','pendidikan'] if c in df.columns]
    non_esensial_cols = [c for c in ['hiburan','perawatan','teknologi','lainnya'] if c in df.columns]

    df['total_esensial']       = df[esensial_cols].sum(axis=1)
    df['total_non_esensial']   = df[non_esensial_cols].sum(axis=1)
    df['rasio_esensial']       = df['total_esensial'] / df['total_pengeluaran'].replace(0, np.nan)
    df['rasio_non_esensial']   = df['total_non_esensial'] / df['total_pengeluaran'].replace(0, np.nan)
    df['rasio_bantuan']        = df['bantuan'] / df['total_pemasukan'].replace(0, np.nan)
    df['sisa_per_usia']        = df['sisa_uang'] / df['usia'].replace(0, np.nan)
    df['leverage_pengeluaran'] = df['total_pengeluaran'] / df['pendapatan'].replace(0, np.nan)
    df['flag_defisit']         = (df['sisa_uang'] < 0).astype(int)
    df['log_total_pemasukan']  = np.log1p(df['total_pemasukan'].clip(lower=0))
    df['log_total_pengeluaran']= np.log1p(df['total_pengeluaran'].clip(lower=0))
    df['log_sisa_uang']        = np.log1p(df['sisa_uang'].clip(lower=0))
    return df


# ==============================================================================
# KELAS TEST
# ==============================================================================

class TestKonversiIDR(unittest.TestCase):
    """Test kalkulasi kolom keuangan dasar."""

    def setUp(self):
        self.df = buat_df_sampel(n=20)

    def test_total_pemasukan_positif(self):
        """Total pemasukan harus selalu >= 0."""
        self.assertTrue((self.df['total_pemasukan'] >= 0).all(),
                        "Ditemukan total_pemasukan negatif!")

    def test_total_pengeluaran_positif(self):
        self.assertTrue((self.df['total_pengeluaran'] >= 0).all())

    def test_sisa_uang_konsisten(self):
        """sisa_uang = total_pemasukan - total_pengeluaran."""
        selisih = (self.df['total_pemasukan'] - self.df['total_pengeluaran'] - self.df['sisa_uang']).abs()
        self.assertTrue((selisih < 1e-6).all(), "sisa_uang tidak konsisten!")

    def test_rasio_pengeluaran_range(self):
        """Rasio pengeluaran umumnya positif (boleh > 1 jika defisit)."""
        self.assertTrue((self.df['rasio_pengeluaran'].dropna() >= 0).all())

    def test_tidak_ada_missing_kolom_utama(self):
        for col in ['total_pemasukan', 'total_pengeluaran', 'sisa_uang']:
            self.assertEqual(self.df[col].isnull().sum(), 0,
                             f"Kolom {col} ada nilai NaN!")

    def test_kurs_diterapkan(self):
        """Nilai pendapatan setelah konversi harus jauh lebih besar dari nilai USD aslinya."""
        # Nilai dalam rentang IDR (rata-rata pendapatan > 1 juta)
        self.assertTrue(self.df['pendapatan'].mean() > 1_000_000,
                        "Kurs IDR sepertinya belum diterapkan.")


class TestFinancialScore(unittest.TestCase):
    """Test fungsi hitung_financial_score dan kategorisasi status."""

    def setUp(self):
        self.df = buat_df_sampel(n=30)
        self.df['financial_score'] = self.df.apply(hitung_financial_score, axis=1)
        self.df['status_finansial'] = self.df['financial_score'].apply(get_status)

    def test_score_range(self):
        """Financial score harus berada di rentang 0–100."""
        self.assertTrue((self.df['financial_score'] >= 0).all() and
                        (self.df['financial_score'] <= 100).all(),
                        "Financial score di luar rentang 0–100!")

    def test_score_tipe_float(self):
        self.assertTrue(self.df['financial_score'].dtype in [np.float64, float])

    def test_status_hanya_empat_nilai(self):
        nilai_valid = {"Bahaya", "Waspada", "Stabil", "Sangat Sehat"}
        nilai_aktual = set(self.df['status_finansial'].unique())
        self.assertTrue(nilai_aktual.issubset(nilai_valid),
                        f"Status tidak valid: {nilai_aktual - nilai_valid}")

    def test_status_sesuai_score_sangat_sehat(self):
        self.assertEqual(get_status(85), "Sangat Sehat")
        self.assertEqual(get_status(80), "Sangat Sehat")

    def test_status_sesuai_score_stabil(self):
        self.assertEqual(get_status(75), "Stabil")
        self.assertEqual(get_status(60), "Stabil")

    def test_status_sesuai_score_waspada(self):
        self.assertEqual(get_status(55), "Waspada")
        self.assertEqual(get_status(40), "Waspada")

    def test_status_sesuai_score_bahaya(self):
        self.assertEqual(get_status(39), "Bahaya")
        self.assertEqual(get_status(0),  "Bahaya")

    def test_mahasiswa_hemat_dapat_score_tinggi(self):
        """Mahasiswa dengan banyak sisa uang seharusnya dapat skor tinggi."""
        row_hemat = {
            'total_pemasukan'  : 5_000_000,
            'total_pengeluaran': 1_000_000,
            'sisa_uang'        : 4_000_000,
            'tempat_tinggal'   : 400_000,
            'makanan'          : 300_000,
            'transportasi'     : 150_000,
        }
        skor = hitung_financial_score(row_hemat)
        self.assertGreater(skor, 60, "Mahasiswa hemat seharusnya skor > 60")

    def test_mahasiswa_defisit_dapat_score_rendah(self):
        """Mahasiswa dengan pengeluaran lebih besar dari pemasukan → skor rendah."""
        row_defisit = {
            'total_pemasukan'  : 1_000_000,
            'total_pengeluaran': 2_000_000,
            'sisa_uang'        : -1_000_000,
            'tempat_tinggal'   : 700_000,
            'makanan'          : 500_000,
            'transportasi'     : 300_000,
        }
        skor = hitung_financial_score(row_defisit)
        self.assertLess(skor, 40, "Mahasiswa defisit seharusnya skor < 40")


class TestFeatureEngineering(unittest.TestCase):
    """Test blok Feature Engineering."""

    def setUp(self):
        self.df = buat_df_sampel(n=40)
        self.df = terapkan_feature_engineering(self.df.copy())

    def test_kolom_rasio_dibuat(self):
        """Semua kolom rasio_<kategori> harus ada."""
        for col in ['pendidikan','tempat_tinggal','makanan','transportasi',
                    'buku','hiburan','perawatan','teknologi','kesehatan','lainnya']:
            self.assertIn(f'rasio_{col}', self.df.columns,
                          f"Kolom rasio_{col} tidak ditemukan!")

    def test_rasio_kategori_antara_0_dan_1(self):
        """Setiap rasio kategori harus antara 0 dan 1."""
        for col in ['pendidikan','tempat_tinggal','makanan','transportasi','hiburan']:
            vals = self.df[f'rasio_{col}'].dropna()
            self.assertTrue((vals >= 0).all() and (vals <= 1).all(),
                            f"rasio_{col} di luar [0, 1]!")

    def test_rasio_esensial_non_esensial_jumlah_tidak_lebih_1(self):
        """rasio_esensial + rasio_non_esensial ≤ 1 (sisanya buku/perawatan/kesehatan)."""
        total = (self.df['rasio_esensial'] + self.df['rasio_non_esensial']).dropna()
        self.assertTrue((total <= 1.01).all(),  # toleransi floating point
                        "rasio_esensial + rasio_non_esensial melebihi 1!")

    def test_flag_defisit_binary(self):
        """flag_defisit harus hanya berisi 0 atau 1."""
        nilai_unik = set(self.df['flag_defisit'].unique())
        self.assertTrue(nilai_unik.issubset({0, 1}),
                        f"flag_defisit berisi nilai selain 0/1: {nilai_unik}")

    def test_flag_defisit_konsisten_dengan_sisa_uang(self):
        """flag_defisit=1 hanya jika sisa_uang < 0."""
        inconsistent = self.df[
            ((self.df['flag_defisit'] == 1) & (self.df['sisa_uang'] >= 0)) |
            ((self.df['flag_defisit'] == 0) & (self.df['sisa_uang'] < 0))
        ]
        self.assertEqual(len(inconsistent), 0,
                         "flag_defisit tidak konsisten dengan sisa_uang!")

    def test_log_transform_non_negatif(self):
        """log1p dari nilai yang di-clip ke 0 harus selalu >= 0."""
        for col in ['log_total_pemasukan', 'log_total_pengeluaran', 'log_sisa_uang']:
            self.assertIn(col, self.df.columns, f"{col} tidak ditemukan!")
            self.assertTrue((self.df[col] >= 0).all(),
                            f"{col} mengandung nilai negatif!")

    def test_leverage_pengeluaran_positif(self):
        """leverage_pengeluaran = total_pengeluaran / pendapatan harus > 0."""
        vals = self.df['leverage_pengeluaran'].dropna()
        self.assertTrue((vals > 0).all())

    def test_rasio_bantuan_antara_0_dan_1(self):
        """rasio_bantuan tidak boleh > 1 (bantuan tidak bisa lebih besar dari total pemasukan)."""
        # Karena total_pemasukan = pendapatan + bantuan, rasio bantuan < 1
        vals = self.df['rasio_bantuan'].dropna()
        self.assertTrue((vals >= 0).all() and (vals < 1).all(),
                        "rasio_bantuan di luar [0, 1)!")

    def test_total_esensial_non_esensial_konsisten(self):
        """total_esensial + total_non_esensial ≤ total_pengeluaran."""
        diff = self.df['total_pengeluaran'] - (self.df['total_esensial'] + self.df['total_non_esensial'])
        self.assertTrue((diff >= -1).all(),   # toleransi floating point
                        "total_esensial + total_non_esensial melebihi total_pengeluaran!")


class TestEncodingDanScaling(unittest.TestCase):
    """Test encoding kategorikal dan standarisasi fitur."""

    def test_label_encoder_menghasilkan_integer(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        data = pd.Series(['Laki-laki', 'Perempuan', 'Laki-laki'])
        result = le.fit_transform(data)
        self.assertTrue(np.issubdtype(result.dtype, np.integer))

    def test_standard_scaler_mean_nol(self):
        from sklearn.preprocessing import StandardScaler
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(scaled.mean(axis=0), [0, 0], decimal=10)

    def test_standard_scaler_std_satu(self):
        from sklearn.preprocessing import StandardScaler
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(scaled.std(axis=0), [1, 1], decimal=10)


class TestKMeansClustering(unittest.TestCase):
    """Smoke test KMeans dengan data sintetis."""

    def setUp(self):
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        self.df = buat_df_sampel(n=100)
        self.df['financial_score'] = self.df.apply(hitung_financial_score, axis=1)
        self.df = terapkan_feature_engineering(self.df.copy())

        fitur = ['total_pemasukan', 'total_pengeluaran', 'sisa_uang',
                 'rasio_pengeluaran', 'financial_score',
                 'rasio_esensial', 'rasio_non_esensial', 'flag_defisit',
                 'log_total_pemasukan', 'log_total_pengeluaran']

        le = LabelEncoder()
        self.df['gender_enc'] = le.fit_transform(self.df['gender'])
        fitur += ['gender_enc']

        fitur = [c for c in fitur if c in self.df.columns]
        X = self.df[fitur].dropna()

        scaler  = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)
        self.X_index  = X.index

    def test_kmeans_jumlah_label_sesuai_k(self):
        from sklearn.cluster import KMeans
        k = 3
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = km.fit_predict(self.X_scaled)
        self.assertEqual(len(np.unique(labels)), k)

    def test_kmeans_label_panjang_sama_dengan_data(self):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=3, n_init=5, random_state=42)
        labels = km.fit_predict(self.X_scaled)
        self.assertEqual(len(labels), len(self.X_scaled))

    def test_silhouette_score_positif(self):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        km = KMeans(n_clusters=3, n_init=5, random_state=42)
        labels = km.fit_predict(self.X_scaled)
        sil = silhouette_score(self.X_scaled, labels)
        self.assertGreater(sil, 0, "Silhouette score negatif — clustering sangat buruk!")

    def test_davies_bouldin_terbatas(self):
        from sklearn.cluster import KMeans
        from sklearn.metrics import davies_bouldin_score
        km = KMeans(n_clusters=3, n_init=5, random_state=42)
        labels = km.fit_predict(self.X_scaled)
        db = davies_bouldin_score(self.X_scaled, labels)
        # DB score tidak boleh sangat besar (jika > 5 ada masalah serius)
        self.assertLess(db, 5, f"Davies-Bouldin Score terlalu tinggi: {db:.4f}")

    def test_pca_variance_explained(self):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        pca.fit(self.X_scaled)
        total_var = pca.explained_variance_ratio_.sum() * 100
        # Minimal 30% variasi harus tertangkap oleh 2 komponen
        self.assertGreater(total_var, 30,
                           f"PCA 2D hanya menjelaskan {total_var:.1f}% variasi!")


class TestDataQuality(unittest.TestCase):
    """Test kualitas data dan edge case."""

    def test_tidak_ada_nilai_negatif_di_pengeluaran(self):
        df = buat_df_sampel(n=50)
        cols = ['pendidikan','tempat_tinggal','makanan','transportasi',
                'buku','hiburan','perawatan','teknologi','kesehatan','lainnya']
        for col in cols:
            if col in df.columns:
                self.assertTrue((df[col] >= 0).all(), f"{col} ada nilai negatif!")

    def test_usia_dalam_range_wajar(self):
        df = buat_df_sampel(n=50)
        self.assertTrue((df['usia'] >= 17).all())
        self.assertTrue((df['usia'] <= 35).all())

    def test_financial_score_tidak_nan_pada_data_valid(self):
        df = buat_df_sampel(n=30)
        df['financial_score'] = df.apply(hitung_financial_score, axis=1)
        self.assertEqual(df['financial_score'].isnull().sum(), 0)

    def test_edge_case_pemasukan_nol(self):
        """Jika total_pemasukan = 0, financial_score tidak boleh error."""
        row = {
            'total_pemasukan'  : 0,
            'total_pengeluaran': 500_000,
            'sisa_uang'        : -500_000,
            'tempat_tinggal'   : 200_000,
            'makanan'          : 150_000,
            'transportasi'     : 100_000,
        }
        try:
            skor = hitung_financial_score(row)
            self.assertIsInstance(skor, float)
        except Exception as e:
            self.fail(f"hitung_financial_score crash pada pemasukan=0: {e}")

    def test_edge_case_semua_pengeluaran_nol(self):
        """Jika total_pengeluaran = 0, tidak boleh division by zero."""
        row = {
            'total_pemasukan'  : 5_000_000,
            'total_pengeluaran': 0,
            'sisa_uang'        : 5_000_000,
            'tempat_tinggal'   : 0,
            'makanan'          : 0,
            'transportasi'     : 0,
        }
        try:
            skor = hitung_financial_score(row)
            self.assertIsInstance(skor, float)
        except Exception as e:
            self.fail(f"hitung_financial_score crash pada pengeluaran=0: {e}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    print("=" * 65)
    print("🧪  UNIT TEST — CLUSTERING FINANSIAL MAHASISWA (IDR)")
    print("=" * 65)
    unittest.main(verbosity=2)