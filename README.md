# 💰 Dashboard Sfinity

Dashboard interaktif berbasis **Streamlit** untuk menganalisis pola keuangan mahasiswa menggunakan pendekatan **K-Means Clustering** dan **Financial Health Score**.

---

## 📌 Deskripsi

Dashboard ini memproses data pengeluaran mahasiswa (dalam USD) dan mengonversinya ke **Rupiah (IDR)**, lalu melakukan segmentasi mahasiswa berdasarkan perilaku finansial mereka. Hasil akhirnya adalah profil cluster yang menggambarkan kondisi keuangan tiap kelompok mahasiswa secara mendalam.

---

## 🗂️ Struktur Proyek

```
.
├── App.py                            # File utama Streamlit
├── Data/
│   └── student_spending (1).csv     # Dataset sumber (Kaggle)
├── requirements.txt                  # Dependensi Python
└── README.md
```

---

## 📊 Dataset

| Kolom Asli         | Kolom Hasil Rename   | Keterangan                        |
|--------------------|----------------------|-----------------------------------|
| `age`              | `usia`               | Usia mahasiswa                    |
| `monthly_income`   | `pendapatan`         | Pendapatan bulanan (USD → IDR)    |
| `financial_aid`    | `bantuan`            | Bantuan keuangan (USD → IDR)      |
| `tuition`          | `pendidikan`         | Biaya kuliah                      |
| `housing`          | `tempat_tinggal`     | Biaya tempat tinggal              |
| `food`             | `makanan`            | Biaya makanan                     |
| `transportation`   | `transportasi`       | Biaya transportasi                |
| `books_supplies`   | `buku`               | Biaya buku & alat tulis           |
| `entertainment`    | `hiburan`            | Biaya hiburan                     |
| `personal_care`    | `perawatan`          | Biaya perawatan diri              |
| `technology`       | `teknologi`          | Biaya teknologi                   |
| `health_wellness`  | `kesehatan`          | Biaya kesehatan                   |
| `miscellaneous`    | `lainnya`            | Pengeluaran lainnya               |


---

## ⚙️ Fitur Utama

### 🩺 Financial Health Score
Skor (0–100) yang dihitung dari tiga komponen:

| Komponen              | Bobot | Keterangan                                    |
|-----------------------|-------|-----------------------------------------------|
| Rasio tabungan        | 40%   | Proporsi sisa uang terhadap total pemasukan   |
| Pengeluaran esensial  | 30%   | Proporsi tempat tinggal + makanan + transport |
| Buffer likuiditas     | 30%   | Kemampuan sisa uang menutup pengeluaran       |

### 🏷️ Status Finansial
Mahasiswa dikelompokkan berdasarkan kuartil Financial Health Score:

| Status        | Kondisi                  |
|---------------|--------------------------|
| 🔴 Bahaya      | Score < Q25              |
| 🟠 Waspada     | Q25 ≤ Score < Q50        |
| 🟡 Stabil      | Q50 ≤ Score < Q75        |
| 🟢 Sangat Sehat| Score ≥ Q75              |

### 🤖 Clustering K-Means
- Jumlah cluster optimal dipilih otomatis menggunakan **Silhouette Score** (k=2 hingga 6)
- Fitur: variabel finansial numerik + variabel kategorikal (ter-encode)
- Evaluasi: **Silhouette Score** dan **Davies-Bouldin Index**
- Visualisasi: **PCA 2D scatter plot**

Nama cluster ditetapkan secara otomatis berdasarkan profil rata-rata:

| Label Cluster                          | Ciri Utama                            |
|----------------------------------------|---------------------------------------|
| Mahasiswa Mapan & Hemat                | Income tinggi, sisa uang terbesar     |
| Mahasiswa Hemat                        | Sisa uang besar meski income biasa    |
| Mahasiswa Berpenghasilan Tinggi & Boros| Income tinggi tapi rasio belanja tinggi|
| Mahasiswa Boros / Defisit              | Rasio pengeluaran tertinggi           |
| Mahasiswa Berpenghasilan Rendah        | Income terendah                       |
| Mahasiswa Rata-rata                    | Profil tengah                         |

---

## 🖥️ Tab Dashboard

| Tab              | Isi                                                                 |
|------------------|---------------------------------------------------------------------|
| 📊 Overview      | Distribusi status finansial, pie chart, histogram financial score   |
| 🔍 EDA           | Distribusi numerik, boxplot, komposisi pengeluaran, korelasi        |
| 🤖 Clustering    | Silhouette per k, evaluasi model, PCA 2D scatter                    |
| 🏷️ Profil Cluster| Kartu ringkasan cluster, heatmap, bar KPI, violin plot             |
| 📥 Export        | Download dataset lengkap & ringkasan cluster dalam format CSV       |

---

## 🚀 Cara Menjalankan

### 1. Clone / download project

```bash
git clone <url-repo>
cd <nama-folder>
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Pastikan dataset tersedia

Letakkan file CSV dataset di dalam folder `Data/`:

```
Data/student_spending (1).csv
```

### 4. Jalankan Streamlit

```bash
streamlit run App.py
```

---

## 📦 Dependensi

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

> Semua dependensi tersedia di `requirements.txt`

---

## 📥 Upload Dataset Lain (Opsional)

Dashboard mendukung **upload CSV kustom** melalui sidebar. Pastikan kolom-kolom yang dibutuhkan tersedia sesuai tabel dataset di atas.

---

## 📁 Output

Dari tab **Export**, pengguna dapat mengunduh:
- `hasil_clustering_finansial_IDR.csv` — dataset lengkap dengan kolom hasil clustering
- `ringkasan_cluster_finansial.csv` — tabel ringkasan per cluster

Kolom tambahan yang dihasilkan oleh pipeline:

| Kolom               | Keterangan                              |
|---------------------|-----------------------------------------|
| `total_pemasukan`   | Pendapatan + bantuan (IDR)              |
| `total_pengeluaran` | Total semua kategori pengeluaran (IDR)  |
| `sisa_uang`         | Pemasukan − pengeluaran (IDR)           |
| `rasio_pengeluaran` | Total pengeluaran / total pemasukan     |
| `financial_score`   | Skor kesehatan finansial (0–100)        |
| `status_finansial`  | Bahaya / Waspada / Stabil / Sangat Sehat|
| `cluster`           | Nomor cluster (K-Means)                 |
| `nama_cluster`      | Label deskriptif cluster                |

---

## 👤 Kredit

- **Data:** [Kaggle — Student Spending Dataset](https://www.kaggle.com/)
- **Dashboard:** Dibangun dengan Streamlit · Matplotlib · Seaborn · Scikit-learn
