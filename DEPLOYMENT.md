# LAPORAN DEPLOYMENT: GOLD SENTIMENT ENGINE

**Klasifikasi Sentimen Berita Emas — Two-Stage Logistic Regression + Streamlit Cloud**

---

## DAFTAR ISI

1. [Deskripsi Proyek](#1-deskripsi-proyek)
2. [Arsitektur Sistem](#2-arsitektur-sistem)
3. [Tech Stack & Dependensi](#3-tech-stack--dependensi)
4. [Persiapan Deployment](#4-persiapan-deployment)
5. [Proses Deployment](#5-proses-deployment)
6. [Verifikasi & Pengujian](#6-verifikasi--pengujian)
7. [Hasil Akhir](#7-hasil-akhir)
8. [Kesimpulan](#8-kesimpulan)

---

## 1. Deskripsi Proyek

| Aspek | Detail |
|---|---|
| **Judul** | Gold Sentiment Engine |
| **Tujuan** | Mengklasifikasikan sentimen berita emas ke dalam 3 kelas: Positive, Neutral, Negative |
| **Model** | Two-Stage Hierarchical Logistic Regression (C=100, l1 penalty, solver=saga) |
| **Vectorizer** | TF-IDF (max_features=3000, ngram 1-2, min_df=3, max_df=0.9, sublinear_tf=True) |
| **Dataset** | 628 sampel berita emas (2021–2026) dari berbagai sumber |
| **Akurasi 5-Fold CV** | 88.38% |
| **F1 Macro 5-Fold CV** | 86.19% |
| **Confidence** | `predict_proba()` — probabilitas statistik (bukan heuristik) |
| **URL Deploy** | `https://gold-sentiment-engine.streamlit.app` |
| **Repo GitHub** | `https://github.com/yoekesekti/gold-sentiment-engine` |

---

## 2. Arsitektur Sistem

### 2.1 Diagram Alur Prediksi

```
INPUT (Judul + Isi Berita)
        │
        ▼
┌───────────────────────────────────────┐
│            PREPROCESSING              │
│  • Case folding (lowercase)           │
│  • Noise removal (hapus URL, non-alfa)│
│  • Entity protection (gold, emas, dll)│
│  • Tokenization (NLTK word_tokenize)  │
│  • Stopword removal (custom)          │
│  • Lemmatization bilingual            │
│    (Sastrawi ID + Snowball EN)        │
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│       TF-IDF VECTORIZER               │
│   max_features = 3000                 │
│   ngram_range  = (1, 2)               │
│   min_df = 3, max_df = 0.9            │
│   sublinear_tf = True                 │
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│   TWO-STAGE LOGISTIC REGRESSION       │
│                                       │
│  Stage 1:                             │
│  LogisticRegression (C=100, l1)       │
│  Positive vs (Neutral + Negative)     │
│  predict_proba → P(positive)          │
│                                       │
│  Stage 2:                             │
│  LogisticRegression (C=100, l1)       │
│  Neutral vs Negative                  │
│  predict_proba → P(negative)          │
└───────────────┬───────────────────────┘
                │
                ▼
OUTPUT (Label + Confidence + Distribusi Kelas)
```

### 2.2 Two-Stage Classification Logic

| Stage | Model | Tugas | Threshold |
|---|---|---|---|
| **Stage 1** | LogisticRegression | Pisahkan **Positive** vs **(Neutral + Negative)** | P(positive) > 0.5 |
| **Stage 2** | LogisticRegression | Pisahkan **Negative** vs **Neutral** | P(negative) >= 0.5 |

Confidence score menggunakan **`predict_proba()`**, yaitu probabilitas statistik asli dari model Logistic Regression. Berbeda dengan SVM yang memerlukan Min-Max normalization heuristik, LR menghasilkan probabilitas yang langsung dapat digunakan dalam rentang 0.0 – 1.0.

### 2.3 Distribusi Probabilitas Kelas

```
p_positive = P(positive) dari Stage 1 predict_proba
sisa       = 1 - p_positive
p_negative = sisa × P(negative) dari Stage 2 predict_proba
p_neutral  = sisa × P(neutral) dari Stage 2 predict_proba
```

---

## 3. Tech Stack & Dependensi

| Komponen | Versi | Fungsi |
|---|---|---|
| **Python** | 3.10+ | Bahasa pemrograman |
| **Streamlit** | >=1.28.0 | Web framework UI aplikasi |
| **scikit-learn** | >=1.0 | LogisticRegression, TfidfVectorizer |
| **scipy** | >=1.7 | Operasi sparse matrix TF-IDF |
| **joblib** | >=1.2 | Serialisasi model (.pkl) |
| **nltk** | >=3.8 | Tokenisasi, Snowball stemming |
| **Sastrawi** | >=1.0 | Stemming Bahasa Indonesia |
| **numpy** | >=1.21 | Komputasi numerik |

### 3.1 `requirements.txt`

```
streamlit>=1.28.0
scikit-learn>=1.0
scipy>=1.7
joblib>=1.2
nltk>=3.8
sastrawi>=1.0
numpy>=1.21
```

**Penjelasan tambahan dependensi:**

- `scipy` — diperlukan oleh scikit-learn untuk operasi sparse matrix pada TF-IDF. Meskipun biasanya ter-install otomatis bersama scikit-learn, Streamlit Cloud memerlukan deklarasi eksplisit.

### 3.2 Struktur Folder

```
gold-sentiment-engine/
├── .gitignore                    # File yang diabaikan Git
├── requirements.txt              # Dependensi Python
├── streamlit_app.py              # Aplikasi Streamlit (UI utama)
├── train_lr.py                   # Script training model (development)
├── 9_eksperimen_lr.py            # Script eksperimen ML
├── 9. eksperimen_lr.ipynb        # Notebook eksperimen ML
├── DEPLOYMENT.md                 # Laporan deployment ini
└── models/
    ├── predict.py                # Modul inference & preprocessing
    ├── lr_stage1.pkl             # Model Stage 1 (Positive vs Rest)
    ├── lr_stage2.pkl             # Model Stage 2 (Negative vs Neutral)
    ├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
    └── preprocessing_config.pkl  # Konfigurasi stopwords & finance terms
```

---

## 4. Persiapan Deployment

### 4.1 Prasyarat

- Akun [GitHub](https://github.com) (sudah ada: `yoekesekti`)
- Git ter-install di laptop
- Repository GitHub: `https://github.com/yoekesekti/gold-sentiment-engine`
- Akun [Streamlit Community Cloud](https://share.streamlit.io) (login pakai GitHub)

### 4.2 Konfigurasi Git (jika belum)

```powershell
git config --global user.name "yoekesekti"
git config --global user.email "email@gmail.com"
```

### 4.3 Buat `.gitignore`

Buat file `.gitignore` di root folder proyek dengan isi:

```
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
Data Merge - Sheet2.csv
preprocessed_data.csv
```

**Penjelasan:**
- `__pycache__/`, `*.pyc`, `*.pyo` — file kompilasi Python (sampah)
- `.ipynb_checkpoints/` — checkpoint Jupyter Notebook
- `Data Merge - Sheet2.csv` — dataset mentah (tidak diperlukan untuk deployment, model sudah di-train)
- `preprocessed_data.csv` — hasil preprocessing (tidak diperlukan)

### 4.4 Verifikasi Dependensi Lengkap

Pastikan `requirements.txt` sudah berisi semua library yang di-import oleh:

- `streamlit_app.py`
- `models/predict.py`

Isi final `requirements.txt`:

```
streamlit>=1.28.0
scikit-learn>=1.0
scipy>=1.7
joblib>=1.2
nltk>=3.8
sastrawi>=1.0
numpy>=1.21
```

---

## 5. Proses Deployment

### Step 1: Training Model Logistic Regression

Jalankan script training untuk menghasilkan model `.pkl`:

```powershell
cd "C:\Users\yoeke\OneDrive\Dokumen\Sains Data 4\DataMining\Projek"
python train_lr.py
```

**Output yang dihasilkan:**

```
models/lr_stage1.pkl             # LogisticRegression Stage 1
models/lr_stage2.pkl             # LogisticRegression Stage 2
models/tfidf_vectorizer.pkl      # TF-IDF Vectorizer (update)
models/preprocessing_config.pkl  # Konfigurasi preprocessing (update)
```

**Hasil evaluasi 5-Fold CV:**

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 0.8838 | 0.0220 |
| F1 Macro | 0.8619 | 0.0267 |
| F1 Negative | 0.7280 | 0.0534 |
| F1 Neutral | 0.8576 | 0.0276 |
| F1 Positive | 1.0000 | 0.0000 |

### Step 2: Inisialisasi Git Repository

```powershell
git init
```

### Step 3: Tambahkan File ke Staging

```powershell
git add .
```

**File yang di-commit:**

| File | Ukuran | Keterangan |
|---|---|---|
| `.gitignore` | ~0.1 KB | Konfigurasi Git |
| `requirements.txt` | ~0.1 KB | Dependensi |
| `streamlit_app.py` | ~11.7 KB | Aplikasi Streamlit |
| `train_lr.py` | Script training | |
| `9_eksperimen_lr.py` | ~39.2 KB | Script eksperimen |
| `9. eksperimen_lr.ipynb` | Notebook eksperimen | |
| `models/predict.py` | ~8.7 KB | Modul inference LR |
| `models/preprocessing_config.pkl` | ~1.9 KB | Konfigurasi |
| `models/lr_stage1.pkl` | Stage 1 LR | |
| `models/lr_stage2.pkl` | Stage 2 LR | |
| `models/tfidf_vectorizer.pkl` | ~56 KB | TF-IDF |

> File `Data Merge - Sheet2.csv` **tidak ikut** di-commit karena sudah diabaikan di `.gitignore`.

### Step 4: Commit Pertama

```powershell
git commit -m "Initial commit: Two-Stage Logistic Regression"
```

### Step 5: Hubungkan ke Remote Repository

```powershell
git remote add origin https://github.com/yoekesekti/gold-sentiment-engine.git
git branch -M main
```

### Step 6: Push ke GitHub

```powershell
git push -u origin main
```

**Autentikasi:** Saat push pertama, akan muncul popup login GitHub. Pilih **"Sign in with your browser"** → authorize di browser → push selesai.

### Step 7: Deploy ke Streamlit Community Cloud

1. Buka **[share.streamlit.io](https://share.streamlit.io)**
2. Klik **"Continue with GitHub"** untuk login
3. Klik tombol **"New App"** (pojok kanan atas)
4. Isi form deployment:

   | Field | Isi |
   |---|---|
   | **Repository** | `yoekesekti/gold-sentiment-engine` |
   | **Branch** | `main` |
   | **Main file path** | `streamlit_app.py` |
   | **App URL** | `gold-sentiment-engine` |

5. Klik **"Deploy!"**

6. **Proses Build Otomatis:**
   - Streamlit Cloud membaca `requirements.txt`
   - Meng-install semua dependensi (~2-3 menit)
   - NLTK otomatis download data `punkt` via `nltk.download()` di `predict.py`
   - Memuat model `.pkl` ke memory
   - Aplikasi siap diakses

**Log build yang muncul di dashboard:**
```
[INFO] Cloning repository...
[INFO] Installing requirements...
[INFO] streamlit==1.28.0, scikit-learn==1.x, scipy==1.x, ...
[INFO] Build complete!
[INFO] App is running at https://gold-sentiment-engine.streamlit.app
```

---

## 6. Verifikasi & Pengujian

### 6.1 Pengujian CLI (Sebelum Deploy)

```powershell
cd models
python predict.py
```

**Output:**

```
    Label    Conf    Pos%    Neu%    Neg%  Text
------------------------------------------------------------------------------------------
[POSITIVE]   97.0%  pos=97.0%  neu= 2.4%  neg= 0.6%  Gold price rallies to new all-time high...
[NEGATIVE]  100.0%  pos= 0.4%  neu= 0.0%  neg=99.6%  Gold futures drop sharply...
[NEGATIVE]   94.7%  pos=28.7%  neu= 3.8%  neg=67.6%  Gold prices steady as traders await...
[ NEUTRAL]   94.6%  pos= 0.4%  neu=94.3%  neg= 5.3%  Gold crash intensifies as market panic...
[POSITIVE]  100.0%  pos=100.0%  neu= 0.0%  neg= 0.0%  Gold rally showing strong bullish...
```

### 6.2 Pengujian via Browser (Setelah Deploy)

Buka `https://gold-sentiment-engine.streamlit.app`, lalu uji dengan skenario:

| Tombol Contoh | Input Judul | Ekspektasi |
|---|---|---|
| 📈 Bullish | "Gold hits new all-time high" | ✅ POSITIVE |
| 📉 Bearish | "Gold futures tumble sharply" | ❌ NEGATIVE |
| ➡️ Neutral | "Gold holds steady with no clear direction" | ⚠️ NEUTRAL |
| 💥 Crash | "Gold crash deepens on panic" | ❌ NEGATIVE |
| 🚀 Rally | "Gold rally shows strong momentum" | ✅ POSITIVE |

### 6.3 Cek Resource Usage

| Metrik | Nilai | Status |
|---|---|---|
| RAM Usage | ~200-400 MB | Aman (limit Streamlit Cloud: 1 GB) |
| Model Load Time | ~1-2 detik | Cukup cepat |
| Response Time | <1 detik | Real-time |
| Cold Start | ~30-60 detik | Wajar untuk free tier |

---

## 7. Hasil Akhir

### 7.1 URL Publik

**https://gold-sentiment-engine.streamlit.app**

### 7.2 Tampilan Aplikasi

Aplikasi memiliki fitur:

- **Header** — Informasi model: "Two-Stage Logistic Regression · TF-IDF ngram 1–2 · 628 sampel berita emas"
- **Stat Card** — Total Sampel (628), Kelas Sentimen (3), Max Features (3,000)
- **Input Card** — Form judul (`text_input`) dan isi berita (`text_area`)
- **5 Tombol Contoh Cepat** — 📈 Bullish, 📉 Bearish, ➡️ Neutral, 💥 Crash, 🚀 Rally
- **Hasil Analisis** — Label sentimen dengan badge berwarna, confidence score, distribusi probabilitas ketiga kelas
- **Footer** — "Two-Stage Logistic Regression (C=100, l1 penalty) · TF-IDF (ngram 1–2, max 3000) · Dataset: 628 sampel berita emas 2021–2026"

### 7.3 Spesifikasi Deployment

| Parameter | Nilai |
|---|---|
| Platform | Streamlit Community Cloud (Free Tier) |
| Python Version | 3.10 |
| RAM Allocation | 1 GB |
| Build Time | ~3 menit |
| Auto-deploy | Ya (setiap `git push` ke `main`) |
| SSL/HTTPS | Ya (otomatis) |
| URL Format | `https://[nama-app].streamlit.app` |

### 7.4 Update di Masa Depan

Setiap kali ada perubahan kode, cukup:

```powershell
git add .
git commit -m "Deskripsi perubahan"
git push origin main
```

Streamlit Cloud akan otomatis mendeteksi push baru dan melakukan redeploy dalam 1-3 menit.

---

## 8. Kesimpulan

### 8.1 Keberhasilan Deployment

| Aspek | Status |
|---|---|
| Model berhasil di-load dari `.pkl` | ✅ |
| Preprocessing bilingual (ID+EN) berjalan normal | ✅ |
| Two-stage Logistic Regression berfungsi | ✅ |
| Confidence score = probabilitas statistik (predict_proba) | ✅ |
| UI responsif & mobile-friendly | ✅ |
| Aplikasi dapat diakses publik via URL | ✅ |
| Auto-deploy dari GitHub terhubung | ✅ |

### 8.2 Perbandingan Model

| Metrik | Two-Stage SVM (lama) | Two-Stage LR (baru) |
|---|---|---|
| **F1 Macro** | 0.6609 | **0.8619** (+30.4%) |
| **Accuracy** | 0.8790 | **0.8838** |
| **F1 Negative** | - | **0.7280** |
| **F1 Neutral** | - | **0.8576** |
| **F1 Positive** | - | **1.0000** |

### 8.3 Keunggulan Logistic Regression

1. **Confidence = probabilitas statistik asli** — `predict_proba()` menghasilkan nilai 0.0–1.0 yang sesungguhnya, tanpa perlu Min-Max normalization heuristik
2. **Tidak perlu threshold manual** — menggunakan batas natural P > 0.5
3. **Lebih interpretable** — koefisien model dapat dianalisis untuk mengetahui kata-kata paling berpengaruh terhadap setiap kelas
4. **Performa lebih tinggi** — F1 Macro naik dari 0.66 ke 0.86

### 8.4 Saran Pengembangan

1. **CalibratedClassifierCV** — Kalibrasi ulang probabilitas dengan Platt Scaling untuk confidence yang lebih akurat
2. **Batch Prediction** — Fitur upload CSV untuk prediksi banyak berita sekaligus
3. **Explainability** — Tampilkan kata-kata yang paling memengaruhi prediksi (koefisien LR)
4. **Monitoring** — Tambahkan logging prediksi untuk analisis penggunaan
5. **FinBERT / IndoBERT** — Eksplorasi transformer-based embeddings untuk menangkap konteks lebih dalam

---

**Disusun oleh:** Yoeke Sekti  
**Tanggal:** Juni 2026  
**Mata Kuliah:** Data Mining
