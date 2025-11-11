#  Analisis Sentimen Review Aplikasi Spotify (UTS NLP)

Repositori ini berisi pengerjaan Ujian Tengah Semester (UTS) mata kuliah *Natural Language Processing* (NLP).

**Nama:** Angga Rianto Sudrajat
**NIM:** 10222173

---

## 📄 Deskripsi Proyek

Proyek ini bertujuan untuk melakukan klasifikasi sentimen biner (Positif/Negatif) pada dataset *Spotify App Reviews 2022*.

Tujuan utamanya adalah membandingkan performa dua jenis model:
1.  **Model Machine Learning (ML):** `LogisticRegression` yang dilatih *dari nol* pada data training.
2.  **Model Deep Learning (DL):** `RoBERTa` (model `SamLowe/roberta-base-go_emotions`) sebagai model *pre-trained* (inferensi) yang canggih.

---

## 📊 Dataset

* **Sumber:** [Spotify App Reviews 2022 (Kaggle)](https://www.kaggle.com/datasets/mfaaris/spotify-app-reviews-2022)
* **File:** `reviews.csv`
* **Label Awal:** Rating Bintang (1-5)

### Pemrosesan Label
Data ini sangat *imbalanced* (tidak seimbang), terutama pada `Rating 3` (Netral). Untuk mendapatkan model klasifikasi biner yang akurat, **data dengan `Rating 3` (Netral) dibuang**.

Label biner dibuat dengan pemetaan berikut:
* **Negatif (0):** Rating 1 dan 2
* **Positif (1):** Rating 4 dan 5

---

## 🤖 Model dan Metodologi

### 1. Model ML: `LogisticRegression`
Model ini dilatih untuk tugas klasifikasi biner (Positif/Negatif).

* **Preprocessing:**
    1.  Teks diubah ke *lowercase*.
    2.  Tanda baca, URL, dan karakter non-alfabet dihapus.
    3.  *Stopwords* (kata-kata umum) dibuang.
    4.  Dilakukan *Lemmatization* (mengubah kata ke bentuk dasar).
    5.  Teks diubah menjadi vektor angka menggunakan **TF-IDF**.
* **Tuning:**
    Untuk mendapatkan akurasi tertinggi, model di-tuning menggunakan `GridSearchCV` untuk mencari parameter terbaik (terutama `C` dan `ngram_range`).

### 2. Model DL: `RoBERTa` (GoEmotions)
Model ini digunakan sebagai pembanding Deep Learning. Kita tidak melatihnya, tapi langsung menggunakan kemampuannya (inferensi) untuk dua tugas:

* **Tugas 1 (Evaluasi Biner):**
    Model ini memprediksi 28 emosi. Kita "memetakan" emosi-emosi tersebut ke label biner (misal: `joy`, `love`, `admiration` = Positif; `anger`, `sadness`, `disgust` = Negatif) untuk membuat *Classification Report*.
* **Tugas 2 (Analisis Kualitatif):**
    Kita menggunakan 28 label emosi untuk analisis mendalam: mencari emosi apa yang paling dominan di setiap *review* dan membandingkannya dengan `Rating` bintang aslinya.

---

## 📈 Hasil dan Evaluasi

Evaluasi utama berfokus pada perbandingan *F1-Score (Weighted)* antara kedua model pada *test set*.

### Perbandingan Akurasi (ML vs DL)

| Model | Tipe | Akurasi | F1-Score (Weighted) |
| :--- | :--- | :---: | :---: |
| **LogisticRegression** | ML (Dilatih & Di-Tuning) | **89%** | **0.89** |
| **RoBERTa (SamLowe)** | DL (Pre-trained) | 85% | 0.85 |

### Analisis Hasil
* **`LogisticRegression` (ML)** secara teknis "menang" (89% vs 85%) dalam tugas klasifikasi biner Positif/Negatif. Hal ini wajar karena model ini **dilatih secara spesifik** (spesialis) pada dataset ini untuk satu tugas tersebut.
* **`RoBERTa` (DL)** adalah model generalis yang tidak dilatih di data ini. Fakta bahwa model ini bisa mencapai akurasi **85%** "tanpa konteks" menunjukkan betapa kuatnya arsitektur Transformer.

### Analisis Emosi Mendalam (Kekuatan RoBERTa)
Kekuatan sesungguhnya dari model DL terlihat pada analisis kualitatif 28 emosinya. Plot di notebook (Cell 10) menunjukkan:
* Review **Bintang 1 & 2** didominasi oleh emosi: `anger`, `disappointment`, `annoyance`.
* Review **Bintang 4 & 5** didominasi oleh emosi: `admiration`, `joy`, `love`, `approval`, `gratitude`.

Ini membuktikan bahwa model RoBERTa benar-benar "mengerti" nuansa di balik sentimen Positif atau Negatif.

---

## 🚀 Cara Menjalankan Proyek

1.  Clone repositori ini.
2.  Pastikan Anda memiliki environment Conda (misal: `uts-nlp`).
3.  Install semua *library* yang dibutuhkan (terutama `pandas`, `scikit-learn`, `tensorflow`, `transformers`, `torch`, `tf-keras`, `nltk`, `seaborn`, `matplotlib`).
    ```bash
    pip install pandas scikit-learn tensorflow transformers torch tf-keras nltk seaborn matplotlib wordcloud
    ```
4.  Jalankan notebook `10222173_Angga Rianto Sudrajat_AS.ipynb` secara berurutan dari cell pertama.

**Catatan:** Cell 8 (Tuning `GridSearchCV`) dan Cell 10 (Analisis RoBERTa pada 1000 sampel) akan memakan waktu cukup lama untuk dieksekusi.
