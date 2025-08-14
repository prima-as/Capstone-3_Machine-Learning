# Capstone-3_Machine-Learning

Travel Insurance Claim Prediction – Capstone 3

## Project Introduction

Proyek ini bertujuan membangun **model machine learning** untuk memprediksi kemungkinan nasabah melakukan **klaim asuransi perjalanan**. Dataset berisi lebih dari **44 ribu transaksi** dengan informasi terkait agen, produk, destinasi, durasi, penjualan, komisi, umur, dan status klaim.

### Latar Belakang

Klaim asuransi perjalanan merupakan kejadian yang jarang (sekitar **1,5%** dari seluruh data). Hal ini menimbulkan tantangan **class imbalance**, di mana model cenderung memprediksi kelas mayoritas ("tidak klaim"), sehingga mengabaikan kasus klaim yang sebenarnya terjadi.

Dampaknya, perusahaan:

**-** Kehilangan potensi deteksi dini terhadap risiko klaim.

**-** Berisiko underestimasi cadangan klaim dan salah strategi underwriting.

**-** Tidak optimal dalam memutuskan premi atau kebijakan penawaran produk.

### Tujuan Proyek

**1.** Mengembangkan model klasifikasi yang dapat **mendeteksi klaim** secara efektif meski data sangat tidak seimbang.

**2.** Mengutamakan metrik yang relevan untuk kasus imbalance (**F2 Score, Recall**, dan **Precision**).

**3.** Menyediakan pipeline pemrosesan data yang reproducible untuk implementasi jangka panjang.

**4.** Memberikan wawasan bisnis berbasis analisis data dan hasil model.

---

## Project Workflow

### 1. Business Problem Understanding

**-** **Target**: memprediksi `Claim` (Yes/No).

**-** **Prioritas Metrik**: F2-Score (memberi bobot lebih pada Recall).

**-** **Alternatif Metrik**: Recall dan Precision.

**-** **Pendekatan**: cost-sensitive learning untuk meminimalkan kerugian dari False Negative (klaim yang lolos tanpa terdeteksi).

### 2. Exploratory Data Analysis (EDA)

**-** **Numerik**:

**  **- `Duration`, `Net Sales`, `Commision (in value)`, `Age` → ditemukan outlier dan skewness.

**-** **Kategorikal**:

**  **- `Agency`, `Product Name`, `Destination` → banyak kategori unik.

**-** **Missing Value**:

**  **- `Gender` kosong ± 71%, tidak signifikan bagi model.

**-** **Class Imbalance**:

**  **- `Yes` = 677 (1,5%), `No` = 43.651 (98,5%).

### 3. Data Cleaning

**-** Menghapus duplikat.

**-** Menangani outlier (hapus beberapa outlier).

**-** Memutuskan drop `Gender` karena terlalu banyak missing dan `Commision (in value)` karena tidak relevan sebagai fitur.

### 4. Feature Engineering

Pada tahap ini dilakukan penambahan dua fitur baru untuk meningkatkan kemampuan model mengenali pola:

1. **`duration_category`**

   - Kategori hasil binning dari `Duration` ke dalam beberapa interval (short, medium, long).
   - Tujuan: menangkap pengaruh durasi perjalanan terhadap probabilitas klaim.
2. **`agency_product`**

   - Penggabungan `Agency` dan `Product Name` menjadi satu fitur baru.
   - Tujuan: mengidentifikasi kombinasi spesifik agen–produk yang berpotensi berisiko tinggi atau rendah.

Fitur tambahan ini diikutkan dalam preprocessing dan modeling seperti fitur lainnya.

### 5. Preprocessing

**-** **Numerik**: Scaling (`Robust Scaler`).

**-** **Kategorikal**: One-Hot Encoding atau Binary Encoding untuk fitur dengan banyak kategori.

**-** **Imbalance**:** **

**  **- ROS (Random oversampling), RUS (Random undersampling), SMOTE, NEARMISS.

### 6. Modeling

**-** **Crossval Model**: Logistic Regression, KNN, Decision Tree, Voting, Stacking, Bagging, Boosting, Gradient Boost, XG Boost.

**-**  **StratifiedKFold**: Menjaga proporsi kelas

**-** **Final Model**: Gradient Boost dengan hyperparameter tuning (GridSearch/RandomizedSearch).

### 7. Evaluation

**-** **Confusion Matrix**

**-** **F2-score, Precision dan Recall**.

**-** **Error Analysis**: profil nasabah yang menjadi False Negative dan False Positive untuk perbaikan fitur.

---

## Conclusion

Berdasarkan hasil confusion matrix dari model **Gradient Boost Classifier**, dapat disimpulkan bahwa jika model ini digunakan untuk memfilter nasabah yang akan dilakukan pemeriksaan klaim, maka model mampu menghindari pemeriksaan pada sebagian besar nasabah yang tidak mengajukan klaim (**recall kelas 0 = 85%**) sehingga berpotensi menghemat biaya pengecekan, sambil tetap menangkap sebagian besar nasabah yang benar-benar mengajukan klaim (**recall kelas 1 = 61%**).

Namun, konsekuensinya adalah tingkat ketepatan prediksi klaim masih rendah (**precision kelas 1 = 7%**) yang mengakibatkan kurang tepatnya penilaian dalam menentukan premi, sehingga berpotensi membuat nasabah berpindah ke kompetitor lain.

Analisis biaya dengan asumsi:

- Cost untuk **False Positive (FP)** = USD 100
- Cost untuk **False Negative (FN)** = USD 350

**Tanpa Model (semua dianggap klaim):**

- FP = 7.664 nasabah → biaya FP = 7.664 × 100 USD = USD 766.400

**Dengan Model:**

- FP = 1.144 nasabah → biaya FP = 1.144 × 100 USD = USD 114.400
- FN = 52 nasabah → biaya FN = 52 × 350 USD = USD 18.200
- Total biaya = USD 114.400 + USD 18.200 = USD 132.600

**Penghematan biaya** dibandingkan tanpa model ≈ **USD 652.000**, yang dapat diproyeksikan sebagai keuntungan perusahaan.

---

## Recommendation

- Memastikan kelengkapan data pada semua entitas nasabah, misalnya jika ada kolom yang kosong seperti `Net Sales` atau `Duration`, diisi dengan informasi pengganti yang relevan daripada dibiarkan kosong.
- Menambah jumlah data untuk nasabah yang klaim agar model dapat memahami lebih baik karakteristiknya.
- Menambahkan fitur baru yang relevan, seperti riwayat jumlah perjalanan sebelumnya.
- Mencoba algoritma machine learning lain yang mungkin menghasilkan kinerja lebih baik setelah tuning.
- Melakukan hyperparameter tuning lanjutan dan membandingkan beberapa teknik oversampling (SMOTE, SMOTENC, ADASYN) maupun undersampling.
- Melakukan analisis kesalahan (false positive dan false negative) untuk menemukan pola yang bisa diperbaiki melalui penambahan atau pengolahan fitur.

---

## Limitation

- **Imbalance Data** – Distribusi klaim (class 1) jauh lebih sedikit dibandingkan non-klaim (class 0), sehingga model cenderung bias memprediksi kelas mayoritas.
- **F2 Score Tinggi Karena Recall** – Nilai F2 yang baik terutama didorong oleh recall tinggi, namun precision rendah sehingga potensi false positive masih besar.
- **Fitur Tidak Informatif** – Beberapa fitur memiliki kontribusi kecil berdasarkan analisis feature importance, sehingga model mungkin terpengaruh noise.
- **Keterbatasan Informasi Fitur** – Data yang digunakan belum mencakup variabel penting lain seperti histori transaksi lengkap, profil pelanggan, atau faktor eksternal.
- **Sensitif Terhadap Perubahan Data** – Jika distribusi data berubah (misalnya tren klaim berubah), performa model dapat menurun tanpa retraining.
