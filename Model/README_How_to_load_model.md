# Panduan Menjalankan Model Machine Learning Travel Insurance di Mesin Lain

Model ini dibuat menggunakan **FunctionTransformer** dengan fungsi kustom untuk tahap *data preparation*.
Agar file model `.sav` dapat dijalankan di mesin lain, fungsi-fungsi tersebut **harus tersedia** di environment saat model di-*load*.

---

## 📂 File yang Disertakan

1. **custom_transformers.py**Berisi seluruh fungsi yang digunakan dalam pipeline:

   - `fill_duration` → mengisi nilai **Duration** yang bernilai 0.
   - `fill_netsales` → mengisi nilai **Net Sales** yang kosong.
   - `fill_age` → mengisi nilai **Age** yang 118 atau kosong.
   - `imput_all` → menjalankan semua proses imputasi di atas secara berurutan.
   - `feature_engineer` → membuat kolom baru `duration_category` dan `agency_product`.
2. **load_model_with_custom_funcs.py**Contoh skrip Python untuk mem-*load* file `Travel Insurance ML.sav` di mesin lain.
3. **Travel Insurance ML.sav**
   File model hasil training.

---

## 🔹 Langkah Menjalankan Model di Mesin Lain

1. Pastikan file berikut berada di folder yang sama:

   ```
   custom_transformers.py
   load_model_with_custom_funcs.py
   Travel Insurance ML.sav
   ```
2. Jalankan perintah berikut di terminal:

   ```bash
   python load_model_with_custom_funcs.py
   ```
3. Jika berhasil, akan muncul daftar *pipeline steps* dan tipe model yang digunakan.
