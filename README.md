# studio-2-kelompok-8
# Studio 2 — Klasifikasi Sampah · Kelompok 8 (MENDALAM BELAJAR)

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap         | NIM       | Kontribusi utama |
|----|----------------------|-----------|------------------|
| 1  | Dzaky pramadhani     | 123140208 | Preprocessing dan augmentasi data|
| 2  | M Farisi Suyitno     | 123140152 | Fine-Tuning dan Submission|
| 3  | Awi Septian Prasetyo | 123140201 | Tuning Model dan Evaluasi|

## Ringkasan Pendekatan

Model yang dikirim untuk **Final** adalah *fine-tuning* **ConvNeXt-Tiny pretrained ImageNet** (`timm`, ±27,8 juta parameter). Model awal pada Preview 1 (ResNet-18 pretrained) dijelaskan di bagian "Riwayat Eksperimen" dan sudah tidak dipakai. 

- **Arsitektur**: `timm.create_model("convnext_tiny", pretrained=False, num_classes=5)`, kepala klasifikasi diganti menjadi 5 kelas. Pada pelatihan, bobot pretrained dimuat dari `timm` (kecuali kepala klasifikasi, karena ukurannya berbeda). `get_model()` hanya membangun arsitektur, bobot dimuat oleh `evaluate.py`.
- **Preprocessing**: `Resize(292)` → `CenterCrop(256)` → `ToTensor()` → normalisasi mean/std ImageNet. Transform yang sama dipakai saat validasi dan inferensi.
- **Pelatihan**:
  - Optimizer AdamW (lr 1e-4, weight decay 0,05), *warm-up* 1 epoch lalu *cosine decay*, 20 epoch, batch 24, *mixed precision* (AMP), *gradient clipping* 1,0, seed 42.
  - Loss `SoftTargetCrossEntropy` dengan *Mixup* (α 0,2) dan *CutMix* (α 1,0) berpeluang 0,5 per batch, serta *label smoothing* 0,1.
  - Augmentasi (hanya train): `RandomResizedCrop` (skala 0,5–1,0), *horizontal flip*, `TrivialAugmentWide`, `RandomErasing` (p 0,25).
  - Bobot yang disimpan adalah rata-rata bergerak eksponensial (*EMA*, decay maksimum 0,995) dari bobot model, diambil pada **epoch dengan Macro-F1 tertinggi** di split validasi.
  - Bobot disimpan dalam fp32 (±111,4 MB) dan diunggah dengan **Git LFS** karena melebihi batas 100 MB GitHub.
- **Data tambahan**: **tidak ada**. Seluruh pelatihan memakai train set resmi dari folder Drive asisten (`train/<kelas>/*.jpg`) beserta `train_sources.csv` yang berisi URL sumber tiap gambar.
- **Validasi**: gambar dikelompokkan per domain website (dari `train_sources.csv`), lalu satu domain utuh masuk ke train atau validasi saja, dengan proporsi validasi 15% dan seed 42. Tujuannya agar gambar yang mirip dari website yang sama tidak masuk ke train dan validasi sekaligus, sehingga skor validasi tidak terlalu bagus. Split dibuat oleh `make_group_split.py` menjadi folder `gtrain/` dan `gval/`.

## Hasil

### Model final (ConvNeXt-Tiny)

| Pengukuran | Macro-F1 | Catatan |
|------------|----------|---------|
| Split per domain 85/15 (`gval`, n = 412), epoch terbaik | **0,9709** | Akurasi 0,9733 (401 dari 412 gambar benar). Optimistis: epoch dipilih dari validasi yang sama. |

Hasil per kelas pada `gval` (dihitung dari confusion matrix `evaluate.py`):

| Kelas          | Precision | Recall | F1     |
|----------------|-----------|--------|--------|
| botol_plastik  | 1,0000    | 0,9802 | 0,9900 |
| gelas_plastik  | 0,9643    | 1,0000 | 0,9818 |
| kertas_kardus  | 0,9870    | 0,9870 | 0,9870 |
| saset_kemasan  | 0,9516    | 0,9077 | 0,9291 |
| styrofoam      | 0,9556    | 0,9773 | 0,9663 |

Model final dilatih hanya pada `gtrain` (85% train set resmi), sehingga `gval` tidak ikut pelatihan. Namun bobot terbaik dipilih berdasarkan skor di `gval`, jadi skor tersebut sedikit optimistis dan bukan estimasi yang sepenuhnya bersih. Skor `evaluate.py` pada `gval` sama dengan angka di tabel di atas.

### Skor putaran evaluasi

| Putaran |Model | Macro-F1 lokal | Macro-F1 spreadsheet |
|---------|--------|-------|---------------------------------|------------------------------|
| Preview 1 (14 Okt) | ResNet-18 pretrained | 0.9117 | 0.9150 |
| Preview 2 (21 Okt) | ResNet-18 pretrained | 0.9117 | 0.9150 |
| Final (25 Okt)     | ConvNeXt-Tiny | 0,9709 | - |

### Catatan eksperimen

- Perubahan terbesar dari Preview 1 ke model final adalah pindah ke *backbone* yang lebih kuat (ResNet-18 → ConvNeXt-Tiny), input yang lebih besar (224 → 256), dan resep pelatihan yang lebih kaya (Mixup/CutMix, `TrivialAugmentWide`, `RandomErasing`, EMA, *warm-up*, *gradient clipping*).
- Split acak per gambar diganti dengan split per domain website. Gambar dari website yang sama sering mirip, sehingga split acak membuat skor validasi terlalu tinggi. 
- Selain ConvNeXt-Tiny, kami menyiapkan eksperimen Vision Transformer (`vit_small_patch16_224.augreg_in21k_ft_in1k`, 15 epoch, lr 5e-5) di `model_transformer.ipynb`. 
- Pada `gval`, 11 dari 412 gambar salah prediksi. Kelas terlemah adalah `saset_kemasan` (F1 0,929): 6 dari 65 gambarnya salah, tertukar dengan `styrofoam` (3), `gelas_plastik` (2), dan `kertas_kardus` (1). Kelas ini juga menerima 3 prediksi salah dari kelas lain (2 dari `styrofoam`, 1 dari `botol_plastik`).
- `gelas_plastik` tidak punya gambar yang terlewat (recall 1,0), tetapi presisinya 0,964 karena 3 gambar kelas lain diprediksi sebagai gelas (2 dari `saset_kemasan`, 1 dari `botol_plastik`).
- Hal yang belum diterapkan pada submission: *test-time augmentation*, *ensemble*, dan pembersihan label.

## Riwayat Eksperimen (model awal, Preview 1)

- **Arsitektur**: ResNet-18 pretrained ImageNet (torchvision), kepala klasifikasi diganti menjadi 5 kelas, seluruh layer di-*fine-tune*.
- **Pelatihan**: AdamW (lr 3e-4, weight decay 1e-4), `CosineAnnealingLR`, 10 epoch, batch 32, `CrossEntropyLoss` dengan *label smoothing* 0,1, seed 42. Augmentasi: `RandomResizedCrop` (skala 0,6–1,0), *horizontal flip*, `ColorJitter`. Split validasi lokal 15% secara acak per gambar, dan checkpoint dipilih dari Macro-F1 tertinggi.
- **Hasil lokal Preview 1**: Macro-F1 · akurasi .
- Notebook: `model.ipynb`. Catatan progres: `docs/progress_mingguan.md`.

## Cara Reproduksi

Eksperimen dijalankan di lokal dengan train set resmi di `train/<kelas>/*.jpg` dan `train_sources.csv` di folder utama repo. Keduanya tidak disertakan di repo.

1. Pasang paket dan unduh bobot dari Git LFS:

```bash
pip install -r requirements.txt
git lfs pull
```

2. Buat split per domain (hasilnya `gtrain/` dan `gval/`; hapus kedua folder tersebut jika ingin split ulang):

```bash
python make_group_split.py
```

3. Latih model. Argumen bawaan sudah sesuai konfigurasi final (20 epoch, lr 1e-4, batch 24, seed 42), dan bobot terbaik ditulis ke `submission/model.pth`:

```bash
python train.py --train gtrain --val gval --out submission
```

4. Uji submission:

```bash
python evaluate.py --submission ./submission --data ./gval
```

Paket yang dibutuhkan ada di `requirements.txt`: `torch`, `torchvision`, `pillow`, `numpy`, dan `timm`. Notebook juga memakai `matplotlib`. Submission sendiri memerlukan `torch`, `torchvision`, `pillow`, dan `timm`, dan `submission/requirements.txt` berisi `timm`.

## Struktur Repositori

```
submission/
  model.py                 # get_model() dan get_transform()
  model.pth                # state_dict ConvNeXt-Tiny (fp32, ±111,4 MB, Git LFS), bobot EMA epoch terbaik
  requirements.txt         # timm
evaluate.py                # salinan skrip evaluasi resmi
train.py                   # pelatihan model final (ConvNeXt-Tiny)
make_group_split.py        # split train/validasi per domain website
model.ipynb                # model awal (ResNet-18), Preview 1
model_transformer.ipynb    # eksperimen Vision Transformer
docs/
  progress_mingguan.md     # catatan progres mingguan
requirements.txt           # paket untuk pelatihan dan evaluasi
```