# studio-2-kelompok-8
# Studio 2 — Klasifikasi Sampah · Kelompok 8 (MENDALAM BELAJAR)

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Dzaky pramadhani | 123140208 |             |
| 2  | M Farisi Suyitno | 123140152 |             |
| 3  | Awi Septian Prasetyo | 123140201 |         |

## Ringkasan Pendekatan

Model yang dikirim untuk **Final** adalah *fine-tuning* **ConvNeXt-Tiny pretrained ImageNet** (`timm`, ±27,8 juta parameter). Model awal pada Preview 1 (ResNet-18 pretrained) dijelaskan di bagian "Riwayat Eksperimen" dan sudah tidak dipakai. <ISI: model yang dipakai pada Preview 2>
 
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

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | | | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

## Cara Reproduksi
 
Eksperimen dijalankan di <ISI: Google Colab / lokal, jenis GPU> dengan train set resmi di `train/<kelas>/*.jpg` dan `train_sources.csv` di folder utama repo. Keduanya tidak disertakan di repo.
 
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
