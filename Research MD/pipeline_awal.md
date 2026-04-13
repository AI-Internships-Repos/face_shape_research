# Face Shape Classification — Hybrid Pipeline Summary

Dataset utama:
- Kaggle `niten19/face-shape-dataset`
- Total 5000 image
- 5 class: `Heart`, `Oblong`, `Oval`, `Round`, `Square`
- Per class: 1000 image
- Split bawaan: 800 train + 200 test per class
- License: CC0/Public Domain

Catatan:
- Karena dataset bawaan sudah punya split, tetap lebih aman bikin `validation split` dari train set.
- Kalau mau evaluasi yang lebih jujur, cek juga kemungkinan `identity leakage` karena dataset berisi celebrity faces.

---

## Approach 1 — Hybrid klasik mirip paper Pasupa et al. (2019)

Tujuan:
Gabungkan **fitur geometri wajah** + **fitur deep learning** lalu klasifikasi final pakai model klasik.

### Pipeline
1. **Input image**
2. **Face detection**
   - Deteksi area wajah
3. **Face alignment**
   - Rapikan pose wajah supaya mata/hidung/mulut lebih sejajar
4. **Landmark extraction**
   - Ambil titik-titik penting wajah
5. **Hand-crafted geometric feature extraction**
   - Hitung fitur rasio dan bentuk dari landmark, misalnya:
     - face width / face height
     - forehead width
     - cheekbone width
     - jaw width
     - chin curvature / jaw angle
6. **Deep feature extraction**
   - Crop wajah masuk ke backbone pretrained face/CNN
   - Ambil embedding deep feature
7. **Feature fusion**
   - Gabungkan:
     - geometric features
     - deep features
8. **Classifier**
   - Pakai SVM / kernel-based classifier / DL
9. **Output**
   - Prediksi salah satu dari 5 bentuk wajah

### Algoritma inti
- **Face detector**: RetinaFace
- **Landmark detector**: 68-point / dense landmarks
- **Geometric feature**: rasio jarak, sudut, contour descriptors
- **Deep feature extractor**: VGG-Face style / CNN pretrained
- **Fusion**:
  - simple concatenation, atau
  - **Multiple Kernel Learning (MKL)**
- **Final classifier**: **SVM**
- **Hyperparameter search**: bisa pakai PSO kalau mau meniru paper lama

### Kenapa approach ini bagus
- Lebih **interpretable**
- Bisa dijelaskan kenapa wajah masuk oval/round/square
- Cocok kalau mau bikin sistem yang bisa kasih reasoning ke user

### Kekurangan
- Sangat tergantung kualitas landmark
- Masih sensitif ke pose, occlusion rambut, dan kontur wajah yang tertutup
- Akurasi biasanya kalah dari pipeline deep modern murni

### Rekomendasi implementasi praktis
- Pakai **RetinaFace** untuk deteksi + alignment
- Pakai **MediaPipe / FaceMesh / landmark detector** untuk fitur geometri
- Pakai **EfficientNetV2-S embedding** atau backbone pretrained lain untuk deep branch
- Fusion paling simpel:
  - `concat([geom_features, deep_embedding]) -> MLP/SVM`

---

## Approach 2 — Hybrid modern: 3D-assisted + Deep Learning

Tujuan:
Jangan cuma lihat wajah dari **2D frontal image**, tapi tambahkan **3D facial geometry extraction** di awal supaya informasi kontur samping, jawline, dan depth wajah ikut terbaca.

### Pipeline
1. **Input image**
2. **Face detection**
3. **Face alignment**
4. **3D face extraction / 3D face geometry estimation**
   - Ambil **468 3D landmarks** atau reconstruct mesh wajah
   - Output:
     - depth-aware facial landmarks
     - face mesh
     - head pose
     - contour geometry
5. **3D geometric feature extraction**
   - Hitung fitur bentuk dari mesh / 3D landmarks:
     - forehead width
     - cheekbone prominence
     - jaw width
     - chin sharpness
     - face length
     - contour curvature
     - side-depth consistency
6. **2D deep branch**
   - Image wajah hasil alignment masuk ke CNN modern
   - Backbone ambil embedding visual global
7. **Feature fusion**
   - Gabungkan:
     - 3D geometry features
     - deep image embedding
8. **Classification head**
   - MLP / Softmax classifier
9. **Output**
   - 5 kelas face shape

### Algoritma inti
- **Face detector**: RetinaFace
- **3D extraction**:
  - paling ringan: **MediaPipe Face Mesh** (468 3D landmarks)
  - kalau mau lebih serius: **3D face reconstruction** model seperti DECA-class pipeline
- **Deep backbone**:
  - **EfficientNetV2-S** sebagai backbone utama
- **Fusion**:
  - concatenation + MLP
  - atau attention fusion
- **Classifier head**:
  - Dense layer + Softmax
- **Loss**:
  - Cross-Entropy
  - opsional: label smoothing

### Kenapa approach ini lebih kuat
- Kontur wajah tidak cuma dibaca dari tampak depan
- Lebih bagus untuk:
  - jawline
  - chin shape
  - cheekbone prominence
- Lebih tahan terhadap ambiguity antara:
  - round vs square
  - oval vs heart

### Kekurangan
- Pipeline lebih kompleks
- Perlu komputasi lebih besar
- Kualitas pseudo-3D dari single image tetap tidak sesempurna scan 3D sungguhan

### Rekomendasi implementasi praktis
- **Versi ringan**:
  - RetinaFace
  - MediaPipe Face Mesh
  - EfficientNetV2-S
  - concat 3D ratios + CNN embedding
  - classifier MLP
- **Versi riset lebih serius**:
  - RetinaFace
  - DECA / 3DMM-based face reconstruction
  - EfficientNetV2-S / Swin / ViT
  - fusion head + ablation study

---

## Saran final yang paling masuk akal

### Kalau targetmu:
#### 1. **Cepat, stabil, gampang dijelaskan**
Pilih **Approach 1**
- landmark/geometric + deep feature fusion
- cocok untuk baseline paper / skripsi / eksperimen awal

#### 2. **Akurasi lebih tinggi dan lebih modern**
Pilih **Approach 2**
- 3D extraction dulu
- lalu gabung dengan CNN backbone modern
- ini lebih cocok kalau tujuanmu benar-benar bikin sistem face-shape yang lebih robust

---

## Recommended final stack

### Baseline stack
- Detector: **RetinaFace**
- Landmark: **MediaPipe Face Mesh**
- Backbone: **EfficientNetV2-S**
- Fusion: `concat(3D_geom, deep_embedding)`
- Head: `MLP + Softmax`

### Kenapa stack ini
- RetinaFace: deteksi dan alignment kuat
- MediaPipe Face Mesh: kasih **468 landmark 3D** real-time
- EfficientNetV2-S: efisien dan di paper DOAJ performanya sangat bagus untuk task face shape
- Fusion branch: bikin model tidak terlalu bergantung pada tekstur 2D saja

---

## Ringkasan 1 kalimat per approach

### Approach 1
`2D face -> landmark geometry + deep features -> fusion -> SVM/MLP -> face shape`

### Approach 2
`2D face -> detect & align -> 3D face geometry extraction -> combine with CNN embedding -> fusion head -> face shape`

---

## Pilihan terbaik untuk project ini
**Approach 2** lebih recommended sebagai final model,  
sedangkan **Approach 1** dipakai sebagai baseline pembanding.