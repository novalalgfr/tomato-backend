# Tomato.Logy - AI Backend API

Backend service untuk sistem **Deteksi Penyakit Daun Tomat** yang dibangun menggunakan **Flask** dan **YOLOv11 (Ultralytics)**. Layanan ini bertugas menerima sampel citra daun dari client, memprosesnya menggunakan model Object Detection canggih (YOLOv11), dan mengembalikan hasil berupa gambar yang sudah dianotasi dengan Bounding Box serta label kelas penyakit.

## 🧠 Tech Stack

- **Language:** Python 3.9+
- **Framework:** Flask & Flask-CORS
- **AI Core:** Ultralytics YOLOv11 (PyTorch)
- **Computer Vision:** OpenCV (cv2)
- **Architecture:** YOLOv11n (Nano) - Custom Trained on Tomato Leaf Disease Dataset

## 📂 Struktur Folder

```text
backend/
├── app.py                   # Entry point server Flask
├── requirements.txt         # Daftar library dependencies
├── models/                  # Folder penyimpanan file model
│   └── best.pt              # File bobot model YOLOv11 hasil training
├── utils/                   # Modul logika aplikasi
│   ├── image_processing.py  # Inference logic & plotting bounding box
│   └── model_loader.py      # Singleton loader untuk model YOLO
├── static/                  # Folder file statis (akses publik)
│   └── results/             # Menyimpan hasil gambar output (sementara)
└── uploads/                 # Menyimpan file upload sementara
```

## 🚀 Cara Menjalankan (Local)

Pastikan Python sudah terinstal di komputer Anda.

### 1. Clone Repository

```bash
git clone https://github.com/novalalgfr/tomato-backend.git
cd frontend
```

### 2. Setup Virtual Environment

Disarankan menggunakan virtual environment agar library tidak konflik.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Model

Pastikan file model `best.pt` sudah diletakkan di dalam folder `models/`.

> **Catatan:** File model tidak disertakan di repository ini karena ukurannya yang besar dan batasan GitHub. Silakan hubungi pengembang untuk mendapatkan file model.

### 5. Jalankan Server

```bash
python app.py
```

Server akan berjalan di `http://127.0.0.1:5000`.

## 📡 API Endpoints

### `POST /predict`

Menerima upload gambar dan mengembalikan hasil deteksi.

- **URL:** `http://127.0.0.1:5000/predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `file`: File gambar (JPG/PNG)

**Response (JSON):**

```json
{
  "status": "success",
  "message": "Analysis Complete",
  "detect_url": "[http://127.0.0.1:5000/static/results/detect_a1b2c3d4.jpg](http://127.0.0.1:5000/static/results/detect_a1b2c3d4.jpg)",
  "detections_count": 2
}
```

## 📝 Lisensi

Project ini dikembangkan untuk keperluan edukasi dan penelitian.

---

**© 2026 Tomato.Logy System**
