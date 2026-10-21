# 🎯 Circle Object Detection with TensorFlow

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.6%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end computer vision pipeline for detecting synthetic and real-world geometric circle primitives. This project parses custom CSV coordinate annotations, normalizes geometric bounding boxes, converts data into optimized binary `TFRecord` formats, and prepares data for object detection model training using the TensorFlow Object Detection API or YOLO models.

---

## 📐 Bounding Box Geometry & Normalization

Bounding box coordinates $(x_{min}, y_{min}, x_{max}, y_{max})$ provided in pixel space are transformed and normalized to relative continuous coordinates $[0.0, 1.0]$ based on image width $W$ and height $H$:

$$x_{norm} = \frac{\max(0, \min(W, x_{min} + \text{box\_length}))}{W}$$

$$y_{norm} = \frac{\max(0, \min(H, y_{min} + \text{box\_width}))}{H}$$

---

## 🏗 System Architecture

```text
       ┌────────────────────────┐
       │   Kaggle CSV Dataset   │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ AST Coordinate Parsing │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  BBox Normalization &  │
       │    Boundary Clipping   │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  TFRecord Binary Dump  │
       └───────────┬────────────┘
                   │
                   ▼
 ┌──────────────────────────────────┐
 │ TensorFlow Object Detection API  │
 └──────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
circle-object-detection/
│
├── dataset/                      # Local dataset storage (git-ignored)
│   ├── train_image/              # Raw image files (.png)
│   └── result.csv                # CSV containing bounding box annotations
│
├── scripts/
│   └── generate_tfrecord.py      # TFRecord conversion & validation pipeline
│
├── label_map.pbtxt               # Protocol buffer label mapping file
├── requirements.txt              # Project dependencies
├── .gitignore                    # Excluded raw datasets and build outputs
└── README.md                     # Documentation
```

---

## 📑 Dataset Specification

The ground truth dataset annotations (`result.csv`) are formatted with the following schema:

| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `img_name` | String | Filename of target image | `img1.png` |
| `box_corner` | String | Tuple string representing top-left $(x_{min}, y_{min})$ | `"(334, 295)"` |
| `box_length` | Float / Int | Width of the bounding box | `50.0` |
| `box_width` | Float / Int | Height of the bounding box | `50.0` |
| `shape` | String | Target object label | `circle` |

---

## 🚀 Quickstart Guide

### 1. Clone Repository & Setup Environment

```bash
git clone https://github.com/your-username/circle-object-detection.git
cd circle-object-detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset from Kaggle

Ensure your `kaggle.json` credentials are configured in `~/.kaggle/`:

```bash
kaggle datasets download -d aman2000jaiswal/circle-object-detection -p dataset --unzip
```

### 3. Generate Label Map and TFRecord Binary

Run the dataset serialization script:

```bash
python scripts/generate_tfrecord.py
```

Upon successful run, `label_map.pbtxt` and `train.record` will be created in your root directory.

---

## ⚡ Future Enhancements & Scaling

- [ ] **Multi-Bounding Box Aggregation:** Group `df.groupby('img_name')` to support images containing multiple target circles.
- [ ] **YOLO Converter:** Add export script for Ultralytics YOLO PyTorch annotation format.
- [ ] **Data Augmentation Pipeline:** Integrate `albumentations` for scale jittering, rotation, and lighting variations.
- [ ] **TFLite Export:** Quantize model weights for edge deployment on Raspberry Pi or mobile devices.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.⚡ Future Enhancements & Scaling[ ] Multi-Bounding Box Aggregation: Group df.groupby('img_name') to support images containing multiple target circles.[ ] YOLO Converter: Add export script for Ultralytics YOLO PyTorch annotation format.[ ] Data Augmentation Pipeline: Integrate albumentations for scale jittering, rotation, and lighting variations.[ ] TFLite Export: Quantize model weights for edge deployment on Raspberry Pi or mobile devices.
