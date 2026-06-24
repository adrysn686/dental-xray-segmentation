# Dental X-Ray Instance Segmentation

> Automated detection and segmentation of dental structures and pathologies in X-ray images using YOLOv11.  
> Developed at **UCI SEPE Lab**, University of California, Irvine.

---

## 📌 About

This project builds and trains an instance segmentation model on professionally annotated dental X-rays. The model identifies and outlines 11 anatomical structures and pathologies — including root canals, apical lesions, and decay — with pixel-level polygon masks.

Annotations were produced by a team of dental professionals who manually traced outlines around each structure of interest.

---

## 🏷️ Classes

| ID | Class |
|----|-------|
| 0 | Apical Lesion |
| 1 | Main Root |
| 2 | Main Canal |
| 3 | Mesial Root |
| 4 | Mesial Canal |
| 5 | Distal Root |
| 6 | Distal Canal |
| 7 | Palatal Canal |
| 8 | Palatal Root |
| 9 | Root Canal Filling |
| 10 | Decay |

---

## 📁 Repository Structure

```
dental_dataset_v4/
|-- data/
|   |-- images/
|   |   |-- train/          # 436 training images (80%)
|   |   +-- val/            # 110 validation images (20%)
|   |-- labels/
|   |   |-- train/          # matching YOLO label files
|   |   +-- val/
|   +-- data.yaml           # dataset config
|-- runs/
|   +-- segment/
|       +-- train-4/        # training outputs, weights, plots
|-- split_data.py           # train/val split utility
|-- filter_data_v3.py       # class filtering and ID remapping
+-- analysis.py             # dataset analysis utilities
```

---

## 📊 Results

| Model | Epochs | Box mAP50 | Mask mAP50 | Box mAP50-95 | Mask mAP50-95 |
|-------|--------|-----------|------------|--------------|---------------|
| YOLOv11n-seg (nano) | 100 | 0.70 | 0.60 | 0.35 | 0.25 |
| YOLOv11s-seg (small) | 100 | TBD | TBD | TBD | TBD |

- No overfitting observed — train and val losses follow consistent downward trends across all runs
- Training uses **transfer learning** from ImageNet-pretrained YOLO weights

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> Requires Python 3.10+, CUDA-compatible GPU recommended

### 1. Prepare the dataset

Run once to split images and labels into train/val folders:

```bash
python split_data.py
```

### 2. Train

```bash
# Nano model
yolo segment train model=yolo11n-seg.pt data=data/data.yaml epochs=100 imgsz=640

# Small model
yolo segment train model=yolo11s-seg.pt data=data/data.yaml epochs=100 imgsz=640 name=train-small
```

### 3. Run inference on a new X-ray

```bash
yolo segment predict model=runs/segment/train-4/weights/best.pt source=path/to/xray.jpg
```

---

## 🛠️ Scripts

| Script | Description |
|--------|-------------|
| `split_data.py` | Randomly splits dataset 80/20 into train/val, moves images and labels into matching subfolders |
| `filter_data_v3.py` | Filters original 40-class dataset to 11 target classes and remaps class IDs |
| `analysis.py` | Dataset analysis and statistics |

---

## 📋 Dataset Format

Each image has a matching `.txt` label file. Each line represents one object:

```
class_id  x1 y1  x2 y2  x3 y3  ...
```

Coordinates are normalized to `[0, 1]` relative to image dimensions. Polygon points trace the exact outline of each structure.

---

## ⚙️ Training Hardware

- GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
- Training time: ~17 min / 100 epochs (nano), ~22 min / 100 epochs (small)

---

## 🏫 Affiliation

**UCI SEPE Lab**  
University of California, Irvine
