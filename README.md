Dental X-Ray Instance Segmentation


Automated detection and segmentation of dental structures and pathologies in X-ray images using YOLOv11.

Developed at UCI SEPE Lab, University of California, Irvine.




📌 About

This project builds and trains an instance segmentation model on professionally annotated dental X-rays. The model identifies and outlines 11 anatomical structures and pathologies — including root canals, apical lesions, and decay — with pixel-level polygon masks.

Annotations were produced by a team of dental professionals who manually traced outlines around each structure of interest.


🏷️ Classes

IDClass0Apical Lesion1Main Root2Main Canal3Mesial Root4Mesial Canal5Distal Root6Distal Canal7Palatal Canal8Palatal Root9Root Canal Filling10Decay


📁 Repository Structure

dental_dataset_v4/
├── data/
│   ├── images/
│   │   ├── train/          # 436 training images (80%)
│   │   └── val/            # 110 validation images (20%)
│   ├── labels/
│   │   ├── train/          # matching YOLO label files
│   │   └── val/
│   └── data.yaml           # dataset config
├── runs/
│   └── segment/
│       └── train-4/        # training outputs, weights, plots
├── split_data.py           # train/val split utility
├── filter_data_v3.py       # class filtering and ID remapping
└── analysis.py             # dataset analysis utilities


🚀 Getting Started

Prerequisites

bashpip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


Requires Python 3.10+, CUDA-compatible GPU recommended



1. Prepare the dataset

Run once to split images and labels into train/val folders:

bashpython split_data.py

2. Train

bash# Nano model
yolo segment train model=yolo11n-seg.pt data=data/data.yaml epochs=100 imgsz=640

# Small model
yolo segment train model=yolo11s-seg.pt data=data/data.yaml epochs=100 imgsz=640 name=train-small

3. Run inference on a new X-ray

bashyolo segment predict model=runs/segment/train-4/weights/best.pt source=path/to/xray.jpg


🛠️ Scripts

ScriptDescriptionsplit_data.pyRandomly splits dataset 80/20 into train/val, moves images and labels into matching subfoldersfilter_data_v3.pyFilters original 40-class dataset to 11 target classes and remaps class IDsanalysis.pyDataset analysis and statistics


📋 Dataset Format

Each image has a matching .txt label file. Each line represents one object:

class_id  x1 y1  x2 y2  x3 y3  ...

Coordinates are normalized to [0, 1] relative to image dimensions. Polygon points trace the exact outline of each structure.


⚙️ Training Hardware


GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
Training time: ~17 min / 100 epochs (nano), ~22 min / 100 epochs (small)



🏫 Affiliation

UCI SEPE Lab

University of California, Irvine
