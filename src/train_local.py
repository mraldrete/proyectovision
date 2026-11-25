"""Entrenamiento YOLOv8n v4 (más datos, STRIDE=128)."""
import random
import numpy as np
import torch
from ultralytics import YOLO

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

model = YOLO("yolov8n.pt")

results = model.train(
    data="data/bona_yolo_v4/data.yaml",
    epochs=150,
    imgsz=640,
    batch=8,
    patience=40,
    lr0=0.01,
    lrf=0.01,
    cls=1.5,
    mosaic=1.0,
    mixup=0.1,
    scale=0.7,
    fliplr=0.5,
    seed=SEED,
    device=device,
    project="runs_bona",
    name="yolov8n_bona_v4",
    exist_ok=True,
    plots=True,
)