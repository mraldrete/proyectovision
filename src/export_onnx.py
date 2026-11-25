"""Exporta best.pt a ONNX y mide latencia antes/después."""
import time, numpy as np
from ultralytics import YOLO

model = YOLO("runs_bona/yolov8n_bona_v4/weights/best.pt")
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

for _ in range(5):
    model.predict(img, imgsz=640, verbose=False, device="mps")
t0 = time.time()
for _ in range(20):
    model.predict(img, imgsz=640, verbose=False, device="mps")
t_pt = (time.time() - t0) / 20 * 1000
print(f"PyTorch MPS: {t_pt:.1f} ms/img")

path = model.export(format="onnx", imgsz=640, simplify=True)
print("ONNX:", path)

import onnxruntime as ort
import numpy as np

# Latencia ONNX
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
inp = {sess.get_inputs()[0].name: np.random.rand(1, 3, 640, 640).astype(np.float32)}

for _ in range(5):
    sess.run(None, inp)
t0 = time.time()
for _ in range(20):
    sess.run(None, inp)
t_onnx = (time.time() - t0) / 20 * 1000
print(f"ONNX CPU: {t_onnx:.1f} ms/img")