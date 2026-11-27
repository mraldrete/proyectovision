"""Inferencia: carga YOLO y predice sobre una imagen.

Modelo final: YOLO11n fine-tuned sobre HARV v2.
  - mAP50 (test) = 0.587, precision = 0.645, recall = 0.555
  - Entrenado con imgsz=512, por eso la inferencia usa 512.
  - conf por defecto 0.3 (el modelo es preciso; no necesita umbral bajo).
"""
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

WEIGHTS = Path("runs_harv/yolo11n_harv_v2/weights/best.pt")


class TreeDetector:
    def __init__(self, weights=WEIGHTS, conf=0.3, imgsz=512):
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, image_path):
        """Devuelve (imagen_anotada PIL, n_arboles, lista_boxes)."""
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        boxes = []
        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            c = float(b.conf[0])
            boxes.append((x1, y1, x2, y2, c))
        annotated_bgr = results.plot()
        annotated_rgb = annotated_bgr[..., ::-1]
        return Image.fromarray(annotated_rgb), len(boxes), boxes