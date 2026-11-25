"""Inferencia: carga YOLO y predice sobre una imagen."""
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

WEIGHTS = Path("runs_bona/yolov8n_bona_v4/weights/best.pt")


class TreeDetector:
    def __init__(self, weights=WEIGHTS, conf=0.01, imgsz=640):
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