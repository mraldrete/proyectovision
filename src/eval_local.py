from ultralytics import YOLO
best = YOLO("runs_bona/yolov8n_bona_v3/weights/best.pt")
metrics = best.val(
    data="data/bona_yolo_v3/data.yaml",
    split="test",
    imgsz=640,
    conf=0.001,
    iou=0.6,
    plots=True,
    project="runs_bona",
    name="eval_test_v3",
    exist_ok=True,
)
print(f"mAP@0.5     = {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95= {metrics.box.map:.4f}")
print(f"Precision   = {metrics.box.mp:.4f}")
print(f"Recall      = {metrics.box.mr:.4f}")