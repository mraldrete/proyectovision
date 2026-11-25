"""Demo en tiempo real: webcam o video. Uso: python app/demo.py --source 0"""
import argparse, time, cv2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--weights", default="runs_bona/yolov8n_bona_v4/weights/best.pt")
args = ap.parse_args()

model = YOLO(args.weights)
src = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(src)
t_prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok: break
    res = model.predict(frame, conf=args.conf, imgsz=640, verbose=False)[0]
    n = len(res.boxes)
    for b in res.boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
        c = float(b.conf[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{c:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    t = time.time()
    fps = 1.0 / (t - t_prev + 1e-9)
    t_prev = t
    cv2.putText(frame, f"Arboles: {n}  FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv2.imshow("Tree detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()