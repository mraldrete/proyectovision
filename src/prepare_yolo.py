"""
Convierte el dataset HF (puntos + radio) a formato YOLO (bbox normalizada)
y genera particiones train/val/test POR TILE (sin fuga).

Salida:
  data/processed/images/{train,val,test}/*.jpg
  data/processed/labels/{train,val,test}/*.txt
  data/processed/data.yaml
"""
import json, shutil
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "data/raw/hf_dataset"
OUT = ROOT / "data/processed"

# Reproducibilidad (A5)
SEED = 42
np.random.seed(SEED)

def bbox_from_point(x, y, r, W, H):
    """Convierte (x,y,radius) -> (xc,yc,w,h) normalizados en [0,1]."""
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(W, x + r), min(H, y + r)
    if x2 <= x1 or y2 <= y1:
        return None
    xc = (x1 + x2) / 2 / W
    yc = (y1 + y2) / 2 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return xc, yc, w, h

def main():
    ds = load_from_disk(str(HF))["train"]
    n = len(ds)
    # Split POR TILE: 70/15/15
    idx = np.arange(n)
    train_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=SEED)
    val_idx,   test_idx = train_test_split(tmp_idx, test_size=0.50, random_state=SEED)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    # Carpetas
    for s in splits:
        (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)

    stats = {}
    for s, idlist in splits.items():
        n_trees = 0
        for i in idlist:
            row = ds[int(i)]
            img = row["images"].convert("RGB")
            W, H = img.size
            name = f"tile_{int(i):06d}"
            img.save(OUT / "images" / s / f"{name}.jpg", quality=95)
            lines = []
            for t in row["metadata"]["transformed_trees"]:
                bb = bbox_from_point(t["x"], t["y"], t["radius"], W, H)
                if bb is None:
                    continue
                xc, yc, w, h = bb
                if w <= 0 or h <= 0 or w > 1 or h > 1:
                    continue
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                n_trees += 1
            (OUT / "labels" / s / f"{name}.txt").write_text("\n".join(lines))
        stats[s] = {"tiles": len(idlist), "trees": n_trees}

    # data.yaml para Ultralytics YOLO
    yaml_txt = f"""path: {OUT.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: tree
"""
    (OUT / "data.yaml").write_text(yaml_txt)

    print(json.dumps(stats, indent=2))
    print(f"data.yaml -> {OUT/'data.yaml'}")

if __name__ == "__main__":
    main()