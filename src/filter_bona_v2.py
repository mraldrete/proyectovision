"""Filtra el dataset v2: se queda solo con imágenes con >= MIN_TREES cajas."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/bona_yolo_v2"
DST = ROOT / "data/bona_yolo_v3"
MIN_TREES = 5

for split in ["train", "val", "test"]:
    (DST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_keep = n_drop = 0
    for lbl in (SRC / "labels" / split).glob("*.txt"):
        lines = [l for l in lbl.read_text().strip().splitlines() if l.strip()]
        if len(lines) < MIN_TREES:
            n_drop += 1
            continue
        # Copia label
        (DST / "labels" / split / lbl.name).write_text("\n".join(lines) + "\n")
        # Copia imagen
        img = SRC / "images" / split / f"{lbl.stem}.jpg"
        if img.exists():
            import shutil
            shutil.copy(img, DST / "images" / split / img.name)
            n_keep += 1
    print(f"{split}: keep={n_keep}, drop={n_drop}")

(DST / "data.yaml").write_text(f"""path: {DST.resolve()}
train: images/train
val: images/val
test: images/test
names:
  0: tree
""")
print("data.yaml listo:", DST / "data.yaml")
