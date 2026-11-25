"""Verifica 3 imágenes con su bbox YOLO. Uso: python src/check_dataset.py"""
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

IMG = Path("data/bona_yolo/images/train")
LBL = Path("data/bona_yolo/labels/train")
files = sorted(IMG.glob("*.jpg"))[:3]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, p in zip(axes, files):
    img = Image.open(p); W, H = img.size
    ax.imshow(img)
    for line in (LBL / f"{p.stem}.txt").read_text().strip().splitlines():
        _, xc, yc, w, h = map(float, line.split())
        ax.add_patch(patches.Rectangle(
            ((xc - w/2) * W, (yc - h/2) * H), w * W, h * H,
            fill=False, color="lime", lw=2))
    ax.set_title(p.name); ax.axis("off")
plt.tight_layout()
plt.savefig("docs/figuras/check_dataset_bona.png", dpi=110)
plt.show()
print("Guardado en docs/figuras/check_dataset_bona.png")