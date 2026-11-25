"""Recomprime los JPG de data/processed/images a calidad 85 para reducir tamaño."""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data/processed/images"

total_before = total_after = 0
for split in ["train", "val", "test"]:
    for f in (IMG_DIR / split).glob("*.jpg"):
        total_before += f.stat().st_size
        img = Image.open(f).convert("RGB")
        img.save(f, quality=85, optimize=True)
        total_after += f.stat().st_size

print(f"Antes:  {total_before/1e9:.2f} GB")
print(f"Después:{total_after/1e9:.2f} GB")
print(f"Reducción: {100*(1-total_after/total_before):.1f}%")