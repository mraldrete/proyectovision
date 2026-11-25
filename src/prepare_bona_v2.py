"""
Recorta parches 512x512 por tile, incluyendo TODAS las coronas dentro.
Train = algorithmic 2021, Val/Test = hand_bbox 2019.
Uso: python src/prepare_bona_v2.py
"""
import re
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GPKG = ROOT / "data/raw/neon/neon_crowns_dta.gpkg"
ORTHO = ROOT / "data/neon/ortho"
OUT = ROOT / "data/bona_yolo_v2"

PATCH = 512
STRIDE = 256   # solape 50% -> más parches, más datos

if not GPKG.exists():
    raise SystemExit(f"Falta {GPKG}. Copia el .gpkg desde Drive o descárgalo.")

gdf = gpd.read_file(GPKG)
bona = gdf[gdf["siteID"] == "BONA"].copy().to_crs(epsg=32606)
cent = bona.geometry.centroid
bona["cx"] = cent.x
bona["cy"] = cent.y
bona["tile_x"] = (bona["cx"] // 1000 * 1000).astype(int)
bona["tile_y"] = (bona["cy"] // 1000 * 1000).astype(int)

# Índice de tiles descargados
tiles = {}
for year in ["2019", "2021"]:
    for f in (ORTHO / year).glob("*.tif"):
        m = re.search(r'_(\d+)_(\d+)_image\.tif$', f.name)
        if m:
            tiles[(year, int(m.group(1)), int(m.group(2)))] = f

# Carpetas
for s in ["train", "val", "test"]:
    (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)

def make_patches(year, split, crown_df):
    n_imgs = n_boxes = 0
    for (tx, ty), grp in crown_df.groupby(["tile_x", "tile_y"]):
        f = tiles.get((year, int(tx), int(ty)))
        if f is None:
            continue
        with rasterio.open(f) as src:
            # Recorrer el tile con stride
            for row_off in range(0, src.height - PATCH + 1, STRIDE):
                for col_off in range(0, src.width - PATCH + 1, STRIDE):
                    win = Window(col_off, row_off, PATCH, PATCH)
                    # Bounds del parche en UTM
                    left = src.transform.c + col_off * src.transform.a
                    top = src.transform.f + row_off * src.transform.e
                    right = left + PATCH * src.transform.a
                    bottom = top + PATCH * src.transform.e
                    # Coronas cuyo centro cae dentro del parche
                    inside = grp[(grp["cx"] >= left) & (grp["cx"] < right) &
                                 (grp["cy"] > bottom) & (grp["cy"] <= top)]
                    if len(inside) < 2:   # descartar parches con 0-1 árbol
                        continue
                    arr = src.read(window=win)
                    if arr.shape[1] != PATCH or arr.shape[2] != PATCH:
                        continue
                    img = np.transpose(arr[:3], (1, 2, 0))
                    name = f"{year}_{int(tx)}_{int(ty)}_{col_off}_{row_off}"
                    Image.fromarray(img).save(OUT / "images" / split / f"{name}.jpg", quality=90)
                    lines = []
                    for _, r in inside.iterrows():
                        minx, miny, maxx, maxy = r.geometry.bounds
                        x1 = (minx - left) / (right - left)
                        x2 = (maxx - left) / (right - left)
                        y1 = (top - maxy) / (top - bottom)
                        y2 = (top - miny) / (top - bottom)
                        x1, y1, x2, y2 = [max(0.0, min(1.0, v)) for v in [x1, y1, x2, y2]]
                        if x2 <= x1 or y2 <= y1:
                            continue
                        xc, yc, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
                        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                        n_boxes += 1
                    if not lines:
                        (OUT / "images" / split / f"{name}.jpg").unlink()
                        continue
                    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
                    n_imgs += 1
    print(f"{split} ({year}): {n_imgs} imágenes, {n_boxes} cajas")

train = bona[bona["crown_source"] == "algorithmic"]
eval_ = bona[bona["crown_source"] == "hand_bbox"]
half = len(eval_) // 2

make_patches("2021", "train", train)
make_patches("2019", "val",   eval_.iloc[:half])
make_patches("2019", "test",  eval_.iloc[half:])

(OUT / "data.yaml").write_text(f"""path: {OUT.resolve()}
train: images/train
val: images/val
test: images/test
names:
  0: tree
""")
print("data.yaml listo:", OUT / "data.yaml")