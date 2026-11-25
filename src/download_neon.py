"""
Descarga el GeoPackage de coronas NEON desde Hugging Face.
Uso: python src/download_neon.py
"""
from pathlib import Path
from huggingface_hub import hf_hub_download

RAW = Path("data/raw/neon")
RAW.mkdir(parents=True, exist_ok=True)

def main():
    path = hf_hub_download(
        repo_id="weecology/neon-tree-crowns-dta",
        filename="neon_crowns_dta.gpkg",
        repo_type="dataset",
        local_dir=str(RAW),
    )
    print(f"Descargado en: {path}")
    print(f"Tamaño: {Path(path).stat().st_size / 1e6:.1f} MB")

if __name__ == "__main__":
    main()