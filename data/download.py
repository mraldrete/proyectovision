
### 1.2 Crea `data/download.py`

#```python
"""
Descarga el dataset Satellite Trees Wroclaw 2024 desde Hugging Face.
Uso: python data/download.py
"""
from datasets import load_dataset
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ds = load_dataset("Filipstrozik/satellite_trees_wroclaw_2024")
    ds.save_to_disk(str(RAW_DIR / "hf_dataset"))
    print(f"Guardado en {RAW_DIR / 'hf_dataset'}")
    print(ds)

if __name__ == "__main__":
    main()