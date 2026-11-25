# proyectovision
Proyecto final para la materia de deep learning para visión por computadora -- Sistema de visión por computadora completo --
# Detección y conteo de árboles en ortofotomapas satelitales

Sistema de visión por computadora que detecta y cuenta árboles en imágenes satelitales en tiempo real.

## 1. Problema y usuario
[2–3 párrafos: qué problema resuelve, quién lo usa, por qué importa. Usa lo que ya discutimos: inventario forestal municipal, ONG, empresa silvícola.]

## 2. Integrantes
- [Nombre completo] — [código]
- [Nombre completo] — [código]

## 3. Conjunto de datos
- **Fuente:** Hugging Face — `Filipstrozik/satellite_trees_wroclaw_2024` (16,880 tiles de 1024×1024 px).
- **Licencia:** [pegar].
- **Datasheet completo:** [`docs/datasheet.md`](docs/datasheet.md).
- **Descarga reproducible:**
  ```bash
  python data/download.py
  python src/prepare_yolo.py

## 4. Como reproducir
  ## Entrenamiento (local, Mac M1/M2 con MPS)
python src/train_local.py