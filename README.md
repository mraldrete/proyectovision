# Detección y conteo de árboles en ortofotomapas satelitales

Sistema de visión por computadora que detecta y cuenta árboles en ortofotos satelitales, con aplicación de escritorio (PySide6) para subir una imagen y obtener el conteo.

## 1. Problema y usuario

El inventario forestal manual sobre ortofotos es lento y costoso: un técnico municipal o una ONG debe contar árboles a mano sobre imágenes de alta resolución, lo que limita la frecuencia y cobertura del monitoreo.

Este proyecto entrena un detector de árboles (YOLOv8n fine-tuned) sobre el dataset **NEON Tree Crowns (sitio BONA)** y lo expone en una aplicación de escritorio que permite subir una ortofoto y contar árboles automáticamente, con umbral de confianza ajustable.

**Usuario previsto:** técnico de medio ambiente municipal, ONG de reforestación o empresa silvícola que necesita inventarios rápidos sobre ortofotos, con revisión humana posterior.

## 2. Equipo

- Mario R Aldrete

## 3. Conjunto de datos

- **Fuente:** Hugging Face — [`weecology/neon-tree-crowns-dta`](https://huggingface.co/datasets/weecology/neon-tree-crowns-dta) (41,738 coronas, 38 sitios NEON).
- **Subconjunto usado:** sitio **BONA** (Alaska) — 913 coronas (407 `algorithmic` + 506 `hand_bbox`).
- **Licencia:** CC-BY-4.0.
- **Splits:** train = `algorithmic` (ortofoto 2021), val/test = `hand_bbox` (ortofoto 2019). **Sin fuga:** años distintos + coronas distintas + tiles distintos.
- **Datasheet completo:** [`docs/datasheet.md`](docs/datasheet.md).
- **Descarga reproducible:**
  ```bash
  python src/download_neon.py
  python src/prepare_bona_v2.py

## 4. Arquitectura
Modelo: YOLOv8n fine-tuned desde pesos COCO (50–150 épocas, imgsz=640, early stopping).

Diagrama: entrada (orto 1024×1024) → YOLOv8n (backbone CSP + FPN + head) → cajas + confianza → NMS → conteo.

Alternativa comparada: YOLOv8n preentrenado en COCO sin fine-tuning.

Comparación de arquitecturas
Modelo	mAP@0.5	mAP@0.5:0.95	Parámetros	Tamaño (MB)	Latencia (ms/img)
YOLOv8n fine-tuned (elegido)	0.0363	0.0114	3.0 M	6.0	12.8 (MPS)
YOLOv8n COCO (sin fine-tuning)	0.0000	0.0000	3.2 M	6.5	12.8 (MPS)
Justificación: El modelo preentrenado en COCO no detecta árboles (mAP=0) porque no conoce la clase. El fine-tuning sobre NEON BONA aporta +0.036 mAP. Se intentó YOLOv8s (más capacidad), pero en MPS tardaba 20+ min/época por presión de memoria, inviable para el tiempo del proyecto. YOLOv8n ofrece el mejor balance latencia/tamaño/exactitud alcanzable con estos datos.

## 5. Resultados principales
Métricas en test (2019, OOD temporal)
Métrica	Valor
mAP@0.5	0.0363
mAP@0.5:0.95	0.0114
Precision	0.0221
Recall	0.532
Comparación de versiones
Versión	Train imgs	mAP@0.5	Recall	Causa
v1 (1 árbol/img)	407	0.0005	0.02	Colapso: fondo domina
v2 (multi-árbol)	127	0.037	0.25	Desbalance 1 vs 55
v3 (filtrado ≥5)	80	0.041	0.186	Pocos datos
v4 (STRIDE=128, elegido)	~300	0.036	0.532	Mejor recall, sobre-detecta
Figuras: docs/figuras/entrenamiento_v2/results.png, confusion_matrix.png.

## 6. Instalación y reproducción (≤ 5 comandos)
bash
git clone https://github.com/mraldrete/proyectovision
python3 -m venv .venv_yolo && source .venv_yolo/bin/activate
pip install -r requirements.txt
python src/download_neon.py && python src/prepare_bona_v2.py
python src/train_local.py

En Mac Apple Silicon, el entrenamiento usa MPS. En otras plataformas, CPU o CUDA.

## 7. Demo (1 comando)

python -m app.main

Muestra una ventana PySide6 para subir una ortofoto, ver las cajas detectadas, el conteo de árboles y ajustar el umbral de confianza (1%–50%).

## 8. Video de respaldo
[Enlace a YouTube/Drive del video de 1–2 min mostrando la app funcionando]

## 9. Limitaciones
mAP@0.5 = 0.036 en test. Recall alto (0.53) pero precision muy baja (0.022): detecta muchos árboles pero con muchos falsos positivos.

Causa principal: 40 tiles NEON BONA con GT 81% algorítmico. El volumen de datos anotados es insuficiente para un detector preciso.

Sesgo geográfico: solo BONA (Alaska). No validado en Latinoamérica.

Sesgo temporal: train 2021, eval 2019. La nieve en 2019 afecta las detecciones.

No usar en producción sin revisión humana.

Trabajo futuro: más anotación humana (los 7,946 hand_* del dataset NEON) o segmentación con los hand_polygon.

Dataset descartado: Wroclaw (Filipstrozik/satellite_trees_wroclaw_2024) se descartó por GT desalineado (auditoría en docs/analisis_errores.md).

## 10. Uso de IA generativa y créditos
IA generativa: ChatGPT/Claude usados para redacción de documentación, revisión de código y generación de plantillas de scripts.

Código de terceros: Ultralytics YOLOv8 (AGPL-3.0), PySide6 (LGPL), Hugging Face datasets, geopandas, rasterio.

Dataset: Weinstein et al. (2024), Individual canopy tree species maps for the National Ecological Observatory Network, PLoS Biology. NEON Tree Crowns DTA (CC-BY-4.0).

NEON: National Ecological Observatory Network (orotofotos y datos VST, CC-BY-4.0).