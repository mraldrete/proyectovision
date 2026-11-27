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
### Comparación de arquitecturas (HARV v2)

| Run | Dataset | Train imgs | Modelo | mAP50 (test) | mAP50-95 (test) | P (test) | R (test) | Params | Tamaño |
|---|---|---|---|---|---|---|---|---|---|
| A | HARV v1 | 245 | YOLOv8n | 0.375 | 0.117 | 0.523 | 0.306 | 3.0 M | 6.2 MB |
| B | HARV v2 | 440 | YOLOv8n | 0.512 | 0.163 | 0.610 | 0.510 | 3.0 M | 6.2 MB |
| **C** | **HARV v2** | **440** | **YOLO11n (elegido)** | **0.587** | **0.192** | **0.645** | **0.555** | **2.58 M** | **5.5 MB** |
| Ref | BONA | ~300 | YOLOv8n | 0.036 | 0.011 | 0.022 | 0.532 | 3.0 M | 6.0 MB |

**Justificación:** YOLO11n sobre HARV v2 es el modelo final. Gana en mAP, mAP50-95, precisión y recall frente a YOLOv8n, con 14% menos parámetros y 11% menos tamaño. La comparación aísla el efecto del dataset (A→B: +0.137 mAP50) y de la arquitectura (B→C: +0.075 mAP50). El dataset explica la mayor parte de la mejora (BONA→HARV: +0.339 mAP50) por el split espacial, el test enmascarado y el GT ForestGEO.

## 5. Resultados principales
M## 5. Resultados principales

### Modelo final: YOLO11n sobre HARV v2

| Métrica | Valor (test, 17 imgs) |
|---|---|
| **mAP@0.5** | **0.587** |
| mAP@0.5:0.95 | 0.192 |
| Precision | 0.645 |
| Recall | 0.555 |
| Parámetros | 2.58 M |
| Tamaño | 5.5 MB |
| Latencia (MPS) | 22.6 ms/img |

**Comparación con baseline BONA:** mAP50 0.036 → 0.587 (**16× mejor**). Causa principal: mejor split (espacial + test enmascarado) y GT más limpio (ForestGEO).


### Modelo final: YOLO11n sobre HARV v2

| Métrica | Valor (test, 17 imgs) |
|---|---|
| mAP@0.5 | **0.587** |
| mAP@0.5:0.95 | 0.192 |
| Precision | 0.645 |
| Recall | 0.555 |
| Parámetros | 2.58 M |
| Tamaño | 5.5 MB |

## 6. Instalación y reproducción (≤ 5 comandos)
    ```bash
    git clone https://github.com/mraldrete/proyectovision
    python3 -m venv .venv_yolo && source .venv_yolo/bin/activate
    pip install -r requirements.txt
    python src/download_neon.py && python src/prepare_bona_v2.py
    python src/train_local.py

|
En Mac Apple Silicon, el entrenamiento usa MPS. En otras plataformas, CPU o CUDA.

## 7. Demo (1 comando)
        ```bash
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

- **Test HARV de 17 imágenes** → alta varianza (mAP ±0.03).

- **Un solo sitio NEON** (HARV). No validado en otros sitios.

- **Train con GT algorítmico** (`deepforest`), no 100% humano.

- **BONA descartado para producción** por fuga espacial y GT ruidoso (mAP=0.036). Se conserva como referencia metodológica.

## 10. Uso de IA generativa y créditos
IA generativa: ChatGPT/Claude usados para redacción de documentación, revisión de código y generación de plantillas de scripts.

Código de terceros: Ultralytics YOLOv8 (AGPL-3.0), PySide6 (LGPL), Hugging Face datasets, geopandas, rasterio.

Dataset: Weinstein et al. (2024), Individual canopy tree species maps for the National Ecological Observatory Network, PLoS Biology. NEON Tree Crowns DTA (CC-BY-4.0).

NEON: National Ecological Observatory Network (orotofotos y datos VST, CC-BY-4.0).