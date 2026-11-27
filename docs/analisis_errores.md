# Análisis de errores — Detector de árboles NEON BONA

## Comparación de baselines

| Modelo | Train imgs | mAP50 | Recall | Precision | Causa del resultado |
|---|---|---|---|---|---|
| v1 (1 árbol/img) | 407 | 0.0005 | 0.02 | 0.002 | Colapso: fondo domina |
| v2 (multi-árbol) | 127 | 0.037 | 0.25 | 0.055 | Desbalance 1 vs 55 |
| v3 (filtrado >=5) | 80 | 0.041 | 0.186 | 0.065 | Pocos datos |
| v4 (STRIDE=128) | ~300 | 0.036 | 0.532 | 0.022 | Sobre-detección |

## Tipos de error identificados (5 ejemplos)

1. **Falsos positivos en suelo desnudo** — el modelo confunde sombras con copas.
   - Hipótesis: el GT algorítmico incluye "fallback" (cuadros pequeños alrededor de tallos sin copa visible).
2. **Falsos negativos en copas solapadas** — árboles densos no se separan.
   - Hipótesis: NMS suprime cajas muy cercanas.
3. **Cajas demasiado grandes** — el modelo predice copas más grandes que el GT.
   - Hipótesis: `box_loss` promedia sobre instancias de distinto tamaño.
4. **Detecciones en nieve** — ortofotos 2019 tienen parches de nieve.
   - Hipótesis: shift temporal (train 2021 sin nieve, eval 2019 con nieve).
5. **Baja confianza general** — `P=0.02` indica que el modelo no está seguro.
   - Hipótesis: 40 tiles es insuficiente para aprender la textura de copas.

## Limitación principal

Con 40 tiles NEON y GT algorítmico, YOLOv8n alcanza mAP50≈0.04. La causa principal es el **volumen de datos anotados** y la **calidad del GT algorítmico** (81% del dataset son detecciones de DeepForest, no anotaciones humanas). Trabajo futuro: anotar más tiles a mano o usar los `hand_polygon` como GT de segmentación.

---

## Experimento 2: HARV (Harvard Forest) — comparación de arquitecturas

### Dataset
- Sitio NEON **HARV** (Massachusetts), 9 tiles de ortofoto 2019.
- Train: ForestGEO 2019, bloque **Oeste** `[731600, 732000]`, STRIDE=192 (v2) / 256 (v1).
- Val: ForestGEO 2019, bloque **Este** `[732055, 732300]`, buffer espacial 55 m.
- Test: parcelas `hand_polygon` 2019 **disjuntas** de ForestGEO, enmascaradas en gris (114) fuera de la zona anotada.
- **Sin fuga:** split espacial (Oeste/Este) + temporal + test disjunto.

### Resultados

| Run | Dataset | Train imgs | Modelo | mAP50 (val) | mAP50 (test) | mAP50-95 (test) | P (test) | R (test) | Params | Tamaño |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | HARV v1 | 245 | YOLOv8n | 0.513 | 0.375 | 0.117 | 0.523 | 0.306 | 3.0 M | 6.2 MB |
| **B** | HARV v2 | 440 | YOLOv8n | 0.565 | 0.512 | 0.163 | 0.610 | 0.510 | 3.0 M | 6.2 MB |
| **C** | HARV v2 | 440 | **YOLO11n** | **0.579** | **0.587** | **0.192** | **0.645** | **0.555** | **2.58 M** | **5.5 MB** |
| Ref | BONA | ~300 | YOLOv8n | 0.036 | 0.036 | 0.011 | 0.022 | 0.532 | 3.0 M | 6.0 MB |

### Análisis (E2)

- **Dataset (A→B):** +0.137 mAP50 test. Más datos (245→440 imgs) y STRIDE=192 (más solape) generan parches más densos → mejor aprendizaje.
- **Arquitectura (B→C):** +0.075 mAP50 test. YOLO11n es más preciso que YOLOv8n con **14% menos parámetros** y **11% menos tamaño**.
- **Sitio + split (Ref→A):** +0.339 mAP50 test. El split espacial (Oeste/Este + 55 m buffer), el test enmascarado (gris 114 fuera de parcelas) y el GT ForestGEO (más limpio) explican la mayor parte de la mejora.

### Modelo final elegido

**YOLO11n sobre HARV v2** (`runs_harv/yolo11n_harv_v2/weights/best.pt`):
- **mAP50 = 0.587** (16× mejor que BONA 0.036).
- **Precision = 0.645** (29× mejor que BONA 0.022).
- **Recall = 0.555** (ligeramente mejor que BONA 0.532).
- **Tamaño = 5.5 MB**, 2.58 M parámetros.
- **Latencia = 22.6 ms/img** en MPS.

### Limitaciones del experimento HARV

- **Test de 17 imágenes** → alta varianza en mAP (±0.03).
- **Solo un sitio NEON** (HARV). No validado en otros sitios.
- **Train con `detection_method == 'deepforest'`** (GT algorítmico), no 100% humano.
- **Máscara gris** fuera de parcelas anotadas: reduce falsos positivos, pero también elimina contexto que el modelo podría usar.