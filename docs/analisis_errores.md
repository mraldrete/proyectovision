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