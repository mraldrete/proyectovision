# Datasheet — Satellite Trees Wroclaw 2024

## 1. Motivación
- **Propósito:** entrenar un detector de árboles sobre ortofotomapas satelitales para conteo automático.
- **Usuario previsto:** técnico municipal de medio ambiente, ONG de reforestación o empresa silvícola que hoy cuenta árboles manualmente sobre ortofotos.
- **Tarea:** detección de objetos (clase única: `tree`).

## 2. Composición
- **Fuente:** Hugging Face — `Filipstrozik/satellite_trees_wroclaw_2024`
- **Licencia:** [PEGAR la licencia exacta de la tarjeta de HF]
- **Número de tiles:** 16,880 imágenes de 1024×1024 px.
- **Clases:** 1 (`tree`).
- **Anotación original:** punto (x, y) + radio en píxeles (`transformed_trees`).
- **Conversión:** `radius` → bbox `[x-r, y-r, x+r, y+r]` (ver `src/prepare_yolo.py`).
- **Árboles totales:** [N del EDA].
- **Árboles por tile:** min=[ ], max=[ ], media=[ ], mediana=[ ].
- **Tiles sin árboles:** [N] ([%]%).
- **Radios (px):** min=[ ], p50=[ ], p95=[ ], max=[ ].

## 3. Proceso de etiquetado
- **Origen:** datos GIS públicos del municipio de Wroclaw, Polonia (2024).
- **Anotador:** sistema GIS municipal (no crowdsourcing).
- **Supuesto:** se confía en el ground truth del GIS; la revisión manual (sección 5) valida esta suposición.

## 4. Distribución y particiones
- **Split:** 70/15/15 **por tile** (no por árbol) → sin fuga de información.
- **Semilla:** 42.
- **Train:** [N] tiles / [N] árboles.
- **Val:** [N] tiles / [N] árboles.
- **Test:** [N] tiles / [N] árboles.
- **Justificación:** árboles del mismo tile comparten contexto visual; partirlos por árbol filtraría información entre train y test.

## 5. Calidad de etiquetas (revisión manual)
- **Mmuestras revisadas:** 20 tiles aleatorios de train.
- **Errores encontrados:** [N]/20 = [%]%.
- **Tipo de error:** [describir: p. ej. árboles faltantes en zonas densas, radios sobredimensionados].
- **Corrección aplicada:** [describir o "ninguna; se documenta como limitación"].
- **Figura:** `docs/figuras/revision_20_muestras.png`.

## 6. Usos previstos y no previstos
- **Previsto:** conteo/inventario forestal urbano y periurbano en imágenes aéreas de resolución similar (~10 cm/px).
- **No previsto:** vigilancia de personas, imágenes de very-high-resolution de dron a baja altura, selvas tropicales (dominio distinto), imágenes nocturnas o con nubes.

## 7. Limitaciones conocidas
- **Dominio:** solo Wroclaw (Polonia), clima templado, zona urbana.
- **Resolución:** ortofoto 1024×1024 a ~10 cm/px.
- **Clase única:** no distingue especie ni estado de salud.
- **Sesgo geográfico:** no validado fuera de Europa central.

## 8. Mantenimiento
- **Responsable:** [tu nombre].
- **Contacto:** [tu correo institucional].
- **Versión:** v1.0-final.