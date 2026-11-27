# Model Card — Detector de árboles (YOLOv8n fine-tuned)

## Detalles del modelo
- **Arquitectura:** YOLOv8n (Ultralytics), fine-tuning desde pesos COCO.
- **Parámetros:** ~3.2 M.
- **Entrada:** imagen RGB 640×640.
- **Salida:** bboxes con clase `tree` y confianza.
- **Entrenamiento:** 30 épocas, batch 16, imgsz 640, early stopping patience=10, semilla 42.
- **Hardware:** Tesla T4 (Colab).
- **Fecha:** [fecha de entrenamiento].

## Uso previsto
- Conteo automático de árboles sobre ortofotomapas satelitales de resolución ~10 cm/px en zonas urbanas templadas.
- Apoyo a inventarios forestales municipales y verificación de reforestación.

## Uso no previsto
- Vigilancia de personas o vehículos.
- Imágenes de dron a baja altura, nocturnas, con nubes o nieve.
- Selvas tropicales, bosques boreales o zonas áridas (fuera de dominio).
- Decisiones legales o de valoración económica sin revisión humana.

## Métricas (test set)

| Métrica | Valor |
|---|---|
| mAP@0.5 | **0.587** |
| mAP@0.5:0.95 | 0.192 |
| Precision | 0.645 |
| Recall | 0.555 |
| F1 (macro) | 0.596 |

## Limitaciones
- Test de 17 imágenes (alta varianza).
- Solo sitio HARV (Massachusetts). No validado en Latinoamérica.
- Train con GT algorítmico (ForestGEO deepforest).
- **Usable con revisión humana**, no producción autónoma.

## Sesgos identificados
- **Geográfico:** entrenado solo en Wroclaw (Polonia); no validado en Latinoamérica.
- **De resolución:** sensible a cambios de GSD (ground sample distance).
- **Estacional:** ortofotos de 2024; no se evaluó nieve, sequía ni otoño extremo.

## Mitigaciones concretas
- **Umbral de confianza ajustable** en la app (C4) para que el usuario controle falsos positivos.
- **Prueba fuera de distribución** con imágenes propias (B6) para cuantificar la caída de desempeño.
- **Advertencia en UI** cuando la confianza media del frame cae por debajo de un umbral.
- Umbral de confianza ajustable en la app (`--conf`).
- Advertencia en UI cuando la confianza media < 0.3.
- Recomendación: revisión humana antes de cualquier decisión.

## Limitaciones
- No distingue especie ni salud del árbol.
- Árboles muy pequeños (< 8 px de radio) pueden perderse.
- Copas solapadas en bosque denso generan falsos negativos.
- mAP50 = 0.036 en test (2019, OOD temporal).
- Recall alto (0.53) pero precision muy baja (0.02): el modelo sobre-detecta.
- Causa principal: 40 tiles NEON BONA, GT 81% algorítmico.
- No usar en producción sin revisión humana.

## Consideraciones éticas
- No se usan datos personales (las ortofotos no contienen rostros identificables).
- El modelo no debe usarse para vigilancia.
- Recomendación: revisión humana antes de cualquier decisión de tala o valoración.

## Uso de IA generativa y créditos
- **Herramienta:** [ChatGPT/Claude/Gemini] usada para [redacción de docs / revisión de código / generación de plantillas].
- **Código de terceros:** Ultralytics YOLOv8 (AGPL-3.0), Hugging Face `datasets`.
- **Dataset:** Filipstrozik et al., Satellite Trees Wroclaw 2024.