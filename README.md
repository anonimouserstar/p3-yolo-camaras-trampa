# P3 — Monitoreo de fauna silvestre con cámaras trampa (YOLOv8 + CCT20)

Examen parcial · Redes Neuronales y Aprendizaje Profundo · UNI FIIS 2026-II

Detección y clasificación de especies en fotos de cámaras trampa con YOLOv8 sobre **Caltech Camera Traps (CCT20, partición oficial ECCV 2018)**, con análisis de generalización a cámaras nunca vistas, Grad-CAM, robustez ante oclusión/tamaño y un módulo de monitoreo ecológico basado en reglas.

## Ejecución (un solo comando)

**Google Colab (recomendado):** abrir `P3_YOLO_CamarasTrampa.ipynb`, elegir *Entorno de ejecución → GPU* y luego *Ejecutar todo*. Los pesos, predicciones y el dataset convertido se guardan en `MyDrive/P3_CCT_YOLO`; si Colab se desconecta, *Ejecutar todo* reanuda donde quedó.

**Local / servidor con GPU:**
```bash
pip install -r requirements.txt
python run_notebook.py            # experimento completo
python run_notebook.py --smoke    # prueba de humo con datos sintéticos (CPU, sin descargas)
```

Al terminar se genera `P3_resultados_<fecha>.zip` con:

| Carpeta | Contenido |
|---|---|
| `00_config/` | configuración completa y entorno (versiones, GPU) |
| `01_dataset/` | imágenes por partición, cajas por especie, ubicaciones, figuras de distribución |
| `02_entrenamiento/` | `results.csv`, `args.yaml`, curvas de pérdida y mAP |
| `03_evaluacion/` | mAP@0.5 y mAP@0.5:0.95 global y por especie, matrices de confusión, desbalance, velocidad |
| `04_generalizacion/` | brecha cis→trans, por especie y por cámara; localización vs. clasificación; FP en vacías |
| `05_gradcam/` | mapas Grad-CAM, energía en caja GT, intercambio de fondo, animal borrado |
| `06_robustez/` | desempeño por oclusión estimada, tamaño relativo e iluminación; muestra para validación manual |
| `07_monitoreo/` | reportes de actividad por especie y periodo en una cámara no vista, comparados con la verdad |
| `08_ablacion/` | base vs. sin vacías vs. Repeat Factor Sampling |
| `RESUMEN.md`, `resumen.json` | todas las cifras clave |

## Protocolo experimental

| Partición | Ubicaciones | Uso |
|---|---|---|
| `train` | vistas | entrenamiento (+10 % de imágenes vacías como fondo) |
| `cis_val` ∪ `trans_val` | vistas + nuevas | selección del checkpoint (mAP50-95) |
| `cis_test` | vistas | evaluación en distribución |
| `trans_test` | nunca vistas | evaluación de generalización |

En LILA BC, `eccv_train` corresponde a las ubicaciones *train/cis* y `eccv_val` a las *trans*.

## Decisiones principales
- **Modelo**: YOLOv8s preentrenado en COCO, 640 px, 30 épocas, early stopping (paciencia 10).
- **Aumento**: mosaico, HSV moderado (muchas fotos nocturnas son IR), traslación/escala, volteo horizontal; sin rotaciones (cámara fija).
- **Vacías**: se incluyen como imágenes de fondo con etiqueta vacía. Si CCT20 trae menos de 5 % de vacías, se completan con vacías reales del CCT completo de las mismas cámaras.
- **Desbalance**: análisis AP vs. frecuencia y mitigación con *Repeat Factor Sampling* (Gupta et al., 2019).
- **Oclusión**: proxy reproducible (solapamiento entre cajas, truncamiento en borde, densidad de bordes por especie) con muestra exportada para validación manual (kappa ponderado).
- **Monitoreo**: reglas de umbral operativo (máx. F1 en validación), MaxN por secuencia, eventos independientes a 30 min e índice de abundancia relativa (RAI).

## Datos
Caltech Camera Traps, LILA BC — https://lila.science/datasets/caltech-camera-traps · Licencia CDLA-Permissive 1.0.
Beery, S., Van Horn, G., & Perona, P. (2018). *Recognition in Terra Incognita*. ECCV.
