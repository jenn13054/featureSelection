# Bitácora de Cambios

Este documento registra los cambios realizados durante la sesión de trabajo sobre el proyecto `data-analysis-tecnm`.

---

## 1. Explicación del Script Original (`script.py`)

**Fecha**: Sesión actual  
**Descripción**: Se explicó el funcionamiento del script original `script.py`. El script era una demostración de selección de características (Filter, Wrapper, Embedded) aplicada a un dataset de clasificación binaria. Se identificó que intentaba cargar datos desde Google Drive (`/content/drive/My Drive/...`) y que carecía del import de `os`.

---

## 2. Corrección de Ruta de Datos

**Archivo modificado**: `script.py`  
**Cambios**:
- Agregado `import os`.
- Reemplazada la ruta de Google Drive por la ruta local: `data_final.csv`.
- Reemplazado el bloque de carga de Google Drive por manejo de errores local (`FileNotFoundError`, `RuntimeError`).
- Agregado mensaje de confirmación con dimensiones del dataset cargado.

---

## 3. Generación de Dependencias

**Archivo creado**: `requirements.txt`  
**Contenido**:
```text
numpy
pandas
scikit-learn
```
**Motivo**: Permitir la instalación rápida de dependencias para ejecutar `script.py` en cualquier entorno.

---

## 4. Impresión Progresiva de Resultados

**Archivo modificado**: `script.py`  
**Cambios**:
- Agregada impresión inmediata de métricas (`print_metrics`) después de cada método (baseline, filters, wrappers, embedded).
- Eliminada la sección redundante de detalle al final que repetía la información.
- Se mantuvo la tabla comparativa final (`Summary table`) ordenada por accuracy.

**Motivo**: Mejorar la experiencia de ejecución al ver resultados en tiempo real en lugar de esperar al final.

---

## 5. Refactorización para Experimento de Tesis

**Archivo modificado**: `script.py` (reescritura completa)  
**Cambios principales**:
- **Argumentos de línea de comandos** (`argparse`):
  - `--runs`: Número de ejecuciones (default: 15).
  - `--methods`: Lista de métodos a ejecutar o `all`.
  - `--data`: Ruta al CSV (default: `data_final.csv`).
  - `--output`: Archivo CSV de resultados acumulados (default: `experiment_results.csv`).
  - `--seed`: Semilla base (default: 42).
- **Semillas diferenciadas**: Cada run usa `seed + run_id` para garantizar variabilidad.
- **Persistencia por iteración**: Después de cada run, los resultados se agregan (append) al CSV de salida.
- **Reanudación automática**: Si el CSV de salida ya existe, el script detecta las runs completadas y las salta.
- **Estadísticas finales**: Al finalizar, genera una tabla con promedio (`mean`) y desviación estándar (`std`) por método.
- **Métodos parametrizables**: Posibilidad de ejecutar todos o una selección de los 9 métodos disponibles.

**Métodos soportados**:
1. `baseline`
2. `variance_threshold`
3. `chi2`
4. `mutual_info`
5. `rfe`
6. `rfecv`
7. `l1`
8. `rf_topk`
9. `rf_median`

---

## 6. Documentación de Uso

**Archivo creado**: `experiment.md`  
**Contenido**: Guía completa de uso del nuevo `script.py`, incluyendo:
- Descripción de parámetros.
- Ejemplos de comandos.
- Lista de métodos disponibles.
- Descripción de archivos generados (`experiment_results.csv`, `_summary.csv`).
- Notas orientadas a trabajos de tesis (reproducibilidad, robustez, reanudación).

---

## 7. Ejecución del Experimento (15 Runs)

**Fecha**: 2026-05-19  
**Descripción**: Se ejecutó `script.py` con los parámetros por defecto (15 runs, semilla 42).

**Proceso**:
- Instalación de dependencias: `pip3 install -r requirements.txt` (numpy, pandas, scikit-learn).
- Ejecución en background con timeout extendido debido a la duración de métodos como `rfecv`.
- El script se reanudó automáticamente tras un timeout inicial, completando las 15 runs.
- Archivos generados:
  - `experiment_results.csv` — resultados individuales por run y método.
  - `experiment_results_summary.csv` — resumen estadístico (mean ± std) por método.

**Resultados principales**:
- Métodos con mayor accuracy: `variance_threshold`, `mutual_info`, `rfecv`, `rf_topk` (0.9358).
- Método más eficiente: `rf_topk` (10 features, 0.034s, accuracy 0.9358).
- El baseline (46 features, 3.49s) no mostró ventaja sobre los métodos de selección.

---

## 8. Análisis y Visualización de Resultados (15 Runs)

**Fecha**: 2026-05-19  
**Descripción**: Inspección de resultados y generación de tablas comparativas y gráficos.

**Archivos creados**:
- `results_analysis.py` — script de análisis automatizado.
- `results_comparison_table.md` — tabla comparativa en formato Markdown.
- `results_comparison.png` — visualización con 4 subplots (accuracy, features, tiempo, scatter).

**Dependencias adicionales instaladas**: `matplotlib` para generación de gráficos.

---

## 9. Ejecución del Experimento con 30 Runs

**Fecha**: 2026-05-19  
**Descripción**: Nueva ejecución de `script.py` con configuración ampliada para mayor robustez estadística.

**Comando ejecutado**:
```bash
python3 script.py --runs 30 --output experiment_results_30runs.csv
```

**Características**:
- 30 ejecuciones independientes con semillas 42–71.
- Archivo de salida separado para no sobrescribir los resultados anteriores.
- Duración total: ~38 minutos en background.
- Reanudación automática habilitada (aunque en esta ejecución no fue necesaria).

**Archivos generados**:
- `experiment_results_30runs.csv` — 270 filas (30 runs × 9 métodos).
- `experiment_results_30runs_summary.csv` — resumen estadístico de 30 runs.

---

## 10. Análisis y Visualización de Resultados (30 Runs)

**Fecha**: 2026-05-19  
**Descripción**: Generación de tabla comparativa y visualización para los 30 runs.

**Archivos creados**:
- `results_analysis_30runs.py` — script de análisis para 30 runs.
- `results_comparison_30runs.md` — tabla comparativa en Markdown.
- `results_comparison_30runs.png` — gráfico comparativo actualizado.

**Hallazgos clave (30 runs)**:
- Consistencia con los resultados de 15 runs: los rankings se mantienen estables.
- `rf_topk` sigue siendo el método recomendado (accuracy 0.9357, 10 features, 0.034s).
- La desviación estándar de accuracy es muy baja (~0.0004), indicando alta estabilidad del dataset.
- `rfecv` muestra alta variabilidad en el número de features seleccionadas (32.9 ± 4.9).

---

## 11. Ejecución del Experimento con 60 Runs

**Fecha**: 2026-05-21  
**Descripción**: Ejecución de `script.py` con configuración ampliada a 60 runs para mayor robustez estadística en el contexto de tesis.

**Comando ejecutado**:
```bash
python3 script.py --runs 60 --output experiment_results_60runs.csv
```

**Características**:
- 60 ejecuciones independientes con semillas 42–101.
- Archivo de salida separado (`experiment_results_60runs.csv`) para no sobrescribir resultados anteriores.
- Duración total: ~77 minutos (primera tanda: 47 runs en 60 min; reanudación: 13 runs en 17 min).
- Reanudación automática habilitada: tras un timeout en la primera ejecución, el script detectó las 47 runs completadas y finalizó las 13 restantes sin pérdida de datos.

**Archivos generados**:
- `experiment_results_60runs.csv` — 540 filas (60 runs × 9 métodos).
- `experiment_results_60runs_summary.csv` — resumen estadístico de 60 runs (mean ± std).

---

## 12. Análisis y Visualización de Resultados (60 Runs)

**Fecha**: 2026-05-21  
**Descripción**: Generación de tabla comparativa y visualización para los 60 runs.

**Archivos creados**:
- `results_analysis_60runs.py` — script de análisis para 60 runs (con límites de gráficos dinámicos).
- `results_comparison_60runs.md` — tabla comparativa en Markdown.
- `results_comparison_60runs.png` — gráfico comparativo actualizado.

**Hallazgos clave (60 runs)**:
- **Mayor accuracy**: `baseline`, `variance_threshold`, `mutual_info`, `l1`, `rf_topk`, `rf_median` (empatados en **0.9356**).
- **Menor número de features**: `chi2`, `mutual_info`, `rfe`, `rf_topk` (todos con **10 features**).
- **Más eficiente (accuracy/tiempo)**: `rf_topk` (**27.68**), seguido de `chi2` (**22.29**) y `rfe` (**22.14**).
- **`chi2` es el método más estable**: desviación estándar de 0 en todas las métricas (selección determinista).
- **`rfecv`** mantiene alta variabilidad en features seleccionadas (**33.0 ± 4.8**) con accuracy ligeramente inferior.
- Los rankings se mantienen consistentes con los experimentos de 15 y 30 runs, confirmando la estabilidad de los resultados.

---

## Resumen de Archivos del Proyecto

| Archivo | Estado | Descripción |
|---|---|---|
| `script.py` | Modificado | Script principal refactorizado para experimentos repetibles. |
| `requirements.txt` | Creado | Dependencias: numpy, pandas, scikit-learn. |
| `experiment.md` | Creado | Guía de uso del experimento. |
| `bitacora.md` | Creado | Este documento. |
| `data_final.csv` | Existente | Dataset original (sin cambios). |
| `explore.py` | Existente | Script auxiliar original (sin cambios). |
| `experiment_results.csv` | Generado | Resultados individuales del experimento (15 runs). |
| `experiment_results_summary.csv` | Generado | Resumen estadístico del experimento (15 runs). |
| `experiment_results_30runs.csv` | Generado | Resultados individuales del experimento (30 runs). |
| `experiment_results_30runs_summary.csv` | Generado | Resumen estadístico del experimento (30 runs). |
| `results_analysis.py` | Creado | Script de análisis y visualización (15 runs). |
| `results_analysis_30runs.py` | Creado | Script de análisis y visualización (30 runs). |
| `results_comparison_table.md` | Creado | Tabla comparativa en Markdown (15 runs). |
| `results_comparison_30runs.md` | Creado | Tabla comparativa en Markdown (30 runs). |
| `results_comparison.png` | Creado | Gráfico comparativo (15 runs). |
| `results_comparison_30runs.png` | Creado | Gráfico comparativo (30 runs). |
| `experiment_results_60runs.csv` | Generado | Resultados individuales del experimento (60 runs). |
| `experiment_results_60runs_summary.csv` | Generado | Resumen estadístico del experimento (60 runs). |
| `results_analysis_60runs.py` | Creado | Script de análisis y visualización (60 runs). |
| `results_comparison_60runs.md` | Creado | Tabla comparativa en Markdown (60 runs). |
| `results_comparison_60runs.png` | Creado | Gráfico comparativo (60 runs). |
