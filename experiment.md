# Guía del Experimento de Selección de Características

Este documento describe cómo utilizar el script `script.py` para ejecutar experimentos de selección de características de forma repetida, con semillas diferentes y persistencia de resultados.

---

## Descripción General

El script ejecuta múltiples métodos de selección de características (`Filter`, `Wrapper` y `Embedded`) un número configurable de veces. En cada ejecución (run) se utiliza una **semilla distinta** para garantizar variabilidad en la división train/test y en los modelos estocásticos.

Los resultados de cada iteración se guardan inmediatamente en un archivo CSV, lo que permite:
- **Reanudar** el experimento si se interrumpe.
- Calcular estadísticas robustas (promedio y desviación estándar) al finalizar.

---

## Parámetros de Ejecución

| Parámetro | Default | Descripción |
|---|---|---|
| `--runs` | `15` | Número total de ejecuciones del experimento. |
| `--methods` | `all` | Lista de métodos a ejecutar. Usa `all` para ejecutar todos. |
| `--data` | `data_final.csv` | Ruta al archivo CSV con el dataset. |
| `--output` | `experiment_results.csv` | Archivo CSV donde se acumulan los resultados por run. |
| `--seed` | `42` | Semilla base. En la run *i* se usa `seed + i`. |

---

## Ejemplos de Uso

### Ejecutar todos los métodos 15 veces (default)
```bash
python3 script.py
```

### Ejecutar 20 runs con métodos específicos
```bash
python3 script.py --runs 20 --methods baseline chi2 l1 rf_topk
```

### Usar un dataset diferente y semilla base distinta
```bash
python3 script.py --runs 30 --data mi_dataset.csv --seed 123 --output resultados.csv
```

### Reanudar un experimento interrumpido
Si el archivo `--output` ya existe, el script detecta automáticamente las runs completadas y salta las que faltan:
```bash
python3 script.py --runs 50 --output experiment_results.csv
```

---

## Métodos Disponibles

| ID | Método | Tipo |
|---|---|---|
| `baseline` | Todas las características | Línea base |
| `variance_threshold` | VarianceThreshold (`threshold=0.01`) | Filter |
| `chi2` | SelectKBest + chi² (`k=10`) | Filter |
| `mutual_info` | SelectKBest + mutual information (`k=10`) | Filter |
| `rfe` | Recursive Feature Elimination (`n=10`) | Wrapper |
| `rfecv` | RFECV (5-fold CV) | Wrapper |
| `l1` | Logistic Regression con penalización L1 (`C=0.1`) | Embedded |
| `rf_topk` | Top 10 características por importancia de Random Forest | Embedded |
| `rf_median` | SelectFromModel(RandomForest, `threshold='median'`) | Embedded |

---

## Archivos Generados

### 1. Resultados por iteración
- **Archivo**: `experiment_results.csv` (o el valor de `--output`)
- **Contenido**: Una fila por método y por run.
- **Columnas**: `run_id`, `seed`, `method`, `n_features`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `training_time`.

### 2. Tabla comparativa final
- **Archivo**: `experiment_results_summary.csv` (se deriva del nombre del output)
- **Contenido**: Estadísticas agregadas por método.
  - `mean`: Promedio de cada métrica.
  - `std`: Desviación estándar de cada métrica.

---

## Notas para Tesis

- **Reproducibilidad**: Cada run tiene su propia semilla (`seed + run_id`). Si necesitas reproducir exactamente una run específica, utiliza su `seed` correspondiente.
- **Robustez**: Se recomienda usar al menos `--runs 15` (o más) para que los promedios y desviaciones estándar sean estadísticamente significativas.
- **Métricas**: Todas las métricas de clasificación (`precision`, `recall`, `f1`) usan `zero_division=0` para evitar errores en casos límite.
- **Interrupciones**: No borres el archivo CSV de resultados si quieres reanudar; el script lo lee automáticamente al inicio.

---

## Requisitos

Instala las dependencias antes de ejecutar:

```bash
pip install -r requirements.txt
```

Dependencias principales:
- `numpy`
- `pandas`
- `scikit-learn`
