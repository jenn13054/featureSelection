# Experimento de Selección de Características

Script reproducible para comparar métodos de selección de características (Filter, Wrapper, Embedded) mediante múltiples ejecuciones con semillas aleatorias distintas.

---

## Requisitos Previos

- Python **3.8 o superior**.
- El dataset `data_final.csv` debe encontrarse en la raíz del proyecto (o indicar su ruta con `--data`).

---

## Instalación con Entorno Virtual (`venv`)

Se recomienda usar un entorno virtual para aislar las dependencias del proyecto.

### 1. Clonar o descargar el proyecto

```bash
cd data-analysis-tecnm
```

### 2. Crear el entorno virtual

```bash
python3 -m venv venv
```

### 3. Activar el entorno virtual

- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

- **Windows (CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```

- **Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate.ps1
  ```

> Una vez activado, tu terminal mostrará el prefijo `(venv)`.

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Uso Básico

### Ejecutar el experimento con valores por defecto (15 runs, todos los métodos)

```bash
python3 script.py
```

### Ejecutar 20 runs seleccionando métodos específicos

```bash
python3 script.py --runs 20 --methods baseline chi2 l1 rf_topk
```

### Usar un dataset o semilla diferentes

```bash
python3 script.py --runs 30 --data mi_dataset.csv --seed 123 --output resultados.csv
```

### Reanudar un experimento interrumpido

Si el experimento se detiene, simplemente vuelve a ejecutar el mismo comando. El script detecta automáticamente las runs ya completadas en el archivo CSV y continúa desde donde se quedó:

```bash
python3 script.py --runs 50 --output experiment_results.csv
```

---

## Estructura del Proyecto

```
data-analysis-tecnm/
├── venv/                       # Entorno virtual (no se sube al repo)
├── data_final.csv              # Dataset principal
├── script.py                   # Script de experimentos
├── requirements.txt            # Dependencias de Python
├── experiment.md               # Guía detallada del experimento
├── bitacora.md                 # Bitácora de cambios
└── README.md                   # Este archivo
```

---

## Archivos Generados

| Archivo | Descripción |
|---|---|
| `experiment_results.csv` | Resultados crudos: una fila por método y por run. |
| `experiment_results_summary.csv` | Tabla comparativa final con promedio y desviación estándar por método. |
| `selected_features_l1.csv` | Lista de features seleccionadas por el método L1 (si aplica). |
| `selected_features_rf_topk.csv` | Lista de features seleccionadas por RF top-k (si aplica). |

---

## Desactivar el Entorno Virtual

Cuando termines de trabajar, puedes salir del entorno virtual con:

```bash
deactivate
```

---

## Métodos Disponibles

Puedes consultar la lista completa de métodos en la guía [`experiment.md`](experiment.md).

Los identificadores para `--methods` son:

- `baseline`
- `variance_threshold`
- `chi2`
- `mutual_info`
- `rfe`
- `rfecv`
- `l1`
- `rf_topk`
- `rf_median`
# featureSelection
# featureSelection
