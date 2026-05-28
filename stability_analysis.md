# Análisis de Estabilidad de Features

## Resumen Ejecutivo

Este análisis mide la **consistencia** con la que cada método de selección de características elige las mismas features a través de 5 folds de Cross-Validation. Un método estable debería tener un **Jaccard Index** cercano a 1.0.

## Resultados de Estabilidad (5-Fold CV)

| Método | CV Accuracy | Jaccard Mean | Jaccard Std | N Features |
|--------|-------------|--------------|-------------|------------|
| baseline | 0.9356 | 1.0000 | 0.0000 | 46.0 |
| variance_threshold | 0.9356 | 0.9909 | 0.0111 | 43.2 |
| chi2 | 0.9341 | 0.9273 | 0.0891 | 10.0 |
| mutual_info | 0.9356 | 0.5531 | 0.0654 | 10.0 |
| l1 | 0.9356 | 0.9422 | 0.0361 | 43.4 |
| rf_topk | 0.9356 | 1.0000 | 0.0000 | 10.0 |
| rf_median | 0.9356 | 1.0000 | 0.0000 | 23.0 |

## Hallazgos Clave

1. **Métodos perfectamente estables (Jaccard = 1.0):**
   - `baseline`, `rf_topk`, `rf_median`
   
2. **Muy estables:**
   - `variance_threshold` (0.99) y `l1` (0.94) — eliminan muy pocas features.
   - `chi2` (0.93) — estable al fijar k=10.

3. **Menos estable:**
   - `mutual_info` (0.55) — el conjunto de 10 features varía significativamente entre folds.

## Features Más Consistentes (Top 10)

Ordenadas por frecuencia promedio de selección entre todos los métodos:

| Feature | Frecuencia Promedio |
|---------|---------------------|
| PNA | 83.33% |
| average.first.period | 83.33% |
| general.math.eval_codes | 66.67% |
| scholarship.type_codes | 66.67% |
| gender_codes | 66.67% |
| mother.education.complete_codes | 66.67% |
| program_codes | 66.67% |
| region_codes | 66.67% |
| scholarship.perc | 66.67% |
| school.cost_codes | 66.67% |

## Features Unánimes

Features seleccionadas en **todos los folds por todos los métodos** evaluados:

_Ninguna feature alcanzó unanimidad absoluta entre todos los métodos._

## Recomendaciones

- Si buscas **estabilidad máxima** con pocos features: usa `rf_topk` (10 features, Jaccard=1.0).
- Si buscas un **balance** entre reducción y estabilidad: `chi2` o `l1` son buenas opciones.
- Evita depender únicamente de `mutual_info` para selección crítica, dado su alta variabilidad.
- Las features `average.first.period`, `general.math.eval_codes`, `scholarship.type_codes` y `gender_codes` son las más robustamente seleccionadas por múltiples métodos.

---
*Generado automáticamente por script.py*
