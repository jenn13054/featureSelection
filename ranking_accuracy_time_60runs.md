# Ranking de Métodos: Accuracy + Tiempo de Ejecución (60 runs)

Este informe ordena los métodos de selección de características del mejor al peor combinando **accuracy** y **tiempo de entrenamiento**. Se utilizan cuatro criterios diferentes.

## 1. Ranking por Eficiencia (Accuracy / Tiempo)

Mide cuánta accuracy se obtiene por segundo de entrenamiento. Mayor es mejor.

| Rank | Método | Accuracy | Tiempo (s) | Eficiencia | Features |
|---|---|---|---|---|---|
| 1 | rf_topk | 0.9356 | 0.0338 | 27.68 | 10.0 |
| 2 | chi2 | 0.9341 | 0.0419 | 22.29 | 10.0 |
| 3 | rfe | 0.9342 | 0.0422 | 22.14 | 10.0 |
| 4 | mutual_info | 0.9356 | 0.1035 | 9.04 | 10.0 |
| 5 | rf_median | 0.9356 | 0.1067 | 8.77 | 23.0 |
| 6 | rfecv | 0.9355 | 0.2052 | 4.56 | 33.0 |
| 7 | l1 | 0.9356 | 0.2911 | 3.21 | 43.1 |
| 8 | variance_threshold | 0.9356 | 2.8370 | 0.33 | 43.1 |
| 9 | baseline | 0.9356 | 3.0762 | 0.30 | 46.0 |

## 2. Ranking por Accuracy (desempate: menor tiempo)

Primero se ordena por accuracy descendente; en empates, gana el método más rápido.

| Rank | Método | Accuracy | Tiempo (s) | Features |
|---|---|---|---|---|
| 1 | rf_topk | 0.9356 | 0.0338 | 10.0 |
| 2 | mutual_info | 0.9356 | 0.1035 | 10.0 |
| 3 | rf_median | 0.9356 | 0.1067 | 23.0 |
| 4 | l1 | 0.9356 | 0.2911 | 43.1 |
| 5 | variance_threshold | 0.9356 | 2.8370 | 43.1 |
| 6 | baseline | 0.9356 | 3.0762 | 46.0 |
| 7 | rfecv | 0.9355 | 0.2052 | 33.0 |
| 8 | rfe | 0.9342 | 0.0422 | 10.0 |
| 9 | chi2 | 0.9341 | 0.0419 | 10.0 |

## 3. Ranking por Score Ponderado (70% accuracy + 30% rapidez)

Se normalizan accuracy y rapidez (inverso del tiempo) al rango [0, 1] y se combinan con pesos 0.7 y 0.3 respectivamente.

| Rank | Método | Accuracy | Tiempo (s) | Score ponderado | Features |
|---|---|---|---|---|---|
| 1 | rf_topk | 0.9356 | 0.0338 | 1.0000 | 10.0 |
| 2 | mutual_info | 0.9356 | 0.1035 | 0.9931 | 10.0 |
| 3 | rf_median | 0.9356 | 0.1067 | 0.9928 | 23.0 |
| 4 | l1 | 0.9356 | 0.2911 | 0.9746 | 43.1 |
| 5 | rfecv | 0.9355 | 0.2052 | 0.9364 | 33.0 |
| 6 | variance_threshold | 0.9356 | 2.8370 | 0.7236 | 43.1 |
| 7 | baseline | 0.9356 | 3.0762 | 0.7000 | 46.0 |
| 8 | rfe | 0.9342 | 0.0422 | 0.3458 | 10.0 |
| 9 | chi2 | 0.9341 | 0.0419 | 0.2992 | 10.0 |

## 4. Ranking por Dominancia de Pareto

Un método domina a otro si tiene mayor o igual accuracy **y** menor o igual tiempo, con al menos una mejora estricta. El Frente 1 contiene las soluciones no dominadas (óptimas).

| Frente | Método | Accuracy | Tiempo (s) | Features |
|---|---|---|---|---|
| 1 | rf_topk | 0.9356 | 0.0338 | 10.0 |
| 2 | mutual_info | 0.9356 | 0.1035 | 10.0 |
| 2 | rfe | 0.9342 | 0.0422 | 10.0 |
| 2 | chi2 | 0.9341 | 0.0419 | 10.0 |
| 3 | rf_median | 0.9356 | 0.1067 | 23.0 |
| 4 | l1 | 0.9356 | 0.2911 | 43.1 |
| 4 | rfecv | 0.9355 | 0.2052 | 33.0 |
| 5 | variance_threshold | 0.9356 | 2.8370 | 43.1 |
| 6 | baseline | 0.9356 | 3.0762 | 46.0 |

## Recomendación Final

Considerando todos los criterios, **`rf_topk`** es consistentemente el mejor método:

- **Accuracy**: 0.9356 (empatado con el baseline y la mayoría de métodos).
- **Tiempo**: 0.0338 s (el más rápido de todos).
- **Features**: 10 (reducción del 78% respecto al baseline).
- **Eficiencia**: 27.68 (la más alta).
- **Frente de Pareto**: Frente 1 (no dominado).

**Conclusión práctica**: Para un trabajo de tesis, `rf_topk` ofrece el mejor compromiso entre accuracy y costo computacional. Si se prioriza únicamente la accuracy sin importar el tiempo, el baseline y varios métodos son equivalentes, pero ninguno supera significativamente a `rf_topk`.