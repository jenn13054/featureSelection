# Análisis Estadístico: Prueba t de Student (30 runs)

Este informe compara los métodos de selección de características usando **pruebas t pareadas** sobre los resultados de 30 runs independientes.

- **Dataset de entrada:** `experiment_results_30runs.csv`
- **Nivel de significancia (α):** 0.05
- **Corrección de Bonferroni:** aplicada para comparaciones múltiples

## 1. Resumen Descriptivo por Métrica

### accuracy

| Método | Media | Desv. Est. | Mín | Máx | N |
|---|---|---|---|---|---|
| mutual_info | 0.935686 | 0.000390 | 0.935042 | 0.936517 | 30 |
| rf_topk | 0.935681 | 0.000372 | 0.935088 | 0.936425 | 30 |
| variance_threshold | 0.935652 | 0.000348 | 0.935088 | 0.936287 | 30 |
| baseline | 0.935629 | 0.000359 | 0.934996 | 0.936287 | 30 |
| rf_median | 0.935629 | 0.000377 | 0.934904 | 0.936333 | 30 |
| l1 | 0.935623 | 0.000365 | 0.934996 | 0.936287 | 30 |
| rfecv | 0.935622 | 0.000500 | 0.933890 | 0.936379 | 30 |
| rfe | 0.934177 | 0.000087 | 0.933982 | 0.934397 | 30 |
| chi2 | 0.934129 | 0.000037 | 0.934120 | 0.934305 | 30 |

### precision

| Método | Media | Desv. Est. | Mín | Máx | N |
|---|---|---|---|---|---|
| baseline | 0.936792 | 0.000280 | 0.936288 | 0.937399 | 30 |
| l1 | 0.936785 | 0.000287 | 0.936288 | 0.937399 | 30 |
| rf_median | 0.936753 | 0.000292 | 0.936124 | 0.937358 | 30 |
| rf_topk | 0.936752 | 0.000296 | 0.936037 | 0.937407 | 30 |
| variance_threshold | 0.936737 | 0.000279 | 0.936124 | 0.937361 | 30 |
| mutual_info | 0.936727 | 0.000293 | 0.936161 | 0.937538 | 30 |
| rfecv | 0.936700 | 0.000557 | 0.934185 | 0.937445 | 30 |
| rfe | 0.934256 | 0.000069 | 0.934157 | 0.934379 | 30 |
| chi2 | 0.934129 | 0.000035 | 0.934120 | 0.934292 | 30 |

### recall

| Método | Media | Desv. Est. | Mín | Máx | N |
|---|---|---|---|---|---|
| chi2 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 30 |
| rfe | 0.999898 | 0.000066 | 0.999753 | 1.000000 | 30 |
| mutual_info | 0.998603 | 0.000224 | 0.998125 | 0.998964 | 30 |
| rf_topk | 0.998567 | 0.000241 | 0.997977 | 0.998914 | 30 |
| rfecv | 0.998562 | 0.000313 | 0.998026 | 0.999655 | 30 |
| variance_threshold | 0.998552 | 0.000226 | 0.998125 | 0.999062 | 30 |
| rf_median | 0.998506 | 0.000266 | 0.997927 | 0.998914 | 30 |
| l1 | 0.998460 | 0.000251 | 0.997927 | 0.998914 | 30 |
| baseline | 0.998459 | 0.000250 | 0.997927 | 0.998914 | 30 |

### f1

| Método | Media | Desv. Est. | Mín | Máx | N |
|---|---|---|---|---|---|
| mutual_info | 0.966676 | 0.000200 | 0.966341 | 0.967091 | 30 |
| rf_topk | 0.966672 | 0.000191 | 0.966366 | 0.967056 | 30 |
| variance_threshold | 0.966657 | 0.000178 | 0.966362 | 0.966974 | 30 |
| rf_median | 0.966644 | 0.000194 | 0.966267 | 0.967010 | 30 |
| baseline | 0.966643 | 0.000184 | 0.966311 | 0.966984 | 30 |
| rfecv | 0.966642 | 0.000250 | 0.965812 | 0.967030 | 30 |
| l1 | 0.966640 | 0.000188 | 0.966311 | 0.966984 | 30 |
| rfe | 0.965963 | 0.000045 | 0.965861 | 0.966076 | 30 |
| chi2 | 0.965943 | 0.000019 | 0.965938 | 0.966030 | 30 |

### roc_auc

| Método | Media | Desv. Est. | Mín | Máx | N |
|---|---|---|---|---|---|
| l1 | 0.706039 | 0.007675 | 0.691447 | 0.722265 | 30 |
| baseline | 0.706032 | 0.007686 | 0.691324 | 0.722263 | 30 |
| variance_threshold | 0.701646 | 0.007689 | 0.684256 | 0.716061 | 30 |
| rfecv | 0.700714 | 0.008669 | 0.676735 | 0.716609 | 30 |
| rf_median | 0.694277 | 0.007663 | 0.679395 | 0.711343 | 30 |
| rf_topk | 0.675710 | 0.007136 | 0.660807 | 0.687843 | 30 |
| chi2 | 0.650383 | 0.009038 | 0.627108 | 0.665923 | 30 |
| rfe | 0.636620 | 0.007665 | 0.622425 | 0.654073 | 30 |
| mutual_info | 0.624832 | 0.009110 | 0.607712 | 0.642069 | 30 |

### n_features

| Método | Media | Desv. Est. | Mín | Máx | N |
|---|---|---|---|---|---|
| baseline | 46.000000 | 0.000000 | 46.000000 | 46.000000 | 30 |
| l1 | 43.433333 | 1.006302 | 42.000000 | 45.000000 | 30 |
| variance_threshold | 43.066667 | 0.253708 | 43.000000 | 44.000000 | 30 |
| rfecv | 32.933333 | 4.933582 | 25.000000 | 44.000000 | 30 |
| rf_median | 23.000000 | 0.000000 | 23.000000 | 23.000000 | 30 |
| chi2 | 10.000000 | 0.000000 | 10.000000 | 10.000000 | 30 |
| mutual_info | 10.000000 | 0.000000 | 10.000000 | 10.000000 | 30 |
| rf_topk | 10.000000 | 0.000000 | 10.000000 | 10.000000 | 30 |
| rfe | 10.000000 | 0.000000 | 10.000000 | 10.000000 | 30 |

### training_time

| Método | Media | Desv. Est. | Mín | Máx | N |
|---|---|---|---|---|---|
| rf_topk | 0.033797 | 0.000768 | 0.032198 | 0.035084 | 30 |
| chi2 | 0.042148 | 0.002953 | 0.037337 | 0.049138 | 30 |
| rfe | 0.042564 | 0.003044 | 0.038015 | 0.049724 | 30 |
| mutual_info | 0.104764 | 0.010211 | 0.088542 | 0.130239 | 30 |
| rf_median | 0.107419 | 0.011093 | 0.091023 | 0.124613 | 30 |
| rfecv | 0.201108 | 0.044287 | 0.131613 | 0.301778 | 30 |
| l1 | 0.291852 | 0.012337 | 0.272727 | 0.318255 | 30 |
| variance_threshold | 2.834495 | 0.262661 | 2.481762 | 3.547951 | 30 |
| baseline | 3.069886 | 0.234498 | 2.694394 | 3.538983 | 30 |

## 2. Comparación de Cada Método vs Baseline

Para cada métrica se realiza una prueba t pareada entre cada método y el `baseline`. Un valor negativo de `mean_diff` indica que el método tiene menor valor que el baseline; positivo, mayor.

### accuracy vs baseline

| Método | Media Dif. | t | p-value | p-value (Bonferroni) | Significativo (α) | Significativo (Bonferroni) | d de Cohen |
|---|---|---|---|---|---|---|---|
| chi2 | -0.001500 | -22.8639 | < 0.001 | < 0.001 | Sí | Sí | -4.1744 |
| l1 | -0.000006 | -1.6820 | 0.1033 | 0.8264 | No | No | -0.3071 |
| mutual_info | 0.000057 | 1.5437 | 0.1335 | 1.0000 | No | No | 0.2818 |
| rf_median | 0.000000 | 0.0000 | 1.0000 | 1.0000 | No | No | 0.0000 |
| rf_topk | 0.000052 | 1.9064 | 0.0665 | 0.5324 | No | No | 0.3481 |
| rfe | -0.001452 | -22.4454 | < 0.001 | < 0.001 | Sí | Sí | -4.0980 |
| rfecv | -0.000008 | -0.1264 | 0.9003 | 1.0000 | No | No | -0.0231 |
| variance_threshold | 0.000023 | 1.1904 | 0.2436 | 1.0000 | No | No | 0.2173 |

### precision vs baseline

| Método | Media Dif. | t | p-value | p-value (Bonferroni) | Significativo (α) | Significativo (Bonferroni) | d de Cohen |
|---|---|---|---|---|---|---|---|
| chi2 | -0.002663 | -52.0741 | < 0.001 | < 0.001 | Sí | Sí | -9.5074 |
| l1 | -0.000007 | -2.3681 | 0.0248 | 0.1981 | Sí | No | -0.4323 |
| mutual_info | -0.000065 | -2.3507 | 0.0258 | 0.2060 | Sí | No | -0.4292 |
| rf_median | -0.000039 | -2.6026 | 0.0144 | 0.1154 | Sí | No | -0.4752 |
| rf_topk | -0.000040 | -1.9310 | 0.0633 | 0.5065 | No | No | -0.3525 |
| rfe | -0.002536 | -49.8248 | < 0.001 | < 0.001 | Sí | Sí | -9.0967 |
| rfecv | -0.000092 | -0.9544 | 0.3478 | 1.0000 | No | No | -0.1743 |
| variance_threshold | -0.000055 | -3.8496 | < 0.001 | 0.0048 | Sí | Sí | -0.7028 |

### recall vs baseline

| Método | Media Dif. | t | p-value | p-value (Bonferroni) | Significativo (α) | Significativo (Bonferroni) | d de Cohen |
|---|---|---|---|---|---|---|---|
| chi2 | 0.001541 | 33.7891 | < 0.001 | < 0.001 | Sí | Sí | 6.1690 |
| l1 | 0.000002 | 1.0000 | 0.3256 | 1.0000 | No | No | 0.1826 |
| mutual_info | 0.000145 | 6.0896 | < 0.001 | < 0.001 | Sí | Sí | 1.1118 |
| rf_median | 0.000048 | 2.5107 | 0.0179 | 0.1430 | Sí | No | 0.4584 |
| rf_topk | 0.000109 | 5.0244 | < 0.001 | < 0.001 | Sí | Sí | 0.9173 |
| rfe | 0.001439 | 32.1919 | < 0.001 | < 0.001 | Sí | Sí | 5.8774 |
| rfecv | 0.000104 | 1.9659 | 0.0589 | 0.4716 | No | No | 0.3589 |
| variance_threshold | 0.000094 | 4.9788 | < 0.001 | < 0.001 | Sí | Sí | 0.9090 |

### f1 vs baseline

| Método | Media Dif. | t | p-value | p-value (Bonferroni) | Significativo (α) | Significativo (Bonferroni) | d de Cohen |
|---|---|---|---|---|---|---|---|
| chi2 | -0.000700 | -20.7644 | < 0.001 | < 0.001 | Sí | Sí | -3.7910 |
| l1 | -0.000003 | -1.6292 | 0.1141 | 0.9128 | No | No | -0.2974 |
| mutual_info | 0.000033 | 1.7524 | 0.0903 | 0.7222 | No | No | 0.3199 |
| rf_median | 0.000002 | 0.1288 | 0.8984 | 1.0000 | No | No | 0.0235 |
| rf_topk | 0.000030 | 2.0959 | 0.0449 | 0.3595 | Sí | No | 0.3827 |
| rfe | -0.000680 | -20.4514 | < 0.001 | < 0.001 | Sí | Sí | -3.7339 |
| rfecv | -0.000000 | -0.0153 | 0.9879 | 1.0000 | No | No | -0.0028 |
| variance_threshold | 0.000015 | 1.4424 | 0.1599 | 1.0000 | No | No | 0.2634 |

### roc_auc vs baseline

| Método | Media Dif. | t | p-value | p-value (Bonferroni) | Significativo (α) | Significativo (Bonferroni) | d de Cohen |
|---|---|---|---|---|---|---|---|
| chi2 | -0.055649 | -45.3848 | < 0.001 | < 0.001 | Sí | Sí | -8.2861 |
| l1 | 0.000008 | 0.9293 | 0.3604 | 1.0000 | No | No | 0.1697 |
| mutual_info | -0.081200 | -51.0727 | < 0.001 | < 0.001 | Sí | Sí | -9.3246 |
| rf_median | -0.011755 | -24.5446 | < 0.001 | < 0.001 | Sí | Sí | -4.4812 |
| rf_topk | -0.030322 | -38.8813 | < 0.001 | < 0.001 | Sí | Sí | -7.0987 |
| rfe | -0.069412 | -62.1471 | < 0.001 | < 0.001 | Sí | Sí | -11.3465 |
| rfecv | -0.005317 | -3.9777 | < 0.001 | 0.0034 | Sí | Sí | -0.7262 |
| variance_threshold | -0.004386 | -15.3253 | < 0.001 | < 0.001 | Sí | Sí | -2.7980 |

## 3. Comparaciones Pareadas entre Métodos

Se realizan todas las comparaciones 2 a 2 para la métrica **accuracy**. Solo se muestran las diferencias estadísticamente significativas (p < 0.05).

| Método 1 | Método 2 | Media Dif. | t | p-value | Significativo (α) | d de Cohen |
|---|---|---|---|---|---|---|
| baseline | chi2 | 0.001500 | 22.8639 | < 0.001 | Sí | 4.1744 |
| baseline | rfe | 0.001452 | 22.4454 | < 0.001 | Sí | 4.0980 |
| chi2 | l1 | -0.001494 | -22.3691 | < 0.001 | Sí | -4.0840 |
| chi2 | mutual_info | -0.001557 | -21.8100 | < 0.001 | Sí | -3.9820 |
| chi2 | rf_median | -0.001500 | -21.8233 | < 0.001 | Sí | -3.9844 |
| chi2 | rf_topk | -0.001552 | -22.8256 | < 0.001 | Sí | -4.1674 |
| chi2 | rfe | -0.000048 | -2.9749 | 0.0059 | Sí | -0.5431 |
| chi2 | rfecv | -0.001492 | -16.2513 | < 0.001 | Sí | -2.9671 |
| chi2 | variance_threshold | -0.001523 | -24.2155 | < 0.001 | Sí | -4.4211 |
| l1 | rfe | 0.001446 | 21.9353 | < 0.001 | Sí | 4.0048 |
| mutual_info | rfe | 0.001509 | 21.9438 | < 0.001 | Sí | 4.0064 |
| rf_median | rf_topk | -0.000052 | -2.2263 | 0.0339 | Sí | -0.4065 |
| rf_median | rfe | 0.001452 | 21.6631 | < 0.001 | Sí | 3.9551 |
| rf_topk | rfe | 0.001504 | 23.1622 | < 0.001 | Sí | 4.2288 |
| rfe | rfecv | -0.001445 | -16.2349 | < 0.001 | Sí | -2.9641 |
| rfe | variance_threshold | -0.001475 | -24.3380 | < 0.001 | Sí | -4.4435 |

## 4. Interpretación de Resultados

- **p-value < 0.05**: existe evidencia estadística suficiente para rechazar la hipótesis nula de igualdad de medias.

- **Corrección de Bonferroni**: reduce la probabilidad de falsos positivos al realizar múltiples comparaciones, pero es conservadora.

- **d de Cohen**: tamaño del efecto. Valores aproximados: |d| < 0.2 (pequeño), 0.2–0.5 (medio), 0.5–0.8 (grande), > 0.8 (muy grande).

- Dado que las desviaciones estándar de accuracy son muy pequeñas (~0.0004), incluso diferencias mínimas pueden resultar estadísticamente significativas, aunque no necesariamente prácticamente relevantes.
