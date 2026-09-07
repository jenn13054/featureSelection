# Prueba t de Student Pareada: Métodos vs Baseline (30 runs)

**Métrica analizada:** accuracy  
**Baseline:** todas las características (46 features)  
**Nivel de significancia:** α = 0.05  
**Hipótesis nula (H₀):** μ_método − μ_baseline = 0  
**Hipótesis alternativa (H₁):** μ_método − μ_baseline ≠ 0

## Tabla de resultados

| Método | Media método | Media baseline | Diferencia | Desv. dif. | t | p-value | IC 95% | d Cohen | Significativo |
|---|---|---|---|---|---|---|---|---|---|
| chi2 | 0.934129 | 0.935629 | -0.001500 | 0.000359 | -22.8639 | < 0.0001 | [-0.001634, -0.001366] | -4.1744 | Sí |
| rfe | 0.934177 | 0.935629 | -0.001452 | 0.000354 | -22.4454 | < 0.0001 | [-0.001585, -0.001320] | -4.0980 | Sí |
| rf_topk | 0.935681 | 0.935629 | 0.000052 | 0.000150 | 1.9064 | 0.066549 | [-0.000004, 0.000108] | 0.3481 | No |
| l1 | 0.935623 | 0.935629 | -0.000006 | 0.000020 | -1.6820 | 0.103300 | [-0.000014, 0.000001] | -0.3071 | No |
| mutual_info | 0.935686 | 0.935629 | 0.000057 | 0.000202 | 1.5437 | 0.133515 | [-0.000018, 0.000132] | 0.2818 | No |
| variance_threshold | 0.935652 | 0.935629 | 0.000023 | 0.000106 | 1.1904 | 0.243561 | [-0.000017, 0.000063] | 0.2173 | No |
| rfecv | 0.935622 | 0.935629 | -0.000008 | 0.000333 | -0.1264 | 0.900274 | [-0.000132, 0.000117] | -0.0231 | No |
| rf_median | 0.935629 | 0.935629 | 0.000000 | 0.000126 | 0.0000 | 1.000000 | [-0.000047, 0.000047] | 0.0000 | No |

## Interpretación

### Métodos significativamente diferentes al baseline (n = 2)

- **chi2**: accuracy menor que baseline (Δ = -0.001500, p = 0.000000, d = -4.1744)
- **rfe**: accuracy menor que baseline (Δ = -0.001452, p = 0.000000, d = -4.0980)

### Métodos no significativamente diferentes al baseline (n = 6)

- **rf_topk**: Δ = 0.000052, p = 0.066549, IC 95% = [-0.000004, 0.000108]
- **l1**: Δ = -0.000006, p = 0.103300, IC 95% = [-0.000014, 0.000001]
- **mutual_info**: Δ = 0.000057, p = 0.133515, IC 95% = [-0.000018, 0.000132]
- **variance_threshold**: Δ = 0.000023, p = 0.243561, IC 95% = [-0.000017, 0.000063]
- **rfecv**: Δ = -0.000008, p = 0.900274, IC 95% = [-0.000132, 0.000117]
- **rf_median**: Δ = 0.000000, p = 1.000000, IC 95% = [-0.000047, 0.000047]

## Notas metodológicas

- Se utiliza la **prueba t pareada** porque cada run es una observación repetida para todos los métodos.
- El **intervalo de confianza del 95%** indica el rango plausible para la verdadera diferencia de medias.
- **d de Cohen** mide el tamaño del efecto: pequeño (<0.2), mediano (0.2–0.5), grande (0.5–0.8), muy grande (>0.8).
- Las diferencias significativas deben interpretarse junto con la relevancia práctica, no solo el p-value.