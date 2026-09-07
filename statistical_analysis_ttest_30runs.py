"""
Análisis estadístico con prueba t de Student para los resultados de 30 runs.

Este script carga 'experiment_results_30runs.csv', realiza pruebas t pareadas
sobre las métricas de cada método y genera un informe en Markdown.

Uso:
    python3 statistical_analysis_ttest_30runs.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError as exc:
    raise ImportError(
        "scipy es necesario para las pruebas t. "
        "Instálalo con: pip install scipy"
    ) from exc


INPUT_CSV = "experiment_results_30runs.csv"
OUTPUT_MD = "statistical_analysis_ttest_30runs.md"
OUTPUT_CSV = "statistical_analysis_ttest_30runs.csv"
ALPHA = 0.05


def load_results(path: str) -> pd.DataFrame:
    """Carga el CSV de resultados y valida su contenido mínimo."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path)
    required = {"run_id", "method", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El CSV debe contener las columnas: {missing}")

    return df


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Devuelve un DataFrame con métodos en columnas y runs en filas."""
    pivot = df.pivot(index="run_id", columns="method", values=metric)
    return pivot


def paired_ttest(method_a: np.ndarray, method_b: np.ndarray) -> tuple:
    """
    Prueba t pareada bilateral para dos muestras relacionadas.

    Retorna (t_statistic, p_value).
    """
    if len(method_a) != len(method_b):
        raise ValueError("Las muestras deben tener el mismo tamaño.")

    # scipy.stats.ttest_rel ya ignora NaNs por defecto en versiones recientes,
    # pero lo hacemos explícito para robustez.
    mask = ~(np.isnan(method_a) | np.isnan(method_b))
    a = method_a[mask]
    b = method_b[mask]

    if len(a) < 2:
        return np.nan, np.nan

    t_stat, p_value = stats.ttest_rel(a, b)
    return float(t_stat), float(p_value)


def cohens_d(method_a: np.ndarray, method_b: np.ndarray) -> float:
    """
    Calcula la d de Cohen para muestras pareadas (tamaño del efecto).

    Fórmula: d = mean(diff) / std(diff)
    """
    mask = ~(np.isnan(method_a) | np.isnan(method_b))
    diff = method_a[mask] - method_b[mask]

    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return 0.0

    return float(diff.mean() / diff.std(ddof=1))


def bonferroni_alpha(n_comparisons: int, alpha: float = ALPHA) -> float:
    """Ajusta el nivel de significancia por el método de Bonferroni."""
    if n_comparisons <= 0:
        return alpha
    return alpha / n_comparisons


def pairwise_ttests(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Genera todas las comparaciones pareadas entre métodos para una métrica.

    Incluye estadístico t, p-value, p-value corregido (Bonferroni) y d de Cohen.
    """
    pivot = pivot_metric(df, metric)
    methods = pivot.columns.tolist()

    rows = []
    n_comparisons = len(methods) * (len(methods) - 1) // 2
    corrected_alpha = bonferroni_alpha(n_comparisons)

    for i, method_i in enumerate(methods):
        for j in range(i + 1, len(methods)):
            method_j = methods[j]
            a = pivot[method_i].values
            b = pivot[method_j].values

            t_stat, p_value = paired_ttest(a, b)
            d_value = cohens_d(a, b)
            mean_diff = float(np.nanmean(a - b))

            rows.append(
                {
                    "method_1": method_i,
                    "method_2": method_j,
                    "mean_diff": mean_diff,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "p_value_bonferroni": min(p_value * n_comparisons, 1.0),
                    "significant_uncorrected": p_value < ALPHA,
                    "significant_bonferroni": p_value < corrected_alpha,
                    "cohens_d": d_value,
                    "n_runs": int(np.sum(~(np.isnan(a) | np.isnan(b)))),
                }
            )

    return pd.DataFrame(rows)


def comparison_vs_baseline(df: pd.DataFrame, metric: str, baseline: str = "baseline") -> pd.DataFrame:
    """
    Compara cada método contra un método de referencia (baseline) con prueba t pareada.
    """
    pivot = pivot_metric(df, metric)
    if baseline not in pivot.columns:
        raise ValueError(f"El método baseline '{baseline}' no existe en los datos.")

    baseline_values = pivot[baseline].values
    methods = [m for m in pivot.columns if m != baseline]

    rows = []
    n_comparisons = len(methods)
    corrected_alpha = bonferroni_alpha(n_comparisons)

    for method in methods:
        method_values = pivot[method].values
        t_stat, p_value = paired_ttest(method_values, baseline_values)
        d_value = cohens_d(method_values, baseline_values)
        mean_diff = float(np.nanmean(method_values - baseline_values))

        rows.append(
            {
                "method": method,
                "baseline": baseline,
                "mean_diff": mean_diff,
                "t_statistic": t_stat,
                "p_value": p_value,
                "p_value_bonferroni": min(p_value * n_comparisons, 1.0),
                "significant_uncorrected": p_value < ALPHA,
                "significant_bonferroni": p_value < corrected_alpha,
                "cohens_d": d_value,
                "n_runs": int(np.sum(~(np.isnan(method_values) | np.isnan(baseline_values)))),
            }
        )

    return pd.DataFrame(rows)


def metric_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Resumen estadístico básico por método para una métrica dada."""
    summary = df.groupby("method")[metric].agg(["mean", "std", "min", "max", "count"])
    summary = summary.reset_index()
    return summary


def format_pvalue(p: float) -> str:
    """Formatea p-values para tablas Markdown."""
    if np.isnan(p):
        return "NaN"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def generate_markdown_report(
    df: pd.DataFrame,
    pairwise_results: dict,
    baseline_results: dict,
    output_path: str,
) -> None:
    """Genera el informe Markdown con todos los resultados."""
    lines = []
    lines.append("# Análisis Estadístico: Prueba t de Student (30 runs)\n")
    lines.append(
        "Este informe compara los métodos de selección de características usando "
        "**pruebas t pareadas** sobre los resultados de 30 runs independientes.\n"
    )
    lines.append(f"- **Dataset de entrada:** `{INPUT_CSV}`")
    lines.append(f"- **Nivel de significancia (α):** {ALPHA}")
    lines.append(f"- **Corrección de Bonferroni:** aplicada para comparaciones múltiples\n")

    # Resumen por métrica
    lines.append("## 1. Resumen Descriptivo por Métrica\n")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "n_features", "training_time"]

    for metric in metrics:
        if metric not in df.columns:
            continue

        lines.append(f"### {metric}\n")
        summary = metric_summary(df, metric)
        summary = summary.sort_values("mean", ascending=False if metric != "training_time" else True)

        lines.append("| Método | Media | Desv. Est. | Mín | Máx | N |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['method']} | {row['mean']:.6f} | {row['std']:.6f} | "
                f"{row['min']:.6f} | {row['max']:.6f} | {int(row['count'])} |"
            )
        lines.append("")

    # Comparación vs baseline por métrica
    lines.append("## 2. Comparación de Cada Método vs Baseline\n")
    lines.append(
        "Para cada métrica se realiza una prueba t pareada entre cada método y el "
        "`baseline`. Un valor negativo de `mean_diff` indica que el método tiene "
        "menor valor que el baseline; positivo, mayor.\n"
    )

    for metric, result_df in baseline_results.items():
        lines.append(f"### {metric} vs baseline\n")
        lines.append(
            "| Método | Media Dif. | t | p-value | p-value (Bonferroni) | "
            "Significativo (α) | Significativo (Bonferroni) | d de Cohen |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|"
        )

        for _, row in result_df.iterrows():
            sig = "Sí" if row["significant_uncorrected"] else "No"
            sig_bonf = "Sí" if row["significant_bonferroni"] else "No"
            lines.append(
                f"| {row['method']} | {row['mean_diff']:.6f} | {row['t_statistic']:.4f} | "
                f"{format_pvalue(row['p_value'])} | {format_pvalue(row['p_value_bonferroni'])} | "
                f"{sig} | {sig_bonf} | {row['cohens_d']:.4f} |"
            )
        lines.append("")

    # Comparaciones pareadas
    lines.append("## 3. Comparaciones Pareadas entre Métodos\n")
    lines.append(
        "Se realizan todas las comparaciones 2 a 2 para la métrica **accuracy**. "
        "Solo se muestran las diferencias estadísticamente significativas (p < 0.05).\n"
    )

    acc_pairwise = pairwise_results["accuracy"]
    significant = acc_pairwise[acc_pairwise["significant_uncorrected"]].copy()

    if significant.empty:
        lines.append("No se encontraron diferencias significativas en accuracy entre ningún par de métodos.\n")
    else:
        lines.append(
            "| Método 1 | Método 2 | Media Dif. | t | p-value | Significativo (α) | d de Cohen |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in significant.iterrows():
            lines.append(
                f"| {row['method_1']} | {row['method_2']} | {row['mean_diff']:.6f} | "
                f"{row['t_statistic']:.4f} | {format_pvalue(row['p_value'])} | "
                f"{'Sí' if row['significant_uncorrected'] else 'No'} | {row['cohens_d']:.4f} |"
            )
        lines.append("")

    # Interpretación
    lines.append("## 4. Interpretación de Resultados\n")
    lines.append(
        "- **p-value < 0.05**: existe evidencia estadística suficiente para rechazar "
        "la hipótesis nula de igualdad de medias.\n"
    )
    lines.append(
        "- **Corrección de Bonferroni**: reduce la probabilidad de falsos positivos "
        "al realizar múltiples comparaciones, pero es conservadora.\n"
    )
    lines.append(
        "- **d de Cohen**: tamaño del efecto. Valores aproximados: |d| < 0.2 (pequeño), "
        "0.2–0.5 (medio), 0.5–0.8 (grande), > 0.8 (muy grande).\n"
    )
    lines.append(
        "- Dado que las desviaciones estándar de accuracy son muy pequeñas (~0.0004), "
        "incluso diferencias mínimas pueden resultar estadísticamente significativas, "
        "aunque no necesariamente prácticamente relevantes.\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    """Punto de entrada principal."""
    warnings.filterwarnings("ignore")

    print(f"Cargando resultados desde {INPUT_CSV}...")
    df = load_results(INPUT_CSV)

    methods = df["method"].unique()
    print(f"Métodos detectados ({len(methods)}): {', '.join(sorted(methods))}")
    print(f"Runs detectados: {df['run_id'].nunique()}\n")

    metrics_for_pairwise = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    pairwise_results = {}
    baseline_results = {}

    for metric in metrics_for_pairwise:
        if metric not in df.columns:
            print(f"Métrica '{metric}' no encontrada, se omite.")
            continue

        print(f"Realizando pruebas t pareadas para '{metric}'...")
        pairwise_results[metric] = pairwise_ttests(df, metric)
        baseline_results[metric] = comparison_vs_baseline(df, metric, baseline="baseline")

    # Guardar CSVs detallados
    for metric, result_df in pairwise_results.items():
        csv_path = f"pairwise_ttest_{metric}_30runs.csv"
        result_df.to_csv(csv_path, index=False)
        print(f"Guardado: {csv_path}")

    for metric, result_df in baseline_results.items():
        csv_path = f"baseline_ttest_{metric}_30runs.csv"
        result_df.to_csv(csv_path, index=False)
        print(f"Guardado: {csv_path}")

    # Generar informe Markdown
    print(f"\nGenerando informe Markdown: {OUTPUT_MD}...")
    generate_markdown_report(df, pairwise_results, baseline_results, OUTPUT_MD)

    # Guardar también un CSV resumen de comparaciones vs baseline para accuracy
    baseline_results["accuracy"].to_csv(OUTPUT_CSV, index=False)
    print(f"Guardado resumen vs baseline (accuracy): {OUTPUT_CSV}")

    print("\nAnálisis estadístico completado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
