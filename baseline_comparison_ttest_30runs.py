"""
Prueba t de Student pareada: cada método de selección de características vs baseline.

Dataset: experiment_results_30runs.csv
Métrica principal: accuracy
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}")
    return pd.read_csv(path)


def paired_ttest_against_baseline(df: pd.DataFrame, metric: str, baseline: str = "baseline") -> pd.DataFrame:
    """Prueba t pareada de cada método contra el baseline."""
    pivot = df.pivot(index="run_id", columns="method", values=metric)
    baseline_values = pivot[baseline].values
    methods = [m for m in pivot.columns if m != baseline]

    rows = []
    for method in methods:
        method_values = pivot[method].values
        valid = ~(np.isnan(method_values) | np.isnan(baseline_values))
        a = method_values[valid]
        b = baseline_values[valid]

        n = len(a)
        mean_diff = float(np.mean(a - b))
        std_diff = float(np.std(a - b, ddof=1))
        se_diff = std_diff / np.sqrt(n)
        t_stat, p_value = stats.ttest_rel(a, b)

        # Intervalo de confianza del 95% para la diferencia de medias
        ci_margin = stats.t.ppf(0.975, df=n - 1) * se_diff
        ci_lower = mean_diff - ci_margin
        ci_upper = mean_diff + ci_margin

        # d de Cohen pareada
        d = mean_diff / std_diff if std_diff > 0 else 0.0

        rows.append({
            "method": method,
            "n": n,
            "method_mean": float(np.mean(a)),
            "baseline_mean": float(np.mean(b)),
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "se_diff": se_diff,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "cohens_d": d,
            "significant_05": p_value < 0.05,
        })

    return pd.DataFrame(rows)


def plot_comparison(result_df: pd.DataFrame, output_path: str) -> None:
    """Genera un gráfico de diferencias de medias con intervalos de confianza."""
    result_df = result_df.sort_values("mean_diff")
    methods = result_df["method"].tolist()
    diffs = result_df["mean_diff"].values
    ci_lower = result_df["ci_95_lower"].values
    ci_upper = result_df["ci_95_upper"].values
    errors = [diffs - ci_lower, ci_upper - diffs]

    colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in diffs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(methods, diffs, xerr=errors, color=colors, alpha=0.75, edgecolor="black", capsize=4)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Diferencia de media de accuracy vs baseline", fontsize=12)
    ax.set_title("Prueba t pareada: método vs baseline (30 runs)", fontsize=14)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Anotar p-values
    for i, (method, p) in enumerate(zip(methods, result_df["p_value"])):
        label = f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001"
        ax.text(diffs[i] + (ci_upper[i] - ci_lower[i]) * 0.05, i, label, va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Gráfico guardado: {output_path}")


def generate_report(result_df: pd.DataFrame, output_path: str) -> None:
    """Genera informe Markdown con la prueba t pareada vs baseline."""
    lines = []
    lines.append("# Prueba t de Student Pareada: Métodos vs Baseline (30 runs)\n")
    lines.append("**Métrica analizada:** accuracy  ")
    lines.append("**Baseline:** todas las características (46 features)  ")
    lines.append("**Nivel de significancia:** α = 0.05  ")
    lines.append("**Hipótesis nula (H₀):** μ_método − μ_baseline = 0  ")
    lines.append("**Hipótesis alternativa (H₁):** μ_método − μ_baseline ≠ 0\n")

    lines.append("## Tabla de resultados\n")
    lines.append("| Método | Media método | Media baseline | Diferencia | Desv. dif. | t | p-value | IC 95% | d Cohen | Significativo |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for _, row in result_df.iterrows():
        sig = "Sí" if row["significant_05"] else "No"
        ci = f"[{row['ci_95_lower']:.6f}, {row['ci_95_upper']:.6f}]"
        p_str = f"{row['p_value']:.6f}" if row["p_value"] >= 0.0001 else "< 0.0001"
        lines.append(
            f"| {row['method']} | {row['method_mean']:.6f} | {row['baseline_mean']:.6f} | "
            f"{row['mean_diff']:.6f} | {row['std_diff']:.6f} | {row['t_statistic']:.4f} | "
            f"{p_str} | {ci} | {row['cohens_d']:.4f} | {sig} |"
        )

    lines.append("\n## Interpretación\n")
    significant = result_df[result_df["significant_05"]]
    not_significant = result_df[~result_df["significant_05"]]

    lines.append(f"### Métodos significativamente diferentes al baseline (n = {len(significant)})\n")
    if significant.empty:
        lines.append("Ningún método difiere significativamente del baseline.\n")
    else:
        for _, row in significant.iterrows():
            direction = "mayor" if row["mean_diff"] > 0 else "menor"
            lines.append(
                f"- **{row['method']}**: accuracy {direction} que baseline "
                f"(Δ = {row['mean_diff']:.6f}, p = {row['p_value']:.6f}, d = {row['cohens_d']:.4f})"
            )
        lines.append("")

    lines.append(f"### Métodos no significativamente diferentes al baseline (n = {len(not_significant)})\n")
    for _, row in not_significant.iterrows():
        lines.append(
            f"- **{row['method']}**: Δ = {row['mean_diff']:.6f}, p = {row['p_value']:.6f}, "
            f"IC 95% = [{row['ci_95_lower']:.6f}, {row['ci_95_upper']:.6f}]"
        )

    lines.append("\n## Notas metodológicas\n")
    lines.append("- Se utiliza la **prueba t pareada** porque cada run es una observación repetida para todos los métodos.")
    lines.append("- El **intervalo de confianza del 95%** indica el rango plausible para la verdadera diferencia de medias.")
    lines.append("- **d de Cohen** mide el tamaño del efecto: pequeño (<0.2), mediano (0.2–0.5), grande (0.5–0.8), muy grande (>0.8).")
    lines.append("- Las diferencias significativas deben interpretarse junto con la relevancia práctica, no solo el p-value.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Informe guardado: {output_path}")


def main():
    warnings.filterwarnings("ignore")

    df = load_data("experiment_results_30runs.csv")
    result_df = paired_ttest_against_baseline(df, metric="accuracy", baseline="baseline")

    # Ordenar por p-value
    result_df = result_df.sort_values("p_value")

    # Guardar CSV
    result_df.to_csv("baseline_comparison_ttest_30runs.csv", index=False)
    print("CSV guardado: baseline_comparison_ttest_30runs.csv")

    # Generar informe y gráfico
    generate_report(result_df, "baseline_comparison_ttest_30runs.md")
    plot_comparison(result_df, "baseline_comparison_ttest_30runs.png")


if __name__ == "__main__":
    main()
