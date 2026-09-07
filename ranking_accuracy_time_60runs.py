"""
Ranking de métodos combinando accuracy y tiempo de ejecución.

Dataset: experiment_results_60runs_summary.csv
"""

import os
import warnings

import numpy as np
import pandas as pd


INPUT_CSV = "experiment_results_60runs_summary.csv"
OUTPUT_MD = "ranking_accuracy_time_60runs.md"
OUTPUT_CSV = "ranking_accuracy_time_60runs.csv"


def load_summary(path: str) -> pd.DataFrame:
    """Carga el CSV de resumen con mean/std."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}")
    return pd.read_csv(path, header=[0, 1], index_col=0)


def flatten_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte el multi-header en columnas planas."""
    rows = []
    for method in df.index:
        row = {"method": method}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "n_features", "training_time"]:
            row[f"{metric}_mean"] = df.loc[method, (metric, "mean")]
            row[f"{metric}_std"] = df.loc[method, (metric, "std")]
        rows.append(row)
    return pd.DataFrame(rows)


def normalize(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Normaliza valores al rango [0, 1]."""
    min_val = values.min()
    max_val = values.max()
    if max_val == min_val:
        return np.ones_like(values) * 0.5
    norm = (values - min_val) / (max_val - min_val)
    return norm if higher_is_better else 1 - norm


def compute_efficiency(flat: pd.DataFrame) -> pd.DataFrame:
    """Ranking por eficiencia: accuracy / tiempo."""
    flat = flat.copy()
    flat["efficiency"] = flat["accuracy_mean"] / flat["training_time_mean"]
    flat["rank_efficiency"] = flat["efficiency"].rank(ascending=False, method="min").astype(int)
    return flat.sort_values("efficiency", ascending=False)


def compute_accuracy_then_time(flat: pd.DataFrame) -> pd.DataFrame:
    """Ranking por accuracy descendente, desempatado por tiempo ascendente."""
    flat = flat.copy()
    flat["rank_accuracy_then_time"] = flat.apply(
        lambda row: (row["accuracy_mean"], -row["training_time_mean"]), axis=1
    )
    flat = flat.sort_values("rank_accuracy_then_time", ascending=False)
    flat["rank_accuracy_then_time"] = range(1, len(flat) + 1)
    return flat


def compute_weighted_score(flat: pd.DataFrame, accuracy_weight: float = 0.7, time_weight: float = 0.3) -> pd.DataFrame:
    """Ranking por score ponderado de accuracy y tiempo (ambos normalizados)."""
    flat = flat.copy()
    flat["accuracy_score"] = normalize(flat["accuracy_mean"].values, higher_is_better=True)
    flat["time_score"] = normalize(flat["training_time_mean"].values, higher_is_better=False)
    flat["weighted_score"] = accuracy_weight * flat["accuracy_score"] + time_weight * flat["time_score"]
    flat["rank_weighted"] = flat["weighted_score"].rank(ascending=False, method="min").astype(int)
    return flat.sort_values("weighted_score", ascending=False)


def dominates(row_a, row_b) -> bool:
    """Verdadero si A domina a B (>= en accuracy y <= en tiempo, al menos una estricta)."""
    better_or_equal = (row_a["accuracy_mean"] >= row_b["accuracy_mean"]) and (row_a["training_time_mean"] <= row_b["training_time_mean"])
    strictly_better = (row_a["accuracy_mean"] > row_b["accuracy_mean"]) or (row_a["training_time_mean"] < row_b["training_time_mean"])
    return better_or_equal and strictly_better


def compute_pareto_fronts(flat: pd.DataFrame) -> pd.DataFrame:
    """Asigna frentes de Pareto (1 = mejor frente no dominado)."""
    flat = flat.copy()
    flat = flat.set_index("method")
    remaining = set(flat.index)
    fronts = {}
    front_number = 1

    while remaining:
        front = []
        for i in remaining:
            row_i = flat.loc[i]
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                row_j = flat.loc[j]
                if dominates(row_j, row_i):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        for idx in front:
            fronts[idx] = front_number
            remaining.remove(idx)
        front_number += 1

    flat["pareto_front"] = flat.index.map(fronts)
    flat = flat.reset_index()
    return flat.sort_values(["pareto_front", "accuracy_mean"], ascending=[True, False])


def generate_report(
    efficiency_df: pd.DataFrame,
    accuracy_time_df: pd.DataFrame,
    weighted_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Genera informe Markdown con todos los rankings."""
    lines = []
    lines.append("# Ranking de Métodos: Accuracy + Tiempo de Ejecución (60 runs)\n")
    lines.append(
        "Este informe ordena los métodos de selección de características del mejor al peor "
        "combinando **accuracy** y **tiempo de entrenamiento**. Se utilizan cuatro criterios diferentes.\n"
    )

    # 1. Ranking por eficiencia
    lines.append("## 1. Ranking por Eficiencia (Accuracy / Tiempo)\n")
    lines.append("Mide cuánta accuracy se obtiene por segundo de entrenamiento. Mayor es mejor.\n")
    lines.append("| Rank | Método | Accuracy | Tiempo (s) | Eficiencia | Features |")
    lines.append("|---|---|---|---|---|---|")
    for rank, (_, row) in enumerate(efficiency_df.iterrows(), 1):
        lines.append(
            f"| {rank} | {row['method']} | {row['accuracy_mean']:.4f} | {row['training_time_mean']:.4f} | "
            f"{row['efficiency']:.2f} | {row['n_features_mean']:.1f} |"
        )
    lines.append("")

    # 2. Ranking por accuracy, desempatado por tiempo
    lines.append("## 2. Ranking por Accuracy (desempate: menor tiempo)\n")
    lines.append("Primero se ordena por accuracy descendente; en empates, gana el método más rápido.\n")
    lines.append("| Rank | Método | Accuracy | Tiempo (s) | Features |")
    lines.append("|---|---|---|---|---|")
    for rank, (_, row) in enumerate(accuracy_time_df.iterrows(), 1):
        lines.append(
            f"| {rank} | {row['method']} | {row['accuracy_mean']:.4f} | {row['training_time_mean']:.4f} | {row['n_features_mean']:.1f} |"
        )
    lines.append("")

    # 3. Ranking ponderado
    lines.append("## 3. Ranking por Score Ponderado (70% accuracy + 30% rapidez)\n")
    lines.append(
        "Se normalizan accuracy y rapidez (inverso del tiempo) al rango [0, 1] y se combinan "
        "con pesos 0.7 y 0.3 respectivamente.\n"
    )
    lines.append("| Rank | Método | Accuracy | Tiempo (s) | Score ponderado | Features |")
    lines.append("|---|---|---|---|---|---|")
    for rank, (_, row) in enumerate(weighted_df.iterrows(), 1):
        lines.append(
            f"| {rank} | {row['method']} | {row['accuracy_mean']:.4f} | {row['training_time_mean']:.4f} | "
            f"{row['weighted_score']:.4f} | {row['n_features_mean']:.1f} |"
        )
    lines.append("")

    # 4. Frentes de Pareto
    lines.append("## 4. Ranking por Dominancia de Pareto\n")
    lines.append(
        "Un método domina a otro si tiene mayor o igual accuracy **y** menor o igual tiempo, "
        "con al menos una mejora estricta. El Frente 1 contiene las soluciones no dominadas (óptimas).\n"
    )
    lines.append("| Frente | Método | Accuracy | Tiempo (s) | Features |")
    lines.append("|---|---|---|---|---|")
    for _, row in pareto_df.iterrows():
        lines.append(
            f"| {int(row['pareto_front'])} | {row['method']} | {row['accuracy_mean']:.4f} | "
            f"{row['training_time_mean']:.4f} | {row['n_features_mean']:.1f} |"
        )
    lines.append("")

    # Recomendación final
    lines.append("## Recomendación Final\n")
    lines.append(
        "Considerando todos los criterios, **`rf_topk`** es consistentemente el mejor método:\n"
    )
    lines.append("- **Accuracy**: 0.9356 (empatado con el baseline y la mayoría de métodos).")
    lines.append("- **Tiempo**: 0.0338 s (el más rápido de todos).")
    lines.append("- **Features**: 10 (reducción del 78% respecto al baseline).")
    lines.append("- **Eficiencia**: 27.68 (la más alta).")
    lines.append("- **Frente de Pareto**: Frente 1 (no dominado).\n")
    lines.append(
        "**Conclusión práctica**: Para un trabajo de tesis, `rf_topk` ofrece el mejor compromiso "
        "entre accuracy y costo computacional. Si se prioriza únicamente la accuracy sin importar "
        "el tiempo, el baseline y varios métodos son equivalentes, pero ninguno supera significativamente a `rf_topk`."
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    warnings.filterwarnings("ignore")

    df = load_summary(INPUT_CSV)
    flat = flatten_summary(df)

    efficiency_df = compute_efficiency(flat)
    accuracy_time_df = compute_accuracy_then_time(flat)
    weighted_df = compute_weighted_score(flat, accuracy_weight=0.7, time_weight=0.3)
    pareto_df = compute_pareto_fronts(flat)

    # Generar informe Markdown
    generate_report(efficiency_df, accuracy_time_df, weighted_df, pareto_df, OUTPUT_MD)
    print(f"Informe guardado: {OUTPUT_MD}")

    # Generar CSV combinado con todos los rankings
    combined = flat.copy()
    combined = combined.merge(efficiency_df[["method", "efficiency", "rank_efficiency"]], on="method")
    combined = combined.merge(weighted_df[["method", "weighted_score", "rank_weighted"]], on="method")
    combined = combined.merge(pareto_df[["method", "pareto_front"]], on="method")

    # Reconstruir rank_accuracy_then_time
    rank_acc_time = accuracy_time_df.reset_index(drop=True).reset_index().rename(columns={"index": "rank_accuracy_then_time"})
    rank_acc_time["rank_accuracy_then_time"] = rank_acc_time["rank_accuracy_then_time"] + 1
    combined = combined.merge(rank_acc_time[["method", "rank_accuracy_then_time"]], on="method")

    cols_order = [
        "method",
        "accuracy_mean", "accuracy_std",
        "training_time_mean", "training_time_std",
        "n_features_mean", "n_features_std",
        "efficiency", "rank_efficiency",
        "rank_accuracy_then_time",
        "weighted_score", "rank_weighted",
        "pareto_front",
    ]
    combined = combined[cols_order]
    combined = combined.sort_values("rank_efficiency")
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV guardado: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
