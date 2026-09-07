"""
Genera un informe HTML interactivo con los resultados de 60 runs.

Fuente: experiment_results_60runs_summary.csv
Salida: results_report_60runs.html
"""

import os
import warnings

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


INPUT_CSV = "experiment_results_60runs_summary.csv"
OUTPUT_HTML = "results_report_60runs.html"


def load_summary(path: str) -> pd.DataFrame:
    """Carga el CSV de resumen con mean/std por métrica."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}")

    # El CSV tiene multi-header: primera fila métrica, segunda mean/std
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.index.name = "method"
    return df


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


def create_summary_table(flat: pd.DataFrame) -> go.Figure:
    """Tabla resumen con medias y desviaciones."""
    columns = ["Método", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Features", "Tiempo (s)"]

    values = [
        flat["method"].tolist(),
        [f"{m:.4f} ± {s:.4f}" for m, s in zip(flat["accuracy_mean"], flat["accuracy_std"])],
        [f"{m:.4f} ± {s:.4f}" for m, s in zip(flat["precision_mean"], flat["precision_std"])],
        [f"{m:.4f} ± {s:.4f}" for m, s in zip(flat["recall_mean"], flat["recall_std"])],
        [f"{m:.4f} ± {s:.4f}" for m, s in zip(flat["f1_mean"], flat["f1_std"])],
        [f"{m:.4f} ± {s:.4f}" for m, s in zip(flat["roc_auc_mean"], flat["roc_auc_std"])],
        [f"{m:.1f} ± {s:.1f}" for m, s in zip(flat["n_features_mean"], flat["n_features_std"])],
        [f"{m:.3f} ± {s:.3f}" for m, s in zip(flat["training_time_mean"], flat["training_time_std"])],
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(values=columns, fill_color="#2c3e50", align="center", font=dict(color="white", size=13)),
        cells=dict(values=values, align="center", font=dict(size=12), height=30)
    )])
    fig.update_layout(title="Resumen de resultados — 60 runs", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_bar_with_error(flat: pd.DataFrame, metric_mean: str, metric_std: str, title: str, yaxis_title: str, color: str) -> go.Figure:
    """Gráfico de barras con barras de error."""
    # Ordenar de mayor a menor
    flat_sorted = flat.sort_values(metric_mean, ascending=False)

    fig = go.Figure(data=[go.Bar(
        x=flat_sorted["method"],
        y=flat_sorted[metric_mean],
        error_y=dict(type="data", array=flat_sorted[metric_std], visible=True),
        marker_color=color,
        text=[f"{v:.4f}" for v in flat_sorted[metric_mean]],
        textposition="outside"
    )])
    fig.update_layout(
        title=title,
        xaxis_title="Método",
        yaxis_title=yaxis_title,
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=60)
    )
    return fig


def create_scatter_accuracy_vs_features(flat: pd.DataFrame) -> go.Figure:
    """Scatter: accuracy vs número de features, tamaño = tiempo."""
    fig = go.Figure(data=[go.Scatter(
        x=flat["n_features_mean"],
        y=flat["accuracy_mean"],
        mode="markers+text",
        text=flat["method"],
        textposition="top center",
        marker=dict(
            size=flat["training_time_mean"] * 500,  # escalar para visibilidad
            color=flat["accuracy_mean"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Accuracy")
        ),
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Features: %{x:.1f}<br>" +
            "Accuracy: %{y:.4f}<br>" +
            "Tiempo: %{marker.size:.4f}s<extra></extra>"
        )
    )])
    fig.update_layout(
        title="Accuracy vs Número de Features (tamaño ∝ tiempo)",
        xaxis_title="Número de features",
        yaxis_title="Accuracy",
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=60)
    )
    return fig


def create_efficiency_chart(flat: pd.DataFrame) -> go.Figure:
    """Gráfico de eficiencia: accuracy / tiempo."""
    flat = flat.copy()
    flat["efficiency"] = flat["accuracy_mean"] / flat["training_time_mean"]
    flat_sorted = flat.sort_values("efficiency", ascending=True)

    fig = go.Figure(data=[go.Bar(
        x=flat_sorted["efficiency"],
        y=flat_sorted["method"],
        orientation="h",
        marker_color="#e67e22",
        text=[f"{v:.2f}" for v in flat_sorted["efficiency"]],
        textposition="outside"
    )])
    fig.update_layout(
        title="Eficiencia: Accuracy / Tiempo de entrenamiento",
        xaxis_title="Accuracy / Tiempo (s⁻¹)",
        yaxis_title="Método",
        template="plotly_white",
        margin=dict(l=120, r=40, t=60, b=60)
    )
    return fig


def create_radar_chart(flat: pd.DataFrame) -> go.Figure:
    """Radar chart con métricas normalizadas (mayor es mejor, excepto tiempo)."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    flat_norm = flat.copy()

    for metric in metrics:
        col_mean = f"{metric}_mean"
        min_val = flat_norm[col_mean].min()
        max_val = flat_norm[col_mean].max()
        flat_norm[metric] = (flat_norm[col_mean] - min_val) / (max_val - min_val) if max_val > min_val else 0.5

    # Invertir tiempo: menor tiempo es mejor
    t_min = flat_norm["training_time_mean"].min()
    t_max = flat_norm["training_time_mean"].max()
    flat_norm["time_score"] = 1 - (flat_norm["training_time_mean"] - t_min) / (t_max - t_min) if t_max > t_min else 0.5

    categories = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Rapidez"]

    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

    for idx, row in flat_norm.iterrows():
        values = [
            row["accuracy"], row["precision"], row["recall"],
            row["f1"], row["roc_auc"], row["time_score"]
        ]
        values.append(values[0])  # cerrar el radar

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["method"],
            line=dict(color=colors[idx % len(colors)])
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Perfil multimetric normalizado",
        template="plotly_white",
        margin=dict(l=80, r=80, t=80, b=40)
    )
    return fig


def create_combined_dashboard(flat: pd.DataFrame) -> go.Figure:
    """Dashboard combinado con 4 subplots."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Accuracy", "Número de features", "Tiempo de entrenamiento", "Eficiencia Acc/Time"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    # Accuracy
    flat_sorted = flat.sort_values("accuracy_mean", ascending=False)
    fig.add_trace(go.Bar(
        x=flat_sorted["method"], y=flat_sorted["accuracy_mean"],
        error_y=dict(type="data", array=flat_sorted["accuracy_std"], visible=True),
        marker_color="#3498db", name="Accuracy", showlegend=False
    ), row=1, col=1)

    # Features
    flat_sorted = flat.sort_values("n_features_mean", ascending=True)
    fig.add_trace(go.Bar(
        x=flat_sorted["method"], y=flat_sorted["n_features_mean"],
        marker_color="#2ecc71", name="Features", showlegend=False
    ), row=1, col=2)

    # Tiempo
    flat_sorted = flat.sort_values("training_time_mean", ascending=True)
    fig.add_trace(go.Bar(
        x=flat_sorted["method"], y=flat_sorted["training_time_mean"],
        marker_color="#e74c3c", name="Tiempo (s)", showlegend=False
    ), row=2, col=1)

    # Eficiencia
    flat["efficiency"] = flat["accuracy_mean"] / flat["training_time_mean"]
    flat_sorted = flat.sort_values("efficiency", ascending=False)
    fig.add_trace(go.Bar(
        x=flat_sorted["method"], y=flat_sorted["efficiency"],
        marker_color="#e67e22", name="Acc/Time", showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        title_text="Dashboard de resultados — 60 runs",
        template="plotly_white",
        height=800,
        margin=dict(l=60, r=20, t=80, b=60)
    )
    return fig


def load_ranking(path: str) -> pd.DataFrame:
    """Carga el CSV con el ranking combinado accuracy + tiempo."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}")
    return pd.read_csv(path)


def create_ranking_table(ranking: pd.DataFrame) -> go.Figure:
    """Tabla con el ranking combinado."""
    ranking = ranking.sort_values("rank_weighted")

    columns = ["Rank", "Método", "Accuracy", "Tiempo (s)", "Features", "Eficiencia", "Score ponderado", "Frente Pareto"]
    values = [
        ranking["rank_weighted"].astype(int).tolist(),
        ranking["method"].tolist(),
        [f"{v:.4f}" for v in ranking["accuracy_mean"]],
        [f"{v:.4f}" for v in ranking["training_time_mean"]],
        [f"{v:.1f}" for v in ranking["n_features_mean"]],
        [f"{v:.2f}" for v in ranking["efficiency"]],
        [f"{v:.4f}" for v in ranking["weighted_score"]],
        ranking["pareto_front"].astype(int).tolist(),
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(values=columns, fill_color="#1a5276", align="center", font=dict(color="white", size=12)),
        cells=dict(values=values, align="center", font=dict(size=11), height=28)
    )])
    fig.update_layout(
        title="Ranking combinado: Accuracy + Tiempo (60 runs)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_weighted_score_chart(ranking: pd.DataFrame) -> go.Figure:
    """Gráfico de barras horizontales del score ponderado."""
    ranking = ranking.sort_values("weighted_score", ascending=True)

    colors = ["#27ae60" if f == 1 else "#3498db" for f in ranking["pareto_front"]]

    fig = go.Figure(data=[go.Bar(
        x=ranking["weighted_score"],
        y=ranking["method"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in ranking["weighted_score"]],
        textposition="outside"
    )])
    fig.update_layout(
        title="Score ponderado: 70% accuracy + 30% rapidez",
        xaxis_title="Score ponderado",
        yaxis_title="Método",
        template="plotly_white",
        height=500,
        margin=dict(l=120, r=60, t=60, b=60)
    )
    return fig


def build_html(figures: dict, output_path: str) -> None:
    """Construye el HTML final combinando todos los gráficos."""
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="es">',
        "<head>",
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "    <title>Resultados del Experimento — 60 Runs</title>",
        '    <script src="https://cdn.tailwindcss.com"></script>',
        "</head>",
        '<body class="bg-gray-50 text-gray-800">',
        '    <div class="max-w-7xl mx-auto px-4 py-8">',
        '        <h1 class="text-3xl font-bold text-center mb-2">Resultados del Experimento</h1>',
        '        <p class="text-center text-gray-600 mb-8">Feature Selection — 60 runs independientes | Dataset: data_final.csv</p>',
    ]

    sections = [
        ("Tabla resumen", "summary_table"),
        ("Ranking combinado Accuracy + Tiempo", "ranking_table"),
        ("Score ponderado por método", "weighted_score_chart"),
        ("Dashboard comparativo", "dashboard"),
        ("Accuracy por método", "accuracy_bar"),
        ("Número de features seleccionadas", "features_bar"),
        ("Tiempo de entrenamiento", "time_bar"),
        ("Accuracy vs Features", "scatter"),
        ("Eficiencia (Accuracy / Tiempo)", "efficiency"),
        ("Perfil multimetric normalizado", "radar"),
    ]

    for title, key in sections:
        html_parts.append(f'        <h2 class="text-2xl font-semibold mt-10 mb-4">{title}</h2>')
        html_parts.append(f'        <div class="bg-white p-4 rounded-lg shadow">')
        html_parts.append(figures[key].to_html(full_html=False, include_plotlyjs=("cdn" if key == "summary_table" else False)))
        html_parts.append('        </div>')

    html_parts.extend([
        '        <footer class="mt-12 text-center text-sm text-gray-500">',
        '            Generado automáticamente desde experiment_results_60runs_summary.csv',
        '        </footer>',
        '    </div>',
        "</body>",
        "</html>"
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


def main():
    warnings.filterwarnings("ignore")

    print(f"Cargando resumen desde {INPUT_CSV}...")
    df = load_summary(INPUT_CSV)
    flat = flatten_summary(df)
    print(f"Métodos cargados: {', '.join(flat['method'].tolist())}")

    ranking = load_ranking("ranking_accuracy_time_60runs.csv")
    print(f"Ranking cargado: {len(ranking)} métodos")

    figures = {
        "summary_table": create_summary_table(flat),
        "ranking_table": create_ranking_table(ranking),
        "weighted_score_chart": create_weighted_score_chart(ranking),
        "dashboard": create_combined_dashboard(flat),
        "accuracy_bar": create_bar_with_error(flat, "accuracy_mean", "accuracy_std", "Accuracy por método", "Accuracy", "#3498db"),
        "features_bar": create_bar_with_error(flat, "n_features_mean", "n_features_std", "Número de features seleccionadas", "Features", "#2ecc71"),
        "time_bar": create_bar_with_error(flat, "training_time_mean", "training_time_std", "Tiempo de entrenamiento", "Tiempo (s)", "#e74c3c"),
        "scatter": create_scatter_accuracy_vs_features(flat),
        "efficiency": create_efficiency_chart(flat),
        "radar": create_radar_chart(flat),
    }

    print(f"Generando informe HTML: {OUTPUT_HTML}...")
    build_html(figures, OUTPUT_HTML)
    print(f"Informe HTML generado exitosamente: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
