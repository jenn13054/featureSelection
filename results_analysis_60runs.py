"""
Genera tabla y visualización para los resultados de 60 runs.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results = pd.read_csv('experiment_results_60runs.csv')
numeric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'n_features', 'training_time']
summary = results.groupby('method')[numeric_cols].agg(['mean', 'std']).round(4)

order = ['baseline', 'variance_threshold', 'chi2', 'mutual_info',
         'rfe', 'rfecv', 'l1', 'rf_topk', 'rf_median']
summary = summary.reindex([m for m in order if m in summary.index])

# Markdown
md_lines = [
    "# Resultados del Experimento — 60 Runs",
    "",
    "**Dataset:** `data_final.csv` | **Runs:** 60 | **Semilla base:** 42",
    "",
    "| Método | Accuracy | Precision | Recall | F1-Score | ROC-AUC | N. Features | Tiempo (s) |",
    "|--------|----------|-----------|--------|----------|---------|-------------|------------|"
]

for method in summary.index:
    row = summary.loc[method]
    md_lines.append(
        f"| **{method}** | "
        f"{row[('accuracy', 'mean')]:.4f} ± {row[('accuracy', 'std')]:.4f} | "
        f"{row[('precision', 'mean')]:.4f} ± {row[('precision', 'std')]:.4f} | "
        f"{row[('recall', 'mean')]:.4f} ± {row[('recall', 'std')]:.4f} | "
        f"{row[('f1', 'mean')]:.4f} ± {row[('f1', 'std')]:.4f} | "
        f"{row[('roc_auc', 'mean')]:.4f} ± {row[('roc_auc', 'std')]:.4f} | "
        f"{row[('n_features', 'mean')]:.1f} ± {row[('n_features', 'std')]:.1f} | "
        f"{row[('training_time', 'mean')]:.3f} ± {row[('training_time', 'std')]:.3f} |"
    )

md_lines.extend(["", "## Ranking por Accuracy", ""])
acc_rank = summary[('accuracy', 'mean')].sort_values(ascending=False)
for i, (method, acc) in enumerate(acc_rank.items(), 1):
    feats = summary.loc[method, ('n_features', 'mean')]
    tiempo = summary.loc[method, ('training_time', 'mean')]
    md_lines.append(f"{i}. **`{method}`** → {acc:.4f} | {feats:.1f} features | {tiempo:.3f}s")

md_lines.extend(["", "## Ranking por Eficiencia (Acc/Tiempo)", ""])
eff = (summary[('accuracy', 'mean')] / summary[('training_time', 'mean')]).sort_values(ascending=False)
for i, (method, val) in enumerate(eff.items(), 1):
    md_lines.append(f"{i}. **`{method}`** → {val:.2f}")

with open('results_comparison_60runs.md', 'w') as f:
    f.write('\n'.join(md_lines))

print("✅ Tabla Markdown guardada en: results_comparison_60runs.md")

# Visualización
methods = summary.index.tolist()
acc_mean = summary[('accuracy', 'mean')].values
acc_std = summary[('accuracy', 'std')].values
feat_mean = summary[('n_features', 'mean')].values
feat_std = summary[('n_features', 'std')].values
time_mean = summary[('training_time', 'mean')].values
time_std = summary[('training_time', 'std')].values

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparación de Métodos — 60 Runs\n(Promedio ± desv. estándar)', fontsize=14, fontweight='bold')
colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

# Accuracy — límites dinámicos con pequeño margen
ax = axes[0, 0]
bars = ax.barh(methods, acc_mean, xerr=acc_std, color=colors, edgecolor='black', capsize=3)
acc_min = max(0, acc_mean.min() - acc_std.max() - 0.005)
acc_max = min(1, acc_mean.max() + acc_std.max() + 0.005)
ax.set_xlim(acc_min, acc_max)
ax.set_xlabel('Accuracy')
ax.set_title('Accuracy (mayor es mejor)', fontweight='bold')
ax.invert_yaxis()
for bar, val in zip(bars, acc_mean):
    ax.text(val + (acc_max - acc_min)*0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=8)

# Features
ax = axes[0, 1]
bars = ax.barh(methods, feat_mean, xerr=feat_std, color=colors, edgecolor='black', capsize=3)
ax.set_xlabel('Número de Features')
ax.set_title('Features seleccionadas (menor es mejor)', fontweight='bold')
ax.invert_yaxis()
for bar, val in zip(bars, feat_mean):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=8)

# Tiempo
ax = axes[1, 0]
bars = ax.barh(methods, time_mean, xerr=time_std, color=colors, edgecolor='black', capsize=3)
ax.set_xlabel('Tiempo (segundos)')
ax.set_title('Tiempo de entrenamiento (menor es mejor)', fontweight='bold')
ax.invert_yaxis()
ax.set_xscale('log')
for bar, val in zip(bars, time_mean):
    ax.text(val * 1.2, bar.get_y() + bar.get_height()/2, f'{val:.3f}s', va='center', fontsize=8)

# Scatter
ax = axes[1, 1]
scatter = ax.scatter(feat_mean, acc_mean, s=time_mean*300, c=range(len(methods)), cmap='tab10', edgecolors='black', alpha=0.8)
for i, method in enumerate(methods):
    ax.annotate(method, (feat_mean[i], acc_mean[i]), textcoords="offset points", xytext=(8, 0), fontsize=8, ha='left')
ax.set_xlabel('Número de Features')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Features (tamaño = tiempo)', fontweight='bold')
ax.set_xlim(0, 50)

plt.tight_layout()
plt.savefig('results_comparison_60runs.png', dpi=200, bbox_inches='tight')
plt.close()

print("✅ Gráfico guardado en: results_comparison_60runs.png")

# Resumen en consola
print("\n" + "="*120)
print("RESUMEN EJECUTIVO — 60 RUNS")
print("="*120)
print(f"{'Método':<20} {'Accuracy':>10} {'Features':>10} {'Tiempo(s)':>12} {'Eficiencia':>12}")
print("-"*120)
for method in summary.index:
    acc = summary.loc[method, ('accuracy', 'mean')]
    feats = summary.loc[method, ('n_features', 'mean')]
    t = summary.loc[method, ('training_time', 'mean')]
    eff_val = acc / t
    print(f"{method:<20} {acc:>10.4f} {feats:>10.1f} {t:>12.3f} {eff_val:>12.2f}")
print("="*120)
