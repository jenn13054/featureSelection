"""
Feature Selection Experiment Runner
====================================
Ejecuta múltiples métodos de selección de características N veces
con semillas diferentes, guarda los resultados por iteración y
al final genera una tabla con promedios y desviaciones estándar.

Incluye análisis de estabilidad por Cross-Validation para medir la
consistencia de las características seleccionadas por cada método.

Uso básico (todos los métodos, 15 runs):
    python script.py --runs 15 --data data_final.csv

Uso seleccionando métodos:
    python script.py --runs 20 --methods baseline chi2 l1 rf_topk

Uso con análisis de estabilidad (5-Fold CV):
    python script.py --runs 15 --stability-cv 5

Métodos disponibles:
    baseline, variance_threshold, chi2, mutual_info,
    rfe, rfecv, l1, rf_topk, rf_median
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    chi2,
    mutual_info_classif,
    RFE,
    RFECV,
    SelectFromModel,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

AVAILABLE_METHODS = [
    "baseline",
    "variance_threshold",
    "chi2",
    "mutual_info",
    "rfe",
    "rfecv",
    "l1",
    "rf_topk",
    "rf_median",
    "consensus",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta experimentos de selección de características de forma repetida."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=15,
        help="Número de ejecuciones del experimento (default: 15).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help=(
            "Lista de métodos a ejecutar. Usa 'all' para todos. "
            f"Opciones: {AVAILABLE_METHODS}"
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data_final.csv",
        help="Ruta al archivo CSV con los datos (default: data_final.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_results.csv",
        help="Archivo CSV donde se acumulan los resultados por run (default: experiment_results.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla base. En cada run se usa seed + run_id (default: 42).",
    )
    parser.add_argument(
        "--stability-cv",
        type=int,
        default=0,
        help=(
            "Si > 0, realiza un análisis de estabilidad de features con "
            "StratifiedKFold usando el número de folds indicado (default: 0)."
        ),
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.5,
        help=(
            "Umbral de votos (0-1) para el método consensus. "
            "Una feature debe ser seleccionada por al menos este porcentaje de métodos "
            "para ser incluida (default: 0.5)."
        ),
    )
    return parser.parse_args()


def get_methods_to_run(methods_arg):
    """Resuelve la lista de métodos a ejecutar a partir de los argumentos."""
    if "all" in methods_arg:
        return AVAILABLE_METHODS.copy()
    valid = [m for m in methods_arg if m in AVAILABLE_METHODS]
    invalid = [m for m in methods_arg if m not in AVAILABLE_METHODS]
    if invalid:
        raise ValueError(
            f"Métodos inválidos: {invalid}. Disponibles: {AVAILABLE_METHODS}"
        )
    if not valid:
        return AVAILABLE_METHODS.copy()
    return valid


def evaluate_feature_set(
    feature_list, X_train, X_test, y_train, y_test, classifier=None, scaler=None
):
    """
    Entrena un clasificador con un subconjunto de características y devuelve métricas.
    """
    start_time = time.time()
    if classifier is None:
        classifier = LogisticRegression(max_iter=5000, solver="sag")
    if scaler is None:
        scaler = StandardScaler()

    if len(feature_list) == 0:
        return {
            "n_features": 0,
            "features": [],
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "training_time": 0.0,
        }

    pipe = Pipeline([("scaler", scaler), ("clf", classifier)])
    pipe.fit(X_train[feature_list], y_train)
    y_pred = pipe.predict(X_test[feature_list])
    y_proba = (
        pipe.predict_proba(X_test[feature_list])[:, 1]
        if hasattr(pipe, "predict_proba")
        else None
    )
    end_time = time.time()

    return {
        "n_features": len(feature_list),
        "features": feature_list,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
        "training_time": end_time - start_time,
    }


def select_features(method, X_train, y_train, seed, consensus_threshold=0.5):
    """
    Selecciona características según el método indicado.
    Devuelve una lista con los nombres de las columnas seleccionadas.
    """
    if method == "baseline":
        return list(X_train.columns)

    if method == "consensus":
        consensus_selectors = [
            "variance_threshold",
            "chi2",
            "mutual_info",
            "l1",
            "rf_topk",
            "rf_median",
        ]
        votes = {f: 0 for f in X_train.columns}
        for sel_method in consensus_selectors:
            sel = select_features(sel_method, X_train, y_train, seed)
            for f in sel:
                votes[f] += 1
        min_votes = max(1, int(len(consensus_selectors) * consensus_threshold))
        selected = [f for f, v in votes.items() if v >= min_votes]
        if not selected:
            selected = sorted(votes, key=votes.get, reverse=True)[:10]
        return selected

    if method == "variance_threshold":
        vt = VarianceThreshold(threshold=0.01)
        vt.fit(X_train)
        return list(X_train.columns[vt.get_support()])

    if method == "chi2":
        mms = MinMaxScaler()
        X_train_min = pd.DataFrame(
            mms.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        skb = SelectKBest(score_func=chi2, k=10)
        skb.fit(X_train_min, y_train)
        return list(X_train.columns[skb.get_support()])

    if method == "mutual_info":
        skb = SelectKBest(score_func=mutual_info_classif, k=10)
        skb.fit(X_train, y_train)
        return list(X_train.columns[skb.get_support()])

    if method == "rfe":
        est = LogisticRegression(
            max_iter=5000, solver="liblinear", random_state=seed
        )
        rfe = RFE(estimator=est, n_features_to_select=10, step=1)
        rfe.fit(X_train, y_train)
        return list(X_train.columns[rfe.get_support()])

    if method == "rfecv":
        est = LogisticRegression(
            max_iter=5000, solver="liblinear", random_state=seed
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rfecv = RFECV(
            estimator=est, step=1, cv=cv, scoring="accuracy", n_jobs=-1
        )
        rfecv.fit(X_train, y_train)
        return list(X_train.columns[rfecv.support_])

    if method == "l1":
        scaler = StandardScaler().fit(X_train)
        X_train_s = pd.DataFrame(
            scaler.transform(X_train), columns=X_train.columns, index=X_train.index
        )
        lr_l1 = LogisticRegression(
            penalty="l1", solver="saga", C=0.1, max_iter=5000, random_state=seed
        )
        lr_l1.fit(X_train_s, y_train)
        mask = np.abs(lr_l1.coef_).ravel() > 1e-6
        return list(X_train.columns[mask])

    if method == "rf_topk":
        rf = RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        return list(X_train.columns[top_idx])

    if method == "rf_median":
        rf = RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        sfm = SelectFromModel(rf, threshold="median", prefit=True)
        return list(X_train.columns[sfm.get_support()])

    raise ValueError(f"Método desconocido: {method}")


def jaccard_index(set_a, set_b):
    """Calcula el índice de Jaccard entre dos conjuntos."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_stability_cv(method, X, y, n_splits, seed, consensus_threshold=0.5):
    """
    Evalúa la estabilidad de un método de selección de características
    mediante StratifiedKFold. Devuelve métricas de estabilidad (Jaccard,
    número de features, etc.) y rendimiento CV.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    selected_sets = []
    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Semilla distinta por fold para que la aleatoriedad interna varíe
        fold_seed = seed + fold_idx
        selected = select_features(
            method, X_train_fold, y_train_fold, fold_seed,
            consensus_threshold=consensus_threshold
        )
        selected_sets.append(set(selected))

        # Evaluación rápida en el fold de validación
        if len(selected) > 0:
            if method == "chi2":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            clf = LogisticRegression(
                max_iter=5000, solver="sag", random_state=fold_seed
            )
            pipe = Pipeline([("scaler", scaler), ("clf", clf)])
            pipe.fit(X_train_fold[selected], y_train_fold)
            acc = accuracy_score(y_val_fold, pipe.predict(X_val_fold[selected]))
            cv_scores.append(acc)
        else:
            cv_scores.append(np.nan)

    # Índice de Jaccard promedio entre todos los pares de folds
    jaccards = []
    for i in range(len(selected_sets)):
        for j in range(i + 1, len(selected_sets)):
            jaccards.append(jaccard_index(selected_sets[i], selected_sets[j]))
    jaccard_mean = np.mean(jaccards) if jaccards else np.nan
    jaccard_std = np.std(jaccards) if jaccards else np.nan

    # Estadísticas del tamaño del conjunto seleccionado
    n_features_list = [len(s) for s in selected_sets]
    n_features_mean = np.mean(n_features_list)
    n_features_std = np.std(n_features_list)

    # Frecuencia de selección de cada feature
    all_features = list(X.columns)
    freq = {f: sum(1 for s in selected_sets if f in s) for f in all_features}

    return {
        "method": method,
        "cv_accuracy_mean": np.mean(cv_scores) if cv_scores else np.nan,
        "cv_accuracy_std": np.std(cv_scores) if cv_scores else np.nan,
        "jaccard_mean": jaccard_mean,
        "jaccard_std": jaccard_std,
        "n_features_mean": n_features_mean,
        "n_features_std": n_features_std,
        "selection_frequencies": freq,
    }


def run_single_experiment(run_id, seed, data_path, methods, consensus_threshold=0.5):
    """
    Ejecuta una sola iteración del experimento con la semilla dada.
    Devuelve un DataFrame con una fila por método ejecutado.
    """
    print(f"\n{'='*60}")
    print(f"RUN {run_id:03d}  |  Seed: {seed}")
    print(f"{'='*60}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Archivo no encontrado: {data_path}")

    data = pd.read_csv(data_path)
    X = data.drop(columns=["target", "dropout.semester"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    records = []

    # ------------------------------------------------------------------
    # 1) Baseline (todas las características)
    # ------------------------------------------------------------------
    if "baseline" in methods:
        selected = select_features("baseline", X_train, y_train, seed)
        m = evaluate_feature_set(
            selected,
            X_train,
            X_test,
            y_train,
            y_test,
            classifier=LogisticRegression(
                max_iter=5000, solver="sag", random_state=seed
            ),
        )
        records.append(
            {
                "run_id": run_id,
                "seed": seed,
                "method": "baseline",
                **{k: v for k, v in m.items() if k != "features"},
            }
        )
        print(f"[baseline]        acc={m['accuracy']:.4f} | feats={m['n_features']}")

    # ------------------------------------------------------------------
    # 2) FILTER METHODS
    # ------------------------------------------------------------------
    if "variance_threshold" in methods:
        selected = select_features("variance_threshold", X_train, y_train, seed)
        m = evaluate_feature_set(
            selected,
            X_train,
            X_test,
            y_train,
            y_test,
            classifier=LogisticRegression(
                max_iter=5000, solver="sag", random_state=seed
            ),
        )
        records.append(
            {
                "run_id": run_id,
                "seed": seed,
                "method": "variance_threshold",
                **{k: v for k, v in m.items() if k != "features"},
            }
        )
        print(
            f"[var_threshold]   acc={m['accuracy']:.4f} | feats={m['n_features']}"
        )

    if "chi2" in methods:
        selected = select_features("chi2", X_train, y_train, seed)
        mms = MinMaxScaler()
        X_train_min = pd.DataFrame(
            mms.fit_transform(X_train), columns=X.columns, index=X_train.index
        )
        X_test_min = pd.DataFrame(
            mms.transform(X_test), columns=X.columns, index=X_test.index
        )
        clf = LogisticRegression(
            max_iter=5000, solver="liblinear", random_state=seed
        )
        m = evaluate_feature_set(
            selected, X_train_min, X_test_min, y_train, y_test, classifier=clf, scaler=None
        )
        records.append(
            {
                "run_id": run_id,
                "seed": seed,
                "method": "chi2",
                **{k: v for k, v in m.items() if k != "features"},
            }
        )
        print(f"[chi2]            acc={m['accuracy']:.4f} | feats={m['n_features']}")

    if "mutual_info" in methods:
        selected = select_features("mutual_info", X_train, y_train, seed)
        m = evaluate_feature_set(
            selected,
            X_train,
            X_test,
            y_train,
            y_test,
            classifier=LogisticRegression(
                max_iter=5000, solver="sag", random_state=seed
            ),
        )
        records.append(
            {
                "run_id": run_id,
                "seed": seed,
                "method": "mutual_info",
                **{k: v for k, v in m.items() if k != "features"},
            }
        )
        print(
            f"[mutual_info]     acc={m['accuracy']:.4f} | feats={m['n_features']}"
        )

    # ------------------------------------------------------------------
    # 3) WRAPPER METHODS
    # ------------------------------------------------------------------
    if "rfe" in methods:
        selected = select_features("rfe", X_train, y_train, seed)
        m = evaluate_feature_set(
            selected,
            X_train,
            X_test,
            y_train,
            y_test,
            classifier=LogisticRegression(
                max_iter=5000, solver="liblinear", random_state=seed
            ),
        )
        records.append(
            {
                "run_id": run_id,
                "seed": seed,
                "method": "rfe",
                **{k: v for k, v in m.items() if k != "features"},
            }
        )
        print(f"[rfe]             acc={m['accuracy']:.4f} | feats={m['n_features']}")

    if "rfecv" in methods:
        selected = select_features("rfecv", X_train, y_train, seed)
        m = evaluate_feature_set(
            selected,
            X_train,
            X_test,
            y_train,
            y_test,
            classifier=LogisticRegression(
                max_iter=5000, solver="liblinear", random_state=seed
            ),
        )
        records.append(
            {
                "run_id": run_id,
                "seed": seed,
                "method": "rfecv",
                **{k: v for k, v in m.items() if k != "features"},
            }
        )
        print(f"[rfecv]           acc={m['accuracy']:.4f} | feats={m['n_features']}")

    # ------------------------------------------------------------------
    # 4) EMBEDDED METHODS
    # ------------------------------------------------------------------
    if "l1" in methods:
        selected = select_features("l1", X_train, y_train, seed)
        m = evaluate_feature_set(
            selected,
            X_train,
            X_test,
            y_train,
            y_test,
            classifier=LogisticRegression(
                max_iter=5000, solver="liblinear", random_state=seed
            ),
            scaler=StandardScaler(),
        )
        records.append(
            {
                "run_id": run_id,
                "seed": seed,
                "method": "l1",
                **{k: v for k, v in m.items() if k != "features"},
            }
        )
        print(f"[l1]              acc={m['accuracy']:.4f} | feats={m['n_features']}")

    if "rf_topk" in methods:
        selected = select_features("rf_topk", X_train, y_train, seed)
        m = evaluate_feature_set(
            selected,
            X_train,
            X_test,
            y_train,
            y_test,
            classifier=LogisticRegression(
                max_iter=5000, solver="liblinear", random_state=seed
            ),
        )
        records.append(
            {
                "run_id": run_id,
                "seed": seed,
                "method": "rf_topk",
                **{k: v for k, v in m.items() if k != "features"},
            }
        )
        print(
            f"[rf_topk]         acc={m['accuracy']:.4f} | feats={m['n_features']}"
        )

    if "rf_median" in methods:
        selected = select_features("rf_median", X_train, y_train, seed)
        m = evaluate_feature_set(
            selected,
            X_train,
            X_test,
            y_train,
            y_test,
            classifier=LogisticRegression(
                max_iter=5000, solver="liblinear", random_state=seed
            ),
        )
        records.append(
            {
                "run_id": run_id,
                "seed": seed,
                "method": "rf_median",
                **{k: v for k, v in m.items() if k != "features"},
            }
        )
        print(
            f"[rf_median]       acc={m['accuracy']:.4f} | feats={m['n_features']}"
        )

    if "consensus" in methods:
        selected = select_features(
            "consensus", X_train, y_train, seed,
            consensus_threshold=consensus_threshold
        )
        m = evaluate_feature_set(
            selected,
            X_train,
            X_test,
            y_train,
            y_test,
            classifier=LogisticRegression(
                max_iter=5000, solver="sag", random_state=seed
            ),
        )
        records.append(
            {
                "run_id": run_id,
                "seed": seed,
                "method": "consensus",
                **{k: v for k, v in m.items() if k != "features"},
            }
        )
        print(
            f"[consensus]       acc={m['accuracy']:.4f} | feats={m['n_features']}"
        )

    return pd.DataFrame(records)


def main():
    args = parse_args()
    methods = get_methods_to_run(args.methods)
    print(f"Configuración: {args.runs} runs | métodos: {methods} | data: {args.data}")

    # Detectar runs ya completadas para permitir reanudación
    completed_runs = set()
    if os.path.exists(args.output):
        try:
            existing = pd.read_csv(args.output)
            completed_runs = set(existing["run_id"].unique())
            print(
                f"Archivo '{args.output}' encontrado. Runs completadas: {len(completed_runs)}"
            )
        except Exception as e:
            print(f"No se pudo leer '{args.output}', se empezará de cero. Error: {e}")

    # Bucle principal de ejecuciones
    for run_idx in range(args.runs):
        if run_idx in completed_runs:
            print(f"Run {run_idx:03d} ya existe en '{args.output}', saltando...")
            continue

        seed = args.seed + run_idx
        df_run = run_single_experiment(
            run_idx, seed, args.data, methods,
            consensus_threshold=args.consensus_threshold
        )

        header = not os.path.exists(args.output)
        df_run.to_csv(args.output, mode="a", header=header, index=False)
        print(f"Run {run_idx:03d} guardado en '{args.output}'")

    # ------------------------------------------------------------------
    # Tabla comparativa final: promedio y desviación estándar
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("TABLA COMPARATIVA FINAL  (Promedio ± Desv. Est.)")
    print(f"{'='*60}")

    df_all = pd.read_csv(args.output)
    # Filtrar solo las runs solicitadas en esta invocación
    df_all = df_all[df_all["run_id"] < args.runs]

    numeric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "n_features",
        "training_time",
    ]

    # Calcular estadísticas
    summary = df_all.groupby("method")[numeric_cols].agg(["mean", "std"]).round(4)
    print(summary)

    # Guardar resumen
    summary_path = args.output.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print(f"\nResumen guardado en: {summary_path}")

    # ------------------------------------------------------------------
    # Análisis de estabilidad por Cross-Validation
    # ------------------------------------------------------------------
    if args.stability_cv > 0:
        print(f"\n{'='*60}")
        print(f"ANÁLISIS DE ESTABILIDAD  ({args.stability_cv}-Fold CV)")
        print(f"{'='*60}")

        data = pd.read_csv(args.data)
        X_full = data.drop(columns=["target", "dropout.semester"])
        y_full = data["target"]

        stability_records = []
        freq_records = []

        for method in methods:
            print(f"\n[{method}] Calculando estabilidad...")
            stab = compute_stability_cv(
                method, X_full, y_full, args.stability_cv, args.seed,
                consensus_threshold=args.consensus_threshold
            )
            stability_records.append(
                {
                    "method": method,
                    "cv_accuracy_mean": stab["cv_accuracy_mean"],
                    "cv_accuracy_std": stab["cv_accuracy_std"],
                    "jaccard_mean": stab["jaccard_mean"],
                    "jaccard_std": stab["jaccard_std"],
                    "n_features_mean": stab["n_features_mean"],
                    "n_features_std": stab["n_features_std"],
                }
            )
            for feature, count in stab["selection_frequencies"].items():
                freq_records.append(
                    {
                        "method": method,
                        "feature": feature,
                        "frequency": count,
                        "frequency_pct": count / args.stability_cv,
                    }
                )

        df_stab = pd.DataFrame(stability_records)
        print("\nResumen de estabilidad:")
        print(df_stab.to_string(index=False))

        stab_path = args.output.replace(".csv", "_stability.csv")
        df_stab.to_csv(stab_path, index=False)
        print(f"\nEstabilidad guardada en: {stab_path}")

        freq_path = args.output.replace(".csv", "_stability_freq.csv")
        pd.DataFrame(freq_records).to_csv(freq_path, index=False)
        print(f"Frecuencias guardadas en: {freq_path}")


if __name__ == "__main__":
    main()
