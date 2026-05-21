"""
Feature Selection Experiment Runner
====================================
Ejecuta múltiples métodos de selección de características N veces
con semillas diferentes, guarda los resultados por iteración y
al final genera una tabla con promedios y desviaciones estándar.

Uso básico (todos los métodos, 15 runs):
    python script.py --runs 15 --data data_final.csv

Uso seleccionando métodos:
    python script.py --runs 20 --methods baseline chi2 l1 rf_topk

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


def run_single_experiment(run_id, seed, data_path, methods):
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
        m = evaluate_feature_set(
            list(X.columns),
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
        vt = VarianceThreshold(threshold=0.01)
        vt.fit(X_train)
        selected = list(X.columns[vt.get_support()])
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
        k_chi = 10
        mms = MinMaxScaler()
        X_train_min = pd.DataFrame(
            mms.fit_transform(X_train), columns=X.columns, index=X_train.index
        )
        X_test_min = pd.DataFrame(
            mms.transform(X_test), columns=X.columns, index=X_test.index
        )
        skb = SelectKBest(score_func=chi2, k=k_chi)
        skb.fit(X_train_min, y_train)
        selected = list(X.columns[skb.get_support()])
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
        k_mi = 10
        skb = SelectKBest(score_func=mutual_info_classif, k=k_mi)
        skb.fit(X_train, y_train)
        selected = list(X.columns[skb.get_support()])
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
        est = LogisticRegression(
            max_iter=5000, solver="liblinear", random_state=seed
        )
        rfe = RFE(estimator=est, n_features_to_select=10, step=1)
        rfe.fit(X_train, y_train)
        selected = list(X.columns[rfe.get_support()])
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
        est = LogisticRegression(
            max_iter=5000, solver="liblinear", random_state=seed
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rfecv = RFECV(
            estimator=est, step=1, cv=cv, scoring="accuracy", n_jobs=-1
        )
        rfecv.fit(X_train, y_train)
        selected = list(X.columns[rfecv.support_])
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
        scaler = StandardScaler().fit(X_train)
        X_train_s = pd.DataFrame(
            scaler.transform(X_train), columns=X.columns, index=X_train.index
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test), columns=X.columns, index=X_test.index
        )
        lr_l1 = LogisticRegression(
            penalty="l1", solver="saga", C=0.1, max_iter=5000, random_state=seed
        )
        lr_l1.fit(X_train_s, y_train)
        mask = np.abs(lr_l1.coef_).ravel() > 1e-6
        selected = list(X.columns[mask])
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
        rf = RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        selected = list(X.columns[top_idx])
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
        rf = RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        sfm = SelectFromModel(rf, threshold="median", prefit=True)
        selected = list(X.columns[sfm.get_support()])
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
        df_run = run_single_experiment(run_idx, seed, args.data, methods)

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


if __name__ == "__main__":
    main()
