"""Reproducible OCF vs Prade-inspired baseline evaluation on the official Monk files.

Run from the repository root:
    python ocf-vs-baseline/runs/run_monk_experiment.py

The official 432-row .test files are used unchanged for every repetition.
Thresholds are selected only on validation data carved from .train.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
OUTPUT = ROOT / "results" / "monk-experiment"
FEATURES = [f"X_{i}" for i in range(1, 7)]
N_REPEATS = 10
VALIDATION_SIZE = 0.10
RANDOM_SEED = 42
CATEGORICAL_MISMATCH_COST = 100
OCF_SHIFT = len(FEATURES) * CATEGORICAL_MISMATCH_COST
METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "jaccard", "youden", "roc_auc"]


def load_official_file(path):
    names = ["y", *FEATURES, "id"]
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=str)
    if frame.shape[1] != 8 or frame.isna().any().any():
        raise ValueError(f"Unexpected or incomplete Monk file: {path}")
    frame["y"] = frame["y"].astype(int)
    if not set(frame["y"].unique()).issubset({0, 1}):
        raise ValueError(f"Labels must be 0/1 in {path}")
    return frame


def youden(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return sensitivity + specificity - 1.0


def choose_threshold(y_true, scores, positive_if):
    """Maximise validation Youden; deterministic middle choice among ties."""
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(np.concatenate(([scores.min() - 1], scores, [scores.max() + 1])))
    values = []
    for threshold in candidates:
        pred = (scores <= threshold).astype(int) if positive_if == "le" else (scores >= threshold).astype(int)
        values.append(youden(y_true, pred))
    best = max(values)
    best_thresholds = candidates[np.isclose(values, best)]
    return float(best_thresholds[len(best_thresholds) // 2]), float(best)


def predict(scores, threshold, positive_if):
    scores = np.asarray(scores, dtype=float)
    return (scores <= threshold).astype(int) if positive_if == "le" else (scores >= threshold).astype(int)


def metrics(y_true, y_pred, ranking_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "jaccard": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "youden": youden(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, ranking_score),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def ocf_scores(query_x, reference_x, reference_y):
    """Same scaled categorical distance used by datadistances.py, minus its shift."""
    mismatch = (query_x[:, None, :] != reference_x[None, :, :]).astype(np.int32)
    mismatch_cost = mismatch * CATEGORICAL_MISMATCH_COST
    # Positive reference: distance; negative reference: MAX_SCALING_VALUE - distance.
    contributions = np.where(reference_y[None, :, None] == 1,
                              mismatch_cost, CATEGORICAL_MISMATCH_COST - mismatch_cost)
    return contributions.sum(axis=(1, 2)).astype(float) - OCF_SHIFT


class DeltaBaseline:
    """Cumulated +1/-1 delta; exact interpretations or individual literals."""
    def __init__(self, variant):
        if variant not in ("exact", "literal"):
            raise ValueError(variant)
        self.variant = variant

    def fit(self, x, y):
        self.delta = {}
        for row, label in zip(x, y):
            increment = 1 if int(label) == 1 else -1
            keys = [tuple(row)] if self.variant == "exact" else [(i, value) for i, value in enumerate(row)]
            for key in keys:
                self.delta[key] = self.delta.get(key, 0) + increment
        return self

    def score(self, x):
        scores = []
        for row in x:
            keys = [tuple(row)] if self.variant == "exact" else [(i, value) for i, value in enumerate(row)]
            scores.append(sum(self.delta.get(key, 0) for key in keys))
        return np.asarray(scores, dtype=float)


def evaluate_dataset(dataset_number):
    train = load_official_file(DATA / f"monks-{dataset_number}.train")
    test = load_official_file(DATA / f"monks-{dataset_number}.test")
    if len(test) != 432:
        raise ValueError(f"Official Monk-{dataset_number} test file must have 432 rows; got {len(test)}")

    # Keep every official test row, including IDs also present in the official train file.
    train_ids = set(train["id"])
    test_ids = set(test["id"])
    overlap_ids = train_ids & test_ids
    exact_overlap = set(map(tuple, train[["y", *FEATURES]].to_numpy())) & set(map(tuple, test[["y", *FEATURES]].to_numpy()))

    x_test = test[FEATURES].to_numpy(dtype=object)
    y_test = test["y"].to_numpy()
    split = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=VALIDATION_SIZE,
                                   random_state=RANDOM_SEED)
    run_rows = []

    for repeat, (fit_idx, val_idx) in enumerate(split.split(train[FEATURES], train["y"]), start=1):
        fit = train.iloc[fit_idx]
        val = train.iloc[val_idx]
        x_fit = fit[FEATURES].to_numpy(dtype=object)
        y_fit = fit["y"].to_numpy()
        x_val = val[FEATURES].to_numpy(dtype=object)
        y_val = val["y"].to_numpy()

        # OCF: threshold chosen on validation, then applied unchanged to official test.
        val_ocf = ocf_scores(x_val, x_fit, y_fit)
        test_ocf = ocf_scores(x_test, x_fit, y_fit)
        ocf_thr, val_ocf_youden = choose_threshold(y_val, val_ocf, "le")
        ocf_pred = predict(test_ocf, ocf_thr, "le")
        row = {"dataset": dataset_number, "repeat": repeat, "model": "OCF",
               "threshold": ocf_thr, "validation_youden": val_ocf_youden}
        row.update(metrics(y_test, ocf_pred, -test_ocf))
        run_rows.append(row)

        # Prade-inspired positive l-cut operationalisation: predict positive iff delta >= l.
        for variant in ("exact", "literal"):
            model = DeltaBaseline(variant).fit(x_fit, y_fit)
            val_delta = model.score(x_val)
            test_delta = model.score(x_test)
            threshold, val_j = choose_threshold(y_val, val_delta, "ge")
            pred = predict(test_delta, threshold, "ge")
            row = {"dataset": dataset_number, "repeat": repeat,
                   "model": "BaselineExact" if variant == "exact" else "BaselineLiteral",
                   "threshold": threshold, "validation_youden": val_j}
            row.update(metrics(y_test, pred, test_delta))
            run_rows.append(row)

    return run_rows, {
        "dataset": dataset_number, "official_train_rows": len(train),
        "official_test_rows": len(test), "train_test_overlapping_ids": len(overlap_ids),
        "identical_train_test_rows": len(exact_overlap),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_runs, dataset_info = [], []
    for number in (1, 2, 3):
        runs, info = evaluate_dataset(number)
        all_runs.extend(runs)
        dataset_info.append(info)
        print(f"Monk-{number}: {info['official_train_rows']} train, "
              f"{info['official_test_rows']} official test rows unchanged; "
              f"{info['train_test_overlapping_ids']} overlapping IDs.")
    run_frame = pd.DataFrame(all_runs)
    run_frame.to_csv(OUTPUT / "per_run_results.csv", index=False)

    summary_rows = []
    for (dataset, model), group in run_frame.groupby(["dataset", "model"], sort=True):
        result = {"dataset": dataset, "model": model, "repeats": len(group)}
        for metric in METRICS:
            result[f"mean_{metric}"] = group[metric].mean()
            result[f"std_{metric}"] = group[metric].std(ddof=1)
        for count in ("tp", "tn", "fp", "fn"):
            result[f"total_{count}"] = int(group[count].sum())
        summary_rows.append(result)
    pd.DataFrame(summary_rows).to_csv(OUTPUT / "summary_results.csv", index=False)
    pd.DataFrame(dataset_info).to_csv(OUTPUT / "benchmark_overlap.csv", index=False)
    print(f"Saved results in {OUTPUT}")


if __name__ == "__main__":
    main()
