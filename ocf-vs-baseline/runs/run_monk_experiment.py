"""Reproducible OCF vs Baseline evaluation on official Monk files.

Run from the repository root:
    python ocf-vs-baseline/runs/run_monk_experiment.py

The official 432-row .test files are used unchanged for every repetition.
Thresholds are selected only on validation data carved from .train.
"""
from itertools import combinations
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
BASELINE_P_CUT = 0   # P = {training interpretations w : delta(w) >= BASELINE_P_CUT}
OCF_SHIFT = len(FEATURES) * CATEGORICAL_MISMATCH_COST
METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "jaccard", "youden", "roc_auc"]


def load_official_file(path):
    names = ["y", *FEATURES, "id"]
    frame = pd.read_csv(path, sep=r"\s+", engine="python", header=None, names=names, dtype=str)
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
    contributions = np.where(reference_y[None, :, None] == 1,
                              mismatch_cost, CATEGORICAL_MISMATCH_COST - mismatch_cost)
    return contributions.sum(axis=(1, 2)).astype(float) - OCF_SHIFT


class Baseline:
    """Baseline of Section 2, learned on the training data in two steps.

    Step 1 (cumulation): over the training examples only, delta(w) = (#positive examples equal to w)
    - (#negative examples equal to w), i.e. +1 for every positive and -1 for every negative example.
    Step 2 (formula learning, after training): P = {training interpretations w : delta(w) >= BASELINE_P_CUT}
    and N = {training interpretations w : delta(w) < 0}. Learn a formula F satisfied by every w in P and
    by no w in N, as a disjunction of terms (conjunctions of at most k literals X_i = v). A term is kept
    iff some w in P satisfies it and no w in N does. k is the smallest size for which every w in P
    satisfies F. Classification: an unseen x is positive iff x satisfies F.
    No threshold. The validation rows are used only by repair(): F is tested on them and changed to match.
    """

    def __init__(self, p_cut=BASELINE_P_CUT):
        self.p_cut = p_cut

    def fit(self, x, y):
        x = np.asarray(x, dtype=object)
        y = np.asarray(y).astype(int)
        self.n_features = x.shape[1]
        # step 1: cumulate +1 / -1 on the training interpretations
        delta = {}
        for row, label in zip(x, y):
            key = tuple(row)
            delta[key] = delta.get(key, 0) + (1 if label == 1 else -1)
        self.delta = delta
        positives = [np.asarray(w, dtype=object) for w, d in delta.items() if d >= self.p_cut]
        negatives = [np.asarray(w, dtype=object) for w, d in delta.items() if d < 0]
        # step 2: learn F
        self.subsets = [cols for size in range(1, self.n_features + 1)
                        for cols in combinations(range(self.n_features), size)]

        def terms_of(ws):
            return {(cols, tuple(w[list(cols)])) for w in ws for cols in self.subsets}

        self.terms_of = terms_of
        self.negatives = negatives
        self.extra = set()          # terms added by repair(); always active, whatever k is
        self.terms = terms_of(positives) - terms_of(negatives)
        self.k = self.n_features
        if positives:
            for k in range(1, self.n_features + 1):
                self.k = k
                if all(self.score(np.asarray(positives, dtype=object)) > 0):
                    break
        return self

    def score(self, x):
        """Number of kept terms (of size <= k) satisfied by each row; > 0 iff the row satisfies F."""
        x = np.asarray(x, dtype=object).reshape(-1, self.n_features)
        subsets = [cols for cols in self.subsets if len(cols) <= self.k]
        return np.asarray([sum((cols, tuple(row[list(cols)])) in self.terms for cols in subsets)
                           + sum((cols, tuple(row[list(cols)])) in self.extra for cols in self.subsets)
                           for row in x], dtype=float)

    def repair(self, x, y):
        """Test F on labelled rows (the validation rows) and change it so that it matches them.

        1. Every kept term satisfied by a negative validation row is removed (those rows join N), so no
           negative validation row satisfies F any more.
        2. Every positive validation row not satisfied by F gets the smallest-size terms that it satisfies
           and no known negative satisfies (a term of that size is added, whatever k is).
        A positive row equal to a known negative interpretation cannot be repaired (conflict).
        Returns (#rows misclassified before repair, #rows misclassified after repair).
        """
        x = np.asarray(x, dtype=object).reshape(-1, self.n_features)
        y = np.asarray(y).astype(int)
        before = int((self.predict(x) != y).sum())
        val_negatives = [row for row, label in zip(x, y) if label == 0]
        self.negatives = self.negatives + val_negatives
        bad_terms = self.terms_of(val_negatives)
        self.terms -= bad_terms
        self.extra -= bad_terms
        n_terms = self.terms_of(self.negatives)
        for row, label in zip(x, y):
            if label == 1 and self.score(row.reshape(1, -1))[0] == 0:
                for size in range(1, self.n_features + 1):
                    candidates = {(cols, tuple(row[list(cols)])) for cols in self.subsets
                                  if len(cols) == size} - n_terms
                    if candidates:
                        self.extra |= candidates
                        break
        after = int((self.predict(x) != y).sum())
        return before, after

    def predict(self, x):
        return (self.score(x) > 0).astype(int)


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

        val_ocf = ocf_scores(x_val, x_fit, y_fit)
        test_ocf = ocf_scores(x_test, x_fit, y_fit)
        ocf_thr, val_ocf_youden = choose_threshold(y_val, val_ocf, "le")
        ocf_pred = predict(test_ocf, ocf_thr, "le")
        row = {"dataset": dataset_number, "repeat": repeat, "model": "OCF",
               "threshold": ocf_thr, "validation_youden": val_ocf_youden}
        row.update(metrics(y_test, ocf_pred, -test_ocf))
        run_rows.append(row)

        # Baseline: formula learned on the 90% fit part, then tested on the validation rows and
        # changed to match them (repair). Baseline_FitOnly is the same formula before the repair.
        model = Baseline().fit(x_fit, y_fit)
        fit_only_score = model.score(x_test)
        val_acc_before = float((model.predict(x_val) == y_val).mean())
        k_fit = model.k
        errors_before, errors_after = model.repair(x_val, y_val)
        test_score = model.score(x_test)
        row = {"dataset": dataset_number, "repeat": repeat, "model": "Baseline_FitOnly",
               "threshold": np.nan, "validation_youden": np.nan, "baseline_k": k_fit,
               "validation_accuracy": val_acc_before}
        row.update(metrics(y_test, (fit_only_score > 0).astype(int), fit_only_score))
        run_rows.append(row)
        row = {"dataset": dataset_number, "repeat": repeat, "model": "Baseline",
               "threshold": np.nan, "validation_youden": np.nan, "baseline_k": k_fit,
               "validation_accuracy": val_acc_before,
               "validation_errors_before": errors_before, "validation_errors_after": errors_after}
        row.update(metrics(y_test, (test_score > 0).astype(int), test_score))
        run_rows.append(row)

    # Additional experiment: Baseline trained on all official training rows.
    x_train = train[FEATURES].to_numpy(dtype=object)
    y_train = train["y"].to_numpy()
    full_model = Baseline().fit(x_train, y_train)
    full_score = full_model.score(x_test)
    row = {"dataset": dataset_number, "repeat": 1, "model": "Baseline_FullTrain",
           "threshold": np.nan, "validation_youden": np.nan, "baseline_k": full_model.k}
    row.update(metrics(y_test, (full_score > 0).astype(int), full_score))
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
