"""Reproducible comparison of OCF, Baselines, and supplementary classifiers on Monk files.

Run from the repository root:
    python ocf-vs-baseline/runs/run_monk_experiment.py

The official 432-row .test files are used unchanged for every repetition.
All models use the same reproducible 90/10 fit/validation splits. Categorical
supplementary models are one-hot encoded. Original
Baseline fits on the 90% partition without refinement; Baseline with
Hypothesis Refinement additionally uses validation constraints to refine.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
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


class BaselineFormulaLearner:
    """Learn a disjunction of conjunctive attribute-value hypotheses.

    Fit first aggregates duplicate interpretations with delta(w) = positive
    count - negative count. Interpretations with delta >= 0 are positive and
    interpretations with delta < 0 are negative.

    Formula construction generates safe terms (terms covering no known
    negative interpretation), then greedily selects terms that cover the
    largest number of still-uncovered positive interpretations. Ties prefer
    shorter terms. Full-feature terms ensure every separable positive can be
    covered.

    Hypothesis refinement adds validation positives and negatives to the
    formula constraints and rebuilds the DNF using the same positive-preserving
    cover. Thus no separable positive interpretation from fit or validation is
    lost, and no known negative interpretation is covered. If an identical
    feature vector has both labels, perfect separation is impossible; the
    conflict is reported rather than hidden.
    """

    def __init__(self):
        self.n_features = len(FEATURES)
        self.positive_rows = set()
        self.negative_rows = set()
        self.terms = set()
        self.uncovered_positives = set()
        self.conflicting_patterns = set()

    @staticmethod
    def _term_matches(term, row):
        cols, values = term
        return all(row[col] == value for col, value in zip(cols, values))

    def _rebuild_formula(self):
        self.conflicting_patterns = self.positive_rows & self.negative_rows
        positives = sorted(self.positive_rows - self.negative_rows)
        negatives = self.negative_rows
        if not positives:
            self.terms = set()
            self.uncovered_positives = set()
            return

        candidates = {}
        for positive_index, row in enumerate(positives):
            for size in range(1, self.n_features + 1):
                for cols in combinations(range(self.n_features), size):
                    values = tuple(row[col] for col in cols)
                    term = (tuple(cols), values)
                    if term in candidates:
                        continue
                    if any(self._term_matches(term, negative) for negative in negatives):
                        continue
                    coverage = {
                        i for i, positive in enumerate(positives)
                        if self._term_matches(term, positive)
                    }
                    if coverage:
                        candidates[term] = coverage

        uncovered = set(range(len(positives)))
        selected = set()
        while uncovered:
            best_term = None
            best_gain = 0
            for term, coverage in candidates.items():
                gain = len(coverage & uncovered)
                if gain == 0:
                    continue
                if (gain > best_gain or
                    (gain == best_gain and best_term is not None and
                     (len(term[0]), term[0], tuple(map(str, term[1]))) <
                     (len(best_term[0]), best_term[0], tuple(map(str, best_term[1]))))):
                    best_term = term
                    best_gain = gain
            if best_term is None:
                break
            selected.add(best_term)
            uncovered -= candidates[best_term]

        self.terms = selected
        self.uncovered_positives = {positives[i] for i in uncovered}

    def fit(self, x, y):
        x = np.asarray(x, dtype=object).reshape(-1, self.n_features)
        y = np.asarray(y).astype(int)
        delta = {}
        for row, label in zip(x, y):
            key = tuple(row)
            delta[key] = delta.get(key, 0) + (1 if label == 1 else -1)

        self.positive_rows = {row for row, value in delta.items() if value >= 0}
        self.negative_rows = {row for row, value in delta.items() if value < 0}
        self._rebuild_formula()
        return self

    def score(self, x):
        x = np.asarray(x, dtype=object).reshape(-1, self.n_features)
        return np.asarray([
            sum(self._term_matches(term, tuple(row)) for term in self.terms)
            for row in x
        ], dtype=float)

    def predict(self, x):
        return (self.score(x) > 0).astype(int)

    def refine(self, x, y):
        """Refine the DNF using fit and validation constraints.

        Rebuilding the formula is equivalent to specializing or splitting
        offending disjuncts: candidate conjunctive terms are retained only
        when they exclude every known negative. Greedy set cover then combines
        those terms with OR to cover every separable positive interpretation.
        Returns validation errors before and after refinement.
        """
        x = np.asarray(x, dtype=object).reshape(-1, self.n_features)
        y = np.asarray(y).astype(int)
        before = int((self.predict(x) != y).sum())

        for row, label in zip(x, y):
            key = tuple(row)
            if label == 1:
                self.positive_rows.add(key)
            else:
                self.negative_rows.add(key)

        self._rebuild_formula()
        after = int((self.predict(x) != y).sum())
        return before, after


def build_supplementary_models():
    """Create fresh categorical-feature pipelines for each Monk repetition."""
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
        "SVM": SVC(probability=False, random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "Neural Network": MLPClassifier(max_iter=2000, random_state=RANDOM_SEED),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Majority Classifier": DummyClassifier(strategy="most_frequent"),
    }
    return {
        name: make_pipeline(OneHotEncoder(handle_unknown="ignore", sparse_output=False), estimator)
        for name, estimator in estimators.items()
    }


def evaluate_dataset(dataset_number):
    train = load_official_file(DATA / f"monks-{dataset_number}.train")
    test = load_official_file(DATA / f"monks-{dataset_number}.test")
    if len(test) != 432:
        raise ValueError(f"Official Monk-{dataset_number} test file must have 432 rows; got {len(test)}")

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

        # Run all nine supplementary classifiers on the same Monk split.
        # One-hot encoding is required because Monk attributes are categorical.
        for model_name, classifier in build_supplementary_models().items():
            classifier.fit(x_fit, y_fit)
            model_pred = classifier.predict(x_test)
            if hasattr(classifier, "predict_proba"):
                model_score = classifier.predict_proba(x_test)[:, 1]
            elif hasattr(classifier, "decision_function"):
                model_score = classifier.decision_function(x_test)
            else:
                model_score = model_pred.astype(float)
            row = {
                "dataset": dataset_number, "repeat": repeat,
                "model": model_name, "threshold": np.nan,
                "validation_youden": np.nan,
            }
            row.update(metrics(y_test, model_pred, model_score))
            run_rows.append(row)

        val_ocf = ocf_scores(x_val, x_fit, y_fit)
        test_ocf = ocf_scores(x_test, x_fit, y_fit)
        ocf_thr, val_ocf_youden = choose_threshold(y_val, val_ocf, "le")
        ocf_pred = predict(test_ocf, ocf_thr, "le")
        row = {"dataset": dataset_number, "repeat": repeat, "model": "OCF",
               "threshold": ocf_thr, "validation_youden": val_ocf_youden}
        row.update(metrics(y_test, ocf_pred, -test_ocf))
        run_rows.append(row)

        # Original Baseline: same fit partition, but no validation refinement.
        original_baseline = BaselineFormulaLearner().fit(x_fit, y_fit)
        original_test_score = original_baseline.score(x_test)
        row = {
            "dataset": dataset_number, "repeat": repeat,
            "model": "Original Baseline",
            "threshold": np.nan, "validation_youden": np.nan,
            "validation_accuracy": float((original_baseline.predict(x_val) == y_val).mean()),
            "uncovered_positive_patterns": len(original_baseline.uncovered_positives),
            "conflicting_patterns": len(original_baseline.conflicting_patterns),
        }
        row.update(metrics(y_test, (original_test_score > 0).astype(int), original_test_score))
        run_rows.append(row)

        # Baseline with Hypothesis Refinement: same fit partition, then refine
        # using the validation partition.
        model = BaselineFormulaLearner().fit(x_fit, y_fit)
        val_acc_before = float((model.predict(x_val) == y_val).mean())
        errors_before, errors_after = model.refine(x_val, y_val)
        test_score = model.score(x_test)
        row = {
            "dataset": dataset_number, "repeat": repeat,
            "model": "Baseline with Hypothesis Refinement",
            "threshold": np.nan, "validation_youden": np.nan,
            "validation_accuracy_before_refinement": val_acc_before,
            "validation_errors_before_refinement": errors_before,
            "validation_errors_after_refinement": errors_after,
            "uncovered_positive_patterns": len(model.uncovered_positives),
            "conflicting_patterns": len(model.conflicting_patterns),
        }
        row.update(metrics(y_test, (test_score > 0).astype(int), test_score))
        run_rows.append(row)

    return run_rows, {
        "dataset": dataset_number, "official_train_rows": len(train),
        "official_test_rows": len(test), "train_test_overlapping_ids": len(overlap_ids),
        "identical_train_test_rows": len(exact_overlap),
    }



def create_plots(summary_frame, run_frame):
    """Save per-Monk comparison charts and aggregated confusion matrices."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_order = [
        "OCF", "Original Baseline", "Baseline with Hypothesis Refinement",
        "Logistic Regression", "SVM", "Decision Tree", "Random Forest",
        "Gradient Boosting", "Neural Network", "KNN", "Naive Bayes",
        "Majority Classifier",
    ]
    display_names = {
        "Baseline with Hypothesis Refinement": "Baseline + Refinement",
        "Logistic Regression": "Logistic Reg.",
        "Gradient Boosting": "Grad. Boosting",
        "Neural Network": "Neural Network",
        "Majority Classifier": "Majority",
    }

    for number in (1, 2, 3):
        data = summary_frame[summary_frame["dataset"] == number].set_index("model")
        models = [name for name in model_order if name in data.index]
        labels = [display_names.get(name, name) for name in models]
        x = np.arange(len(models))

        # Core classification metrics.
        fig, ax = plt.subplots(figsize=(15, 6))
        width = 0.19
        for offset, metric in enumerate(("mean_accuracy", "mean_precision", "mean_recall", "mean_f1")):
            ax.bar(x + (offset - 1.5) * width, [data.loc[m, metric] for m in models],
                   width, label=metric.replace("mean_", "").title())
        ax.set(title=f"Monk-{number}: classification metrics", ylabel="Score",
               ylim=(0, 1.05), xticks=x, xticklabels=labels)
        ax.tick_params(axis="x", labelrotation=35)
        ax.legend(ncol=4)
        fig.tight_layout()
        fig.savefig(OUTPUT / f"monk_{number}_classification_metrics.png", dpi=180)
        plt.close(fig)

        # MCC and ROC AUC are shown separately from thresholded metrics.
        fig, ax = plt.subplots(figsize=(15, 6))
        width = 0.36
        ax.bar(x - width / 2, [data.loc[m, "mean_mcc"] for m in models],
               width, label="MCC")
        ax.bar(x + width / 2, [data.loc[m, "mean_roc_auc"] for m in models],
               width, label="ROC AUC")
        ax.axhline(0, linewidth=0.8)
        ax.set(title=f"Monk-{number}: MCC and ROC AUC", ylabel="Score",
               ylim=(-1.05, 1.05), xticks=x, xticklabels=labels)
        ax.tick_params(axis="x", labelrotation=35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT / f"monk_{number}_mcc_roc_auc.png", dpi=180)
        plt.close(fig)

        # Summed confusion counts across the 10 repetitions, one panel per model.
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        for ax, model in zip(axes.flat, models):
            row = data.loc[model]
            matrix = np.array([[row["total_tn"], row["total_fp"]],
                               [row["total_fn"], row["total_tp"]]], dtype=int)
            image = ax.imshow(matrix)
            ax.set_title(display_names.get(model, model), fontsize=9)
            ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
            ax.set_yticks([0, 1], labels=["True 0", "True 1"])
            threshold = matrix.max() / 2 if matrix.max() else 0
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:,}", ha="center", va="center",
                        color="white" if value > threshold else "black", fontsize=9)
        for ax in axes.flat[len(models):]:
            ax.axis("off")
        fig.suptitle(f"Monk-{number}: aggregated confusion matrices (10 repeats)")
        fig.tight_layout()
        fig.savefig(OUTPUT / f"monk_{number}_confusion_matrices.png", dpi=180)
        plt.close(fig)


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
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(OUTPUT / "summary_results.csv", index=False)

    # Three separate, easy-to-compare tables: one for each Monk problem.
    for number in (1, 2, 3):
        monk_table = summary_frame[summary_frame["dataset"] == number].drop(
            columns=["dataset"]
        ).sort_values("model")
        monk_table.to_csv(OUTPUT / f"monk_{number}_results.csv", index=False)

    pd.DataFrame(dataset_info).to_csv(OUTPUT / "benchmark_overlap.csv", index=False)
    create_plots(summary_frame, run_frame)
    print(f"Saved CSV tables and comparison plots for all three Monk datasets in {OUTPUT}")


if __name__ == "__main__":
    main()
