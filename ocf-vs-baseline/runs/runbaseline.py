"""
Baseline (cumulated information, generalized possibility distribution delta) on the SAME
train/test samplings as the OCF model. Put this file in runs/ and run it after createsamplings.py.

Two variants are evaluated and saved as two rows in each <dataset>-all-results.csv:

  BaselineExact   every training row is the piece of information "exactly this interpretation":
                  +1 if the row is an example (y=1), -1 if it is a counter-example (y=0).
                  delta(w) = sum of the increments received by the interpretation w.
                  (This is the baseline exactly as defined in the note.)

  BaselineLiteral every training row is the piece of information "feature X_i has value v"
                  for each of its features (a formula = a literal; its models are all the
                  interpretations satisfying it): +1 if y=1, -1 if y=0.
                  delta(w) = sum over the features of the cumulated increments of (X_i = w_i).

Prediction:  delta > 0 -> positive,  delta < 0 -> negative,  delta == 0 -> FALLBACK (see below).
Ranking score for ROC AUC: delta itself.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from constants import *
from dataop import my_from_csv, separate_id_x_y, save_test_results_into_dataset, sorted_by_increasing_size
from to_file import *

MODEL_NAME_BASELINE_EXACT = 'BaselineExact'
MODEL_NAME_BASELINE_LITERAL = 'BaselineLiteral'

# What to predict when delta == 0 (no guarantee either way):
# 'majority' = majority class of the training set (ties -> 1), or 0 / 1 for a constant answer.
FALLBACK = 'majority'


def fallback_label(y_train):
    if FALLBACK == 'majority':
        return 1 if (y_train == 1).sum() >= (y_train == 0).sum() else 0
    return int(FALLBACK)


class DeltaBaseline:
    """Generalized possibility distribution delta, built by cumulating +1 / -1 increments."""

    def __init__(self, variant):
        assert variant in ('exact', 'literal')
        self.variant = variant

    def fit(self, x, y):
        self.fallback = fallback_label(y)
        self.delta = {}
        for row, label in zip(x.itertuples(index=False, name=None), y):
            inc = 1 if label == 1 else -1
            for key in self._keys(row):
                self.delta[key] = self.delta.get(key, 0) + inc
        return self

    def _keys(self, row):
        if self.variant == 'exact':
            return [tuple(row)]
        return [(i, v) for i, v in enumerate(row)]

    def score(self, x):
        return np.array([sum(self.delta.get(k, 0) for k in self._keys(row))
                         for row in x.itertuples(index=False, name=None)])

    def predict(self, x):
        s = self.score(x)
        return np.where(s > 0, 1, np.where(s < 0, 0, self.fallback)), s


def evaluate(variant, model_name, dataset_name, sampling_ids):
    metric_sums = {m: 0.0 for m in ALL_METRIC_NAMES + [METRIC_ROCAUC]}
    TP = TN = FP = FN = 0
    coverage = []
    for id_sampling in sampling_ids:
        _, x_tr, y_tr = separate_id_x_y(my_from_csv(get_sampling_train_ocfhamming_file_path(dataset_name, id_sampling)))
        _, x_te, y_te = separate_id_x_y(my_from_csv(get_sampling_test_ocfhamming_file_path(dataset_name, id_sampling)))
        y_tr, y_te = y_tr.values, y_te.values

        model = DeltaBaseline(variant).fit(x_tr, y_tr)
        y_pred, scores = model.predict(x_te)
        coverage.append(float((scores != 0).mean()))

        for m in ALL_METRIC_NAMES:
            metric_sums[m] += calculate_metric(y_te, y_pred, m)
        metric_sums[METRIC_ROCAUC] += roc_auc_score(y_te, scores) if len(np.unique(y_te)) > 1 else np.nan

        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        TP, TN, FP, FN = TP + tp, TN + tn, FP + fp, FN + fn

    n = len(sampling_ids)
    result = {MODEL_NAME: model_name, TOTAL_TP: int(TP), TOTAL_TN: int(TN), TOTAL_FP: int(FP), TOTAL_FN: int(FN)}
    for m, total in metric_sums.items():
        result[METRIC_NAME_TO_AVG_METRIC_FEATURE_NAME[m]] = total / n
    save_test_results_into_dataset(get_all_results_file_name(dataset_name), result)
    return result, float(np.mean(coverage))


if __name__ == "__main__":
    for dataset_name in sorted_by_increasing_size(get_all_datasets_names()):
        dirs = get_all_sampling_dir_paths(dataset_name)
        if not dirs:
            print(f'No sampling for {dataset_name}, skipping (run createsamplings.py first)')
            continue
        ids = [get_id_sampling_from_dir_path(d) for d in dirs]
        print(f'\n{dataset_name}: {len(ids)} samplings')
        for variant, name in (('exact', MODEL_NAME_BASELINE_EXACT), ('literal', MODEL_NAME_BASELINE_LITERAL)):
            res, cov = evaluate(variant, name, dataset_name, ids)
            print(f'  {name:16s} acc={res[AVG_ACCURACY]:.3f} f1={res[AVG_F1_SCORE]:.3f} mcc={res[AVG_MCC]:.3f} '
                  f'youden={res[AVG_YOUDEN]:.3f} auc={res[AVG_ROC_AUC]:.3f}  (test rows with delta != 0: {cov:.0%})')
