"""
Re-created constants.py (the original file was not shipped with the code archive).

Every name below is one that the code base uses via `from constants import *`.
Values were inferred from how the code uses them. Anything that is only a naming
choice (file suffixes, column labels, model labels) is arbitrary but internally
consistent. The two choices that can influence numerical results are marked
"[AFFECTS RESULTS]".
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, confusion_matrix)

# ----------------------------------------------------------------------------
# Environment / directories
# ----------------------------------------------------------------------------
DATASET_ROOT_PATH_ENV_VAR_NAME = 'KRML_DATASETS'
ALL_RESULTS_DIR_NAME = 'all-results-plots'

# ----------------------------------------------------------------------------
# Column names in the dataset files
# ----------------------------------------------------------------------------
TARGET_DATA_COLUMN_NAME = 'y'
INDEX_COLUMN_NAME = 'ID'
OCF_DATA_COLUMN_NAME = 'OCF'
DISTANCE_DATA_COLUMN_NAME = 'distance'

# feature-type file (columns are read as literals 'FEATURE_NAME' / 'FEATURE_TYPE' in the code)
FEATURE_NAME_COLUMN_NAME = 'FEATURE_NAME'
FEATURE_TYPE_COLUMN_NAME = 'FEATURE_TYPE'

# scaling-parameter file
SHIFT_COLUMN_NAME = 'SHIFT'
COEFFICIENT_COLUMN_NAME = 'COEFFICIENT'
MIN_VALUE_COLUMN_NAME = 'MIN_VALUE'
MAX_VALUE_COLUMN_NAME = 'MAX_VALUE'
MEAN_VALUE_COLUMN_NAME = 'MEAN_VALUE'

# ----------------------------------------------------------------------------
# Scaling / distances
# ----------------------------------------------------------------------------
MIN_SCALING_VALUE = 0
MAX_SCALING_VALUE = 100                                    # "normalized to, let's say 100" (retrieve_uci_datasets.py)
# [AFFECTS RESULTS] distance between two different categorical/binary values.
# Taken equal to the largest per-feature distance of numeric features.
DISTANCE_BETWEEN_TWO_CATEGORICAL_OR_BINARY_DIFFERENT_VALUES = MAX_SCALING_VALUE
UNKNOWN_CATEGORY_NAME_TRAINING = 'UNKNOWN_TRAIN_'
UNKNOWN_CATEGORY_NAME_TEST = 'UNKNOWN_TEST_'

HAMMING_NAME = 'hamming'
DRASTIC_NAME = 'drastic'

# ----------------------------------------------------------------------------
# File / directory name pieces (see to_file.py)
# ----------------------------------------------------------------------------
DATASET_ORIGINAL = '-original'
DATASET_TYPE = '-type'
DATASET_SCALING_PARAMETERS = '-scaling-param'

DATASET_SAMPLING_NAME = '-sampling'
DATASET_SAMPLING_TRAIN_NAME = '-train'
DATASET_SAMPLING_TEST_NAME = '-test'
DATASET_STANDARDIMPUTEDSCALED = '-isc'
DATASET_ONEHOTENCODED = '-ohe'
DATASET_SCALED_FOR_OCFHAMMING = '-ocfhamming'
DATASET_OCF_NAME = '-ocf'

SHIFT_VALUE_NAME = '-shift'
BEST_THRESHOLD_NAME = '-best-threshold'
WITH_THRESHOLD_NAME = '-with-threshold'
OCF_FIG_NAME_WITHOUT_THRESHOLD = '-no-threshold'
CURVE_FIG_NAME = '-curve'
ROC_FIG_NAME = '-roc'
LOG_NAME = '-log'
ALL_RESULTS = '-all-results'

FILLED_LINE = '-' * 60

# ----------------------------------------------------------------------------
# Figure settings (used by ocf_plot.py)
# ----------------------------------------------------------------------------
FIG_DOT_STANDARD_DEVIATION = 0.08
FIG_DOT_TRANSPARENCY = 0.5
FIG_TITLE_FONT_SIZE = 8

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
METRIC_ACCURACY = 'accuracy'
METRIC_PRECISION = 'precision'
METRIC_RECALL = 'recall'
METRIC_F1_SCORE = 'f1_score'
METRIC_MCC = 'mcc'
METRIC_JACCARD = 'jaccard'
METRIC_YOUDEN = 'youden'
METRIC_ROCAUC = 'roc_auc'        # runexistingmodels.py compares against the literal 'roc_auc'

# threshold selection on the training set (see comments in createOCFsfortrainingsets.py)
METRIC_NAME_USED_FOR_BEST_THRESHOLD = METRIC_YOUDEN

# ROC AUC is NOT part of this list: scripts add it separately
ALL_METRIC_NAMES = [METRIC_ACCURACY, METRIC_PRECISION, METRIC_RECALL, METRIC_F1_SCORE,
                    METRIC_MCC, METRIC_JACCARD, METRIC_YOUDEN]

AVG_ACCURACY = 'avg_accuracy'
AVG_PRECISION = 'avg_precision'
AVG_RECALL = 'avg_recall'
AVG_F1_SCORE = 'avg_f1_score'
AVG_MCC = 'avg_mcc'
AVG_JACCARD = 'avg_jaccard'
AVG_YOUDEN = 'avg_youden'
AVG_ROC_AUC = 'avg_roc_auc'

METRIC_NAME_TO_AVG_METRIC_FEATURE_NAME = {
    METRIC_ACCURACY: AVG_ACCURACY,
    METRIC_PRECISION: AVG_PRECISION,
    METRIC_RECALL: AVG_RECALL,
    METRIC_F1_SCORE: AVG_F1_SCORE,
    METRIC_MCC: AVG_MCC,
    METRIC_JACCARD: AVG_JACCARD,
    METRIC_YOUDEN: AVG_YOUDEN,
    METRIC_ROCAUC: AVG_ROC_AUC,
}

# names used in the figures (from_test_results_to_plot_2025.py)
NEW_AVG_ACCURACY = 'Accuracy'
NEW_AVG_PRECISION = 'Precision'
NEW_AVG_RECALL = 'Recall'
NEW_AVG_F1_SCORE = 'F1_score'
NEW_AVG_MCC = 'MCC'
NEW_AVG_JACCARD = 'Jaccard'
NEW_AVG_YOUDEN = 'Youden'
NEW_AVG_ROC_AUC = 'Roc_auc'

TOTAL_TP = 'total_tp'
TOTAL_TN = 'total_tn'
TOTAL_FP = 'total_fp'
TOTAL_FN = 'total_fn'

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
MODEL_NAME = 'model'
MODEL_NAME_LOGISTIC_REGRESSION = 'LogisticRegression'
MODEL_NAME_SVM = 'SVM'
MODEL_NAME_DECISION_TREE = 'DecisionTree'
MODEL_NAME_RANDOM_FOREST = 'RandomForest'
MODEL_NAME_GRADIENT_BOOSTING = 'GradientBoosting'
MODEL_NAME_NEURAL_NETWORK = 'NeuralNetwork'
MODEL_NAME_KNN = 'KNN'
MODEL_NAME_NAIVE_BAYES = 'NaiveBayes'
MODEL_NAME_TRIVIAL_MAJ = 'Maj'
MODEL_NAME_OCF_HAMMING = 'OCFHamming'
NEW_MODEL_NAME_OCF_HAMMING = 'OCF'

ALL_MODEL_NAMES = [MODEL_NAME_LOGISTIC_REGRESSION, MODEL_NAME_SVM, MODEL_NAME_DECISION_TREE,
                   MODEL_NAME_RANDOM_FOREST, MODEL_NAME_GRADIENT_BOOSTING, MODEL_NAME_NEURAL_NETWORK,
                   MODEL_NAME_KNN, MODEL_NAME_NAIVE_BAYES, MODEL_NAME_TRIVIAL_MAJ, MODEL_NAME_OCF_HAMMING]

# columns of every <dataset>-all-results.csv (dataop.save_test_results_into_dataset asserts equality of key sets)
ALL_FEATURE_NAME_FOR_TEST_RESULTS = (
    [MODEL_NAME]
    + [METRIC_NAME_TO_AVG_METRIC_FEATURE_NAME[m] for m in ALL_METRIC_NAMES + [METRIC_ROCAUC]]
    + [TOTAL_TP, TOTAL_TN, TOTAL_FP, TOTAL_FN]
)


# ----------------------------------------------------------------------------
# Metric helpers (called by runtests.py, createOCFsfortrainingsets.py, runexistingmodels.py)
# ----------------------------------------------------------------------------
def _tp_tn_fp_fn(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tp, tn, fp, fn


def my_jaccard_index(y_true, y_pred):
    """Jaccard index of the positive class: TP / (TP + FP + FN); 0 if undefined."""
    tp, tn, fp, fn = _tp_tn_fp_fn(y_true, y_pred)
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 0.0


def my_youden_index(y_true, y_pred):
    """Youden's J = sensitivity + specificity - 1, in [-1, 1]."""
    tp, tn, fp, fn = _tp_tn_fp_fn(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sens + spec - 1)


def calculate_metric(y_true, y_pred, metric_name):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if metric_name == METRIC_ACCURACY:
        return accuracy_score(y_true, y_pred)
    if metric_name == METRIC_PRECISION:
        return precision_score(y_true, y_pred, zero_division=0)
    if metric_name == METRIC_RECALL:
        return recall_score(y_true, y_pred, zero_division=0)
    if metric_name == METRIC_F1_SCORE:
        return f1_score(y_true, y_pred, zero_division=0)
    if metric_name == METRIC_MCC:
        return matthews_corrcoef(y_true, y_pred)
    if metric_name == METRIC_JACCARD:
        return my_jaccard_index(y_true, y_pred)
    if metric_name == METRIC_YOUDEN:
        return my_youden_index(y_true, y_pred)
    raise ValueError(f'Unknown metric {metric_name}')
