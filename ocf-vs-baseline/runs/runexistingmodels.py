"""Run the original supplementary scikit-learn model comparison.

This is a faithful port of IteratedBeliefChangeML/python-code/runs/runexistingmodels.py.
"""
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
)
from constants import *
from custom_models.majority_classifier import MajorityClassifier
from dataop import (
    my_from_csv, save_test_results_into_dataset, sorted_by_increasing_size,
    separate_id_x_y,
)
from to_file import (
    get_all_datasets_names, get_all_results_file_name,
    get_all_sampling_dir_paths, get_id_sampling_from_dir_path,
    get_sampling_train_existingmodel_file_path,
    get_sampling_test_existingmodel_file_path,
)


def build_models():
    # Preserve the estimators and parameters used in the original experiment.
    return {
        MODEL_NAME_LOGISTIC_REGRESSION: LogisticRegression(max_iter=5000),
        MODEL_NAME_SVM: SVC(probability=True),
        MODEL_NAME_DECISION_TREE: DecisionTreeClassifier(),
        MODEL_NAME_RANDOM_FOREST: RandomForestClassifier(),
        MODEL_NAME_GRADIENT_BOOSTING: GradientBoostingClassifier(),
        MODEL_NAME_NEURAL_NETWORK: MLPClassifier(max_iter=5000),
        MODEL_NAME_KNN: KNeighborsClassifier(),
        MODEL_NAME_NAIVE_BAYES: GaussianNB(),
        MODEL_NAME_TRIVIAL_MAJ: MajorityClassifier(),
    }


def evaluate(y_true, y_pred, y_probability):
    values = {
        METRIC_ACCURACY: accuracy_score(y_true, y_pred),
        METRIC_PRECISION: precision_score(y_true, y_pred, zero_division=0),
        METRIC_RECALL: recall_score(y_true, y_pred, zero_division=0),
        METRIC_F1_SCORE: f1_score(y_true, y_pred, zero_division=0),
        METRIC_MCC: matthews_corrcoef(y_true, y_pred),
        METRIC_JACCARD: my_jaccard_index(y_true, y_pred),
        METRIC_YOUDEN: my_youden_index(y_true, y_pred),
    }
    if y_probability is not None and len(np.unique(y_true)) > 1:
        values[METRIC_ROCAUC] = roc_auc_score(y_true, y_probability)
    else:
        values[METRIC_ROCAUC] = np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return values, {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}


def main():
    models = build_models()
    metrics = ALL_METRIC_NAMES + [METRIC_ROCAUC]
    datasets = sorted_by_increasing_size(get_all_datasets_names())
    for number, dataset_name in enumerate(datasets, start=1):
        started = time.time()
        print(f"Processing #{number}/#{len(datasets)} dataset {dataset_name}...")
        results = {name: {metric: [] for metric in metrics} for name in models}
        confusion = {
            name: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for name in models
        }
        samplings = get_all_sampling_dir_paths(dataset_name)
        if not samplings:
            print(f"No sampling folders for {dataset_name}; skipping.")
            continue
        for sampling_dir in samplings:
            sampling_id = get_id_sampling_from_dir_path(sampling_dir)
            train = my_from_csv(
                get_sampling_train_existingmodel_file_path(dataset_name, sampling_id)
            )
            test = my_from_csv(
                get_sampling_test_existingmodel_file_path(dataset_name, sampling_id)
            )
            _, x_train, y_train = separate_id_x_y(train)
            _, x_test, y_test = separate_id_x_y(test)
            for model_name, model in models.items():
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                probability = (
                    model.predict_proba(x_test)[:, 1]
                    if hasattr(model, "predict_proba") else None
                )
                metric_values, cm_values = evaluate(y_test, prediction, probability)
                for metric, value in metric_values.items():
                    results[model_name][metric].append(value)
                for key, value in cm_values.items():
                    confusion[model_name][key] += value
            print(f"  completed {sampling_dir}")
        for model_name, model_results in results.items():
            row = {MODEL_NAME: model_name}
            for metric, scores in model_results.items():
                row[METRIC_NAME_TO_AVG_METRIC_FEATURE_NAME[metric]] = (
                    np.nanmean(scores) if metric == METRIC_ROCAUC else np.mean(scores)
                )
            row.update({
                TOTAL_TP: confusion[model_name]["TP"],
                TOTAL_TN: confusion[model_name]["TN"],
                TOTAL_FP: confusion[model_name]["FP"],
                TOTAL_FN: confusion[model_name]["FN"],
            })
            save_test_results_into_dataset(get_all_results_file_name(dataset_name), row)
        print(f"Finished {dataset_name} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
