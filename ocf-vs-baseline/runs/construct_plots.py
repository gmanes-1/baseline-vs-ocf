"""Port of the original 2025 OCF-vs-classifiers plotting workflow."""
import argparse
import os
from math import pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from constants import *
from dataop import my_from_csv, sorted_by_increasing_size
from to_file import (
    get_all_datasets_names, get_all_results_file_name, get_all_results_plot_dir_path,
)

METRIC_LABELS = [
    (METRIC_ACCURACY, NEW_AVG_ACCURACY),
    (METRIC_PRECISION, NEW_AVG_PRECISION),
    (METRIC_RECALL, NEW_AVG_RECALL),
    (METRIC_F1_SCORE, NEW_AVG_F1_SCORE),
    (METRIC_MCC, NEW_AVG_MCC),
    (METRIC_JACCARD, NEW_AVG_JACCARD),
    (METRIC_YOUDEN, NEW_AVG_YOUDEN),
    (METRIC_ROCAUC, NEW_AVG_ROC_AUC),
]
METRIC_LIST = [label for _, label in METRIC_LABELS]
MODEL_SHORT_NAMES = {
    MODEL_NAME_LOGISTIC_REGRESSION: "LogReg",
    MODEL_NAME_SVM: "SVM",
    MODEL_NAME_DECISION_TREE: "DecTree",
    MODEL_NAME_RANDOM_FOREST: "RandFor",
    MODEL_NAME_GRADIENT_BOOSTING: "GradBoost",
    MODEL_NAME_NEURAL_NETWORK: "NeurNet",
    MODEL_NAME_KNN: "KNN",
    MODEL_NAME_NAIVE_BAYES: "NaivBay",
    MODEL_NAME_TRIVIAL_MAJ: "Maj",
    NEW_MODEL_NAME_OCF_HAMMING: "OCF",
}
MODEL_COLORS = {
    "OCF": "mediumblue",
    MODEL_NAME_LOGISTIC_REGRESSION: "wheat",
    MODEL_NAME_SVM: "yellowgreen",
    MODEL_NAME_DECISION_TREE: "skyblue",
    MODEL_NAME_RANDOM_FOREST: "saddlebrown",
    MODEL_NAME_GRADIENT_BOOSTING: "coral",
    MODEL_NAME_NEURAL_NETWORK: "orangered",
    MODEL_NAME_KNN: "darkturquoise",
    MODEL_NAME_NAIVE_BAYES: "greenyellow",
    MODEL_NAME_TRIVIAL_MAJ: "grey",
}
COLUMN_RENAMES = {
    AVG_ACCURACY: NEW_AVG_ACCURACY,
    AVG_F1_SCORE: NEW_AVG_F1_SCORE,
    AVG_PRECISION: NEW_AVG_PRECISION,
    AVG_RECALL: NEW_AVG_RECALL,
    AVG_JACCARD: NEW_AVG_JACCARD,
    AVG_MCC: NEW_AVG_MCC,
    AVG_YOUDEN: NEW_AVG_YOUDEN,
    AVG_ROC_AUC: NEW_AVG_ROC_AUC,
    MODEL_NAME_OCF_HAMMING: NEW_MODEL_NAME_OCF_HAMMING,
}


def load_all_available_result_csv_files(dataset_names):
    loaded = {}
    for dataset_name in dataset_names:
        path = get_all_results_file_name(dataset_name)
        if not os.path.exists(path):
            continue
        frame = my_from_csv(path)
        if all(column in frame.columns for column in ALL_FEATURE_NAME_FOR_TEST_RESULTS):
            if all(model in frame[MODEL_NAME].values for model in ALL_MODEL_NAMES):
                frame = frame.rename(columns=COLUMN_RENAMES)
                frame[MODEL_NAME] = frame[MODEL_NAME].replace(COLUMN_RENAMES)
                loaded[dataset_name] = frame
    return loaded


def generate_scatter_plots(all_results):
    os.makedirs(get_all_results_plot_dir_path(), exist_ok=True)
    points = []
    for dataset_name, frame in all_results.items():
        ocf = frame[frame[MODEL_NAME] == "OCF"].iloc[0]
        for model_name in frame[MODEL_NAME].unique():
            if model_name == "OCF":
                continue
            other = frame[frame[MODEL_NAME] == model_name].iloc[0]
            for metric in METRIC_LIST:
                points.append({
                    "OCF": ocf[metric], "OtherModel": other[metric],
                    "Metric": metric, "Model": model_name,
                })
    scatter = pd.DataFrame(points)
    for metric in METRIC_LIST:
        for model_name in scatter["Model"].unique():
            subset = scatter[(scatter["Metric"] == metric) & (scatter["Model"] == model_name)]
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(subset["OCF"], subset["OtherModel"], alpha=0.8, s=150)
            low = min(subset["OCF"].min(), subset["OtherModel"].min())
            high = max(subset["OCF"].max(), subset["OtherModel"].max())
            ax.plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1)
            ax.set(xlabel="OCF", ylabel=model_name,
                   title=f"{metric} (OCF vs {model_name})")
            ax.grid(True)
            fig.savefig(os.path.join(get_all_results_plot_dir_path(),
                                     f"scatter_{metric}_{model_name}.png"),
                        bbox_inches="tight")
            plt.close(fig)


def generate_box_plots(all_results):
    os.makedirs(get_all_results_plot_dir_path(), exist_ok=True)
    combined = pd.concat(all_results.values(), ignore_index=True)
    combined = combined.copy()
    combined[MODEL_NAME] = combined[MODEL_NAME].map(MODEL_SHORT_NAMES)
    for metric in METRIC_LIST:
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(x=MODEL_NAME, y=metric, data=combined, ax=ax)
        ax.set(title=f"Box Plot: {metric}", xlabel="Model Name", ylabel=metric)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y")
        fig.subplots_adjust(bottom=0.23)
        fig.savefig(os.path.join(get_all_results_plot_dir_path(), f"box_{metric}.png"),
                    bbox_inches="tight")
        plt.close(fig)


def _draw_radar(ax, frame, title):
    categories = METRIC_LIST
    n = len(categories)
    angles = [i / float(n) * 2 * pi for i in range(n)]
    angles += angles[:1]
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    for _, row in frame.iterrows():
        model = row[MODEL_NAME]
        values = [float(row[m]) for m in categories]
        values += values[:1]
        style = "--" if model == MODEL_NAME_TRIVIAL_MAJ else "-"
        width = 3 if model == "OCF" else 2
        color = MODEL_COLORS.get(model)
        ax.plot(angles, values, linewidth=width, linestyle=style,
                label=model, color=color)
        ax.fill(angles, values, color=color, alpha=0.08)
    ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.48, 1.12), fontsize=8)


def generate_spider_plots(all_results):
    output = get_all_results_plot_dir_path()
    os.makedirs(output, exist_ok=True)

    def split_models(frame):
        focus = frame[frame[MODEL_NAME] == "OCF"]
        maj = frame[frame[MODEL_NAME] == MODEL_NAME_TRIVIAL_MAJ]
        others = frame[
            (frame[MODEL_NAME] != "OCF") &
            (frame[MODEL_NAME] != MODEL_NAME_TRIVIAL_MAJ)
        ]
        midpoint = len(others) // 2
        return (
            pd.concat([others.iloc[:midpoint], focus]),
            pd.concat([others.iloc[midpoint:], maj, focus]),
        )

    for dataset_name, frame in all_results.items():
        first, second = split_models(frame)
        for part_number, part in enumerate((first, second), start=1):
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(111, polar=True)
            _draw_radar(ax, part, f"{dataset_name} (#{part_number})")
            fig.savefig(os.path.join(output,
                         f"spider_dataset_{dataset_name}-part{part_number}.png"),
                        bbox_inches="tight")
            plt.close(fig)

    average = pd.concat(all_results.values(), ignore_index=True)
    average = average.groupby(MODEL_NAME, as_index=False)[METRIC_LIST].mean()
    first, second = split_models(average)
    for part_number, part in enumerate((first, second), start=1):
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, polar=True)
        _draw_radar(ax, part, f"Spider Plot: Average Results (Part {part_number})")
        fig.savefig(os.path.join(output, f"spider_avg-part{part_number}.png"),
                    bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-plots", action="store_true",
                        help="Also create scatter and box plots (spider plots are always created).")
    args = parser.parse_args()
    names = sorted_by_increasing_size(get_all_datasets_names())
    results = load_all_available_result_csv_files(names)
    if not results:
        raise SystemExit("No datasets have complete results for all models and OCF.")
    print(f"Complete results available for {len(results)} datasets:")
    print(list(results))
    generate_spider_plots(results)
    if args.all_plots:
        generate_scatter_plots(results)
        generate_box_plots(results)


if __name__ == "__main__":
    main()
