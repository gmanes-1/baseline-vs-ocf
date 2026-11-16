"""Re-created ocf_plot.py: scatter plot of OCF values, positives vs negatives (+ optional threshold)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from constants import *


def create_ocf_plot(dataset_ocf, fig_file, fig_title, threshold=None):
    positive = dataset_ocf[dataset_ocf[TARGET_DATA_COLUMN_NAME] == 1].copy()
    negative = dataset_ocf[dataset_ocf[TARGET_DATA_COLUMN_NAME] == 0].copy()

    # horizontal jitter: positives around 0, negatives around 1
    pos_x = np.random.normal(0, FIG_DOT_STANDARD_DEVIATION, size=len(positive))
    neg_x = np.random.normal(1, FIG_DOT_STANDARD_DEVIATION, size=len(negative))

    plt.figure(figsize=(6, 8))
    plt.scatter(pos_x, positive[OCF_DATA_COLUMN_NAME], color='blue', alpha=FIG_DOT_TRANSPARENCY, label='pos')
    plt.scatter(neg_x, negative[OCF_DATA_COLUMN_NAME], color='red', alpha=FIG_DOT_TRANSPARENCY, label='neg')

    if threshold is not None:
        plt.axhline(y=threshold, color='gray', linestyle='--')
        low = min(dataset_ocf[OCF_DATA_COLUMN_NAME].min(), threshold)
        high = max(dataset_ocf[OCF_DATA_COLUMN_NAME].max(), threshold)
    else:
        low = dataset_ocf[OCF_DATA_COLUMN_NAME].min()
        high = dataset_ocf[OCF_DATA_COLUMN_NAME].max()

    plt.xlabel('pos vs. neg')
    plt.xticks([0, 1], ['pos', 'neg'])
    plt.ylabel(OCF_DATA_COLUMN_NAME)
    plt.title(f'{fig_title} ({len(positive)}pos/{len(negative)}neg)', fontsize=FIG_TITLE_FONT_SIZE)
    margin = max(1, (high - low) * 0.05)
    plt.ylim(low - margin, high + margin)
    plt.xlim(-1, 2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_file)
    plt.close()
