# Monk benchmark experiment: OCF vs Baseline

## Run

From the repository root:

```bash
python ocf-vs-baseline/runs/run_monk_experiment.py
```

Requires Python, NumPy, pandas, and scikit-learn.

## Benchmark protocol

- Reads the official `monks-1/2/3.train` and `.test` files in `ocf-vs-baseline/datasets/`.
- **Preserves every official test row unchanged (432 rows per Monk task).** The test files are never filtered or rewritten.
- The official training file is split into 10 reproducible, stratified 90% fit / 10% validation partitions (random seed 42).
- The same fit/validation partition is used for OCF and Baseline in each repetition.
- OCF is trained on the 90% fit partition and its threshold is selected by maximising Youden's J on the 10% validation partition. The chosen threshold is then applied to all 432 official test rows.
- Baseline learns its formula F on the same 90% fit partition, then F is tested on the 10% validation rows and changed to match them (see Models). `Baseline_FitOnly` is the formula before that repair. An additional **Baseline_FullTrain** run per dataset trains Baseline on 100% of the official training file.
- No test labels are used for fitting, threshold selection, choosing k, or repairing the formula.

## Models

- **OCF:** uses the categorical scaled-distance calculation in `datadistances.py`: a matching category costs 0, a mismatch costs 100; a positive reference example contributes that distance and a negative reference contributes 100 minus that distance. The OCF shift is 600 for six categorical features. Lower OCF is the positive direction.
- **Baseline:** Section 2 is learned on the training data in two steps. (1) Cumulation: over the training examples only, `delta(w)` = #positive examples equal to w minus #negative ones, i.e. +1 for every positive and -1 for every negative. (2) After training, formula learning: P = training interpretations with `delta >= 0`, N = training interpretations with `delta < 0`. The learned formula F is a disjunction of terms, each a conjunction of at most k literals `X_i = v`; a term is kept iff some interpretation in P satisfies it and none in N does. k is the smallest size for which every interpretation in P satisfies F. An unseen example is classified positive iff it satisfies F. The ROC AUC score is the number of kept terms the example satisfies. Requiring F to reject N is an implementation choice: without it F = true would satisfy every example in P. Section 2 itself does not define a classification rule.
  **Repair on validation:** F is tested on the validation rows (`validation_accuracy` is measured before the repair). Every kept term satisfied by a negative validation row is removed, and for every positive validation row not satisfied by F the smallest-size terms that it satisfies and no known negative satisfies are added. After the repair no validation row is misclassified (except a positive row identical to a known negative interpretation). The validation rows are therefore used as training data in this second stage; the official test file stays untouched until the final evaluation.

## Important benchmark caveat

The official Monk files contain overlap between training and test IDs. This experiment **does not remove that overlap**, because doing so would change the official test benchmark. The script reports the number of overlapping IDs and exact duplicated labelled rows in `benchmark_overlap.csv`. Results should therefore be interpreted as performance on the official benchmark files, not as a fully independent unseen-example estimate.

## Outputs

Written to `ocf-vs-baseline/results/monk-experiment/`:

- `per_run_results.csv`: one row per model, Monk task, and repetition, including the validation-selected threshold (OCF only), the term size `baseline_k` (Baseline only) and official-test metrics.
- `summary_results.csv`: mean and sample standard deviation of each metric across repetitions, plus summed confusion counts. OCF and Baseline have 10 repetitions; `Baseline_FullTrain` has one full-training run per dataset (its standard deviations are undefined).
- `benchmark_overlap.csv`: official train/test row counts and overlap diagnostics.

Metrics: accuracy, precision, recall, F1, MCC, positive-class Jaccard, Youden's J, and ROC AUC. ROC AUC uses `-OCF` for OCF and the number of satisfied kept terms for Baseline.
