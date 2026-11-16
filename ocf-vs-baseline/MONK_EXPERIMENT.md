# Monk benchmark experiment: OCF vs Prade-inspired baseline

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
- The same fit/validation partition is used for OCF, BaselineExact, and BaselineLiteral in each repetition.
- Each model's threshold is selected by maximising Youden's J on validation only. The chosen threshold is then applied to all 432 official test rows.
- No test labels are used for fitting or threshold selection.

## Models

- **OCF:** uses the categorical scaled-distance calculation in `datadistances.py`: a matching category costs 0, a mismatch costs 100; a positive reference example contributes that distance and a negative reference contributes 100 minus that distance. The OCF shift is 600 for six categorical features. Lower OCF is the positive direction.
- **BaselineExact:** Prade-style cumulative delta with each complete feature tuple as one piece of information (+1 for a positive training example, -1 for a negative one).
- **BaselineLiteral:** cumulative delta with each individual feature/value literal as a piece of information. Its score is the sum of the six literal deltas.
- For both baseline variants, the operational positive-class rule is the positive `l`-cut: predict positive iff `delta(x) >= l`. Section 2 defines the cut as a set of interpretations; this explicit rule makes it usable as a binary classifier and is an implementation convention, not a claim that the paper specifies all classification details.

## Important benchmark caveat

The official Monk files contain overlap between training and test IDs. This experiment **does not remove that overlap**, because doing so would change the official test benchmark. The script reports the number of overlapping IDs and exact duplicated labelled rows in `benchmark_overlap.csv`. Results should therefore be interpreted as performance on the official benchmark files, not as a fully independent unseen-example estimate.

## Outputs

Written to `ocf-vs-baseline/results/monk-experiment/`:

- `per_run_results.csv`: one row per model, Monk task, and repetition, including the validation-selected threshold and official-test metrics.
- `summary_results.csv`: mean and sample standard deviation of each metric across 10 repetitions, plus summed confusion counts.
- `benchmark_overlap.csv`: official train/test row counts and overlap diagnostics.

Metrics: accuracy, precision, recall, F1, MCC, positive-class Jaccard, Youden's J, and ROC AUC. ROC AUC uses `-OCF` for OCF and the delta score for the baselines.
