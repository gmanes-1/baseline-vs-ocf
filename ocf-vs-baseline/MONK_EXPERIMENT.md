# Monk benchmark experiment: OCF vs the two Baseline variants

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
- The same fit/validation partition is used for OCF, Original Baseline, and Baseline with Hypothesis Refinement in each repetition.
- OCF is trained on the 90% fit partition and its threshold is selected by maximising Youden's J on the 10% validation partition. The chosen threshold is then applied to all 432 official test rows.
- The Baseline is evaluated in two variants on the same 10 splits. **Original Baseline** is fitted on each 90% fit partition and does not use validation labels or call `refine()`. **Baseline with Hypothesis Refinement** starts from the same 90% fit partition and additionally incorporates the 10% validation labels through `refine()`. This makes the comparison isolate the effect of hypothesis refinement. OCF remains unchanged.
- No test labels are used for fitting, threshold selection, formula repair, or model selection.

## Models

The same 10 stratified 90/10 repetitions are used to test all models on each
Monk task. The nine supplementary classifiers are Logistic Regression, SVM,
Decision Tree, Random Forest, Gradient Boosting, Neural Network, KNN, Naive
Bayes, and Majority Classifier. Since Monk attributes are categorical, these
classifiers are evaluated through a one-hot-encoding pipeline. Each classifier
is freshly instantiated for every repetition.

The comparison also includes OCF, Original Baseline, and Baseline with
Hypothesis Refinement. The Majority Classifier is only an additional reference
model; it does not replace either Baseline variant.

- **OCF:** uses the categorical scaled-distance calculation in `datadistances.py`: a matching category costs 0, a mismatch costs 100; a positive reference example contributes that distance and a negative reference contributes 100 minus that distance. The OCF shift is 600 for six categorical features. Lower OCF is the positive direction.
- **Both Baseline variants:** first cumulate duplicate training interpretations: `delta(w)` = number of positive rows equal to `w` minus number of negative rows equal to `w`. Interpretations with `delta >= 0` form the positive set P; interpretations with `delta < 0` form the negative set N. The classifier is a disjunction (OR) of conjunctive terms, where each condition is an attribute-value equality such as `X_2 = 1`.

### Positive-preserving formula construction and hypothesis refinement

The formula learner generates candidate conjunctive terms from attribute subsets of positive interpretations. A candidate is **safe** only if it covers no known negative interpretation. It then uses greedy set cover:

1. Start with all positive interpretations uncovered.
2. Select the safe term covering the largest number of still-uncovered positives; ties prefer fewer conditions.
3. Add that term to the disjunction and mark its covered positives as covered.
4. Repeat until all separable positives are covered or no safe candidate remains.

Because the complete six-feature conjunction is among the candidates, every positive interpretation can be covered unless the same complete feature vector is also labelled negative. In that conflicting case, no deterministic formula using these attributes can satisfy both labels; the script records the conflict rather than claiming perfect separation.

**False-positive refinement:** validation negatives are added to the negative constraints. Terms that would cover them are therefore not eligible in the rebuilt formula. If one broad term would otherwise need to be discarded, the greedy cover can replace it with multiple narrower conjunctive terms (a split), preserving positive coverage through the OR of those terms.

**False-negative refinement:** validation positives are added to the positive constraints. The cover adds one or more safe conjunctive terms to include them, while retaining coverage of all previously separable fit positives.

After refinement, the script reports validation errors before and after refinement, plus counts of uncovered positive patterns and conflicting patterns. A zero error count is guaranteed only when the labelled fit/validation interpretations are separable using the available attributes.

### Original Baseline

**Original Baseline** is fitted separately on each of the 10 reproducible 90% fit partitions. It uses the same initial DNF construction and greedy safe-term cover, but **does not call `refine()` and does not use validation labels**. Its formula is applied unchanged to the official test file. The **Baseline with Hypothesis Refinement** starts from the same fit partition, then calls `refine()` using the corresponding 10% validation partition. Therefore, both variants are evaluated on matched splits, and their difference isolates the effect of refinement.

The greedy set-cover step is a heuristic: it produces a valid cover when one exists in the candidate language, but it does not guarantee the globally smallest DNF. Candidate generation enumerates attribute subsets, so its worst-case dependence on the number of attributes is exponential; the Monk experiments use six attributes.

## Important benchmark caveat

The official Monk files contain overlap between training and test IDs. This experiment **does not remove that overlap**, because doing so would change the official test benchmark. The script reports the number of overlapping IDs and exact duplicated labelled rows in `benchmark_overlap.csv`. Results should therefore be interpreted as performance on the official benchmark files, not as a fully independent unseen-example estimate.

## Outputs

Written to `ocf-vs-baseline/results/monk-experiment/`:

- `monk_1_results.csv`, `monk_2_results.csv`, and `monk_3_results.csv`: three separate summary tables, one per Monk task, with each model's mean and sample standard deviation across 10 repetitions and summed confusion counts.
- `per_run_results.csv`: detailed metrics for every model, Monk task, and repetition.
- `summary_results.csv`: combined summary for all three Monk tasks.
- `benchmark_overlap.csv`: official train/test row counts and overlap diagnostics.

Metrics: accuracy, precision, recall, F1, MCC, positive-class Jaccard, Youden's J, and ROC AUC. ROC AUC uses `-OCF` for OCF and the number of satisfied DNF terms for Baseline.
