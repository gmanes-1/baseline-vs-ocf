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
- Baseline learns a DNF formula on the 90% fit partition. It is evaluated in two variants: `Baseline_NoRepair` uses the fit-only formula unchanged, while `Baseline` incorporates validation labels through constrained formula repair. OCF remains unchanged.
- No test labels are used for fitting, threshold selection, formula repair, or model selection.

## Models

- **OCF:** uses the categorical scaled-distance calculation in `datadistances.py`: a matching category costs 0, a mismatch costs 100; a positive reference example contributes that distance and a negative reference contributes 100 minus that distance. The OCF shift is 600 for six categorical features. Lower OCF is the positive direction.
- **Baseline:** first cumulates duplicate training interpretations: `delta(w)` = number of positive rows equal to `w` minus number of negative rows equal to `w`. Interpretations with `delta >= 0` form the positive set P; interpretations with `delta < 0` form the negative set N. The classifier is a disjunction (OR) of conjunctive terms, where each condition is an attribute-value equality such as `X_2 = 1`.

### Positive-preserving formula construction and repair

The formula learner generates candidate conjunctive terms from attribute subsets of positive interpretations. A candidate is **safe** only if it covers no known negative interpretation. It then uses greedy set cover:

1. Start with all positive interpretations uncovered.
2. Select the safe term covering the largest number of still-uncovered positives; ties prefer fewer conditions.
3. Add that term to the disjunction and mark its covered positives as covered.
4. Repeat until all separable positives are covered or no safe candidate remains.

Because the complete six-feature conjunction is among the candidates, every positive interpretation can be covered unless the same complete feature vector is also labelled negative. In that conflicting case, no deterministic formula using these attributes can satisfy both labels; the script records the conflict rather than claiming perfect separation.

**False-positive repair:** validation negatives are added to the negative constraints. Terms that would cover them are therefore not eligible in the rebuilt formula. If one broad term would otherwise need to be discarded, the greedy cover can replace it with multiple narrower conjunctive terms (a split), preserving positive coverage through the OR of those terms.

**False-negative repair:** validation positives are added to the positive constraints. The cover adds one or more safe conjunctive terms to include them, while retaining coverage of all previously separable fit positives.

After repair, the script reports validation errors before and after repair, plus counts of uncovered positive patterns and conflicting patterns. A zero error count is guaranteed only when the labelled fit/validation interpretations are separable using the available attributes.

### Baseline without repair

`Baseline_NoRepair` is fitted only on the 90% fit partition. It uses the same DNF construction and greedy safe-term cover, but **does not call formula repair and does not incorporate validation labels**. The validation partition is used only to report a diagnostic accuracy/error count; the learned formula is applied unchanged to the official test file. This provides a direct comparison between fit-only Baseline and validation-repaired Baseline under the same split.

The greedy set-cover step is a heuristic: it produces a valid cover when one exists in the candidate language, but it does not guarantee the globally smallest DNF. Candidate generation enumerates attribute subsets, so its worst-case dependence on the number of attributes is exponential; the Monk experiments use six attributes.

## Important benchmark caveat

The official Monk files contain overlap between training and test IDs. This experiment **does not remove that overlap**, because doing so would change the official test benchmark. The script reports the number of overlapping IDs and exact duplicated labelled rows in `benchmark_overlap.csv`. Results should therefore be interpreted as performance on the official benchmark files, not as a fully independent unseen-example estimate.

## Outputs

Written to `ocf-vs-baseline/results/monk-experiment/`:

- `per_run_results.csv`: one row per model, Monk task, and repetition, including OCF's validation-selected threshold, fit-only Baseline validation diagnostics, repaired Baseline repair diagnostics, and official-test metrics.
- `summary_results.csv`: mean and sample standard deviation of each metric across the 10 repetitions, plus summed confusion counts.
- `benchmark_overlap.csv`: official train/test row counts and overlap diagnostics.

Metrics: accuracy, precision, recall, F1, MCC, positive-class Jaccard, Youden's J, and ROC AUC. ROC AUC uses `-OCF` for OCF and the number of satisfied DNF terms for Baseline.
