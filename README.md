# Baseline vs OCF: Experiments, Theory, and Reproduction

This repository compares ordinal conditional functions (OCFs), symbolic learning, and conventional machine-learning classifiers. **This root-level README is the single project guide**: it covers the conceptual foundations, the MONK benchmark, the broader UCI workflow, implementation details, and reproduction commands.

## 1. Research motivation

The project is motivated by *Iterated Belief Change as Learning* (Schwind et al., IJCAI 2025), which studies belief-change operators as a basis for binary classification. The central comparison is between:

- **OCF classifiers**, which use ordinal scores derived from labelled examples;
- **symbolic Baseline learners**, which construct logical descriptions of the positive class;
- **standard supervised classifiers**, used as reference models.

The MONK experiments make the symbolic representation especially transparent: six categorical attributes define a finite domain, and each target class can be described by a Boolean formula.

This repository contains two related but distinct experiment tracks:
1. the MONK benchmark comparison, including a fit-only DNF Baseline learner;
2. the broader supplementary-model/OCF workflow on processed UCI datasets, ported from the original supplementary repository.

The MONK runner is self-contained and does not require the broader UCI workflow.

## 2. Logical foundations: examples, counterexamples, and version spaces

Let W be the set of feature interpretations. Write p_i for the descriptions of positive examples, n_j for descriptions of negative examples (counterexamples), and v_C for a formula describing membership in the positive class.

A formula consistent with all labelled descriptions must satisfy:

\[
\bigvee_i p_i \models v_C
\qquad\text{and}\qquad
v_C \models \bigwedge_j \neg n_j.
\]

With P = OR_i p_i and N = OR_j n_j, this is:

\[
P \models v_C \models \neg N.
\]

A consistent class formula exists exactly when positive and negative descriptions do not overlap:

\[
P \land N \models \bot.
\]

The version space is the set of formulas satisfying these constraints; it is nonempty precisely when the constraints are jointly satisfiable.

This logical consistency condition should not be conflated with a pointwise ordering between numerical evidence distributions. If one function counts positive descriptions satisfied by an interpretation and another counts negative descriptions satisfied by it, consistency implies disjoint supports, not in general an inequality between the two functions.

**Implementation boundary:** the MONK runner uses this logical motivation to build a DNF classifier from the fitting partition. It does not implement the generalized pair of possibility distributions, nor the alternative construction that counts negations of counterexamples. Those are separate theoretical proposals.

## 3. MONK benchmark

The MONK's Problems were introduced by Thrun et al. as three artificial binary-classification tasks over six categorical attributes. The official UCI data describe a domain of 432 possible feature vectors.

| Feature | Cardinality |
|---|---:|
| X_1 | 3 |
| X_2 | 3 |
| X_3 | 2 |
| X_4 | 3 |
| X_5 | 4 |
| X_6 | 2 |

Thus the domain size is 3 × 3 × 2 × 3 × 4 × 2 = 432. The supplied data files are under `ocf-vs-baseline/datasets/`.

The conventional target concepts are:

- **MONK-1:** positive if head shape equals body shape OR jacket colour is red.
- **MONK-2:** positive if exactly two of the six attributes take their first value.
- **MONK-3:** positive if NOT (body-shape = octagon OR jacket-colour = blue) OR (holding = sword AND jacket-colour = green). The standard MONK-3 task includes label noise.

The files encode values categorically; the experiment uses those encoded values directly.

## 4. Methods in the MONK experiment

The MONK runner evaluates eleven models per task: OCF, one DNF Baseline trained on the complete official training file, and nine supplementary classifiers.

### 4.1 OCF

For query x and reference row z, the runner uses categorical mismatch cost:

\[
d(x,z)=100\sum_{r=1}^{6}\mathbf{1}[x_r\ne z_r].
\]

For each fit reference row, a positive label contributes the mismatch cost, while a negative label contributes 100 minus the mismatch cost. Contributions are summed over features and fit rows, then a constant shift of 600 is subtracted. **Smaller raw scores favour the positive class.**

The classification threshold is selected on the validation partition by maximizing Youden's J = sensitivity + specificity − 1, then applied to the official test file. This describes the implementation in `runs/run_monk_experiment.py`, not every possible OCF formulation.

### 4.2 Symbolic Baseline DNF learner

The single implementation, `BaselineFormulaLearner`, is defined in `ocf-vs-baseline/runs/baseline.py` and imported by the MONK experiment runner. It learns a disjunctive normal form: an OR of conjunctions of feature-value equality literals.

1. **Aggregate repeated vectors.** For each distinct vector w, compute Δ(w) = number of positive rows at w minus number of negative rows at w. Assign Δ(w) ≥ 0 to the positive set (ties are positive); otherwise assign it to the negative set.
2. **Generate safe terms.** For each positive vector, generate conjunctions from every nonempty subset of its six feature positions. Retain a term only if it matches no known negative vector.
3. **Greedy set cover.** Repeatedly select the safe term covering the most uncovered positive vectors. Ties prefer fewer literals and then deterministic ordering. OR the selected terms.

This is a greedy heuristic, not a guarantee of a globally minimum DNF. The score is the number of selected terms satisfied by x; prediction is positive when that score is greater than zero. Conflicting labels for an identical feature vector cannot be perfectly separated by any deterministic classifier using only those features.

### 4.3 Baseline fitting protocol

Only one DNF Baseline is retained in the MONK benchmark. It is fitted once on **all rows in the official training file** (100% of the training data), with no train/validation split for the Baseline, and then applied unchanged to the official test file.

### 4.4 Supplementary classifiers

The nine additional classifiers are Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting, MLP/Neural Network, KNN, Gaussian Naive Bayes, and a majority-class DummyClassifier. They use one-hot encoding via `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. Fresh pipelines are fitted for each repetition. Random-state parameters are set to 42 where supported; KNN and GaussianNB use their scikit-learn defaults.

The majority classifier is a reference model, not one of the symbolic Baseline variants.

## 5. MONK experimental protocol

For each of MONK-1, MONK-2, and MONK-3, the runner:

1. Loads the official training and test files (eight columns: label, six features, ID).
2. Checks that the official test file has 432 rows.
3. Fits the DNF Baseline once on **100% of the official training rows**, with no validation split for this model, and evaluates it on the official test file.
4. Creates ten stratified 90/10 fit/validation partitions for OCF and the nine supplementary classifiers, using `StratifiedShuffleSplit(n_splits=10, test_size=0.10, random_state=42)`.
5. Fits the nine supplementary classifiers on each fit partition.
6. Computes OCF validation/test scores from each fit partition, chooses the threshold on validation, and evaluates on test.
7. Saves the single full-training Baseline result and the per-repetition OCF/supplementary-model results, then computes summaries and creates plots.

| Setting | Value |
|---|---:|
| Repetitions | 10 |
| Baseline training data | 100% of official training file; no split |
| OCF / supplementary-model fit-validation split | 90% / 10% |
| Split seed | 42 |
| Categorical mismatch cost | 100 |
| OCF shift | 600 |

Test labels are used only for final metrics, not fitting or OCF threshold selection.

### Important limitation: official MONK test data overlap with training data

The official 432-row test file represents the full domain and contains the official training IDs. The committed overlap diagnostic reports:

| Task | Train rows | Test rows | Shared IDs | Identical labelled feature rows |
|---|---:|---:|---:|---:|
| MONK-1 | 124 | 432 | 124 | 124 |
| MONK-2 | 169 | 432 | 169 | 169 |
| MONK-3 | 122 | 432 | 122 | 116 |

Consequently, metrics are performance over the complete official 432-instance domain, which includes fit instances in each repetition. They are **not** estimates from a fully independent unseen test set. For MONK-3, six shared IDs have non-identical labelled feature rows, consistent with the task's noise.

The same 432 test rows are reused across all ten repetitions. Standard deviations therefore describe variability across fit/validation resamplings, not independent-test confidence intervals.

## 6. Metrics and score conventions

The runner records accuracy, precision, recall, F1, Matthews correlation coefficient (MCC), positive-class Jaccard, Youden's J, ROC AUC, and TP/TN/FP/FN.

For ROC AUC, larger scores indicate stronger positive evidence:
- OCF: negative raw OCF score, because smaller raw values favour positive;
- Baseline: number of selected DNF terms satisfied;
- supplementary classifiers: predicted probabilities where available, otherwise decision function, otherwise hard predictions.

Confusion matrices sum counts over ten repetitions. Spider plots rescale MCC and Youden's J from [−1, 1] to [0, 1] for display only; CSV values remain on their original scales.

## 7. Repository map

Paths are relative to the repository root.

| Path | Purpose |
|---|---|
| `ocf-vs-baseline/runs/baseline.py` | The single implementation of the symbolic DNF Baseline learner. |\n| `ocf-vs-baseline/runs/run_monk_experiment.py` | MONK experiment orchestration, OCF, reference models, metrics, and plots. |
| `ocf-vs-baseline/datasets/monks-1.train`, `monks-2.train`, `monks-3.train` | Official MONK training files. |
| `ocf-vs-baseline/datasets/monks-1.test`, `monks-2.test`, `monks-3.test` | Official 432-row test/domain files. |
| `ocf-vs-baseline/results/monk-experiment/` | Result tables, overlap diagnostic, and figures. |
| `ocf-vs-baseline/requirements.txt` | Python dependencies. |
| `ocf-vs-baseline/runs/runbaseline.py` | Exact-only `BaselineExact` implementation for the broader workflow, not the MONK DNF learner. |
| `ocf-vs-baseline/runs/run_all_experiments.py` | Orchestrator for the broader UCI stages and MONK. |

## 8. Installation and run instructions

Use Python 3.9 or newer and scikit-learn 1.2 or newer (for `sparse_output=False`). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ocf-vs-baseline/requirements.txt
python ocf-vs-baseline/runs/run_monk_experiment.py
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

The MONK runner locates datasets and outputs relative to its own file. It does not require `KRML_DATASETS` or a separate MONK download.

The runner writes to `ocf-vs-baseline/results/monk-experiment/`:

- `per_run_results.csv`: metrics and diagnostics for each task/repetition/model;
- `summary_results.csv`: means, sample standard deviations, and summed confusion counts;
- `monk_1_results.csv`, `monk_2_results.csv`, `monk_3_results.csv`: task-specific summaries;
- `benchmark_overlap.csv`: training/test overlap diagnostics;
- `monk_N_classification_metrics.png`, `monk_N_mcc_roc_auc.png`, `monk_N_confusion_matrices.png`, `monk_N_spider_plot.png`: figures.

Rerunning overwrites generated outputs.

## 9. Broader UCI / supplementary-model workflow (optional)

The repository also ports the original supplementary-model comparison on processed UCI datasets from `IteratedBeliefChangeML/python-code.zip`. This is a separate workflow; it is not needed for MONK-only reproduction.

Set `KRML_DATASETS` to the absolute path of the processed UCI dataset root. It must contain the per-dataset folders/files expected by `to_file.py`. The original source README describes 58 selected UCI datasets.

From the repository root:

```bash
export KRML_DATASETS="/absolute/path/to/processed/dataset"
python ocf-vs-baseline/process_external_datasets/retrieve_uci_datasets.py
cd ocf-vs-baseline
python runs/createsamplings.py
python runs/runexistingmodels.py
python runs/createOCFsfortrainingsets.py
python runs/runtests.py
python runs/construct_plots.py
```

The supplementary models are Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting, Neural Network, KNN, Naive Bayes, and the source project's Majority Classifier (`Maj`). The plotter creates radar plots by default; add `--all-plots` for the OCF-vs-model scatter plots and metric box plots.

The one-command orchestrator is:

```bash
python ocf-vs-baseline/runs/run_all_experiments.py
```

It runs the UCI workflow and MONK, then plotting, and stops at the first failed stage. It accepts `--skip-sampling`, `--skip-existing-models`, `--skip-ocf`, `--skip-monk`, `--skip-plots`, and `--all-plots`. Note that the broader plotter expects complete UCI results; for a MONK-only run use the dedicated MONK command above.

UCI outputs are stored in the dataset folders using the original `*-all-results.csv` convention; comparison figures are written under `all-results-plots/` in `KRML_DATASETS`.

## 10. Reporting recommendations

1. Report the three MONK tasks separately; they have different target concepts and MONK-3 includes noise.
2. Describe the MONK DNF Baseline as trained once on the complete official training file, without a validation split.
3. Explain that the OCF threshold is selected on validation data.
4. Always disclose the official train/test overlap when discussing MONK test metrics.
5. Do not interpret the ten-run standard deviation as an independent-test confidence interval.
6. Describe the DNF learner as a greedy safe-term construction, not an exact minimum-formula solver.
7. Keep the implemented greedy DNF learner distinct from the generalized possibility-distribution theory; the repository contains only this one Baseline implementation.

## 11. References and source repositories

### Papers and benchmark

- S. Thrun et al. **The MONK's Problems: A Performance Comparison of Different Learning Algorithms.** Carnegie Mellon University, Technical Report CMU-CS-91-197, December 1991. [Paper and benchmark description](https://www.cs.cmu.edu/~thrun/papers/thrun.MONK.html).
- J. Wnek. **MONK's Problems** [Dataset]. UCI Machine Learning Repository, 1993. DOI: [10.24432/C5R30R](https://doi.org/10.24432/C5R30R). [UCI dataset page](https://archive.ics.uci.edu/dataset/70/monk%2Bs%2Bproblems).
- N. Schwind, K. Inoue, S. Konieczny, and P. Marquis. **Iterated Belief Change as Learning.** Proceedings of IJCAI-25, 2025, pp. 4669–4677. DOI: [10.24963/ijcai.2025/520](https://doi.org/10.24963/ijcai.2025/520). [Official IJCAI paper page](https://ijcai.org/proceedings/2025/520).

### Code and research material

- [This project: gmanes-1/baseline-vs-ocf](https://github.com/gmanes-1/baseline-vs-ocf).
- [Original supplementary repository: nicolas-schwind/IteratedBeliefChangeML](https://github.com/nicolas-schwind/IteratedBeliefChangeML), accompanying Schwind et al. and containing the supplementary code and experimental material.
- [Bundled PDF: BaselineML.pdf](BaselineML.pdf). For formal citation, use the bibliographic details on the PDF title page; author, venue, and year are not inferred from the filename.

---

**MONK reproduction entry point:** install the requirements and run `python ocf-vs-baseline/runs/run_monk_experiment.py` from the repository root.
