# MONK Experiment: Conceptual Rationale, Method, Implementation, and Reproduction

> **Scope.** This document is the single, self-contained guide to the MONK benchmark experiment in this repository. It explains the learning problem, the comparison methods, the exact protocol implemented by the code, the interpretation and limitations of the results, and how to reproduce the run. It describes the implementation currently present on the `main` branch; it does not claim that every method is a verbatim reproduction of a method in the cited papers.

## 1. Research question and conceptual overview

This experiment compares a symbolic, example-based classifier called **Baseline** with an **OCF (ordinal conditional function)** classifier and nine conventional machine-learning classifiers on the three MONK binary-classification problems.

The experiment is motivated by two complementary perspectives:

1. **Belief change and learning.** The paper *Iterated Belief Change as Learning* studies how belief-change operators can be used to construct learning procedures for binary classification. Its supplementary repository provides the original experimental code and data-processing workflow used in that work.
2. **Logical descriptions of classes.** A class can be described by a Boolean formula over feature-value propositions. Positive examples constrain the class formula to include their descriptions; counterexamples constrain it to exclude their descriptions. This makes symbolic rules and their refinement a natural point of comparison with distance-based and statistical classifiers.

The MONK benchmark is useful because it has only six categorical features, a finite and well-defined input domain, and three distinct target concepts. This makes it possible to inspect and learn explicit logical descriptions, while also comparing them against conventional classifiers.

The main experimental questions are:

- How does OCF compare with a symbolic Baseline formula learner?
- Does using validation examples to refine the Baseline hypothesis improve its test performance compared with leaving the initial hypothesis unchanged?
- How do these methods compare with standard classifiers on the same data partitions?

The experiment evaluates predictive performance. It is **not** a proof that one representation is theoretically superior, and it does not by itself establish that the learned formula is the unique intended concept.

## 2. MONK benchmark

The MONK's Problems were introduced by Thrun et al. as three artificial binary classification tasks over the same six-feature domain. The official UCI dataset contains 432 possible instances per problem, six categorical attributes, a binary class label, and an instance identifier.

The six features are represented in this repository as `X_1` through `X_6`. Their categorical cardinalities are:

| Feature | Number of values |
|---|---:|
| `X_1` | 3 |
| `X_2` | 3 |
| `X_3` | 2 |
| `X_4` | 3 |
| `X_5` | 4 |
| `X_6` | 2 |

The domain therefore contains (3\times3\times2\times3\times4\times2=432) possible feature vectors. The files are stored in `datasets/monks-N.train` and `datasets/monks-N.test`, for (N\in\{1,2,3\}).

The three tasks use different target concepts. In the conventional description of the benchmark:

- **MONK-1:** positive when the head shape equals the body shape, or the jacket colour is red.
- **MONK-2:** positive when exactly two of the six attributes take their first value.
- **MONK-3:** positive when the body shape is not octagonal or the jacket colour is not blue, or when the robot is holding a sword and the jacket colour is green. This task includes label noise in the standard benchmark.

The exact encodings of attribute values are those in the supplied UCI files; the experiment does not recode the feature values into semantic labels such as “red” or “round”.

## 3. Logical perspective: examples, counterexamples, and a class formula

Let (W) be the set of all feature interpretations. For a given task, let (p_1,\ldots,p_m) denote the descriptions of positive examples and (n_1,\ldots,n_{m'}) the descriptions of counterexamples. Let (v_C) be a propositional formula representing membership in the target class.

A concept that agrees with all labelled examples must satisfy

\[
\bigvee_{i=1}^{m}p_i\models v_C
\qquad\text{and}\qquad
v_C\models\bigwedge_{j=1}^{m'}\neg n_j.
\]

Equivalently,

\[
P\models v_C\models\neg N,
\qquad
P=\bigvee_i p_i,\quad N=\bigvee_j n_j.
\]

A consistent concept exists exactly when the positive and negative descriptions do not overlap:

\[
P\land N\models\bot.
\]

This is the logical consistency condition underlying the version-space view: the version space is the set of candidate class formulas satisfying all positive and negative constraints, and it is nonempty precisely when those constraints are jointly satisfiable.

This logical condition should not be confused with a pointwise ordering between two numerical distributions. In particular, if a positive-evidence function counts descriptions satisfied by an interpretation and a negative-evidence function counts counterexample descriptions satisfied by it, logical consistency gives disjoint supports, not a general inequality between the two functions.

The present MONK implementation uses this logical motivation to construct and refine a disjunctive normal form (DNF) classifier. It does **not** implement the generalized possibility-distribution proposal as a separate pair of possibility distributions. Nor does it use the alternative “count the negations of counterexamples” construction discussed during the theoretical analysis. Those are distinct mathematical constructions and should not be attributed to the current experiment code.

## 4. Methods compared

There are twelve evaluated models for each MONK task: OCF, two Baseline variants, and nine supplementary classifiers.

### 4.1 OCF

OCF is a distance-based score constructed from the fit partition. The implementation in `runs/run_monk_experiment.py` uses a categorical mismatch cost of 100 per feature:

\[
d(x,z)=100\sum_{r=1}^{6}\mathbf{1}[x_r\ne z_r].
\]

For a query (x) and a fit reference row (z) with label (y_z\in\{0,1\}), the per-feature contribution is:

- if (y_z=1), the contribution is the mismatch cost;
- if (y_z=0), the contribution is (100) minus the mismatch cost.

The contributions are summed over all six features and all fit reference rows. The implementation then subtracts a constant shift of (600) (six features times 100). The resulting score is used with the convention **smaller scores favour the positive class**.

The OCF threshold is selected using the validation partition by maximizing Youden's (J):

\[
J=\text{sensitivity}+\text{specificity}-1.
\]

The selected threshold is then applied to the official 432-row test file. The test labels are not used to choose this threshold.

This is the categorical distance implementation in the Monk runner; it should not be confused with every possible OCF distance or scaling choice in the wider source project.

### 4.2 Baseline formula learner

The Monk experiment implements `BaselineFormulaLearner` inside `runs/run_monk_experiment.py`. It learns a DNF formula: an OR of conjunctive terms, where each literal is an equality between one feature and one categorical value. For example, a term may be

\[
(X_1=1)\land(X_4=2),
\]

and a DNF hypothesis is an OR of such terms.

**Step A — aggregate duplicate feature vectors.** For each distinct feature vector (w) in the fit partition, the learner computes

\[
\Delta(w)=\#\{\text{positive rows equal to }w\}
          -\#\{\text{negative rows equal to }w\}.
\]

If (Delta(w)\geq0), the vector is put in the positive set; if (Delta(w)<0), it is put in the negative set. Thus a tie is assigned to the positive set. This aggregation means repeated occurrences are not treated as separate formula terms, although their labels contribute to the signed count.

**Step B — generate safe conjunctive terms.** For every positive vector, the learner generates terms from every nonempty subset of the six feature positions, using the values of that vector on the selected positions. A term is retained only if it matches no known negative vector. Its coverage is the set of positive vectors it matches.

**Step C — greedy set cover.** Starting with all positive vectors uncovered, the learner repeatedly chooses a safe term that covers the largest number of still-uncovered positives. Ties prefer terms with fewer literals, followed by a deterministic ordering. The selected terms are combined with OR. This is a greedy heuristic; it is not a guarantee of a globally minimum-size DNF.

Because the full six-feature conjunction is among the candidates, a positive vector can be covered unless that exact feature vector is also in the negative set. If the same vector has conflicting labels, no deterministic classifier based only on those six features can satisfy both labels. The implementation records such patterns as conflicts rather than silently claiming that they are separable.

The Baseline score of a query is the number of selected DNF terms it satisfies:

\[
s_{\mathrm{B}}(x)=\sum_{t\in T}\mathbf{1}[x\models t].
\]

The prediction is positive exactly when (s_{\mathrm{B}}(x)>0).

### 4.3 Original Baseline versus Baseline with Hypothesis Refinement

Both variants start from exactly the same fit partition in each repetition.

- **Original Baseline:** fits the DNF using only the 90% fit partition. It does not use validation labels and does not call `refine()`. Its formula is applied unchanged to the official test file.
- **Baseline with Hypothesis Refinement:** starts from the same fit partition, then incorporates the 10% validation examples and counterexamples as additional constraints and rebuilds the DNF.

The refinement procedure is implemented by adding validation feature vectors to the positive or negative sets and rerunning safe-term generation and greedy set cover. In logical terms, this can specialize or split broad disjuncts: a term that would cover a newly observed negative is no longer safe, while narrower terms can still cover positive examples. Validation positives are also added to the positive coverage requirement.

The code records validation errors before and after refinement, uncovered positive patterns, and conflicting patterns. A zero validation error is attainable only when the labelled patterns are separable in the available feature representation. Refinement uses validation labels by design; this is why the two Baseline variants must be reported separately.

### 4.4 Nine supplementary classifiers

The comparison also includes:

1. Logistic Regression (`max_iter=5000`)
2. Support Vector Machine (SVM)
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. Multilayer Perceptron / Neural Network (`max_iter=2000`)
7. K-Nearest Neighbours (KNN)
8. Gaussian Naive Bayes
9. Majority Classifier (scikit-learn `DummyClassifier(strategy="most_frequent")`)

The MONK features are categorical, so each supplementary classifier is wrapped in a pipeline with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. A fresh model pipeline is instantiated and fitted for every repetition. Random-state parameters are set to 42 where supported; KNN and Gaussian Naive Bayes use their scikit-learn defaults.

The Majority Classifier is a reference baseline only. It is not either of the symbolic Baseline variants.

## 5. Exact experimental protocol

The protocol below follows the current implementation, not an idealized train/validation/test design.

For each task MONK-1, MONK-2, and MONK-3:

1. Load the official `.train` and `.test` files from `datasets/`. The parser expects eight columns: class label, six feature values, and ID.
2. Keep the official test file unchanged. The runner checks that it contains exactly 432 rows.
3. Create ten stratified random partitions of the official training file using `StratifiedShuffleSplit(n_splits=10, test_size=0.10, random_state=42)`.
4. In each repetition, use 90% of the official training rows as the fit partition and 10% as the validation partition. Stratification preserves the class proportions as closely as possible.
5. Fit all nine supplementary classifiers on the fit partition and evaluate them on the same official test rows.
6. Compute OCF scores for the validation and test rows using only the fit partition. Select the OCF threshold by maximizing validation Youden's (J), then evaluate on the official test rows.
7. Fit Original Baseline on the fit partition and evaluate its unchanged formula on the official test rows.
8. Fit a separate Baseline instance on the same fit partition, refine it using the validation partition, and evaluate the refined formula on the official test rows.
9. Save per-repetition metrics. After all ten repetitions, calculate the mean and sample standard deviation ((ddof=1)) of each metric for each task/model. Sum the confusion-matrix counts across repetitions and generate plots.

The split seed, number of repetitions, validation fraction, mismatch cost, and shift are constants near the top of `runs/run_monk_experiment.py`:

| Setting | Value |
|---|---:|
| Repetitions | 10 |
| Fit fraction of official training file | 90% |
| Validation fraction of official training file | 10% |
| Validation split seed | 42 |
| Categorical mismatch cost | 100 |
| OCF shift | 600 |

**No official test labels are used** for fitting, refinement, or OCF threshold selection. They are used only to calculate the final reported test metrics.

### 5.1 Important benchmark limitation: the official test file contains the training instances

The official MONK test file has 432 rows, i.e. the full 432-instance domain. The official training file is a subset of that domain. Therefore, the official training IDs are also present in the 432-row test file. The overlap diagnostic already committed in `results/monk-experiment/benchmark_overlap.csv` reports:

| Task | Official train rows | Official test rows | Shared IDs | Identical labelled feature rows |
|---|---:|---:|---:|---:|
| MONK-1 | 124 | 432 | 124 | 124 |
| MONK-2 | 169 | 432 | 169 | 169 |
| MONK-3 | 122 | 432 | 122 | 116 |

This is an important property of the MONK benchmark files, not a bug in the split code. It means that the reported metrics are performance on the **complete official 432-instance domain**, which includes instances used for fitting in each repetition. They must not be described as estimates on a fully independent unseen test set. In MONK-3, six shared IDs do not have identical labelled feature rows, consistent with the task's noise.

The ten repetitions also reuse the same 432 test rows. Their scores are therefore repeated resampling measurements over different fit/validation partitions, not ten independent test sets. The reported standard deviations describe variability across these partitions; they are not confidence intervals from independent test samples.

## 6. Metrics and score conventions

For every task, model, and repetition, the runner calculates:

- **Accuracy**
- **Precision**
- **Recall**
- **F1**
- **Matthews correlation coefficient (MCC)**
- **Positive-class Jaccard index**
- **Youden's (J)**
- **ROC AUC**

It also stores (TP,TN,FP,FN).

For ROC AUC, the runner uses a ranking score whose larger values indicate stronger positive evidence:

- OCF: (-s_{\mathrm{OCF}}(x)), because smaller raw OCF scores favour positive predictions.
- Baseline: the number of satisfied selected DNF terms.
- Supplementary models: predicted probabilities when available, otherwise the decision function, otherwise the hard predictions (for the Majority Classifier).

The confusion matrices in the plots sum the counts across all ten repetitions. The spider plots put metrics on a common radial scale; MCC and Youden's (J), whose natural range is ([-1,1]), are mapped to ([0,1]) using ((v+1)/2). This rescaling is only for visualization; CSV values remain on their original scales.

## 7. Code map

All paths below are relative to the repository root.

| Path | Purpose |
|---|---|
| `ocf-vs-baseline/runs/run_monk_experiment.py` | Complete MONK experiment: loading, splitting, OCF, DNF Baseline, refinement, supplementary models, metrics, summaries, and plots. |
| `ocf-vs-baseline/datasets/monks-1.train`, `monks-2.train`, `monks-3.train` | Official training files. |
| `ocf-vs-baseline/datasets/monks-1.test`, `monks-2.test`, `monks-3.test` | Official 432-row test/domain files. |
| `ocf-vs-baseline/results/monk-experiment/` | Committed result tables, overlap diagnostic, and figures from a previous run. |
| `ocf-vs-baseline/requirements.txt` | Project dependencies. |
| `ocf-vs-baseline/MONK_EXPERIMENT.md` | Earlier, shorter MONK note; this document is the consolidated guide. |
| `ocf-vs-baseline/EXPERIMENT_WORKFLOW.md` | Separate workflow for the broader UCI/OCF supplementary experiment. |

**Important distinction:** `runs/runbaseline.py` is a separate implementation of `BaselineExact` and `BaselineLiteral` for the broader processed-UCI workflow. It is not the DNF `BaselineFormulaLearner` used by the MONK runner. Likewise, the broader `run_all_experiments.py` orchestrator includes the UCI workflow and is not needed to run MONK alone.

## 8. Installation and reproduction

### 8.1 Requirements

Use Python 3.9 or newer and a scikit-learn release that supports `OneHotEncoder(sparse_output=False)` (scikit-learn 1.2 or newer). The MONK runner directly imports NumPy, pandas, Matplotlib, and scikit-learn.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ocf-vs-baseline/requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The requirements file also contains `seaborn` and `ucimlrepo` for other project workflows; they are not directly imported by the standalone MONK runner.

### 8.2 Run the MONK experiment

Run from the repository root:

```bash
python ocf-vs-baseline/runs/run_monk_experiment.py
```

The script locates its data and output directories relative to its own file, so it does not require `KRML_DATASETS` and does not require downloading the MONK data separately.

To run only MONK-1, MONK-2, or MONK-3, edit the loop `for number in (1, 2, 3)` in `main()` and the corresponding loop in `create_plots()`; the committed default runs all three tasks.

### 8.3 Output files

The script writes to `ocf-vs-baseline/results/monk-experiment/`:

| File | Contents |
|---|---|
| `per_run_results.csv` | One row per task, repetition, and model, including metrics and available diagnostics. |
| `summary_results.csv` | Combined mean, sample standard deviation, and summed confusion counts for every task/model. |
| `monk_1_results.csv`, `monk_2_results.csv`, `monk_3_results.csv` | Task-specific summary tables. |
| `benchmark_overlap.csv` | Training/test ID and exact labelled-row overlap diagnostics. |
| `monk_N_classification_metrics.png` | Accuracy, precision, recall, and F1 comparison. |
| `monk_N_mcc_roc_auc.png` | MCC and ROC AUC comparison. |
| `monk_N_confusion_matrices.png` | Summed confusion matrices across ten repetitions. |
| `monk_N_spider_plot.png` | Eight-metric radar plot for all twelve models. |

Re-running the script overwrites these generated result files.

### 8.4 Reproduce the broader project workflow (optional)

The repository also contains a separate UCI workflow ported from the supplementary material of *Iterated Belief Change as Learning*. That workflow requires a processed UCI dataset root configured through `KRML_DATASETS`. It is not a prerequisite for the MONK benchmark.

See `EXPERIMENT_WORKFLOW.md` for its dataset preparation and stage commands. The one-command orchestrator `runs/run_all_experiments.py` runs both the UCI workflow and MONK unless stages are skipped; it should not be used when the goal is to reproduce only the MONK experiment.

## 9. Interpretation and reporting recommendations

When reporting the results:

1. Report all three MONK tasks separately because they encode different target concepts and MONK-3 includes noise.
2. Compare Original Baseline and Baseline with Hypothesis Refinement as distinct methods. The latter has access to validation labels and should not be presented as if it were trained only on the fit partition.
3. Report OCF's validation-selected thresholding procedure; do not describe the threshold as fixed in advance.
4. Include the train/test overlap caveat whenever discussing test metrics. The official 432-row test file contains the training instances, so these results are not a conventional independent holdout estimate.
5. Treat the ten-run standard deviation as sensitivity to the repeated stratified fit/validation partitions, not as a standard error or confidence interval based on independent test sets.
6. Describe the Baseline learner as a greedy DNF construction with a safe-term constraint. Do not claim it finds a globally minimal formula.
7. Distinguish the experiment's implemented DNF learner from the separate `BaselineExact`/`BaselineLiteral` implementation and from the generalized possibility-distribution discussion.

## 10. References and source repositories

### Papers and benchmark documentation

- S. Thrun, J. Bala, E. Bloedorn, I. Bratko, B. Cestnik, J. Cheng, K. De Jong, S. Dzeroski, S. E. Fahlman, D. Fisher, R. Hamann, K. Kaufman, S. Keller, I. Kononenko, J. Kreuziger, R. S. Michalski, T. Mitchell, P. Pachowicz, Y. Reich, H. Vafaie, W. Van de Velde, W. Wenzel, J. Wnek, and J. Zhang. **The MONK's Problems: A Performance Comparison of Different Learning Algorithms.** Carnegie Mellon University, School of Computer Science, Technical Report CMU-CS-91-197, December 1991. [Paper and benchmark description](https://www.cs.cmu.edu/~thrun/papers/thrun.MONK.html).
- J. Wnek. **MONK's Problems** [Dataset]. UCI Machine Learning Repository, 1993. DOI: [10.24432/C5R30R](https://doi.org/10.24432/C5R30R). [UCI dataset page](https://archive.ics.uci.edu/dataset/70/monk%2Bs%2Bproblems).
- N. Schwind, K. Inoue, S. Konieczny, and P. Marquis. **Iterated Belief Change as Learning.** Proceedings of IJCAI-25, 2025, pp. 4669–4677. DOI: [10.24963/ijcai.2025/520](https://doi.org/10.24963/ijcai.2025/520). [Official IJCAI paper page](https://ijcai.org/proceedings/2025/520).

### Repositories and included research material

- [Current experiment repository: `gmanes-1/baseline-vs-ocf`](https://github.com/gmanes-1/baseline-vs-ocf). This repository contains the MONK-specific implementation documented above.
- [Original supplementary repository: `nicolas-schwind/IteratedBeliefChangeML`](https://github.com/nicolas-schwind/IteratedBeliefChangeML). It accompanies Schwind et al. (IJCAI 2025) and provides the original supplementary code archive, proofs, and experimental material. The broader UCI workflow in this project is based on material from that source.
- [Bundled PDF: `BaselineML.pdf`](../BaselineML.pdf). This file is included at the root of the current repository. Its full bibliographic metadata should be taken from the PDF title page when citing it in a publication; the repository listing alone does not provide author, venue, or year, so those details are deliberately not guessed here.

---

**Reproducibility entry point:** from the repository root, install the dependencies and execute `python ocf-vs-baseline/runs/run_monk_experiment.py`. This reproduces the three-task MONK comparison and writes all CSV summaries and figures to `ocf-vs-baseline/results/monk-experiment/`.
