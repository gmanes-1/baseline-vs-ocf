# Experiment workflow

This project has two experiment tracks: the original supplementary-model
comparison on processed UCI datasets, and the bundled Monk benchmark.

## Original supplementary models

The model suite and evaluation procedure are ported from
`IteratedBeliefChangeML/python-code.zip`. The original source README specifies
this sequence:

1. Retrieve and preprocess the selected UCI datasets.
2. Generate 10 stratified 90/10 train/test samplings per dataset.
3. Train and test the supplementary classifiers.
4. Construct OCF representations for the training samplings.
5. Test OCF on the corresponding held-out samplings.
6. Generate the comparison figures.

The supplementary classifiers are Logistic Regression, SVM, Decision Tree,
Random Forest, Gradient Boosting, Neural Network, KNN, Naive Bayes, and the
source project's Majority Classifier benchmark (`Maj`). This last classifier
is retained as one of the original comparison models; it is not a replacement
for either Baseline variant in the Monk experiment. The scikit-learn estimator
choices and explicit `max_iter=5000` settings follow the source runner.

### Dataset setup

Set `KRML_DATASETS` to the absolute path of the processed UCI dataset root.
It must contain the per-dataset folders and files expected by `to_file.py`.
The source README describes 58 selected UCI datasets. Dataset retrieval and
preprocessing must be completed before running the experiment stages.

```bash
export KRML_DATASETS="/absolute/path/to/processed/dataset"
```

### Run the original stages

From `ocf-vs-baseline/`:

```bash
python runs/createsamplings.py
python runs/runexistingmodels.py
python runs/createOCFsfortrainingsets.py
python runs/runtests.py
python runs/construct_plots.py
```

The plotter creates spider/radar plots by default. Add `--all-plots` to also
create the original OCF-vs-other-model scatter plots and metric box plots.

## Monk benchmark

The Monk runner uses the official `monks-1/2/3.train` and `.test` files
bundled under `datasets/`. Run it with:

```bash
python runs/run_monk_experiment.py
```

The comparison includes OCF, Original Baseline, and Baseline with Hypothesis
Refinement. The experiment protocol, matched 90/10 splits, and official
train/test overlap caveat are described in `MONK_EXPERIMENT.md`.

## One-command workflow

```bash
python runs/run_all_experiments.py
```

The orchestrator runs the original UCI sampling, supplementary-model and OCF
stages, then the Monk benchmark, then plotting. It stops on the first failed
stage. If samplings already exist, pass `--skip-sampling`. To run only Monk
and the plotter (which will still need complete UCI results for its plots):

```bash
python runs/run_all_experiments.py --skip-sampling --skip-existing-models --skip-ocf --skip-plots
```

Use `--skip-monk` to omit Monk, `--skip-plots` to omit plotting, and
`--all-plots` to include scatter and box plots.

## Output locations

- Original per-dataset result CSVs: each dataset folder, named using the
  original `*-all-results.csv` convention.
- Original comparison figures: `all-results-plots/` under `KRML_DATASETS`.
- Monk CSV results: `results/monk-experiment/`.

## Reproducibility

The Monk runner uses its documented random seed. The original supplementary
model workflow retains the source's sampling and estimator behavior; the
source runner does not set a common random seed for all scikit-learn
estimators or for `train_test_split`.
