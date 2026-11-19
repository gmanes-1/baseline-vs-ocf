"""Orchestrate the optional UCI workflow and the standalone MONK benchmark.

This script coordinates stages; the actual MONK implementation lives in
run_monk_experiment.py so it can also be executed independently.

Run from any directory:
    python /path/to/ocf-vs-baseline/runs/run_all_experiments.py

The UCI workflow requires KRML_DATASETS to point to the processed dataset
root. The bundled Monk experiment does not require that environment variable.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_stage(label, filename, extra_args=None):
    command = [sys.executable, str(ROOT / "runs" / filename)]
    if extra_args:
        command.extend(extra_args)
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-existing-models", action="store_true")
    parser.add_argument("--skip-ocf", action="store_true")
    parser.add_argument("--skip-monk", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--all-plots", action="store_true",
                        help="Also create scatter and box plots.")
    args = parser.parse_args()

    needs_uci = not (args.skip_sampling and args.skip_existing_models and args.skip_ocf)
    if needs_uci and not os.environ.get("KRML_DATASETS"):
        parser.error("Set KRML_DATASETS to the processed UCI dataset root first.")

    # Preserve the stage order in the original python-code/README.txt.
    if not args.skip_sampling:
        run_stage("Create train/test samplings", "createsamplings.py")
    if not args.skip_existing_models:
        run_stage("Run supplementary classifiers", "runexistingmodels.py")
    if not args.skip_ocf:
        run_stage("Create OCFs for training samplings", "createOCFsfortrainingsets.py")
        run_stage("Test OCF", "runtests.py")
    if not args.skip_monk:
        run_stage("Run Monk 1/2/3 benchmark", "run_monk_experiment.py")
    if not args.skip_plots:
        run_stage("Construct comparison plots", "construct_plots.py",
                  ["--all-plots"] if args.all_plots else None)
    print("\nAll requested stages completed.", flush=True)


if __name__ == "__main__":
    main()
