"""
run_experiments.py

Runs the full experimental grid:

    3 optimizers (sgd, adam, lbfgs) x 5 seeds (0-4) = 15 training runs

For each run:
  1. Train the fixed MLP on the fixed WDBC split (Experiment 1 logs saved).
  2. Evaluate on train/val/test -> loss, accuracy, F1, AUROC (Experiment 2).
  3. Estimate lambda_max(H) via Hessian-vector products + power iteration,
     evaluated on the training set (Experiment 3).
  4. (Optional) local perturbation sharpness (Section 17 extension).

Outputs, written to ../results/:
  - per_run_results.csv       one row per (optimizer, seed) with all scalar
                               summary quantities needed for Experiments 2-4
  - training_logs/<opt>_seed<seed>.json   full per-step logs for Figures 1-2

Usage:
    cd experiments
    python run_experiments.py --seeds 0 1 2 3 4
    python run_experiments.py --seeds 0 1 2   # faster dev run, 3 seeds
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import load_wdbc
from train import train_one_run
from metrics import evaluate, generalization_gap
from hessian import power_iteration_lambda_max, perturbation_sharpness


OPTIMIZERS = ["sgd", "adam", "lbfgs"]


def run_all(seeds, results_dir: Path, run_perturbation: bool = True):
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = results_dir / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    data = load_wdbc()  # fixed split, computed once, reused for every run
    rows = []

    total = len(OPTIMIZERS) * len(seeds)
    run_idx = 0

    for optimizer_name in OPTIMIZERS:
        for seed in seeds:
            run_idx += 1
            print(f"[{run_idx}/{total}] Training {optimizer_name} seed={seed} ...")

            model, log = train_one_run(optimizer_name, seed, data)

            # Save full per-step log for Figures 1 & 2
            log_path = logs_dir / f"{optimizer_name}_seed{seed}.json"
            with open(log_path, "w") as f:
                json.dump(
                    {
                        "optimizer": optimizer_name,
                        "seed": seed,
                        "train_loss": log.train_loss,
                        "val_loss": log.val_loss,
                        "grad_norm": log.grad_norm,
                        "update_norm": log.update_norm,
                    },
                    f,
                )

            # Experiment 2: generalization metrics
            train_metrics = evaluate(model, data.X_train, data.y_train)
            val_metrics = evaluate(model, data.X_val, data.y_val)
            test_metrics = evaluate(model, data.X_test, data.y_test)
            gap = generalization_gap(train_metrics["loss"], test_metrics["loss"])

            # Experiment 3: Hessian geometry (evaluated on training set,
            # consistent with lambda_max being a property of the training
            # loss surface at the found solution theta*)
            hessian_result = power_iteration_lambda_max(
                model, data.X_train, data.y_train, num_iters=100, tol=1e-6, seed=seed
            )

            row = {
                "optimizer": optimizer_name,
                "seed": seed,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "test_loss": test_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "test_f1": test_metrics["f1"],
                "train_auroc": train_metrics["auroc"],
                "val_auroc": val_metrics["auroc"],
                "test_auroc": test_metrics["auroc"],
                "gen_gap": gap,
                "lambda_max": hessian_result["lambda_max"],
                "lambda_max_n_iters": hessian_result["n_iters"],
                "lambda_max_converged": hessian_result["converged"],
                "n_train_steps": len(log.train_loss),
            }

            # Optional extension: perturbation-based sharpness
            if run_perturbation:
                sharp = perturbation_sharpness(
                    model, data.X_train, data.y_train,
                    epsilon=1e-2, n_directions=10, seed=seed,
                )
                row.update({
                    "mean_sharpness": sharp["mean_sharpness"],
                    "max_sharpness": sharp["max_sharpness"],
                    "std_sharpness": sharp["std_sharpness"],
                })

            rows.append(row)
            print(
                f"    train_loss={row['train_loss']:.4f} "
                f"test_loss={row['test_loss']:.4f} "
                f"gap={row['gen_gap']:.4f} "
                f"lambda_max={row['lambda_max']:.4f} "
                f"(converged={row['lambda_max_converged']}, "
                f"n_iters={row['lambda_max_n_iters']})"
            )

    df = pd.DataFrame(rows)
    csv_path = results_dir / "per_run_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Random seeds to run. Use 3 seeds (e.g. 0 1 2) for a fast dev run; "
             "the final experiment should use all 5 (0-4) per the plan.",
    )
    parser.add_argument(
        "--no-perturbation", action="store_true",
        help="Skip the optional perturbation-sharpness extension.",
    )
    parser.add_argument(
        "--results-dir", type=str, default="../results",
        help="Directory to write results into.",
    )
    args = parser.parse_args()

    run_all(
        seeds=args.seeds,
        results_dir=Path(args.results_dir),
        run_perturbation=not args.no_perturbation,
    )