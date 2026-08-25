"""
analyze_results.py

Consumes results/per_run_results.csv and results/training_logs/*.json to
produce exactly the four required figures (Section 15) and the final
results table (Section 14), plus the correlation analysis (Experiment 4).

Outputs written to ../figures/:
    figure1_training_loss.png
    figure2_gradient_norm.png
    figure3_generalization_gap.png
    figure4_lambda_max_vs_gap.png   <- central research figure

And to ../results/:
    final_results_table.csv
    correlation_analysis.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis import summarize_by_optimizer, format_mean_std_table, geometry_vs_generalization

OPTIMIZER_COLORS = {"sgd": "#1f77b4", "adam": "#ff7f0e", "lbfgs": "#2ca02c"}
OPTIMIZER_LABELS = {"sgd": "SGD", "adam": "Adam", "lbfgs": "L-BFGS"}


def load_training_logs(logs_dir: Path, optimizers, seeds):
    logs = {}
    for opt in optimizers:
        logs[opt] = []
        for seed in seeds:
            path = logs_dir / f"{opt}_seed{seed}.json"
            if path.exists():
                with open(path) as f:
                    logs[opt].append(json.load(f))
    return logs


def figure1_training_loss(logs, out_path: Path):
    """
    Figure 1: Training loss vs. iteration, one line per seed, colored by optimizer.

    Two-panel layout: SGD and Adam share a left panel (both logged once per
    full-batch step, 500 steps), and L-BFGS gets its own right panel (logged
    once per outer optimizer.step() call, 20 calls, each internally running
    up to 20 quasi-Newton iterations). These two units are not directly
    comparable, so plotting them on one shared x-axis compresses L-BFGS's
    entire trajectory into a few pixels near x=0 and hides its dynamics.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for opt in ["sgd", "adam"]:
        for i, run in enumerate(logs.get(opt, [])):
            label = OPTIMIZER_LABELS[opt] if i == 0 else None
            ax1.plot(run["train_loss"], color=OPTIMIZER_COLORS[opt], alpha=0.6, label=label)
    ax1.set_xlabel("Training step (full-batch update)")
    ax1.set_ylabel("Training loss (BCE)")
    ax1.set_title("SGD / Adam")
    ax1.set_yscale("log")
    ax1.legend()

    for i, run in enumerate(logs.get("lbfgs", [])):
        label = OPTIMIZER_LABELS["lbfgs"] if i == 0 else None
        ax2.plot(run["train_loss"], color=OPTIMIZER_COLORS["lbfgs"], alpha=0.6,
                  marker="o", markersize=3, label=label)
    ax2.set_xlabel("Outer step (each up to 20 internal L-BFGS iterations)")
    ax2.set_ylabel("Training loss (BCE)")
    ax2.set_title("L-BFGS")
    ax2.set_yscale("log")
    ax2.legend()

    fig.suptitle("Figure 1: Training Loss vs. Iteration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure2_gradient_norm(logs, out_path: Path):
    """
    Figure 2: Gradient norm vs. iteration, one line per seed, colored by optimizer.

    Same two-panel rationale as figure1_training_loss: L-BFGS's outer step
    unit is not comparable to SGD/Adam's, so it gets its own panel.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for opt in ["sgd", "adam"]:
        for i, run in enumerate(logs.get(opt, [])):
            label = OPTIMIZER_LABELS[opt] if i == 0 else None
            ax1.plot(run["grad_norm"], color=OPTIMIZER_COLORS[opt], alpha=0.6, label=label)
    ax1.set_xlabel("Training step (full-batch update)")
    ax1.set_ylabel(r"Gradient norm $\|\nabla_\theta L\|_2$")
    ax1.set_title("SGD / Adam")
    ax1.set_yscale("log")
    ax1.legend()

    for i, run in enumerate(logs.get("lbfgs", [])):
        label = OPTIMIZER_LABELS["lbfgs"] if i == 0 else None
        ax2.plot(run["grad_norm"], color=OPTIMIZER_COLORS["lbfgs"], alpha=0.6,
                  marker="o", markersize=3, label=label)
    ax2.set_xlabel("Outer step (each up to 20 internal L-BFGS iterations)")
    ax2.set_ylabel(r"Gradient norm $\|\nabla_\theta L\|_2$")
    ax2.set_title("L-BFGS")
    ax2.set_yscale("log")
    ax2.legend()

    fig.suptitle("Figure 2: Gradient Norm vs. Iteration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure3_generalization_gap(df: pd.DataFrame, out_path: Path):
    """Figure 3: Generalization gap by optimizer (mean +/- std bar chart with individual points)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    optimizers = ["sgd", "adam", "lbfgs"]
    means = [df[df.optimizer == o]["gen_gap"].mean() for o in optimizers]
    stds = [df[df.optimizer == o]["gen_gap"].std() for o in optimizers]

    ax.bar(
        [OPTIMIZER_LABELS[o] for o in optimizers], means, yerr=stds, capsize=6,
        color=[OPTIMIZER_COLORS[o] for o in optimizers], alpha=0.7,
    )
    for o in optimizers:
        sub = df[df.optimizer == o]
        x_pos = optimizers.index(o)
        ax.scatter([x_pos] * len(sub), sub["gen_gap"], color="black", zorder=3, s=20)

    ax.set_ylabel(r"Generalization gap $G = L_{test} - L_{train}$")
    ax.set_title("Figure 3: Generalization Gap by Optimizer")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure4_lambda_vs_gap(df: pd.DataFrame, corr_results: dict, out_path: Path):
    """Figure 4 (central research figure): lambda_max(H) vs. generalization gap."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for opt in ["sgd", "adam", "lbfgs"]:
        sub = df[df.optimizer == opt]
        ax.scatter(
            sub["lambda_max"], sub["gen_gap"],
            color=OPTIMIZER_COLORS[opt], label=OPTIMIZER_LABELS[opt], s=60, alpha=0.85,
        )
    ax.set_xlabel(r"$\lambda_{max}(H)$")
    ax.set_ylabel(r"Generalization gap $G = L_{test} - L_{train}$")

    pooled = corr_results["pooled"]
    ax.set_title(
        "Figure 4: Largest Hessian Eigenvalue vs. Generalization Gap\n"
        f"(pooled Spearman r={pooled['spearman_r']:.2f}, p={pooled['spearman_p']:.3f}; "
        f"Pearson r={pooled['pearson_r']:.2f}, p={pooled['pearson_p']:.3f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(results_dir: Path, figures_dir: Path, seeds):
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_dir / "per_run_results.csv")

    logs = load_training_logs(results_dir / "training_logs", ["sgd", "adam", "lbfgs"], seeds)

    figure1_training_loss(logs, figures_dir / "figure1_training_loss.png")
    figure2_gradient_norm(logs, figures_dir / "figure2_gradient_norm.png")
    figure3_generalization_gap(df, figures_dir / "figure3_generalization_gap.png")

    corr_results = geometry_vs_generalization(df, gap_col="gen_gap")
    corr_results_acc = geometry_vs_generalization(df, gap_col="acc_gen_gap")
    figure4_lambda_vs_gap(df, corr_results, figures_dir / "figure4_lambda_max_vs_gap.png")

    with open(results_dir / "correlation_analysis.json", "w") as f:
        json.dump(
            {"loss_based": corr_results, "accuracy_based": corr_results_acc},
            f, indent=2, default=float,
        )

    metrics = ["test_loss", "test_f1", "test_auroc", "gen_gap", "acc_gen_gap", "lambda_max"]
    summary = summarize_by_optimizer(df, metrics)
    table = format_mean_std_table(summary, metrics)
    table.to_csv(results_dir / "final_results_table.csv")

    print("Final Results Table (mean ± std):")
    print(table.to_string())
    print("\nCorrelation analysis, loss-based gap (pooled):")
    print(json.dumps(corr_results["pooled"], indent=2))
    print("\nCorrelation analysis, accuracy-based gap (pooled):")
    print(json.dumps(corr_results_acc["pooled"], indent=2))
    print(f"\nFigures written to {figures_dir}")
    print(f"Tables written to {results_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="../results")
    parser.add_argument("--figures-dir", type=str, default="../figures")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    main(Path(args.results_dir), Path(args.figures_dir), args.seeds)