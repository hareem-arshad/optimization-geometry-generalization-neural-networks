"""
analysis.py

Statistical analysis tying together Experiment 2 (generalization) and
Experiment 3 (Hessian geometry) into Experiment 4 (geometry vs.
generalization).

Per the plan: the goal is NOT to manufacture statistical significance.
It is to honestly characterize the relationship between lambda_max(H) and
the generalization gap G as strong / weak / inconsistent / absent.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def summarize_by_optimizer(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """
    Collapse a per-run results dataframe into mean +/- std per optimizer,
    for the given list of metric column names. Produces the Final Results
    Table (Section 14 of the plan).
    """
    rows = []
    for opt_name, group in df.groupby("optimizer"):
        row = {"optimizer": opt_name, "n_runs": len(group)}
        for m in metrics:
            row[f"{m}_mean"] = group[m].mean()
            row[f"{m}_std"] = group[m].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("optimizer")


def format_mean_std_table(summary_df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """
    Render a summary_df (from summarize_by_optimizer) as "mean ± std"
    display strings, matching the reporting convention in the plan.
    """
    out = pd.DataFrame(index=summary_df.index)
    for m in metrics:
        out[m] = summary_df.apply(
            lambda r: f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}", axis=1
        )
    return out


def geometry_vs_generalization(df: pd.DataFrame, gap_col: str = "gen_gap") -> dict:
    """
    Correlate lambda_max(H) against a generalization gap column across ALL
    runs (pooled across optimizers and seeds), as specified for Figure 4.

    gap_col defaults to "gen_gap" (loss-based, L_test - L_train), the
    plan's primary measure. Pass gap_col="acc_gen_gap" to instead check the
    accuracy-based gap (accuracy_train - accuracy_test) as a robustness
    check -- useful when the loss-based gap may be dominated by prediction
    confidence/calibration rather than correctness (see metrics.py).

    Spearman is reported as primary (robust to nonlinearity / outliers,
    appropriate given we are not assuming a linear relationship a priori).
    Pearson is reported as secondary.

    Also reports correlations computed WITHIN each optimizer, since a
    pooled correlation can be driven entirely by between-optimizer
    differences rather than a genuine within-optimizer geometry ->
    generalization relationship. Both views are needed for honest
    interpretation.
    """
    results = {}

    lam = df["lambda_max"].values
    gap = df[gap_col].values

    spearman_r, spearman_p = spearmanr(lam, gap)
    pearson_r, pearson_p = pearsonr(lam, gap)

    results["gap_col"] = gap_col
    results["pooled"] = {
        "n": len(df),
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
    }

    within = {}
    for opt_name, group in df.groupby("optimizer"):
        if len(group) < 3:
            within[opt_name] = {"n": len(group), "note": "too few runs for correlation"}
            continue
        # Guard against degenerate (constant) columns within a small group,
        # e.g. accuracy-based gap can coincide across a few seeds on a small
        # test set -- correlation is undefined there, not a bug.
        if group["lambda_max"].nunique() < 2 or group[gap_col].nunique() < 2:
            within[opt_name] = {
                "n": len(group),
                "note": "constant lambda_max or gap within this optimizer; correlation undefined",
            }
            continue
        s_r, s_p = spearmanr(group["lambda_max"], group[gap_col])
        p_r, p_p = pearsonr(group["lambda_max"], group[gap_col])
        within[opt_name] = {
            "n": len(group),
            "spearman_r": s_r,
            "spearman_p": s_p,
            "pearson_r": p_r,
            "pearson_p": p_p,
        }
    results["within_optimizer"] = within

    return results


def interpret_correlation(r: float) -> str:
    """
    Qualitative bucket for |r|, used only as a plain-language aid in the
    report -- not a substitute for reporting the numeric value.
    """
    a = abs(r)
    if a < 0.1:
        return "absent / negligible"
    elif a < 0.3:
        return "weak"
    elif a < 0.5:
        return "moderate"
    elif a < 0.7:
        return "strong"
    else:
        return "very strong"


if __name__ == "__main__":
    # Smoke test with synthetic data
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "optimizer": ["sgd"] * 5 + ["adam"] * 5 + ["lbfgs"] * 5,
        "lambda_max": rng.uniform(0, 10, 15),
        "gen_gap": rng.uniform(-0.1, 0.3, 15),
    })
    print(geometry_vs_generalization(df))