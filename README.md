# Optimization Geometry and Generalization in Neural Networks

A Computational Study of Optimizer Dynamics and Local Loss Curvature

**Type:** Independent research project
**Tools:** Python, PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, Pandas
**Dataset:** UCI Breast Cancer Wisconsin Diagnostic (WDBC)
https://uci-ics-mlr-prod.aws.uci.edu/dataset/17/breast%2Bcancer%2Bwisconsin%2Bdiagnostic  
Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.


---

## 1. Overview

This project asks whether different neural-network optimizers converge to solutions with different local loss-landscape curvature, and whether that curvature is associated with generalization:

```
Optimizer  ->  Optimization Dynamics  ->  Local Loss Geometry  ->  Generalization
```

Three optimizers, SGD, Adam, and L-BFGS, are trained on an identical, fixed MLP (30 -> 32 -> 16 -> 1) using an identical, fixed, leakage-free WDBC train/val/test split. After training, local curvature at each solution is estimated using the largest Hessian eigenvalue (lambda_max). This is computed with Hessian-vector products (Pearlmutter's trick) and power iteration, so the full Hessian is never formed explicitly.

## 2. Research Questions and Hypotheses

- **RQ1 / H1.** Do SGD, Adam, and L-BFGS exhibit measurably different optimization dynamics? Tested in Experiment 1.
- **RQ2 / H2.** Do they converge to solutions with different local curvature? Tested in Experiment 3.
- **RQ3 / H3.** Is local curvature associated with the generalization gap? Tested in Experiment 4.

Per the original project plan, the project is considered successful even if H2 or H3 is not supported. Honest reporting of weak, inconsistent, or null findings is treated as a valid outcome, not a failure.

## 3. Headline Result

Full details are in `report/`. Summary, based on 5 seeds x 3 optimizers (15 runs):

- **H1: supported.** SGD, Adam, and L-BFGS show clearly different optimization trajectories. SGD and Adam decay gradually over 500 steps; L-BFGS converges in around 20 steps through quasi-Newton updates.
- **H2: supported.** lambda_max differs by roughly three orders of magnitude across optimizers (SGD around 0.30, Adam around 0.014, L-BFGS around 0.0001), and this difference is consistent across seeds.
- **H3: not robustly supported.** There is a strong pooled correlation between lambda_max and the loss-based generalization gap (Spearman r about -0.81, p < 0.001). But this correlation is driven mostly by differences between optimizers rather than a consistent relationship within each optimizer. The within-optimizer correlations are inconsistent in sign and mostly not significant. The effect also weakens a lot when the generalization gap is measured by accuracy instead of loss. Taken together, this is evidence against a simple, measurement-invariant sharpness-generalization relationship, and it lines up with cautions raised by Dinh et al. (2017) and Kaur et al. (2022).

## 4. Repository Structure

```
optimization-geometry/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   └── README.md              # data provenance notes, no raw file stored
├── src/
│   ├── data.py                 # WDBC loading and leakage-free 70/15/15 stratified split
│   ├── model.py                 # fixed 30->32->16->1 MLP
│   ├── train.py                  # unified SGD / Adam / L-BFGS training loop with logging
│   ├── metrics.py                 # loss, accuracy, F1, AUROC, generalization gaps
│   ├── hessian.py                  # Hessian-vector products and power iteration for lambda_max
│   └── analysis.py                  # correlation analysis, summary tables
├── experiments/
│   ├── run_experiments.py            # runs the full optimizer x seed grid
│   └── analyze_results.py             # produces all figures and the final results table
├── notebooks/
│   └── results_analysis.ipynb          # exploratory analysis, report figures
├── results/                              # generated: per_run_results.csv, correlation_analysis.json, etc.
├── figures/                                # generated: figure1-4 PNGs
└── report/
    └── Optimization_Geometry_and_Generalization.pdf
```

## 5. Setup

```bash
pip install -r requirements.txt
```

No dataset download is needed. WDBC is loaded directly with `sklearn.datasets.load_breast_cancer()`, which contains the same 569x30 WDBC data, minus an ID column that isn't used anyway.

## 6. Running the Experiments

First, sanity-check the individual modules:

```bash
cd src
python data.py     # prints train/val/test split sizes and class balance
python model.py     # prints model architecture and parameter count
python train.py     # trains all 3 optimizers for seed 0, prints final losses
```

Then run the full experimental grid, 3 optimizers x 5 seeds, 15 runs total:

```bash
cd experiments
python run_experiments.py --seeds 0 1 2 3 4
python analyze_results.py --seeds 0 1 2 3 4
```

This produces:
- `results/per_run_results.csv`, one row per optimizer/seed with all metrics
- `results/training_logs/*.json`, full per-step logs (loss, grad norm, update norm)
- `results/final_results_table.csv`, mean ± std summary table
- `results/correlation_analysis.json`, lambda_max vs generalization-gap correlations, both loss-based and accuracy-based, both pooled and within-optimizer
- `figures/figure1_training_loss.png` through `figure4_lambda_max_vs_gap.png`

For a faster dev run, use fewer seeds, for example `--seeds 0 1 2`.

## 7. Key Methodological Notes

- **No data leakage.** `StandardScaler` is fit only on the training split. Validation and test sets are transformed using the training statistics, never re-fit.
- **Fixed split, varying seed.** The same `random_state=42` train/val/test split is reused for every optimizer and seed combination. Only model initialization varies across seeds.
- **Full-batch training for all optimizers.** L-BFGS is inherently full-batch and closure-based in PyTorch, so SGD and Adam are also run full-batch here, to avoid confounding optimizer identity with batch size.
- **L-BFGS step semantics.** Each `optimizer.step(closure)` call uses `max_iter=20` internal quasi-Newton iterations rather than 1, since PyTorch's L-BFGS line search performs poorly when forced to restart from scratch on every outer call.
- **The Hessian is never explicitly formed.** lambda_max is estimated using Hessian-vector products (Pearlmutter, 1994) and power iteration, the same general approach used by PyHessian (Yao et al., 2020).
- **Two generalization-gap definitions are reported**, loss-based and accuracy-based, because the loss-based gap can be dominated by prediction confidence rather than correctness. This matters especially for L-BFGS, which drives training loss down to nearly zero.
- **PCA was considered and rejected**, to keep the analysis pipeline clean and avoid confounding the curvature estimates with a dimensionality-reduction step.

## 8. Scope and Limitations

This is a small, intentionally controlled study: one dataset, one architecture, five seeds per optimizer. It is not meant to make general claims about optimizer behavior across architectures or datasets, and sharpness-generalization claims are reported with explicit caveats, following Dinh et al. (2017) and Kaur et al. (2022). See `report/` for the full discussion of limitations.

## 9. References
Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2017). On large-batch training for deep learning: Generalization gap and sharp minima. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

Dinh, L., Pascanu, R., Bengio, S., & Bengio, Y. (2017). Sharp minima can generalize for deep nets. In *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:1019-1028.

Jiang, Y., Neyshabur, B., Mobahi, H., Krishnan, D., & Bengio, S. (2020). Fantastic generalization measures and where to find them. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

Kaur, S., Cohen, J., & Lipton, Z. C. (2022). On the maximum Hessian eigenvalue and generalization. In *Proceedings of Machine Learning Research (PMLR)*, NeurIPS 2022 Workshop track.

Wen, K., Ma, T., & Li, Z. (2023). Sharpness minimization algorithms do not only minimize sharpness to achieve better generalization. In *Advances in Neural Information Processing Systems (NeurIPS)*.

Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). Sharpness-aware minimization for efficiently improving generalization. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

Yao, Z., Gholami, A., Keutzer, K., & Mahoney, M. W. (2020). PyHessian: Neural networks through the lens of the Hessian. In *2020 IEEE International Conference on Big Data (Big Data)*, pp. 581-590.

Pearlmutter, B. A. (1994). Fast exact multiplication by the Hessian. *Neural Computation*, 6(1), 147-160.
