# Data

No raw data file is stored in this repository.

The UCI Breast Cancer Wisconsin Diagnostic (WDBC) dataset is loaded programmatically via:

```python
from sklearn.datasets import load_breast_cancer
```

This provides the identical 569-sample, 30-feature WDBC dataset distributed by UCI, with the
non-predictive ID column already removed (it is not used in any analysis here regardless).

Original source: UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Label convention used in this project: `benign = 0`, `malignant = 1` (see `src/data.py`, which
flips scikit-learn's default encoding to match this convention).

Preprocessing (see `src/data.py`):
- Stratified 70/15/15 train/validation/test split, fixed `random_state=42`, reused identically
  across every optimizer and seed.
- `StandardScaler` fit **only** on the training split; validation and test sets are transformed
  using the training statistics only (no leakage).