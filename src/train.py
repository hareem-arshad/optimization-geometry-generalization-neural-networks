"""
train.py

Unified training loop for SGD, Adam, and L-BFGS on the fixed MLP / WDBC
setup. Logs everything Experiment 1 needs:

    - training loss L_train(t)
    - validation loss L_val(t)
    - gradient norm ||grad L||_2
    - parameter update magnitude ||theta_{t+1} - theta_t||_2

L-BFGS is a full-batch, closure-based optimizer in PyTorch, so all three
optimizers are run full-batch here for a fair, apples-to-apples comparison
of optimization dynamics (this is documented explicitly, since SGD/Adam
are more commonly used with minibatches -- doing so here would confound
"optimizer" with "batch size").
"""

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F

from model import build_model


OPTIMIZER_CONFIGS = {
    # Reasonable, documented hyperparameters per optimizer.
    # Kept fixed across seeds; NOT tuned per-run (see Section 8 of the plan:
    # optimizer hyperparameters should be reasonable and documented, not
    # subject to a sweep).
    "sgd": dict(lr=0.1, momentum=0.9),
    "adam": dict(lr=1e-2, betas=(0.9, 0.999)),
    "lbfgs": dict(lr=1.0, max_iter=1, history_size=10, line_search_fn="strong_wolfe"),
}

N_STEPS = {
    # Full-batch "steps" = epochs here. L-BFGS typically needs far fewer
    # steps to converge on a small convex-ish problem; SGD/Adam need more.
    # Using a shared, generous step budget keeps the comparison fair while
    # letting faster optimizers plateau early (visible in Figure 1).
    "sgd": 500,
    "adam": 500,
    "lbfgs": 100,
}


@dataclass
class TrainingLog:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    update_norm: List[float] = field(default_factory=list)


def _flat_params(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def make_optimizer(name: str, model: torch.nn.Module):
    cfg = OPTIMIZER_CONFIGS[name]
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **cfg)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), **cfg)
    elif name == "lbfgs":
        return torch.optim.LBFGS(model.parameters(), **cfg)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_one_run(
    optimizer_name: str,
    seed: int,
    data,
    n_steps: int = None,
) -> tuple:
    """
    Train one (optimizer, seed) combination on the fixed WDBC split.

    Returns (model, TrainingLog).
    """
    if n_steps is None:
        n_steps = N_STEPS[optimizer_name]

    model = build_model(seed=seed, in_features=data.X_train.shape[1])
    optimizer = make_optimizer(optimizer_name, model)
    log = TrainingLog()

    X_train, y_train = data.X_train, data.y_train
    X_val, y_val = data.X_val, data.y_val

    prev_params = _flat_params(model)

    for step in range(n_steps):

        if optimizer_name == "lbfgs":
            # LBFGS requires a closure that re-evaluates loss (and grad).
            def closure():
                optimizer.zero_grad()
                logits = model(X_train)
                loss = F.binary_cross_entropy_with_logits(logits, y_train)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            # grad norm after the step's last internal evaluation
            g_norm = _grad_norm(model)
            train_loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
        else:
            optimizer.zero_grad()
            logits = model(X_train)
            loss = F.binary_cross_entropy_with_logits(logits, y_train)
            loss.backward()
            g_norm = _grad_norm(model)
            optimizer.step()
            train_loss_val = loss.item()

        with torch.no_grad():
            val_logits = model(X_val)
            val_loss_val = F.binary_cross_entropy_with_logits(val_logits, y_val).item()

        new_params = _flat_params(model)
        update_norm = torch.norm(new_params - prev_params).item()
        prev_params = new_params

        log.train_loss.append(train_loss_val)
        log.val_loss.append(val_loss_val)
        log.grad_norm.append(g_norm)
        log.update_norm.append(update_norm)

    return model, log


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import load_wdbc

    data = load_wdbc()
    for opt_name in ["sgd", "adam", "lbfgs"]:
        model, log = train_one_run(opt_name, seed=0, data=data)
        print(f"{opt_name}: final train_loss={log.train_loss[-1]:.4f}, "
              f"final val_loss={log.val_loss[-1]:.4f}, steps={len(log.train_loss)}")