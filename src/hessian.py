"""
hessian.py

Mathematical centerpiece of the project.

We never construct the full Hessian H(theta) explicitly (it would be an
n x n matrix, n = number of parameters -- intractable to store/build for
even modest networks). Instead we compute Hessian-vector products

    Hv = nabla_theta ( (nabla_theta L)^T v )

using Pearlmutter's (1994) double-backpropagation trick, which needs only
two backward passes and never materializes H. We then use power iteration
to estimate the largest eigenvalue lambda_max(H) from these Hv products
alone.

This mirrors the approach used by PyHessian (Yao et al., 2020).
"""

from typing import List

import torch
import torch.nn.functional as F


def _params(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _flatten(vectors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in vectors])


def _unflatten_like(flat: torch.Tensor, like: List[torch.Tensor]) -> List[torch.Tensor]:
    out, idx = [], 0
    for t in like:
        n = t.numel()
        out.append(flat[idx: idx + n].view_as(t))
        idx += n
    return out


def compute_loss(model, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(X)
    return F.binary_cross_entropy_with_logits(logits, y)


def hessian_vector_product(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Compute H(theta) @ v without forming H.

    Steps (Pearlmutter's trick):
      1. Forward pass, compute loss L(theta).
      2. Backward pass with create_graph=True to get grad = nabla_theta L,
         keeping the computation graph so grad is itself differentiable.
      3. Compute the scalar s = grad . v  (dot product with the given vector).
      4. Backward pass on s w.r.t. theta gives nabla_theta (grad . v) = H v.

    v is a flat 1-D tensor with the same total size as the model's
    trainable parameters.
    """
    params = _params(model)

    loss = compute_loss(model, X, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = _flatten(grads)

    v_unflat = _unflatten_like(v, params)
    dot = sum((g * vi).sum() for g, vi in zip(grads, v_unflat))

    hvp = torch.autograd.grad(dot, params, retain_graph=False)
    return _flatten(hvp).detach()


@torch.no_grad()
def _normalize(v: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(v)
    if norm < 1e-12:
        return torch.randn_like(v)
    return v / norm


def power_iteration_lambda_max(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    num_iters: int = 100,
    tol: float = 1e-6,
    seed: int = 0,
) -> dict:
    """
    Estimate lambda_max(H) via power iteration using only Hessian-vector
    products (never forming H).

    Update rule:
        v_{k+1} = H v_k / || H v_k ||
        lambda_k = v_k^T H v_k   (Rayleigh quotient, since ||v_k|| = 1)

    Iterate until the Rayleigh-quotient estimate changes by less than `tol`
    (relative), or num_iters is reached.

    Returns a dict with the eigenvalue estimate, number of iterations used,
    and the convergence history (useful for validating convergence, as the
    schedule calls for).
    """
    params = _params(model)
    n_params = sum(p.numel() for p in params)

    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n_params, generator=g)
    v = _normalize(v)

    history = []
    lambda_prev = None

    for i in range(num_iters):
        Hv = hessian_vector_product(model, X, y, v)
        lambda_est = torch.dot(v, Hv).item()
        history.append(lambda_est)

        v = _normalize(Hv)

        if lambda_prev is not None:
            rel_change = abs(lambda_est - lambda_prev) / (abs(lambda_prev) + 1e-12)
            if rel_change < tol:
                lambda_prev = lambda_est
                break
        lambda_prev = lambda_est

    return {
        "lambda_max": lambda_prev,
        "n_iters": len(history),
        "history": history,
        "converged": len(history) < num_iters,
    }


@torch.no_grad()
def perturbation_sharpness(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 1e-2,
    n_directions: int = 10,
    seed: int = 0,
) -> dict:
    """
    Optional extension (Section 17 of the plan): local perturbation-based
    sharpness.

    For n_directions random unit directions u, compute
        S_u = L(theta + epsilon * u) - L(theta)
    and report the mean and max over directions as a complementary,
    Hessian-free sensitivity measure.
    """
    params = _params(model)
    base_loss = compute_loss(model, X, y).item()

    g = torch.Generator().manual_seed(seed)
    sharpness_values = []

    original = [p.detach().clone() for p in params]

    for _ in range(n_directions):
        direction = [torch.randn(p.shape, generator=g) for p in params]
        norm = torch.sqrt(sum((d ** 2).sum() for d in direction))
        direction = [d / (norm + 1e-12) for d in direction]

        with torch.no_grad():
            for p, d in zip(params, direction):
                p.add_(epsilon * d)

        perturbed_loss = compute_loss(model, X, y).item()
        sharpness_values.append(perturbed_loss - base_loss)

        with torch.no_grad():
            for p, orig in zip(params, original):
                p.copy_(orig)

    sharpness_values = torch.tensor(sharpness_values)
    return {
        "mean_sharpness": sharpness_values.mean().item(),
        "max_sharpness": sharpness_values.max().item(),
        "std_sharpness": sharpness_values.std().item(),
        "base_loss": base_loss,
    }