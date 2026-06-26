import torch
import torch.nn as nn

from .sampling import (
    random_split_context_query,
    sample_uniform_num_context,
)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dist,
    x_all: torch.Tensor,   # [B, N, x_dim]
    y_all: torch.Tensor,   # [B, N, 1]
    n_context: int | None = None,
    max_context: int | None = None,
    max_grad_norm: float | None = None,
) -> float:
    """
    One PFN training step.

    Pipeline:
      1. sample number of context points
      2. randomly split each dataset into context/query
      3. predict query targets
      4. compute mean NLL
      5. backprop + optimizer step

    Returns:
      scalar Python float loss
    """
    model.train()

    _, total_points, _ = x_all.shape
    device = x_all.device

    if n_context is None:
        n_context = sample_uniform_num_context(
            total_points,
            max_context=max_context,
            device=device,
        )

    x_ctx, y_ctx, x_query, y_query = random_split_context_query(
        x_all=x_all,
        y_all=y_all,
        n_context=n_context,
    )

    logits = model(x_ctx, y_ctx, x_query)                  # [B, m, K]
    loss = dist.nll(logits, y_query.squeeze(-1)).mean()    # scalar

    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def eval_step(
    model: nn.Module,
    dist,
    x_eval: torch.Tensor,   # [B, N, x_dim]
    y_eval: torch.Tensor,   # [B, N, 1]
    n_context: int | None = None,
    max_context: int | None = None,
) -> dict:
    """
    One evaluation step using provided evaluation tensors.

    - Uniformly samples n_context
    - Randomly splits each dataset into context/query
    - Computes predictive logits, NLL, and MAE
    """
    model.eval()

    B, N, _ = x_eval.shape
    device = x_eval.device

    if n_context is None:
        n_context = sample_uniform_num_context(
            N,
            max_context=max_context,
            device=device,
        )

    x_ctx, y_ctx, x_query, y_query = random_split_context_query(
        x_all=x_eval,
        y_all=y_eval,
        n_context=n_context,
    )

    logits = model(x_ctx, y_ctx, x_query)              # [B, m, K]
    y_true = y_query.squeeze(-1)                       # [B, m]

    nll = dist.nll(logits, y_true)                     # [B, m]
    pred_mean = dist.mean(logits)                      # [B, m]

    abs_error = torch.abs(pred_mean - y_true)          # [B, m]

    loss_per_dataset = nll.mean(dim=1)                 # [B]
    mae_per_dataset = abs_error.mean(dim=1)            # [B]

    loss = loss_per_dataset.mean()                     # scalar
    mae = mae_per_dataset.mean()                       # scalar

    return {
        "loss": float(loss.item()),
        "loss_per_dataset": loss_per_dataset,
        "logits": logits,
        "pred_mean": pred_mean,
        "abs_error": abs_error,
        "mae": float(mae.item()),
        "mae_per_dataset": mae_per_dataset,
        "n_context": n_context,
        "x_context": x_ctx,
        "y_context": y_ctx,
        "x_query": x_query,
        "y_query": y_query,
    }
