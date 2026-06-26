import torch
from typing import Tuple


def sample_uniform_num_context(
    total_points: int,
    max_context: int | None = None,
    device: torch.device | None = None,
) -> int:
    """
    Uniformly sample the number of context points from {1, ..., max_context}.

    max_context is clipped to total_points - 1, so every step has a non-empty
    target set and n_query = total_points - n_context.
    """
    if total_points < 2:
        raise ValueError("total_points must be >= 2.")

    max_context = total_points - 1 if max_context is None else max_context
    max_context = min(max_context, total_points - 1)
    if max_context < 1:
        raise ValueError("max_context must be >= 1 after clipping.")

    return int(torch.randint(1, max_context + 1, (1,), device=device).item())


def random_split_context_query(
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    n_context: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly split datasets into context and query sets.

    Inputs:
        x_all : [B, N, x_dim]
        y_all : [B, N, 1]

    Outputs:
        x_context : [B, n_context, x_dim]
        y_context : [B, n_context, 1]
        x_query   : [B, N - n_context, x_dim]
        y_query   : [B, N - n_context, 1]
    """

    B, N, x_dim = x_all.shape
    device = x_all.device

    if not (1 <= n_context < N):
        raise ValueError("n_context must be in [1, N - 1].")

    perm = torch.rand(B, N, device=device).argsort(dim=1)
    x_perm = x_all.gather(1, perm.unsqueeze(-1).expand(-1, -1, x_dim))
    y_perm = y_all.gather(1, perm.unsqueeze(-1).expand(-1, -1, y_all.shape[-1]))

    x_context = x_perm[:, :n_context]
    y_context = y_perm[:, :n_context]

    x_query = x_perm[:, n_context:]
    y_query = y_perm[:, n_context:]

    return x_context, y_context, x_query, y_query
