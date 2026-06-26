import math

import torch
import torch.nn.functional as F


class RiemannDistribution:
    """
    Piecewise-uniform bar distribution with optional half-normal tails.

    With use_tails=False, every bucket is uniform over its finite interval.

    With use_tails=True, interior buckets are uniform and the edge buckets
    become half-normal tails. Interior buckets use:
        p(y) = p_bucket / bucket_width

    Left tail:
        p(y) = p_bucket_0 * HalfNormal(boundaries[1] - y; left_sigma)

    Right tail:
        p(y) = p_bucket_K * HalfNormal(y - boundaries[-2]; right_sigma)
    """

    def __init__(self, boundaries: torch.Tensor, use_tails: bool = True):
        boundaries = boundaries.detach().to(torch.float64)

        if boundaries.ndim != 1 or boundaries.numel() < 2:
            raise ValueError("boundaries must be a 1D tensor of length >= 2.")

        widths = boundaries[1:] - boundaries[:-1]
        if torch.any(widths <= 0):
            raise ValueError("Bucket boundaries must be strictly increasing.")

        self.boundaries = boundaries              # [K+1]
        self.widths = widths                      # [K]
        self.num_buckets = widths.numel()
        self.use_tails = use_tails

        self.left_edge = boundaries[0].item()
        self.right_edge = boundaries[-1].item()

        # Choose sigma so HalfNormal puts 50% mass inside one edge bucket width:
        # erf(w / (sigma * sqrt(2))) = 0.5
        erfinv_half = torch.special.erfinv(torch.tensor(0.5)).item()
        self.left_sigma = widths[0].item() / (math.sqrt(2.0) * erfinv_half)
        self.right_sigma = widths[-1].item() / (math.sqrt(2.0) * erfinv_half)

    @staticmethod
    def fit_equal_mass_buckets(
        y_samples: torch.Tensor,
        num_buckets: int,
        use_tails: bool = True,
    ) -> "RiemannDistribution":
        """
        Fit bucket boundaries using quantiles so each bucket has roughly equal mass.
        """
        if num_buckets < 1:
            raise ValueError("num_buckets must be >= 1.")

        y = y_samples.detach().flatten().float().cpu()
        if y.numel() == 0:
            raise ValueError("y_samples must not be empty.")

        qs = torch.linspace(0.0, 1.0, num_buckets + 1, dtype=y.dtype, device=y.device)
        boundaries = torch.quantile(y, qs)

        # Make strictly increasing using unique_consecutive after small perturbation
        boundaries = boundaries.clone()
        eps = 1e-6
        for i in range(1, boundaries.numel()):
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = boundaries[i - 1] + eps

        return RiemannDistribution(boundaries, use_tails=use_tails)

    def to(self, device: torch.device | str) -> "RiemannDistribution":
        """Move stored tensors to a device."""
        self.boundaries = self.boundaries.to(device)
        self.widths = self.widths.to(device)
        return self

    def bucket_index(self, y: torch.Tensor) -> torch.Tensor:
        """
        Return the bucket index for each y, clipped to [0, K-1].

        Values below the left edge map to bucket 0.
        Values above the right edge map to bucket K-1.
        """
        b = self.boundaries.to(y.device)
        idx = torch.bucketize(y.contiguous(), b[1:-1], right=False)
        return idx.clamp(0, self.num_buckets - 1)

    def bucket_width(
        self,
        idx: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return bucket widths for the provided bucket indices."""
        return self.widths.to(device=device, dtype=dtype)[idx]

    @staticmethod
    def _halfnormal_logpdf(x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Log-density of a HalfNormal(sigma) at x >= 0."""
        return (
            math.log(math.sqrt(2.0 / math.pi))
            - math.log(sigma)
            - 0.5 * (x / sigma) ** 2
        )

    def log_prob(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute log density log p(y | logits).

        logits: [..., K]
        y:      [...]

        Returns:
            tensor with shape [...]
        """
        if logits.shape[-1] != self.num_buckets:
            raise ValueError(
                f"logits last dimension {logits.shape[-1]} does not match "
                f"num_buckets {self.num_buckets}."
            )

        if logits.shape[:-1] != y.shape:
            raise ValueError(
                f"Shape mismatch: logits.shape[:-1]={logits.shape[:-1]} vs y.shape={y.shape}"
            )

        probs = F.softmax(logits, dim=-1).clamp_min(1e-12)
        idx = self.bucket_index(y)
        pb = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

        if not self.use_tails:
            in_support = (y >= self.boundaries[0].to(y.device, y.dtype)) & (
                y <= self.boundaries[-1].to(y.device, y.dtype)
            )
            widths = self.bucket_width(idx, y.device, logits.dtype)
            out = torch.log(pb) - torch.log(widths)
            return out.masked_fill(~in_support, -torch.inf)

        b = self.boundaries.to(y.device, y.dtype)

        # Use idx-based conditions to match quantile/mean anchor points
        is_left  = idx == 0
        is_right = idx == self.num_buckets - 1
        is_mid   = ~(is_left | is_right)

        out = torch.empty_like(y, dtype=logits.dtype)

        if is_mid.any():
            widths = self.bucket_width(idx[is_mid], y.device, logits.dtype)
            out[is_mid] = torch.log(pb[is_mid]) - torch.log(widths)

        # Anchor at b[1] — matches mean and quantile
        if is_left.any():
            dist = (b[1] - y[is_left]).clamp_min(1e-8)
            out[is_left] = torch.log(pb[is_left]) + self._halfnormal_logpdf(dist, self.left_sigma)

        # Anchor at b[-2] — matches mean and quantile
        if is_right.any():
            dist = (y[is_right] - b[-2]).clamp_min(1e-8)
            out[is_right] = torch.log(pb[is_right]) + self._halfnormal_logpdf(dist, self.right_sigma)

        return out

    def nll(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood."""
        return -self.log_prob(logits, y)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Approximate predictive mean E[y | logits].

        Buckets use centers unless use_tails=True, where edge buckets use
        shifted means based on the half-normal tails.
        """
        if logits.shape[-1] != self.num_buckets:
            raise ValueError(
                f"logits last dimension {logits.shape[-1]} does not match "
                f"num_buckets {self.num_buckets}."
            )

        probs = torch.softmax(logits, dim=-1)

        b = self.boundaries.to(logits.device, logits.dtype)
        centers = b[:-1] + self.widths.to(logits.device, logits.dtype) / 2

        if self.use_tails:
            centers = centers.clone()
            centers[0]  = b[1]  - self.left_sigma  * math.sqrt(2.0 / math.pi)
            centers[-1] = b[-2] + self.right_sigma * math.sqrt(2.0 / math.pi)

        return (probs * centers).sum(dim=-1)

    def quantile(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """
        Quantile function for the distribution.

        With use_tails=False, all buckets are uniform on [b_i, b_{i+1}].
        With use_tails=True:
        - bucket 0: left half-normal tail ending at boundaries[1]
        - buckets 1..K-2: uniform on [b_i, b_{i+1}]
        - bucket K-1: right half-normal tail starting at boundaries[-2]
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")
        if logits.shape[-1] != self.num_buckets:
            raise ValueError(
                f"logits last dimension {logits.shape[-1]} does not match "
                f"num_buckets {self.num_buckets}."
            )

        probs = torch.softmax(logits, dim=-1)                       # [..., K]
        cdf = probs.cumsum(dim=-1)                                  # [..., K]

        target = torch.full(
            (*cdf.shape[:-1], 1),
            fill_value=p,
            device=logits.device,
            dtype=logits.dtype,
        )

        idx = torch.searchsorted(cdf, target).squeeze(-1)
        idx = idx.clamp(0, self.num_buckets - 1)                    # [...]

        cdf_prev = torch.cat(
            [torch.zeros_like(cdf[..., :1]), cdf[..., :-1]],
            dim=-1,
        )

        prev_mass = cdf_prev.gather(-1, idx.unsqueeze(-1)).squeeze(-1)          # [...]
        bucket_mass = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)

        # within-component probability in [0,1]
        u = ((p - prev_mass) / bucket_mass).clamp(0.0, 1.0)

        b = self.boundaries.to(logits.device, logits.dtype)

        out = torch.empty_like(u)

        if not self.use_tails:
            left = b[idx]
            right = b[idx + 1]
            return left + u * (right - left)

        # Left tail: Y = b[1] - H, H ~ HalfNormal(left_sigma)
        # CDF: F(y) = 1 - erf((b[1] - y)/(sigma*sqrt(2)))
        # So if u is within the left-tail component:
        # y = b[1] - sigma*sqrt(2)*erfinv(1 - u)
        is_left = idx == 0
        if is_left.any():
            erf_arg = (1.0 - u[is_left]).clamp(0.0, 1.0 - 1e-7)
            dist = self.left_sigma * math.sqrt(2.0) * torch.erfinv(erf_arg)
            out[is_left] = b[1] - dist

        # Right tail: Y = b[-2] + H, H ~ HalfNormal(right_sigma)
        # CDF within this component: u = erf((y - b[-2])/(sigma*sqrt(2)))
        # so y = b[-2] + sigma*sqrt(2)*erfinv(u)
        is_right = idx == self.num_buckets - 1
        if is_right.any():
            erf_arg = u[is_right].clamp(0.0, 1.0 - 1e-7)
            dist = self.right_sigma * math.sqrt(2.0) * torch.erfinv(erf_arg)
            out[is_right] = b[-2] + dist

        # Interior buckets: uniform interpolation
        is_mid = ~(is_left | is_right)
        if is_mid.any():
            left = b[idx[is_mid]]
            right = b[idx[is_mid] + 1]
            out[is_mid] = left + u[is_mid] * (right - left)

        return out
