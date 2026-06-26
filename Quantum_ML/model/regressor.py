import torch
import torch.nn as nn

from .transformer import TransformerBlock, make_attention_mask


class PFNBackbone(nn.Module):
    """
    PFN token encoder + Transformer backbone.

    Returns one embedding per query point with shape [B, n_query, emb_size].
    The bucket decoder is intentionally separate so the representation can be
    reused or inspected before converting to bar-distribution logits.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int = 1,
        emb_size: int = 512,
        n_heads: int = 8,
        ff_hidden: int = 1024,
        n_layers: int = 6,
        dropout: float = 0.0,
        use_x_norm: bool = True,
    ):
        super().__init__()

        if y_dim != 1:
            raise ValueError("PFNBackbone expects y_dim=1.")

        self.use_x_norm = use_x_norm
        self.emb_size = emb_size

        # x -> embedding
        self.x_encoder = nn.Linear(x_dim, emb_size, bias=True)

        # [y_value, is_nan] -> embedding
        self.y_encoder = nn.Linear(2, emb_size, bias=True)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                emb_size=emb_size,
                n_heads=n_heads,
                ff_hidden=ff_hidden,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    @staticmethod
    def _normalize_x(
        x: torch.Tensor,
        x_test: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize x per dataset using both context and query inputs.
        """
        all_x = torch.cat([x, x_test], dim=1)  # [B, n+m, x_dim]
        mu = all_x.mean(dim=1, keepdim=True)
        std = all_x.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)

        x = (x - mu) / std
        x_test = (x_test - mu) / std
        return x, x_test

    @staticmethod
    def _make_y_features(
        train_y: torch.Tensor,
        n_query: int,
    ) -> torch.Tensor:
        """
        Build PFN-style y features over full sequence.

        Context:
          [y, 0]

        Query:
          [0, 1]
        """
        batch_size = train_y.shape[0]
        device = train_y.device
        dtype = train_y.dtype

        if train_y.ndim != 3 or train_y.shape[-1] != 1:
            raise ValueError("train_y must have shape [B, n_context, 1].")  

        # Unknown y values for query tokens
        query_y = torch.full(
            (batch_size, n_query, 1),
            float("nan"),
            device=device,
            dtype=dtype,
        )

        all_y = torch.cat([train_y, query_y], dim=1)  # [B, n+m, 1]
        y_value = torch.nan_to_num(all_y, nan=0.0).squeeze(-1)   # [B, n+m] (context_y,0...
        y_is_nan = torch.isnan(all_y).squeeze(-1).to(dtype)      # [B, n+m] (0...., 1...)

        return torch.stack([y_value, y_is_nan], dim=-1)          # [B, n+m, 2] [(context_y,0...);
                                                                 #               (0...., 1...)]

    def forward(
        self,
        x: torch.Tensor,       # [B, n, x_dim]
        y: torch.Tensor,       # [B, n, 1]
        x_test: torch.Tensor,  # [B, m, x_dim]
    ) -> torch.Tensor:
        _, n, _ = x.shape
        m = x_test.shape[1]
        device = x.device

        # Optional normalization inside the model
        if self.use_x_norm:
            x, x_test = self._normalize_x(x, x_test)

        # Full x sequence
        all_x = torch.cat([x, x_test], dim=1)                    # [B, n+m, x_dim]
        hx = self.x_encoder(all_x)                               # [B, n+m, E]

        # Full y sequence with PFN-style query masking
        y_features = self._make_y_features(y, n_query=m)         # [B, n+m, 2]
        hy = self.y_encoder(y_features)                          # [B, n+m, E]

        # Combine x and y token information
        h = hx + hy                                              # [B, n+m, E]

        # Mask:
        # ctx->ctx allowed
        # qry->ctx allowed
        # ctx->qry blocked
        # qry->qry blocked
        attn_mask = make_attention_mask(
            n_context=n,
            n_query=m,
            device=device,
        )

        # Transformer
        for block in self.blocks:
            h = block(h, attn_mask=attn_mask)


        # Keep only query tokens
        query_h = h[:, n:, :]                                    # [B, m, E]
        return query_h


class BucketDecoder(nn.Sequential):
    """MLP head that maps PFN query embeddings to bucket logits."""

    def __init__(
        self,
        emb_size: int,
        num_buckets: int,
        decoder_hidden: int = 1024,
    ):
        super().__init__(
            nn.Linear(emb_size, decoder_hidden, bias=True),
            nn.GELU(),
            nn.Linear(decoder_hidden, num_buckets, bias=True),
        )


class PFNRegressor(nn.Module):
    """
    PFN-style regressor with separate backbone and bucket decoder.

    Use:
      features = model.backbone(x_ctx, y_ctx, x_query)
      logits = model.decoder(features)

    Calling model(x_ctx, y_ctx, x_query) still returns bucket logits.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int = 1,
        emb_size: int = 512,
        n_heads: int = 8,
        ff_hidden: int = 1024,
        n_layers: int = 6,
        num_buckets: int = 1000,
        dropout: float = 0.0,
        use_x_norm: bool = True,
        decoder_hidden: int = 1024,
    ):
        super().__init__()

        self.num_buckets = num_buckets
        self.backbone = PFNBackbone(
            x_dim=x_dim,
            y_dim=y_dim,
            emb_size=emb_size,
            n_heads=n_heads,
            ff_hidden=ff_hidden,
            n_layers=n_layers,
            dropout=dropout,
            use_x_norm=use_x_norm,
        )
        self.decoder = BucketDecoder(
            emb_size=emb_size,
            num_buckets=num_buckets,
            decoder_hidden=decoder_hidden,
        )

    def forward(
        self,
        x: torch.Tensor,       # [B, n, x_dim]
        y: torch.Tensor,       # [B, n, 1]
        x_test: torch.Tensor,  # [B, m, x_dim]
    ) -> torch.Tensor:
        query_h = self.backbone(x, y, x_test)                    # [B, m, E]

        # Predict bar-distribution logits
        logits = self.decoder(query_h)                           # [B, m, K]
        return logits

    def encode(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_test: torch.Tensor,
    ) -> torch.Tensor:
        """Return query embeddings before bucket decoding."""
        return self.backbone(x, y, x_test)

    def decode(self, query_h: torch.Tensor) -> torch.Tensor:
        """Return bucket logits from query embeddings."""
        return self.decoder(query_h)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """
        Accept checkpoints from the old single-class layout.

        Old checkpoints stored x_encoder/y_encoder/blocks at the model root.
        The split layout stores them under backbone.*.
        """
        if any(key.startswith(("x_encoder.", "y_encoder.", "blocks.")) for key in state_dict):
            state_dict = {
                f"backbone.{key}" if key.startswith(("x_encoder.", "y_encoder.", "blocks.")) else key: value
                for key, value in state_dict.items()
            }

        return super().load_state_dict(state_dict, strict=strict, assign=assign)
    
    def print_parameter_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return total, trainable
