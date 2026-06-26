import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        n_heads: int,
        ff_hidden: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(emb_size, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(emb_size, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(emb_size, elementwise_affine=True)

        self.mlp = nn.Sequential(
            nn.Linear(emb_size, ff_hidden, bias=True),
            nn.GELU(),
            nn.Linear(ff_hidden, emb_size, bias=True),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Attention block: LN -> Attn -> Residual
        n = self.norm1(x)
        attn_out, _ = self.attn(n, n, n, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)

        # MLP block: LN -> MLP -> Residual
        x = x + self.dropout(self.mlp(self.norm2(x)))
        x = self.norm3(x)

        return x


def make_attention_mask(
    n_context: int,
    n_query: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Allowed:
    - context -> context
    - query   -> context

    Blocked:
    - context -> query
    - query   -> query

    PyTorch bool mask convention:
    True  = blocked
    False = allowed
    """
    total_len = n_context + n_query
    # True = blocked
    mask = torch.ones((total_len, total_len), dtype=torch.bool, device=device)
    
    # context attends to context
    mask[:n_context, :n_context] = False
    # query attends to context  
    mask[n_context:, :n_context] = False
    
    return mask