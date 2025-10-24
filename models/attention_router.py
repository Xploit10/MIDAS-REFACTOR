import torch
import torch.nn as nn
import torch.nn.functional as F


class _TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        x = x + self.dropout(attn_output)
        x = self.ln1(x)
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.ln2(x)
        return x


class AttentionRouter(nn.Module):
    """
    Lightweight transformer encoder that maps hand-crafted gating features
    to an escalation logit.  We synthesise two tokens from the same features
    to let self-attention reason about complementary views (magnitude vs.
    interaction terms).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_primary = nn.Linear(input_dim, hidden_dim)
        self.embed_interact = nn.Linear(input_dim, hidden_dim)
        self.positional = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (batch_size, input_dim)
        Returns:
            logits of shape (batch_size,)
        """
        primary = self.embed_primary(x)
        interact = self.embed_interact(x * x)
        tokens = torch.stack([primary, interact], dim=1)
        tokens = tokens + self.positional
        for block in self.blocks:
            tokens = block(tokens)
        pooled = tokens.mean(dim=1)
        logits = self.readout(pooled).squeeze(-1)
        return logits
