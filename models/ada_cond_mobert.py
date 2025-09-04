import torch
import torch.nn as nn
import torch.nn.functional as F
from .adaptive import AdaptiveLayerNorm, ConditionalTransformerEncoderLayer, ConditionalTransformerEncoder
# Make sure AdaptiveLayerNorm and ConditionalTransformerEncoderLayer/ConditionalTransformerEncoder are imported

class CondFlowMolBERT(nn.Module):
    def __init__(
        self,
        vocab=173,
        cond_dim=11,
        time_dim=1,
        d_model=767,
        n_layers=12,
        n_heads=12,
        mlp_dim=2048,
        max_len=72,
        dropout=0.4,
        use_gates=False,
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.time_emb = nn.Linear(1, time_dim)

        # Condition embedding MLP: cond_dim -> d_model
        # self.cond_proj = nn.Sequential(
        #     nn.Linear(cond_dim, 4096),
        #     nn.ReLU(),
        #     nn.LayerNorm(4096),
        #     nn.Linear(4096, d_model),
        #     nn.ReLU(),
        #     nn.LayerNorm(d_model),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model, d_model),
        # )

        total_dim = d_model + time_dim
        # Conditional transformer encoder layer
        layer = ConditionalTransformerEncoderLayer(
            d_model=total_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            cond_dim=total_dim,  # condition added to d_model+time_dim
            # use_gates=use_gates, #used with .adaptive instance instead of .adaptiv2
        )
        self.encoder = ConditionalTransformerEncoder(layer, n_layers)

        self.lm_head = nn.Linear(total_dim, vocab, bias=False)

    def forward(self, x, t, cond=None, force_uncond=False):
        B, L = x.shape

        # Token & positional embeddings
        tok_embed = self.tok_emb(x)                            # [B, L, d_model]
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_embed = self.pos_emb(pos_ids)                      # [B, L, d_model]
        x_embed = tok_embed + pos_embed                        # [B, L, d_model]

        # Time embedding
        t_embed = self.time_emb(t.unsqueeze(-1))               # [B, time_dim]
        t_embed = t_embed.unsqueeze(1).expand(-1, L, -1)       # [B, L, time_dim]

        # Combine embeddings
        h = torch.cat([x_embed, t_embed], dim=-1)             # [B, L, d_model + time_dim]

        # # # Condition projection (optional)
        # if cond is not None and not force_uncond:
        #     cond_embed = self.cond_proj(cond)                 # [B, d_model]
        # else:
        #     cond_embed = None        

        # Pass through conditional encoder
        h = self.encoder(h, condition=cond if not force_uncond else None)   # h = d_model + time (128) , cond (127)

        return self.lm_head(h)
