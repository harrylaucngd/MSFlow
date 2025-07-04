import torch
import torch.nn as nn
import torch.nn.functional as F

class CondFlowMolBERT(nn.Module):
    def __init__(self, vocab = 173, cond_dim = 1449, time_dim=1, d_model=127, n_layers=4, n_heads=4, mlp_dim=256, max_len=72, dropout=0.4):
        super().__init__()
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.time_emb = nn.Linear(1, time_dim)

        # Condition embedding MLP: cond_dim -> 1024 -> 512 -> 256 -> d_model
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024,d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        total_dim = d_model + time_dim
        layer = nn.TransformerEncoderLayer(total_dim, n_heads, mlp_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

        self.lm_head = nn.Linear(total_dim, vocab, bias=False)

    def forward(self, x, t, cond=None, force_uncond=False):
        B, L = x.shape

        tok_embed = self.tok_emb(x)                            # [B, L, d_model]
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_embed = self.pos_emb(pos_ids)                      # [B, L, d_model]
        x_embed = tok_embed + pos_embed                        # [B, L, d_model]

        t_embed = self.time_emb(t.unsqueeze(-1))               # [B, time_dim]
        t_embed = t_embed.unsqueeze(1).expand(-1, L, -1)       # [B, L, time_dim]

        if cond is None:
            # No conditioning at all
            h = torch.cat([x_embed, t_embed], dim=-1)
        else:
            # print("cond stats:", cond.min().item(), cond.max().item(), cond.mean().item(), cond.std().item())
            # Create mask for zero cond rows
            zero_cond_mask = (cond.abs().sum(dim=1) == 0)  # [B], True if zero cond vector
            cond_embed = self.cond_proj(cond)               # [B, d_model]
            cond_embed = cond_embed.unsqueeze(1).expand(-1, L, -1)  # [B, L, d_model]
    
            # For zero condition rows, zero out cond_embed
            cond_embed = cond_embed * (~zero_cond_mask).unsqueeze(1).unsqueeze(2)
            
            h = torch.cat([x_embed + cond_embed, t_embed], dim=-1)

        h = self.encoder(h)
        return self.lm_head(h)


    # def forward(self, x, t, cond=None, force_uncond=False):
    #     B, L = x.shape

    #     tok_embed = self.tok_emb(x)                            # [B, L, d_model]
    #     pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
    #     pos_embed = self.pos_emb(pos_ids)                      # [B, L, d_model]
    #     x_embed = tok_embed + pos_embed                        # [B, L, d_model]

    #     t_embed = self.time_emb(t.unsqueeze(-1))               # [B, time_dim]
    #     t_embed = t_embed.unsqueeze(1).expand(-1, L, -1)       # [B, L, time_dim]

    #     if force_uncond or cond is None or cond.abs().sum() == 0:
    #         # No conditioning: just concat x_embed and t_embed
    #         h = torch.cat([x_embed, t_embed], dim=-1)
    #     else:
    #         cond_embed = self.cond_proj(cond)                   # [B, d_model]
    #         cond_embed = cond_embed.unsqueeze(1).expand(-1, L, -1)  # [B, L, d_model]
    #         h = torch.cat([x_embed + cond_embed, t_embed], dim=-1)  # add cond then concat time

    #     h = self.encoder(h)
    #     return self.lm_head(h)
