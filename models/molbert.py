import torch.nn as nn

class MolBERT(nn.Module):
    def __init__(self, vocab, d_model=128, n_layers=4, n_heads=4, mlp=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, mlp, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        h = self.tok_emb(x)
        h = self.encoder(h)
        return self.lm_head(h)
