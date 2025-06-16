import torch
from utils import *
import json
from configs import *
vocab_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/vocab.json'


with open(vocab_path, "rb") as f:
    vocab = json.load(f)
TOK2ID, ID2TOK = vocab['tok2id'], vocab['id2tok']
TOK2ID = {k: int(v) for k, v in TOK2ID.items()} 
ID2TOK = {int(k): v for k, v in ID2TOK.items()}
VOCAB_SIZE = len(TOK2ID)   ##174 tokens
# @torch.no_grad()
# def sample_diffusion(model, tok2id, max_len, device):
#     x = torch.full((1, max_len), tok2id["[MASK]"], dtype=torch.long, device=device)
#     for t in range(T_STEPS, 0, -1):
#         logits = model(x)
#         x_pred = logits.argmax(-1)
#         keep_prob = 1.0 - t / T_STEPS
#         keep_mask = torch.bernoulli(torch.full_like(x, keep_prob)).bool()
#         x = torch.where(keep_mask, x_pred, x)
#     return x.cpu().tolist()[0]

# @torch.no_grad()
# def sample_mfm(model, tok2id, max_len, device):
#     x = torch.full((1, max_len), tok2id["[MASK]"], dtype=torch.long, device=device)
#     logits = model(x)
#     return logits.argmax(-1).cpu().tolist()[0]

@torch.no_grad()
def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """
    Sample token indices from logits with optional temperature, top_k, top_p nucleus sampling.
    logits: (..., V) float tensor
    returns: (...,) long tensor of sampled indices
    """
    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k)[0][..., -1, None]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float('-inf')), logits)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # mask tokens past nucleus threshold
        mask = cum_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits[mask] = float('-inf')
        # scatter back to original ordering
        logits = torch.gather(sorted_logits, -1, torch.argsort(sorted_idx, dim=-1))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

@torch.no_grad()
def sample_diffusion(model, num_samples=1, seed_tokens=None, steps=T_STEPS, temperature=1.0, top_k=None, top_p=None):
    """Iterative denoising from all‑MASK to sequence using stochastic sampling."""
    B = num_samples
    x = torch.full((B, MAX_LEN), TOK2ID[MASK], dtype=torch.long, device=device)
    
    for t in range(steps, 0, -1):
        logits = model(x)               # (B, L, V)
        mask_idx = (x == TOK2ID[MASK])  # (B, L)
        
        if not mask_idx.any():
            break

        logits_masked = logits[mask_idx]  # (N_masked, V)
        sampled_tokens = sample_from_logits(
            logits_masked,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        x[mask_idx] = sampled_tokens

        keep_prob = 1.0 - mask_sched(t)
        keep_mask = torch.bernoulli(torch.full((B, MAX_LEN), keep_prob, device=device)).bool() & (x != TOK2ID[MASK])
        x = torch.where(keep_mask, x, torch.full_like(x, TOK2ID[MASK]))

    return x.cpu().tolist()  # returns List[List[int]], one list per sample

# def sample_diffusion(model, seed_tokens=None, steps=T_STEPS, temperature=1.0, top_k=None, top_p=None):
    """Iterative denoising from all‑MASK to sequence using stochastic sampling."""
    B = 1
    x = torch.full((B, MAX_LEN), TOK2ID[MASK], dtype=torch.long, device=device)
    for t in range(steps, 0, -1):
        logits = model(x)               # (B,L,V)
        # Only sample tokens currently masked
        mask_idx = (x == TOK2ID[MASK])  # (B,L)
        logits_masked = logits[mask_idx]  # (N_masked, V)

        if logits_masked.numel() == 0:   # all tokens predicted early exit
            break

        sampled_tokens = sample_from_logits(
            logits_masked,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        x[mask_idx] = sampled_tokens

        keep_prob = 1.0 - mask_sched(t)
        keep_mask = torch.bernoulli(torch.full((B, MAX_LEN), keep_prob, device=device)).bool() & (x != TOK2ID[MASK])
        x = torch.where(keep_mask, x, torch.full_like(x, TOK2ID[MASK]))

    return x.cpu().tolist()[0]

@torch.no_grad()
def sample_mfm(model, num_samples=1, temperature=1.0, top_k=None, top_p=None):
    """One‑shot decoding from all‑MASK using MFM vector field with stochastic sampling."""
    x = torch.full((num_samples, MAX_LEN), TOK2ID[MASK], dtype=torch.long, device=device)
    logits = model(x)  # (num_samples, L, V)
    sampled = sample_from_logits(
        logits.view(-1, logits.size(-1)),  # (num_samples * L, V)
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    sampled = sampled.view(num_samples, MAX_LEN)
    return sampled.cpu().tolist()
# def sample_mfm(model , temperature=1.0, top_k=None, top_p=None):
    x = torch.full((1, MAX_LEN), TOK2ID[MASK], dtype=torch.long, device=device)
    logits = model(x)                     # (1,L,V)
    sampled = sample_from_logits(
        logits.squeeze(0),                # (L,V)
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    return sampled.cpu().tolist()

def decode(ids): return "".join(ID2TOK[i] for i in ids if i!=TOK2ID[PAD])