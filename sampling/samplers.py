import torch
from utils import *
import torch.nn.functional as F
import json
from configs import *
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return torch.softmax(self.model(x, t), dim=-1)
    
#35% validity    
@torch.no_grad()
def sample_flow(num_samples,
    model,
    path,
    seq_len,
    vocab_size,
    mask_token_id=0,
    source_distribution="masked",
    steps=128,
    epsilon=1e-3,
    device="cpu",
    return_intermediates=False
):
    # Step size and time grid
    step_size = 1.0 / steps
    time_grid = torch.linspace(0, 1 - epsilon, steps, device=device)

    # Sample initial tokens
    n_samples = num_samples  # increase as needed
    if source_distribution == "uniform":
        x_init = torch.randint(size=(n_samples, seq_len), high=vocab_size, device=device)
    elif source_distribution == "masked":
        x_init = torch.full(size=(n_samples, seq_len), fill_value=mask_token_id, device=device)
    else:
        raise NotImplementedError(f"Unknown source_distribution: {source_distribution}")

    # Wrap model in probability denoiser wrapper
    wrapped_model = WrappedModel(model)

    # Create solver with model, path, and vocabulary size
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_model,
        path=path,
        vocabulary_size=vocab_size
    )

    # Run the solver to generate samples
    samples = solver.sample(
        x_init=x_init,
        step_size=step_size,
        time_grid=time_grid,
        return_intermediates=return_intermediates,
        verbose=True
    )

    return samples.detach().cpu()  # final sample or full trajectory depending on return_intermediates


## worked, I trained and got 20% validity
@torch.no_grad()
# def sample_flow_custom(model, vocab_size, seq_len, num_samples=1,source_distribution="masked",mask_token_id=0, steps=128,epsilon=1e-3, device='cuda'):
    
#     n_samples = num_samples  # increase as needed
#     if source_distribution == "uniform":
#         x_t = torch.randint(size=(n_samples, seq_len), high=vocab_size, device=device)
#     elif source_distribution == "masked":
#         x_t = torch.full(size=(n_samples, seq_len), fill_value=mask_token_id, device=device)
#     else:
#         raise NotImplementedError(f"Unknown source_distribution: {source_distribution}")
#     t = 0.0
#     results = [(x_t.clone(), t)]
#     for _ in range(steps):
#         t_tensor = torch.full((num_samples,), t, device=device)
        
#         # Predict logits from model and get p_t(x)
#         logits = model(x_t, t_tensor)
#         p_t = torch.softmax(logits, dim=-1)  # [B, L, V]

#         # One-hot encode x_t
#         x_one_hot = F.one_hot(x_t, num_classes=vocab_size).float()

#         # Compute update direction u
#         u = (p_t - x_one_hot) / (1.0 - t + 1e-5)

#         # Step size h
#         h = min(1.0 - t, 1.0 / steps)

#         # Update probability distribution and sample next x_t
#         new_probs = x_one_hot + h * u
#         new_probs = torch.clamp(new_probs, min=1e-6)  # Prevent negatives
#         new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)  # Re-normalize

#         x_t = torch.distributions.Categorical(probs=new_probs).sample()

#         t += h
#         results.append((x_t.clone(), t))
    
#     return x_t.detach().cpu()  # or return x_t for just final samples

## with some hacks, haven't tried yet
@torch.no_grad()
def sample_flow_custom(
    model,
    vocab_size,
    seq_len,
    num_samples=1,
    source_distribution="masked",
    mask_token_id=0,
    pad_token_id=1,  # Add your actual [PAD] token ID here
    steps=128,
    epsilon=1e-3,
    entropy_threshold=1.0,  # adjust as needed
    freeze_start_step_ratio=0.5,  # start freezing after this % of steps
    device='cuda'
):
    n_samples = num_samples

    # Initialization
    if source_distribution == "uniform":
        x_t = torch.randint(size=(n_samples, seq_len), high=vocab_size, device=device)
    elif source_distribution == "masked":
        x_t = torch.full(size=(n_samples, seq_len), fill_value=mask_token_id, device=device)
    else:
        raise NotImplementedError(f"Unknown source_distribution: {source_distribution}")
    
    t = 0.0
    results = [(x_t.clone(), t)]

    for step in range(steps):
        t_tensor = torch.full((n_samples,), t, device=device)

        # Model logits
        logits = model(x_t, t_tensor)

        # Suppress [MASK] and [PAD] tokens
        logits[..., mask_token_id] = -1e6
        logits[..., pad_token_id] = -1e6

        # Temperature annealing
        temperature = max(1e-2, 1.0 - t)
        p_t = torch.softmax(logits / temperature, dim=-1)

        # One-hot encoding of current x_t
        x_one_hot = F.one_hot(x_t, num_classes=vocab_size).float()

        # Compute entropy for each position
        entropy = -(p_t * torch.log(p_t + 1e-8)).sum(dim=-1)  # [B, L]

        # Decide which tokens to freeze (low entropy)
        freeze_mask = (entropy < entropy_threshold).unsqueeze(-1)  # [B, L, 1]

        # Compute update direction u
        u = (p_t - x_one_hot) / (1.0 - t + 1e-5)

        # Step size
        h = min(1.0 - t, 1.0 / steps)

        # Update token distributions
        new_probs = x_one_hot + h * u
        new_probs = torch.clamp(new_probs, min=1e-6)
        new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)

        # Optional freezing in later steps
        if step >= int(freeze_start_step_ratio * steps):
            new_probs = torch.where(freeze_mask, x_one_hot, new_probs)

        # Sample next tokens
        x_t = torch.distributions.Categorical(probs=new_probs).sample()

        t += h
        results.append((x_t.clone(), t))

    return x_t.detach().cpu() 



































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

