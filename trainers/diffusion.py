import torch
import random
from utils import *
from torch.nn import CrossEntropyLoss
T_STEPS = 100
mask_sched = lambda t: t / T_STEPS
ce_loss = CrossEntropyLoss(ignore_index=None)  # Set dynamically later

def diffusion_corrupt(x, t, mask_token_id):
    p = mask_sched(t)
    noise = torch.bernoulli(torch.full_like(x, p, dtype=torch.float)).bool()
    return torch.where(noise, torch.full_like(x, mask_token_id), x)

def diffusion_train_step(batch, model,vocab_size, device, mask_token_id, pad_token_id):
    t = random.randint(1, T_STEPS)
    x0 = batch.to(device)
    xt = diffusion_corrupt(x0, t, mask_token_id)
    logits = model(xt)
    loss_fn = CrossEntropyLoss(ignore_index=pad_token_id)
    loss = loss_fn(logits.view(-1, vocab_size), x0.view(-1))
    return loss 
