import torch
import random
from utils import *
from torch.nn import CrossEntropyLoss
from configs.trainers import *
mask_sched = lambda t: t / T_STEPS
ce_loss = CrossEntropyLoss(ignore_index=None)  # Set dynamically later

def diffusion_corrupt(x, t, mask_token_id):
    p = mask_sched(t)
    noise = torch.bernoulli(torch.full_like(x, p, dtype=torch.float)).bool()
    return torch.where(noise, torch.full_like(x, mask_token_id), x)

def diffusion_train_step(batch, model, optimizer, device, mask_token_id, vocab_size, pad_token_id):
    t = random.randint(1, T_STEPS)
    x0 = batch.to(device)
    xt = diffusion_corrupt(x0, t, mask_token_id)
    logits = model(xt)
    loss_fn = CrossEntropyLoss(ignore_index=pad_token_id)
    loss = loss_fn(logits.view(-1, vocab_size), x0.view(-1))
    # optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss #loss.item()
