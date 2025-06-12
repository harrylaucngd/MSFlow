import torch
import random
from torch.nn import CrossEntropyLoss

def mfm_flow_loss(x_clean, x_corrupt, logits, pad_token_id):
    mask_ratio = (x_corrupt == pad_token_id).float().mean(dim=1, keepdim=True)
    weight = 1.0 / torch.clamp(1 - mask_ratio, min=1e-3)
    loss_fn = CrossEntropyLoss(ignore_index=pad_token_id)
    loss = loss_fn(logits.view(-1, logits.size(-1)), x_clean.view(-1))
    return weight.mean() * loss

def mfm_train_step(batch, model, optimizer, device, mask_token_id, pad_token_id):
    m = random.uniform(0.1, 0.6)
    x0 = batch.to(device)
    noise = torch.bernoulli(torch.full_like(x0, m, dtype=torch.float)).bool()
    xc = torch.where(noise, torch.full_like(x0, mask_token_id), x0)
    logits = model(xc)
    loss = mfm_flow_loss(x0, xc, logits, pad_token_id)
    # optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss #loss.item()
