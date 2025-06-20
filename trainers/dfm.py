import torch
from torch.nn import CrossEntropyLoss
from flow_matching.loss import MixturePathGeneralizedKL
from configs import *

#worked correctly
# def dfm_step(batch, model, loss_fn, path, device,mask_token_id):
#     B = batch.size(0)
#     x1 = batch.to(device)
#     x0 = torch.full_like(x1, fill_value=mask_token_id) #fully_masked source

#     # x0 = torch.randint_like(x1, vocab)  # uniform source
#     t = torch.rand(B, device=device) * (1 - 1e-3) #singularity at 1 if ELBO is used

#     sample = path.sample(t=t, x_0=x0, x_1=x1)
#     logits = model(sample.x_t, sample.t)
#     loss = loss_fn(logits=logits, x_1=x1, x_t=sample.x_t, t=sample.t)
#     return loss #loss.item()



def dfm_step(batch, model, source, loss_fn, path, device, mask_token_id):
    B = batch.size(0)
    x1 = batch.to(device)
    if source == 'masked':
        x0 = torch.full_like(x1, fill_value=mask_token_id)  # fully masked input
    else:
        x0 = torch.randint_like(x1, vocab)  # uniform source
    t = torch.rand(B, device=device) * (1 - 1e-3)  # avoid t=1
    path_sample = path.sample(t=t, x_0=x0, x_1=x1)

    logits = model(x=path_sample.x_t, t=path_sample.t)

    if isinstance(loss_fn, CrossEntropyLoss):
        loss = loss_fn(logits.view(-1, logits.size(-1)), x1.view(-1)).mean()   # Reshape for CrossEntropy: [B * T, vocab]

    elif isinstance(loss_fn, MixturePathGeneralizedKL):
        loss = loss_fn(
            logits=logits,
            x_1=x1,
            x_t=path_sample.x_t,
            t=path_sample.t
        ).mean()
    else:
        raise ValueError("Invalid loss function type: {}".format(type(loss_fn)))

    return loss
























# def mfm_flow_loss(x_clean, x_corrupt, logits, pad_token_id):
#     mask_ratio = (x_corrupt == pad_token_id).float().mean(dim=1, keepdim=True)
#     weight = 1.0 / torch.clamp(1 - mask_ratio, min=1e-3)
#     loss_fn = CrossEntropyLoss(ignore_index=pad_token_id)
#     loss = loss_fn(logits.view(-1, logits.size(-1)), x_clean.view(-1))
#     return weight.mean() * loss

# def mfm_train_step(batch, model, optimizer, device, mask_token_id, pad_token_id):
#     m = random.uniform(0.1, 0.6)
#     x0 = batch.to(device)
#     noise = torch.bernoulli(torch.full_like(x0, m, dtype=torch.float)).bool()
#     xc = torch.where(noise, torch.full_like(x0, mask_token_id), x0)
#     logits = model(xc)
#     loss = mfm_flow_loss(x0, xc, logits, pad_token_id)
#     # optimizer.zero_grad(); loss.backward(); optimizer.step()
#     return loss #loss.item()