from torch.optim import AdamW
from transformers import get_scheduler

def get_optimizers_and_schedulers(model_diff, model_mfm, loader_length, epochs, lr=5e-4, warmup_ratio=0.06):
    num_training_steps = loader_length * epochs
    warmup_steps = int(warmup_ratio * num_training_steps)

    optim_diff = AdamW(model_diff.parameters(), lr=lr)
    optim_mfm = AdamW(model_mfm.parameters(), lr=lr)

    scheduler_diff = get_scheduler(
        name="constant_with_warmup",
        optimizer=optim_diff,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    scheduler_mfm = get_scheduler(
        name="constant_with_warmup",
        optimizer=optim_mfm,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    return optim_diff, scheduler_diff, optim_mfm, scheduler_mfm
