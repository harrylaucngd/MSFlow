import pytorch_lightning as pl
import torch
from models.molbert import MolBERT
from utils.optimizer import get_optimizers_and_schedulers
from trainers import diffusion, mfm

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import random

class MolBERTLitModule(pl.LightningModule):
    def __init__(
        self,
        model_name='mfm',  # or 'diffusion'
        vocab_size=173,
        hidden_dim=128,
        n_layers=4,
        n_heads=4,
        mlp=256,
        lr=1e-3,
        warmup_ratio=0.1,
        pad_token_id=1,
        mask_token_id=0,
        device='cuda'
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.model = MolBERT(vocab_size, hidden_dim, n_layers, n_heads, mlp)
        self.lr_scheduler = None  # Initialized in on_fit_start
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    def on_fit_start(self):
        max_steps = self.trainer.max_steps
        warmup_steps = int(self.hparams.warmup_ratio * max_steps)

        optimizer = self.optimizers()
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            num_cycles=0.5
        )

    def training_step(self, batch):
        optimizer = self.optimizers()
        loss = None
        if self.model_name == 'diffusion':
            loss = diffusion.diffusion_train_step(
                batch, self.model, optimizer,
                self.hparams.device,
                self.hparams.mask_token_id,
                self.hparams.vocab_size,
                self.hparams.pad_token_id
            )
            self.log("diffusion_loss", loss, prog_bar=True)
        elif self.model_name == 'mfm':
            loss = mfm.mfm_train_step(
                batch, self.model, optimizer,
                self.hparams.device,
                self.hparams.mask_token_id,
                self.hparams.pad_token_id
            )
            self.log("mfm_loss", loss.item(), prog_bar=True)
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

        self.manual_backward(loss)
        optimizer.step()
        self.lr_scheduler.step()
        optimizer.zero_grad()
        self.log("lr", optimizer.param_groups[0]["lr"], prog_bar=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch):
        x0 = batch.to(self.device)
        # Use a fixed masking ratio or the same random masking you do in training for validation
        mask_ratio = 0.3  # e.g. fixed or tuned for validation random.uniform(0.1, 0.6)

        noise = torch.bernoulli(torch.full_like(x0, mask_ratio, dtype=torch.float)).bool()
        xc = torch.where(noise, torch.full_like(x0, self.hparams.mask_token_id), x0)
        logits = self.model(xc)
        loss = mfm.mfm_flow_loss(x0, xc, logits, self.hparams.pad_token_id)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss