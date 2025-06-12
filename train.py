import os
import json
import torch
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from data import MolDataset
from modules.lit_model import MolBERTLitModule
from configs import data,lit_model



def main():
    df = pd.read_parquet(data.data_path)
    with open(data.vocab_path, "r") as f:
        vocab = json.load(f)
    TOK2ID = {k: int(v) for k, v in vocab["tok2id"].items()}
    VOCAB_SIZE = len(TOK2ID)

    df = df[df["seq_len"] <= data.MAX_LEN]
    encoded = df["encoded"].apply(lambda x: x[:data.MAX_LEN]).tolist()
    dataset = MolDataset(encoded)
    loader = DataLoader(dataset, batch_size=data.batch_size, shuffle=True, num_workers=8)

    wandb_base_dir = "wandb"
    run_id = None
    if os.path.exists(wandb_base_dir):
        run_dirs = [d for d in os.listdir(wandb_base_dir) if d.startswith("run-")]
        if len(run_dirs)>0:
            run_dirs.sort()  # sort to get latest run last (assuming lex order works for your timestamp)

            latest_run_dir = run_dirs[-1]
            run_id = latest_run_dir[-8:]
            print(f"Latest wandb run ID: {run_id}")

    wandb_logger = WandbLogger(
        project="morflow",
        name="FM_large",
        save_dir=wandb_base_dir,
        resume="allow",
        id=run_id
    )

   
    model = MolBERTLitModule(
        model_name= lit_model.model_name,
        vocab_size=VOCAB_SIZE,
        hidden_dim= lit_model.d_model,
        n_layers= lit_model.n_layers,
        n_heads= lit_model.n_heads,
        lr=lit_model.lr,
        warmup_ratio=lit_model.warmup_ratio,
        pad_token_id=TOK2ID[data.PAD],
        mask_token_id=TOK2ID[data.MASK],
        device=lit_model.device
    )

    checkpoint_dir = data.output_path
    last_ckpt_path = os.path.join(checkpoint_dir, "FM_last.ckpt")
    resume_ckpt = last_ckpt_path if os.path.exists(last_ckpt_path) else None

    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="last",
        save_top_k=1,
        save_last=True,
        # every_n_epochs=1,
        save_weights_only=False,
    )

    trainer = Trainer(
        max_steps = lit_model.max_steps,
        accelerator="gpu",
        strategy="ddp",
        devices=1,   # or 2 for multi-GPU
        logger=wandb_logger,
        callbacks=[ckpt_callback],
    )

    # --- Train ---
    trainer.fit(model, loader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
