import os
import json
import torch
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from data import MolDataset
from modules.lit_model import FlowMolBERTLitModule
from configs import data,lit_model
from pytorch_lightning.callbacks import EarlyStopping
from configs.data import TOK2ID,ID2TOK


torch.set_float32_matmul_precision('medium')


def main():
    df = pd.read_parquet(data.data_path)
    VOCAB_SIZE = len(TOK2ID)
    df = df[df["seq_len"] <= data.MAX_LEN]
    encoded = df["encoded"].apply(lambda x: x[:data.MAX_LEN]).tolist()
    dataset = MolDataset(encoded)
    # train_val random split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=data.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=data.batch_size, shuffle=False, num_workers=8)
    print("Length of training set: ", train_size)
    print("Length of validation set: ", val_size)
    # loader = DataLoader(dataset, batch_size=data.batch_size, shuffle=True, num_workers=8)

    wandb_base_dir = "wandb"
    run_id = None
    # if os.path.exists(wandb_base_dir):
    #     run_dirs = [d for d in os.listdir(wandb_base_dir) if d.startswith("run-")]
    #     if len(run_dirs)>0:
    #         run_dirs.sort()  # sort to get latest run last (assuming lex order works for your timestamp)

    #         latest_run_dir = run_dirs[-1]
    #         run_id = latest_run_dir[-8:]
    #         print(f"Latest wandb run ID: {run_id}")
    name = f'{lit_model.loss}_L={data.MAX_LEN}_{lit_model.source}_layers={lit_model.n_layers}_dim={lit_model.d_model}'
    wandb_logger = WandbLogger(
        project="morflow",
        name=f"FM_{name}",
        save_dir=wandb_base_dir,
        resume="allow",
        id=run_id
    )

   

    model = FlowMolBERTLitModule(
        model_name= lit_model.model_name,
        vocab_size=VOCAB_SIZE,
        time_dim= 1,
        hidden_dim= lit_model.d_model,
        n_layers= lit_model.n_layers,
        n_heads= lit_model.n_heads,
        mlp = lit_model.mlp,
        lr=lit_model.lr,
        warmup_ratio=lit_model.warmup_ratio,
        pad_token_id=TOK2ID[data.PAD],
        mask_token_id=TOK2ID[data.MASK],
        device=lit_model.device,
        source= lit_model.source,
        scheduler = lit_model.scheduler,
        path = lit_model.path,
        loss_fn = lit_model.loss,
        weighted=lit_model.weighted
    )
    early_stop_callback = EarlyStopping(
    monitor="val_loss",      
    patience=5,              
    mode="min",             
    verbose=True)

    checkpoint_dir = data.output_path
    last_ckpt_path = os.path.join(checkpoint_dir, "FM_last.ckpt")
    resume_ckpt = last_ckpt_path if os.path.exists(last_ckpt_path) else None


    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename= name + '_best-validity-{epoch:02d}-{validity:.4f}',
        monitor="validity",
        mode="max",
        save_top_k=1,
        save_last = True,
        save_weights_only=False,  # save full checkpoint, or True to save only model weights
    )


    trainer = Trainer(
        max_steps = lit_model.max_steps,
        accelerator="gpu",
        strategy="ddp",
        devices=1,   # or 2 for multi-GPU
        logger=wandb_logger,
        callbacks=[ckpt_callback,early_stop_callback],
    )

    # --- Train ---
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
