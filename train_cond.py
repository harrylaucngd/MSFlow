import os
import json
import torch
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from data import  CondMolDataset
from modules.cond_lit_model import CondFlowMolBERTLitModule
from configs import data,lit_model
from pytorch_lightning.callbacks import EarlyStopping
from configs.data import TOK2ID,ID2TOK
from torch.utils.data import Subset
import random

torch.set_float32_matmul_precision('medium')


def main():
    df = pd.read_parquet(data.data_path)
    VOCAB_SIZE = len(TOK2ID)
    df = df[df["seq_len"] <= data.MAX_LEN]
    df_conditioned = df[df['has_condition']==True]   #train with samples that have conditions only
    df_unconditioned = df[df['has_condition']==False].sample(n=df_conditioned.shape[0]//10 + 1, random_state=42) 
    df_balanced = pd.concat([df_unconditioned,df_conditioned],axis=0)
    encoded = df_balanced["encoded"].apply(lambda x: x[:data.MAX_LEN]).tolist()
    condition = df_balanced.iloc[:, -1450:-1]
    label = df.iloc[:, -1]
    dataset = CondMolDataset(encoded,condition,label)
    # train_val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    false_indices = [i for i in range(len(dataset)) if dataset[i][2] == False]
    true_indices = [i for i in range(len(dataset)) if dataset[i][2] == True]
    val_false_indices = random.sample(false_indices, val_size)
    train_false_indices = list(set(false_indices) - set(val_false_indices))
    train_indices = true_indices + train_false_indices

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_false_indices)

    train_loader = DataLoader(train_dataset, batch_size=data.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=data.batch_size, shuffle=False, num_workers=8)
    print("Length of training set: ", train_size)
    print("Length of validation set: ", val_size)

    wandb_base_dir = "wandb"
    run_id = None
    name = f'Guided_{lit_model.loss}_L={data.MAX_LEN}_{lit_model.source}_layers={lit_model.n_layers}_dim={lit_model.d_model+1}'
    wandb_logger = WandbLogger(
        project="morflow",
        name=f"FM_{name}",
        save_dir=wandb_base_dir,
        resume="allow",
        id=run_id
    )

   

    model = CondFlowMolBERTLitModule(
        model_name= lit_model.model_name,
        vocab_size=VOCAB_SIZE,
        time_dim= 1,
        hidden_dim= lit_model.d_model,
        cond_dim=lit_model.COND_DIM,
        n_layers= lit_model.n_layers,
        n_heads= lit_model.n_heads,
        mlp = lit_model.mlp,
        uncond_prob=lit_model.uncond_prob,
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
    patience=15,              
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
