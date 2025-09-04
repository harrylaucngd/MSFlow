import os
import json
import torch
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger,  MLFlowLogger

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from data import  CondMolDataset
from modules.cond_lit_model import CondFlowMolBERTLitModule
from modules.cond_lit_model import CondFlowMolBERT
from modules.lit_model import FlowMolBERTLitModule
from configs import data,lit_model
from pytorch_lightning.callbacks import EarlyStopping
from configs.data import TOK2ID,ID2TOK
from torch.utils.data import Subset
import random
import torch
from utils.functions import transfer_weights, transfer_weights_with_adaptive_ln


local_rank = int(os.environ.get("LOCAL_RANK", 0))
print(f"Process {local_rank} using device: cuda:{local_rank}")


checkpoint_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/checkpoints/CrossEntropyLoss()_L=72_uniform_layers=12_dim=768_best-validity-epoch=121-validity=0.9350.ckpt'
torch.set_float32_matmul_precision('medium')

def main():
    df = pd.read_parquet(data.data_path)
    VOCAB_SIZE = len(TOK2ID)
    df = df[df["seq_len"] <= data.MAX_LEN]
    generator = torch.Generator().manual_seed(42)
    df = df.iloc[:100000,:]   #similar size to the perturbation data sizels -ls
    # df_conditioned = df[df['has_condition']==True]   #train with samples that have conditions only
    encoded = df["encoded"].apply(lambda x: x[:data.MAX_LEN]).tolist()
    condition = df.SMILES_standard # iloc[:,-1450:-1]  or #conditions_are 11 chem_props or 1449 CP features
    label = [True] * df.shape[0]
    dataset = CondMolDataset(encoded,condition,label,df.index)
    # train_val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator=generator)


    train_loader = DataLoader(train_dataset, batch_size=data.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=data.batch_size, shuffle=False, num_workers=8)
    print("Length of training set: ", train_size)
    print("Length of validation set: ", val_size)

    wandb_base_dir = "wandb"
    mlflow_base_dir = 'mlflow_base'
    run_id = None
    name = f'MFP_adaptiveLN_proj_r3_L1024_{lit_model.COND_DIM}_{lit_model.loss}_L={data.MAX_LEN}_{lit_model.source}_layers={lit_model.n_layers}_dim={lit_model.d_model+1}'
    wandb_logger = WandbLogger(
        project="morflow",
        name=f"{name}",
        save_dir=wandb_base_dir,
        resume="allow",
        id=run_id
    )
    mlflow_logger = MLFlowLogger(
    experiment_name="morflow",            # like wandb project
    run_name=f"{name}",                   # like wandb name
    tracking_uri="file:./mlruns",         # or "http://127.0.0.1:8080" if running MLflow server
    )

   
    checkpoint = FlowMolBERTLitModule.load_from_checkpoint(checkpoint_path)
    uncond_model = checkpoint.model
    cond_model= CondFlowMolBERTLitModule(
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
# Transfer shared weights
    with torch.no_grad():
        transfer_weights_with_adaptive_ln(uncond_model,cond_model.model,freeze_pretrained=False)
    
    early_stop_callback = EarlyStopping(
    monitor="val_loss",      
    patience=8,              
    mode="min",             
    verbose=True)

    checkpoint_dir = data.output_path
    last_ckpt_path = os.path.join(checkpoint_dir, "FM_last.ckpt")
    resume_ckpt = last_ckpt_path if os.path.exists(last_ckpt_path) else None


    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename= name + 'fine_tuned_best_val_loss-{epoch:02d}-{cond_validity:.4f}',
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last = True,
        save_weights_only=False,  # save full checkpoint, or True to save only model weights
    )


    trainer = Trainer(
        max_steps = lit_model.max_steps,
        accelerator="gpu",
        strategy="ddp",
        devices=1,   # or 2 for multi-GPU
        logger=[wandb_logger,mlflow_logger],
        callbacks=[ckpt_callback,early_stop_callback],
    )

    # --- Train ---
    trainer.fit(cond_model, train_loader, val_loader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
