import os
import torch
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger,  MLFlowLogger

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from data import  CondMolDataset
from modules.cond_lit_model import CondFlowMolBERTLitModule
from configs import data,lit_model
from pytorch_lightning.callbacks import EarlyStopping
from configs.data import TOK2ID,ID2TOK
import torch
from utils.functions import create_finetune_strategy

local_rank = int(os.environ.get("LOCAL_RANK", 0))
print(f"Process {local_rank} using device: cuda:{local_rank}")
checkpoint_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/checkpoints/new/context_128/CFG-MFP_1.4M_canonical_context_len=128_uncond_prob=0.1_4096_r=2_LR=0.0001_uniform_dim=1536_4gpusCFG_best_val_loss-epoch=81-cond_validity=0.8516.ckpt'
torch.set_float32_matmul_precision('medium')

def main():
    df = pd.read_parquet(data.data_path)
    VOCAB_SIZE = len(TOK2ID)
    df = df[df["seq_len"] <= data.MAX_LEN]
    df.reset_index(inplace=True)
    df.drop(columns=["index"], inplace=True)
    generator = torch.Generator().manual_seed(42)
    encoded = df["encoded"].apply(lambda x: x[:lit_model.MAX_LEN]).tolist()
    condition = df.fingerprint.tolist()   
    label = [True] * df.shape[0]
    dataset = CondMolDataset(encoded,condition,label,df.index)
    # train_val split
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator=generator)


    train_loader = DataLoader(train_dataset, batch_size=data.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=data.batch_size, shuffle=False, num_workers=8)
    print("Length of training set: ", train_size)
    print("Length of validation set: ", val_size)


    cond_model= CondFlowMolBERTLitModule(
        model_name= lit_model.model_name,
        vocab_size=VOCAB_SIZE,
        time_dim= 1,
        hidden_dim= lit_model.d_model,
        cond_dim=lit_model.COND_DIM,
        n_layers= lit_model.n_layers,
        n_heads= lit_model.n_heads,
        mlp = lit_model.mlp,
        max_len= lit_model.max_len,
        dropout= lit_model.dropout,
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
    checkpoint = CondFlowMolBERTLitModule.load_from_checkpoint(checkpoint_path)
    cond_model.model.load_state_dict(checkpoint.model.state_dict())
    # model =  checkpoint.model
    ft_strategy = "freeze_encoder"
    create_finetune_strategy(cond_model.model, strategy=ft_strategy, unfreeze_last_n=1)
    early_stop_callback = EarlyStopping(
    monitor="val_loss",      
    patience=4,              
    mode="min",             
    verbose=True)

    checkpoint_dir = data.output_path
    last_ckpt_path = os.path.join(checkpoint_dir, "CFG_FM_last.ckpt")
    resume_ckpt = last_ckpt_path if os.path.exists(last_ckpt_path) else None


    wandb_base_dir = "wandb"
    mlflow_base_dir = 'mlflow_base'
    run_id = None
    name = f'Finetune_{ft_strategy}_msg_context_len={lit_model.max_len}_uncond_prob={lit_model.uncond_prob}_{lit_model.COND_DIM}_r=2_LR={lit_model.lr}_{lit_model.source}_dim={lit_model.d_model+1}_4gpus'
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





    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename= f"{name}_{ft_strategy}_best_cond_val-{{epoch:02d}}-{{cond_validity:.2f}}",
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
        # precision=16,
        devices=1, 
        logger=[wandb_logger,mlflow_logger],
        callbacks=[ckpt_callback,early_stop_callback],
    )

    # --- Train ---
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"Rank {rank}, Local Rank {local_rank}, GPU {torch.cuda.current_device()}")
    if torch.distributed.is_initialized():
        print(f"[RANK {torch.distributed.get_rank()}] world_size={torch.distributed.get_world_size()} "
            f"local_rank={os.environ.get('LOCAL_RANK')} "
            f"cuda={torch.cuda.current_device()}")
    trainer.fit(cond_model, train_loader, val_loader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
