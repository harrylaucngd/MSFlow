import os
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, MLFlowLogger

from data import CondMolDataset
from modules.spectra_lit_model import SpectraFlowMolBERTLitModule
from configs import data, lit_model
from configs.data import TOK2ID

torch.set_float32_matmul_precision('medium')

def main():
    # ----------------------------
    # Dataset preparation
    # ----------------------------
    df = pd.read_parquet(data.data_path)
    df = df[df["seq_len"] <= data.MAX_LEN]

    generator = torch.Generator().manual_seed(42)
    encoded = df["encoded"].apply(lambda x: x[:lit_model.MAX_LEN]).tolist()
    condition = df.canon_smiles
    label = [True] * df.shape[0]

    dataset = CondMolDataset(encoded, condition, label, df.index)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=data.batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=data.batch_size, shuffle=False, num_workers=16)

    # ----------------------------
    # Logger and Callbacks
    # ----------------------------
    wandb_logger = WandbLogger(project="morflow", save_dir="wandb", name="stacked_model_finetune")
    mlflow_logger = MLFlowLogger(experiment_name="morflow", run_name="stacked_model_finetune", tracking_uri="file:./mlruns")

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
    ckpt_callback = ModelCheckpoint(dirpath=data.output_path, filename='stacked_best-{epoch:02d}-{val_loss:.4f}', monitor="val_loss", mode="min", save_top_k=1, save_last=True)

    # ----------------------------
    # Flow model config for initialization
    # ----------------------------
    flow_config = dict(
        model_name=lit_model.model_name,
        vocab_size=len(TOK2ID),
        time_dim=1,
        hidden_dim=lit_model.d_model,
        cond_dim=lit_model.COND_DIM,
        n_layers=lit_model.n_layers,
        n_heads=lit_model.n_heads,
        mlp=lit_model.mlp,
        max_len=lit_model.max_len,
        dropout=lit_model.dropout,
        uncond_prob=lit_model.uncond_prob,
        lr=lit_model.lr,
        warmup_ratio=lit_model.warmup_ratio,
        pad_token_id=TOK2ID[data.PAD],
        mask_token_id=TOK2ID[data.MASK],
        device=lit_model.device,
        source=lit_model.source,
        scheduler=lit_model.scheduler,
        path=lit_model.path,
        loss_fn=lit_model.loss,
        weighted=lit_model.weighted
    )

    # ----------------------------
    # Initialize stacked model
    # ----------------------------
    model = SpectraFlowMolBERTLitModule(
        encoder_ckpt='/home/icb/ghaith.mqawass/projs/morflow2.0/pretrained_mist/checkpoints/encoder_canopus.pt',
        encoder_hidden_dim=256,
        encoder_magma_modulo=512,
        flow_config=flow_config
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # ----------------------------
    # Trainer
    # ----------------------------
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        strategy="ddp",
        max_steps=lit_model.max_steps,
        callbacks=[ckpt_callback, early_stop_callback],
        logger=[wandb_logger, mlflow_logger],
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
