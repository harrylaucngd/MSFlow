import numpy as np
from src.datasets import spec2mol_dataset
from hydra import compose, initialize
import torch
from src.mist.models.spectra_encoder import SpectraEncoderGrowing
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from hydra import compose, initialize
import warnings
warnings.filterwarnings('ignore')
from rdkit import rdBase
blocker = rdBase.BlockLogs()
import wandb
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from utils.functions import replace_sigmoid_with_tanh, batch_to_device



wandb.init(project="MIST Pretraining CDDD", name="MS-CDDD")
with initialize(version_base=None, config_path="./configs", job_name="test_app"):
    cfg = compose(config_name="config")

dataset_config = cfg["dataset"]

if dataset_config["name"] not in ("canopus", "msg"):
    raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

datamodule = spec2mol_dataset.Spec2MolDataModule(cfg)

path_data_cddds = f'../{dataset_config["name"]}_cddd.csv'
df_cddds = pd.read_csv(path_data_cddds)

# Storing cddds in the diffms datasets as the graph label
for idx in range(len(datamodule.train_dataset)):
    datamodule.train_dataset[idx]['graph'][0].y = df_cddds[df_cddds['split'] == 'train'].iloc[idx,6:].to_numpy()
for idx in range(len(datamodule.val_dataset)):
    datamodule.val_dataset[idx]['graph'][0].y = df_cddds[df_cddds['split'] == 'val'].iloc[idx,6:].to_numpy()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

encoder_hidden_dim= 512          # Large Model Default (MSG)
encoder_magma_modulo= 2048       # Large Model Default (MSG)
encoder = SpectraEncoderGrowing(
            inten_transform='float',
            inten_prob=0.1,
            remove_prob=0.5,
            peak_attn_layers=2,
            num_heads=8,
            pairwise_featurization=True,
            embed_instrument=False,
            cls_type='ms1',
            set_pooling='cls',
            spec_features='peakformula',
            mol_features='fingerprint',
            form_embedder='pos-cos',
            output_size=512,
            hidden_size=encoder_hidden_dim,
            spectra_dropout=0.1,
            top_layers=1,
            refine_layers=4,
            magma_modulo=encoder_magma_modulo,
        )
replace_sigmoid_with_tanh(encoder)

criterion = nn.MSELoss()
optimizer = optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=1e-5)
best_model = None
num_epochs = 100
batch_size = 512  
sim = 0 
old_sim = 0

encoder.to(device)
for epoch in range(num_epochs):
    encoder.train()
    total_loss = 0.0
    for idx,data in tqdm(enumerate(datamodule.train_dataloader_custom(datamodule.train_dataset,bs=batch_size))):
        data = batch_to_device(data,device)
        cddds = data['graph'].y
        cddds = torch.tensor(cddds, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        outputs,aux = encoder(data)
        loss = criterion(outputs, cddds)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data['graph'].size(0)

    avg_train_loss = total_loss / len(datamodule.train_dataset)
    encoder.eval()
    all_sims = []
    with torch.no_grad():
        total_val_loss = 0.0
        for data in tqdm(datamodule.val_dataloader_custom(datamodule.val_dataset,bs=1)):
            data = batch_to_device(data,device)
            cddds = data['graph'].y
            cddds = torch.tensor(cddds, dtype=torch.float32).to(device)
            outputs,_ = encoder(data) 

            loss = criterion(outputs,cddds)
            total_val_loss  = loss.item() * data['graph'].size(0)
            all_sims.append(F.cosine_similarity(outputs, cddds, dim=-1).detach().cpu().numpy())
    avg_val_loss = total_val_loss / len(datamodule.val_dataset) 
    wandb.log({
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "Cosine Sim": np.mean(np.concatenate(all_sims))
    })

    new_sim = np.mean(np.concatenate(all_sims))
    if(new_sim > old_sim):
        best_model = encoder
        sim = np.mean(np.concatenate(all_sims))
        old_sim = new_sim

torch.save({
    "model_state_dict": best_model.state_dict(),
}, f'../checkpoints/encoder_{dataset_config["name"]}_cddd.pt')

print("✅ Fine-tuning complete and model saved with cosine sim = ", sim)