from omegaconf import OmegaConf
import numpy as np
from src.datasets import spec2mol_dataset
from src import utils
from omegaconf import DictConfig
from hydra import compose, initialize
from omegaconf import OmegaConf
import torch
from src.mist.models.spectra_encoder import SpectraEncoderGrowing
from rdkit import DataStructs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from hydra import compose, initialize
from omegaconf import OmegaConf
import warnings
from utils.functions import gumbel_sigmoid, tanimoto_similarity, weighted_bce, batch_to_device
warnings.filterwarnings('ignore')
from rdkit import rdBase
blocker = rdBase.BlockLogs()






with initialize(version_base=None, config_path="./configs", job_name="test_app"):
    cfg = compose(config_name="config")
dataset_config = cfg["dataset"]
if dataset_config["name"] not in ("canopus", "msg"):
    raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))
datamodule = spec2mol_dataset.Spec2MolDataModule(cfg)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_diff = f'../checkpoints/encoder_{dataset_config["name"]}.pt'
fp_model = torch.load(checkpoint_diff, map_location=torch.device(device))

encoder_hidden_dim= 512          # Large Model Default (MSG)
encoder_magma_modulo= 2048       # Large Model Default (MSG)
# encoder_magma_modulo= 512         # Small Model Default (CANOPUS)
# encoder_hidden_dim= 256           # Small Model Default (CANOPUS)
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
            output_size=4096,
            hidden_size=encoder_hidden_dim,
            spectra_dropout=0.1,
            top_layers=1,
            refine_layers=4,
            magma_modulo=encoder_magma_modulo,
        )
encoder.load_state_dict(fp_model)
optimizer = optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=1e-5)
best_model = None
num_epochs = 100
batch_size = 64
old_avg_loss = 1e12
sim = 0.0
encoder.train()
for epoch in range(num_epochs):
    encoder.train()
    total_loss = 0.0
    temp = max(1 * (0.95 ** epoch), 0.3)  # Gumbel temperature annealing

    for data in tqdm(datamodule.train_dataloader(bs=batch_size)):
        data = batch_to_device(data,device)
        fingerprints = torch.tensor(data['mols'].clone(), dtype=torch.float32).to(device)
        optimizer.zero_grad()
        outputs,_ = encoder(data)  

        outputs = gumbel_sigmoid(outputs, temperature=temp, hard=True).to(device)
        pos_w = (fingerprints.numel() - fingerprints.sum())/(fingerprints.sum()+ 1e-8)
        pos_w = torch.clamp(pos_w, max = 10.0)
        loss_bce = weighted_bce(outputs, fingerprints, pos_weight= pos_w)
        loss = loss_bce 
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data['graph'].size(0)

    avg_loss = total_loss / len(datamodule.train_dataset)
    encoder.eval()
    all_sims = []
    with torch.no_grad():
        for data in tqdm(datamodule.val_dataloader(bs=batch_size)):
            data = batch_to_device(data,device)
            fingerprints = torch.tensor(data['mols'].clone(), dtype=torch.float32).to(device)
            outputs,_ = encoder(data)
            all_sims.append(tanimoto_similarity(outputs, fingerprints))

    print(f"Epoch {epoch+1:02d}/{num_epochs} | "
          f"Loss: {avg_loss:.3f} | "
          f"Tanimoto(bin): {np.mean(all_sims):.3f} | "
          f"temp(bin): {np.mean(temp):.4f} | ")
    
    if(old_avg_loss > avg_loss):
        best_model = encoder
        sim = np.mean(all_sims)
        old_avg_loss = avg_loss

torch.save({
    "model_state_dict": best_model.state_dict(),
}, f'../checkpoints/encoder_{dataset_config["name"]}_cddd.pt')

print("✅ Fine-tuning complete and model saved with sim = ", sim)