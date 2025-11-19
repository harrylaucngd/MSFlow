from omegaconf import OmegaConf
import numpy as np
from DiffMS.src.datasets import spec2mol_dataset
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
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles, MolFromInchi, MolToSmiles, RemoveStereochemistry, MolToInchi
warnings.filterwarnings('ignore')
from rdkit import rdBase
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
blocker = rdBase.BlockLogs()

with initialize(version_base=None, config_path="./configs", job_name="test_app"):
    cfg = compose(config_name="config")

dataset_config = cfg["dataset"]

if dataset_config["name"] not in ("canopus", "msg"):
    raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

datamodule = spec2mol_dataset.Spec2MolDataModule(cfg)
data_with_cddds = '../msg_cddd.csv'
df_cddds = pd.read_csv(data_with_cddds)  # path to pandas df with cddds

for idx in range(len(datamodule.test_dataset)):
    datamodule.test_dataset[idx]['graph'][0].y = df_cddds[df_cddds['split'] == 'test'].iloc[idx,6:].to_numpy()








def batch_to_device(batch, device):
    """
    Recursively move a batch (dict, list, tuple, tensor) to the given device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(batch_to_device(v, device) for v in batch)
    else:
        return batch  # leave other types (ints, strings) unchanged
    
            
device = 'cuda' if torch.cuda.is_available() else 'cpu'
           
checkpoint_diff = '../checkpoints/encoder_msg_cddd.pt' 
fp_model = torch.load(checkpoint_diff, map_location=torch.device(device))
# encoder_hidden_dim= 256           # Small Model Default (CANOPUS)
# encoder_magma_modulo= 512         # Small Model Default (CANOPUS)
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
            spectra_dropout=0.0,
            top_layers=1,
            refine_layers=4,
            magma_modulo=encoder_magma_modulo,
        )
encoder.load_state_dict(fp_model['model_state_dict'])
replace_sigmoid_with_tanh(encoder)

results = []
sims = []
cdd = []
encoder.to(device)
encoder.eval()
for data in tqdm(datamodule.test_dataloader_custom(datamodule.test_dataset,bs=1)):
    data = batch_to_device(data,device)  
    outputs,aux = encoder(data)
    results.append(outputs)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    del outputs
    del cddds
    del data
predictions = torch.cat(results)
predictions = predictions.detach().cpu().numpy()
inchis = []
smiles = []
for i, _ in tqdm(enumerate(range(len(datamodule.test_dataset)))):
    inchi = datamodule.test_dataset[i]['graph'][0].inchi
    inchis.append(inchi)
    smiles.append(MolToSmiles(MolFromInchi(inchi),canonical=True, isomericSmiles=False))

df_test = pd.DataFrame({'inchi_keys': inchis,
                   'canon_smiles': smiles,
                   'cddd': [row for row in predictions]
                   })
df_test.to_parquet('../inference/msg_test_cddd.parquet')