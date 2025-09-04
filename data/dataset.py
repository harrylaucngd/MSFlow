import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, RDKFingerprint
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm import tqdm
from sklearn.decomposition import PCA


# pca = PCA(n_components=32)

# Standardize first

class MolDataset(Dataset):
    def __init__(self, encoded_seqs):
        self.data = torch.tensor(np.array(encoded_seqs),dtype=torch.long)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    


def fast_smiles_to_fps(smiles_list, radius=3, fp_size=1024):
    # Ensure input is a list
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    # Initialize Morgan fingerprint generator
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=fp_size)

    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]  # filter invalid

    fps_np = []
    for mol in tqdm(mols, desc='Converting SMILES to fingerprints'):
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((fp_size,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_np.append(arr)

    return np.array(fps_np)  # shape: (n_mols, fp_size)


class CondMolDataset(Dataset):
    def __init__(self, encoded_seqs,conditions,labels, indices = None):
        self.data = torch.tensor(np.array(encoded_seqs),dtype=torch.long)
        # scaler = StandardScaler()
        cond_scaled = fast_smiles_to_fps(conditions)
        self.conditions = torch.tensor(cond_scaled, dtype = torch.float32)
        # self.conditions = torch.tensor(fast_smiles_to_fps(conditions), dtype = torch.float32)
        self.labels =  torch.tensor(np.array(labels), dtype=torch.bool)
        self.indices = indices
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.conditions[idx], self.labels[idx], self.indices[idx]
    




# scaler = StandardScaler()
# cond_scaled = scaler.fit_transform(np.array(conditions))
# pca = PCA(n_components=128)  
# cond_pca = pca.fit_transform(cond_scaled)