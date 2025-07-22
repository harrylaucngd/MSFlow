import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
# Standardize first

class MolDataset(Dataset):
    def __init__(self, encoded_seqs,conditions):
        self.data = torch.tensor(np.array(encoded_seqs),dtype=torch.long)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    



# Set up generator once
morgan_gen = GetMorganGenerator(radius=2, fpSize=512)

def fast_smiles_to_fps(smiles_list, n_bits=512, radius=2):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]  # filter invalid

    n_mols = len(mols)
    fps = np.zeros((n_mols, n_bits), dtype=np.uint8)

    for i, mol in enumerate(mols):
        fp = morgan_gen.GetFingerprint(mol)
        fps[i]= np.array(fp)  # faster than ConvertToNumpyArray

    return fps


class CondMolDataset(Dataset):
    def __init__(self, encoded_seqs,conditions,labels, indices = None):
        self.data = torch.tensor(np.array(encoded_seqs),dtype=torch.long)
        # scaler = StandardScaler()
        # cond_scaled = scaler.fit_transform(np.array(conditions))
        # self.conditions = torch.tensor(np.array(cond_scaled), dtype = torch.float32)
        self.conditions = torch.tensor(fast_smiles_to_fps(conditions), dtype = torch.float32)
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