import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize first

class MolDataset(Dataset):
    def __init__(self, encoded_seqs,conditions):
        self.data = torch.tensor(np.array(encoded_seqs),dtype=torch.long)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CondMolDataset(Dataset):
    def __init__(self, encoded_seqs,conditions,labels):
        self.data = torch.tensor(np.array(encoded_seqs),dtype=torch.long)
        scaler = StandardScaler()
        cond_scaled = scaler.fit_transform(np.array(conditions))
        # pca = PCA(n_components=128)  
        # cond_pca = pca.fit_transform(cond_scaled)
        self.conditions = torch.tensor(np.array(cond_scaled), dtype = torch.float32)
        self.labels =  torch.tensor(np.array(labels), dtype=torch.bool)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.conditions[idx], self.labels[idx]