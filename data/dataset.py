import torch
from torch.utils.data import Dataset
import numpy as np
class MolDataset(Dataset):
    def __init__(self, encoded_seqs):
        self.data = torch.tensor(np.array(encoded_seqs),dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
