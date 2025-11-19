import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, RDKFingerprint
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import AllChem
from tqdm import tqdm
from rdkit.Chem import MolFromInchi, MolFromSmiles



class MolDataset(Dataset):
    def __init__(self, encoded_seqs):
        self.data = torch.tensor(np.array(encoded_seqs),dtype=torch.long)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    


def smiles_to_fps(smiles_list, radius=3, fp_size=1024):
    # Ensure input is a list
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    # Initialize Morgan fingerprint generator
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=fp_size)

    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]  # filter invalid

    fps_np = []
    for mol in tqdm(mols, desc='Converting SMILES to bit fingerprints'):
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((fp_size,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_np.append(arr)

    return np.array(fps_np)  


def smiles_to_cfps(smiles_list, radius=2, fp_size=4096):
    # Ensure input is a list
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]  # filter invalid

    fps_np = []
    for mol in tqdm(mols, desc='Converting SMILES to count fingerprints'):
        fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=fp_size)
        arr = np.zeros((0,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_np.append(arr)

    return np.array(fps_np)  


class CondMolDataset(Dataset):
    def __init__(self, encoded_seqs,conditions):
        self.data = torch.tensor(np.array(encoded_seqs),dtype=torch.long)
        # cond_fp = smiles_to_cfps(conditions, radius=2,fp_size=4096)
        # self.conditions = torch.tensor(np.array(cond_fp), dtype = torch.float32)
        self.conditions = torch.tensor(conditions, dtype = torch.float32)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.conditions[idx]
    



 # scaler = StandardScaler()
 # cond_scaled = scaler.fit_transform(conditions.to_numpy())