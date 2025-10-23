import random
from collections import Counter

import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors



random.seed(42)

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def read_from_sdf(path):
    res = []
    app = False
    with open(path, 'r') as f:
        for line in tqdm(f.readlines(), desc='Loading SDF structures', leave=False):
            if app:
                res.append(line.strip())
                app = False
            if line.startswith('> <SMILES>'):
                app = True

    return res

def filter(mol):
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
        mol = Chem.MolFromSmiles(smi)

        if "." in smi:
            return False
        
        if Descriptors.MolWt(mol) >= 1500:
            return False
        
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                return False
    except:
        return False
    
    return True

FILTER_ATOMS = {'C', 'N', 'S', 'O', 'F', 'Cl', 'H', 'P'}

def filter_with_atom_types(mol):
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
        mol = Chem.MolFromSmiles(smi)

        if "." in smi:
            return False
        
        if Descriptors.MolWt(mol) >= 1500:
            return False
        
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                return False
            if atom.GetSymbol() not in FILTER_ATOMS:
                return False
    except:
        return False
    
    return True

########## CANOPUS DATASET ##########

# canopus_split = pd.read_csv('../data/canopus/splits/canopus_hplus_100_0.tsv', sep='\t')

# canopus_labels = pd.read_csv('../data/canopus/labels.tsv', sep='\t')
# canopus_labels["name"] = canopus_labels["spec"]
# canopus_labels = canopus_labels[["name", "smiles"]].reset_index(drop=True)

# canopus_labels = canopus_labels.merge(canopus_split, on="name")


path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/ms_data'
# === Load data ===
canopus_split = pd.read_csv(path + '/canopus/splits/canopus_hplus_100_0.tsv', sep='\t')
canopus_labels = pd.read_csv(path +'/canopus/labels.tsv', sep='\t')

# Create unified label DataFrame
canopus_labels["name"] = canopus_labels["spec"]
canopus_labels = canopus_labels[["name", "smiles"]].reset_index(drop=True)
canopus_labels = canopus_labels.merge(canopus_split, on="name")

# === Prepare containers ===
canopus_train = {}
canopus_test = {}
canopus_val = {}

# === Conversion loop ===
for i in tqdm(range(len(canopus_labels)), desc="Converting CANOPUS SMILES to InChI", leave=False):
    smi_orig = canopus_labels.loc[i, "smiles"]
    mol = Chem.MolFromSmiles(smi_orig)
    if mol is None:
        continue

    smi = Chem.MolToSmiles(mol, isomericSmiles=False)  # remove stereochemistry
    mol = Chem.MolFromSmiles(smi)
    inchi = Chem.MolToInchi(mol)

    row = {"smiles": smi, "inchi": inchi}
    split = canopus_labels.loc[i, "split"]

    if split == "train" and filter(mol):
        canopus_train[inchi] = row
    elif split == "test":
        canopus_test[inchi] = row
    elif split == "val":
        canopus_val[inchi] = row

# === Convert dictionaries to DataFrames and save ===
canopus_train_df = pd.DataFrame(canopus_train.values())
canopus_train_df.to_csv(path + "/fp2mol/canopus/preprocessed/canopus_train.csv", index=False)

canopus_test_df = pd.DataFrame(canopus_test.values())
canopus_test_df.to_csv(path + "/fp2mol/canopus/preprocessed/canopus_test.csv", index=False)

canopus_val_df = pd.DataFrame(canopus_val.values())
canopus_val_df.to_csv(path + "/fp2mol/canopus/preprocessed/canopus_val.csv", index=False)

print("✅ CANOPUS preprocessing completed.")
print(f"Train: {len(canopus_train_df)} | Test: {len(canopus_test_df)} | Val: {len(canopus_val_df)}")



# excluded_inchis = set(canopus_test_inchis + canopus_val_inchis)

########## MSG DATASET ##########

msg_split = pd.read_csv(path + '/msg/split.tsv', sep='\t')

msg_labels = pd.read_csv(path + '/msg/labels.tsv', sep='\t')
msg_labels["name"] = msg_labels["spec"]
msg_labels = msg_labels[["name", "smiles"]].reset_index(drop=True)

msg_labels = msg_labels.merge(msg_split, on="name")

msg_train = {}
msg_test = {}
msg_val = {}

for i in tqdm(range(len(msg_labels)), desc="Converting MSG SMILES to InChI", leave=False):
    mol = Chem.MolFromSmiles(msg_labels.loc[i, "smiles"])
    smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    mol = Chem.MolFromSmiles(smi)
    inchi = Chem.MolToInchi(mol)

    row = {"smiles": smi, "inchi": inchi}
    split = msg_labels.loc[i, "split"]

    if split == "train" and filter(mol):
        msg_train[inchi] = row
    elif split == "test":
        msg_test[inchi] = row
    elif split == "val":
        msg_val[inchi] = row

# Save
msg_train_df = pd.DataFrame(msg_train.values())
msg_train_df.to_csv(path + "/fp2mol/msg/preprocessed/msg_train.csv", index=False)

msg_test_df = pd.DataFrame(msg_test.values())
msg_test_df.to_csv(path + "/fp2mol/msg/preprocessed/msg_test.csv", index=False)

msg_val_df = pd.DataFrame(msg_val.values())
msg_val_df.to_csv(path + "/fp2mol/msg/preprocessed/msg_val.csv", index=False)
print("✅ MSG preprocessing completed.")
print(f"Train: {len(msg_train_df)} | Test: {len(msg_test_df)} | Val: {len(msg_val_df)}")


# excluded_inchis.update(msg_test_inchis + msg_val_inchis)
