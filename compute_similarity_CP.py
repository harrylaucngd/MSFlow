import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from modules.cond_lit_model import CondFlowMolBERTLitModule
from data import CondMolDataset
from utils.metrics import decode_tokens_to_smiles
from utils.sample import cond_generate_mols
from configs import *
from sklearn.preprocessing import StandardScaler

# -------------------------
# Fingerprint helper
# -------------------------
morgan_gen1 = GetMorganGenerator(radius=3, fpSize=1024)

def fast_smiles_to_fps(smiles_list, radius=3, fp_size=1024):
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=fp_size)
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]
    fps_list = []
    for mol in mols:
        fp = morgan_gen.GetFingerprint(mol)
        fps_list.append(fp)
    # convert to numpy array
    fps_np = []
    for fp in fps_list:
        arr = np.zeros((fp_size,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_np.append(arr)
    return np.array(fps_np), fps_list

def compute_tanimoto_to_reference(smiles_list, reference_smiles):
    ref_mol = Chem.MolFromSmiles(reference_smiles)
    if ref_mol is None:
        return np.zeros(len(smiles_list))
    ref_fp = morgan_gen1.GetFingerprint(ref_mol)
    _, fps_list = fast_smiles_to_fps(smiles_list)
    sims = [DataStructs.TanimotoSimilarity(fp, ref_fp) for fp in fps_list]
    return np.array(sims)
def main():
    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_parquet("/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_main_wconditions.parquet")   
    # -------------------------
    # Reproducible train/val split
    # -------------------------
    df = df[df["seq_len"] <= 72]
    df_conditioned = df[df['has_condition']==True]   #train with samples that have conditions only
    encoded = df_conditioned["encoded"].apply(lambda x: x[:data.MAX_LEN]).tolist()
    condition = df_conditioned.iloc[:,-1450:-1]#[:,-1451:-2]  #conditions_are 11 chem_props
    label = [True] * df.shape[0]
    dataset = CondMolDataset(encoded,condition,label,df_conditioned.index)
    # train_val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator=generator)
    train_indices = [item[3] for item in train_dataset]
    val_indices = [item[3] for item in val_dataset]
    df_train = df.loc[train_indices]
    df_val = df.loc[val_indices]
    
    # -------------------------
    # Load pretrained model
    # -------------------------
    checkpoint_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/checkpoints/CP_adaptiveLN_1449_CrossEntropyLoss()_L=72_uniform_layers=12_dim=768fine_tuned_best_val_loss-epoch=52-cond_validity=0.5771.ckpt'
    model = CondFlowMolBERTLitModule.load_from_checkpoint(checkpoint_path)
    cfm = model.model
    cfm.eval()
    device = 'cuda'

    # -------------------------
    # Prepare conditions
    # -------------------------
    train_cp_conditions = np.array(df_train.iloc[:,-1450:-1])
    val_cp_conditions = np.array(df_val.iloc[:,-1450:-1])
    scaler = StandardScaler()
    train_cond_scaled = scaler.fit_transform(train_cp_conditions)
    val_cond_scaled = scaler.transform(val_cp_conditions)
    conditions = torch.tensor(val_cond_scaled, dtype = torch.float32)
    query_smiles = df_val['SMILES_standard']

    # -------------------------
    # Generate molecules and compute similarities
    # -------------------------
    all_means = []
    all_top1 = []
    all_top5 = []
    all_top10 = []
    valid = []
    unique = []
    for i in tqdm(range(len(query_smiles)), desc="Generating molecules"):
        cond = conditions[i]
        samples = cond_generate_mols(
            cfm,
            cond=cond,
            source_distribution='uniform',
            guidance_scale=0.0,
            num_samples=100,
            steps=100,
            device=device,
            temperature=1
        )
        _, smiles = decode_tokens_to_smiles(samples, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
        torch.cuda.empty_cache()
        if len(smiles) > 0:
            sims = compute_tanimoto_to_reference(smiles, query_smiles.iloc[i])
        sims_sorted = np.sort(sims)[::-1]  # descending
        all_means.append(sims.mean())
        all_top1.append(sims_sorted[0])
        all_top5.append(sims_sorted[:5].mean())
        all_top10.append(sims_sorted[:10].mean())
        valid.append(len(smiles))
        unique.append(len(set(smiles)))

    # -------------------------
    # Report results
    # -------------------------
    print("Mean Tanimoto over all queries:", np.mean(all_means))
    print("Mean top-1 Tanimoto:", np.mean(all_top1))
    print("Mean top-5 Tanimoto:", np.mean(all_top5))
    print("Mean top-10 Tanimoto:", np.mean(all_top10))

    results_df = pd.DataFrame({
        "query_smiles": query_smiles[:len(all_means)],
        "mean_tanimoto": all_means,
        "top1_tanimoto": all_top1,
        "top5_tanimoto": all_top5,
        "top10_tanimoto": all_top10,
        "validity": valid,
        "uniqueness": unique,

    })

    # Save to CSV
    results_df.to_csv("CP_conditional_generation_tanimoto_results.csv", index=False)
    print("✅ Results saved to 'CP_conditional_generation_tanimoto_results.csv'")


if __name__ == "__main__":
    main()