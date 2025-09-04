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

# -------------------------
# Fingerprint helper
# -------------------------
morgan_gen1 = GetMorganGenerator(radius=3, fpSize=1024)

def fast_smiles_to_fps(smiles_list, radius=3, fp_size=1024):
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]
    fps_list = []
    for mol in mols:
        fp = morgan_gen1.GetFingerprint(mol)
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
    df = pd.read_parquet('/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_with_chem_props.parquet')
    df = df[df["seq_len"] <= 72].reset_index(drop=True)
    df = df.iloc[:100000]
    df["original_index"] = df.index

    encoded = df["encoded"].apply(lambda x: x[:data.MAX_LEN]).tolist()
    condition = df.SMILES_standard
    label = [True]*len(df)
    indices = df["original_index"].tolist()

    dataset = CondMolDataset(encoded, condition, label, indices)

    # -------------------------
    # Reproducible train/val split
    # -------------------------
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_indices = [item[3] for item in train_dataset]
    val_indices = [item[3] for item in val_dataset]
    df_train = df.loc[train_indices].reset_index(drop=True)
    df_val = df.loc[val_indices].reset_index(drop=True)

    # -------------------------
    # Load pretrained model
    # -------------------------
    checkpoint_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/checkpoints/last-v27.ckpt'
    model = CondFlowMolBERTLitModule.load_from_checkpoint(checkpoint_path)
    cfm = model.model
    cfm.eval()
    device = 'cuda'

    # -------------------------
    # Prepare conditions
    # -------------------------
    query_smiles = df_val.SMILES_standard[:10]
    fps_np, _ = fast_smiles_to_fps(query_smiles)
    conds = torch.tensor(fps_np, dtype=torch.float32).to(device)

    # -------------------------
    # Generate molecules and compute similarities
    # -------------------------
    all_means = []
    all_top1 = []
    all_top5 = []
    all_top10 = []

    for i in tqdm(range(len(query_smiles)), desc="Generating molecules"):
        cond = conds[i]
        samples = cond_generate_mols(
            cfm,
            cond=cond,
            source_distribution='uniform',
            guidance_scale=0.0,
            num_samples=500,
            steps=100,
            device=device,
            temperature=0.5
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
        "top10_tanimoto": all_top10
    })

    # Save to CSV
    results_df.to_csv("conditional_generation_tanimoto_results.csv", index=False)
    print("✅ Results saved to 'conditional_generation_tanimoto_results.csv'")


if __name__ == "__main__":
    main()