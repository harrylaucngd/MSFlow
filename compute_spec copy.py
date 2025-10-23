import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import torch
import swifter
from modules.cond_lit_model import CondFlowMolBERTLitModule
from utils.metrics import decode_tokens_to_smiles
from utils.sample import cond_generate_mols
from configs import *
from rdkit.Chem import rdFMCS
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity
from utils.functions import canonicalize, canonicalize_safe
import os
from multiprocessing import Process
from tqdm import tqdm



def compute_maccs_similarity(query_smiles, target_smiles_list):
    """
    Compute MACCS fingerprint Tanimoto similarity between a query molecule
    and a list of target molecules.

    Args:
        query_smiles (str): SMILES string of the query molecule.
        target_smiles_list (list of str): List of SMILES strings for target molecules.

    Returns:
        list of float: Tanimoto similarity values for each target.
    """
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        raise ValueError(f"Invalid query SMILES: {query_smiles}")

    query_fp = MACCSkeys.GenMACCSKeys(query_mol)
    similarities = []

    for smi in target_smiles_list:
        target_mol = Chem.MolFromSmiles(smi)
        if target_mol is None:
            similarities.append(0.0)
            continue

        target_fp = MACCSkeys.GenMACCSKeys(target_mol)
        sim = TanimotoSimilarity(query_fp, target_fp)
        similarities.append(sim)

    return np.array(similarities)

def compute_mcs_fractions(query_smiles, target_smiles_list):
    """
    Compute MCS similarity (fraction of query atoms covered) between a query molecule 
    and a list of target molecules.

    Args:
        query_smiles (str): SMILES string of the query molecule.
        target_smiles_list (list of str): List of SMILES strings for target molecules.

    Returns:
        list of float: MCS similarity values (fraction of query atoms covered) for each target.
    """
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        raise ValueError(f"Invalid query SMILES: {query_smiles}")
    
    n_query_atoms = query_mol.GetNumAtoms()
    similarities = []

    for smi in target_smiles_list:
        target_mol = Chem.MolFromSmiles(smi)
        if target_mol is None:
            similarities.append(0.0)
            continue
        
        mcs_result = rdFMCS.FindMCS([query_mol, target_mol],
                                    bondCompare=rdFMCS.BondCompare.CompareAny,
                                    atomCompare=rdFMCS.AtomCompare.CompareAny,
                                    ringMatchesRingOnly=True,
                                    completeRingsOnly=True)
        
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        n_mcs_atoms = mcs_mol.GetNumAtoms() if mcs_mol else 0
        fraction_query_covered = n_mcs_atoms / n_query_atoms if n_query_atoms > 0 else 0.0
        similarities.append(fraction_query_covered)
    
    return np.array(similarities)

# -------------------------
# Fingerprint helper
# -------------------------
def fast_smiles_to_fps(smiles_list, radius=3, fp_size=1024):
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=fp_size)
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    fps_list = []
    for smi in smiles_list:
        if not isinstance(smi, str) or smi.strip() == "" or smi.lower() == "none":
            fps_list.append(None)
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps_list.append(None)
            continue

        fp = morgan_gen.GetFingerprint(mol)
        fps_list.append(fp)

    if all(fp is None for fp in fps_list):
        return None, None

    fps_np = []
    for fp in fps_list:
        if fp is None:
            fps_np.append(np.zeros((fp_size,), dtype=np.float32))
        else:
            arr = np.zeros((fp_size,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps_np.append(arr)

    return np.array(fps_np), fps_list


morgan_gen1 = GetMorganGenerator(radius=2, fpSize=1024)

def compute_tanimoto_to_reference(smiles_list, reference_smiles):
    ref_mol = Chem.MolFromSmiles(reference_smiles)
    if ref_mol is None:
        return np.zeros(len(smiles_list))

    ref_fp = morgan_gen1.GetFingerprint(ref_mol)
    _, fps_list = fast_smiles_to_fps(smiles_list, radius=2, fp_size=1024)
    if fps_list is None:
        return np.zeros(len(smiles_list))

    sims = []
    for fp in fps_list:
        if fp is None:
            sims.append(0.0)
        else:
            sims.append(DataStructs.TanimotoSimilarity(fp, ref_fp))

    return np.array(sims)





# import your compute_* functions here (as in your script)

def process_chunk(chunk_df, gpu_id, chunk_id, ckpt_dir, checkpoint_path):
    """Worker process that runs on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")

    print(f"[GPU {gpu_id}] Starting chunk {chunk_id} ({len(chunk_df)} samples)")

    # Load model on this GPU
    model = CondFlowMolBERTLitModule.load_from_checkpoint(ckpt_dir + checkpoint_path)
    cfm = model.model
    cfm.eval().to(device)

    chunk_df['canon_smiles'] = chunk_df['smiles'].apply(canonicalize_safe)
    query_smiles = chunk_df.canon_smiles
    fps_list = chunk_df["fingerprint"].tolist()
    fps_np = np.array(fps_list, dtype=np.float32)
    conds = torch.tensor(fps_np, dtype=torch.float32).to(device)

    results = []

    for i in tqdm(range(len(query_smiles)), desc=f"GPU {gpu_id} - Chunk {chunk_id}"):
        cond = conds[i]
        samples = cond_generate_mols(
            cfm,
            cond=cond,
            source_distribution='uniform',
            num_samples=100,
            steps=128,
            device=device,
            temperature=1.0,
            guidance_scale=1.5,
        )

        _, smiles = decode_tokens_to_smiles(samples, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
        smiles = [canonicalize_safe(canonicalize(s)) for s in smiles if s]

        if not smiles:
            results.append({
                "query_smiles": query_smiles.iloc[i],
                "mean_tanimoto": 0.0,
                "top1_tanimoto": 0.0,
                "validity": 0,
                "uniqueness": 0,
                "maxsim_smiles": None
            })
            continue

        sims = compute_tanimoto_to_reference(smiles, query_smiles.iloc[i])
        sims_sorted = np.sort(sims)[::-1]
        maxsim_smiles = smiles[np.argmax(sims)]

        results.append({
            "query_smiles": query_smiles.iloc[i],
            "mean_tanimoto": sims.mean(),
            "top1_tanimoto": sims_sorted[0],
            "validity": len(smiles),
            "uniqueness": len(set(smiles)),
            "maxsim_smiles": maxsim_smiles
        })

    results_df = pd.DataFrame(results)
    out_path = f"results/FP/mass_spec/chunk_{chunk_id}_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"[GPU {gpu_id}] ✅ Saved results to {out_path}")


def main():
    df_test = pd.read_parquet(
        '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/ms_data/fingerprints_from_pretrained_encoder/msg/msg_test_safe_encoded.parquet'
    )
    df_test = df_test[df_test.seq_len <= 128][:8000]  # example

    ckpt_dir = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/checkpoints/'
    checkpoint_path = 'new/context_128/CFG-MFP_1.4M_canonical_context_len=128_uncond_prob=0.1_4096_r=2_LR=0.0001_uniform_dim=1536_4gpusCFG_best_val_loss-epoch=81-cond_validity=0.8516.ckpt'

    # Split into 8 roughly equal chunks
    num_gpus = 8
    chunks = np.array_split(df_test, num_gpus)

    processes = []
    for gpu_id, chunk_df in enumerate(chunks):
        p = Process(target=process_chunk, args=(chunk_df, gpu_id, gpu_id, ckpt_dir, checkpoint_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge results
    all_results = []
    for i in range(num_gpus):
        df_chunk = pd.read_csv(f"results/FP/mass_spec/chunk_{i}_results.csv")
        all_results.append(df_chunk)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("results/FP/mass_spec/final_results.csv", index=False)
    print("✅ All chunks done and merged.")


if __name__ == "__main__":
    main()
