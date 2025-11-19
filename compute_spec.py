import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import torch
import swifter
from tqdm import tqdm

from modules.cond_lit_model import CondFlowMolBERTLitModule
from utils.metrics import decode_tokens_to_smiles
from utils.sample import cond_generate_mols
from configs import *
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity
from utils.functions import canonicalize, canonicalize_safe
from collections import Counter


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


morgan_gen1 = GetMorganGenerator(radius=2, fpSize=2048)

def compute_tanimoto_to_reference(smiles_list, reference_smiles):
    ref_mol = Chem.MolFromSmiles(reference_smiles)
    if ref_mol is None:
        return np.zeros(len(smiles_list))

    ref_fp = morgan_gen1.GetFingerprint(ref_mol)
    _, fps_list = fast_smiles_to_fps(smiles_list, radius=2, fp_size=2048)
    if fps_list is None:
        return np.zeros(len(smiles_list))

    sims = []
    for fp in fps_list:
        if fp is None:
            sims.append(0.0)
        else:
            sims.append(DataStructs.TanimotoSimilarity(fp, ref_fp))

    return np.array(sims)


def main():
    df_test = pd.read_parquet('/hpfs/userws/mqawag/output/data/canopus_test_128_encoded.parquet') #its called chembl but actually zinc
    ckpt_dir = '/hpfs/userws/mqawag/output/checkpoints/'
    checkpoint_path = 'MSFlow_2.8M_canonical_context_len=128_uncond_prob=0.1_4096_r=2_LR=0.0008_uniform_dim=1536_8gpus_best_cond_val-epoch=53-cond_validity=0.9707.ckpt'
    model = CondFlowMolBERTLitModule.load_from_checkpoint(ckpt_dir + checkpoint_path)
    cfm = model.model
    cfm.eval()
    device = 'cuda'

    # -------------------------
    # Prepare conditions
    # -------------------------
    df_test = df_test[df_test.seq_len<=128]

    # df_test.iloc[:5].append(df_test.iloc[6:], ignore_index=True)
    # df_test['canon_smiles']= df_test['smiles'].swifter.apply(canonicalize_safe)
    query_smiles = df_test.canon_smiles
    fps_list = df_test["fingerprint_ft"].tolist() 
    fps_np = np.array(fps_list, dtype=np.float32) 
    # fps_np, _ = fast_smiles_to_fps(query_smiles, radius=2, fp_size=4096)
    conds = torch.tensor(fps_np, dtype=torch.float32).to(device)

    # -------------------------
    # Generate molecules and compute similarities
    # -------------------------
    all_means = []
    all_top1 = []
    all_top5 = []
    all_top10 = []
    maxsim_smiles_list = []

    all_means_maccs = []
    all_top1_maccs = []
    all_top5_maccs = []
    all_top10_maccs = []


    valid = []
    unique = []
   
    for i in tqdm(range(len(query_smiles)), desc="Generating molecules"):
        cond = conds[i]
        samples = cond_generate_mols(
            cfm,
            cond=cond,
            source_distribution='uniform',
            num_samples=100,
            steps=128,
            device=device,
            temperature=1,
            guidance_scale=1.5,
        )
        _, smiles = decode_tokens_to_smiles(samples, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
        smiles = [canonicalize(s) for s in smiles]
        # smiles = [canonicalize_safe(s) for s in smiles]
        smiles = [s for s in smiles if s is not None]  # remove failed molecules

        torch.cuda.empty_cache()
        if len(smiles) > 0:
            print(f"{len(smiles)} mols are generated")
            sims = compute_tanimoto_to_reference(smiles, query_smiles.iloc[i])
            max_idx = np.argmax(sims)
            maxsim_smiles_list.append(smiles[max_idx])
            sims_sorted = np.sort(sims)[::-1] # descending
            all_means.append(sims.mean())
            all_top1.append(sims_sorted[0])
            all_top5.append(sims_sorted[:5].mean())
            all_top10.append(sims_sorted[:10].mean())

            # MACCS similarity
            sims_maccs = compute_maccs_similarity(query_smiles.iloc[i], smiles)
            sims_maccs_sorted = np.sort(sims_maccs)[::-1]
            all_means_maccs.append(sims_maccs.mean())
            all_top1_maccs.append(sims_maccs_sorted[0])
            all_top5_maccs.append(sims_maccs_sorted[:5].mean())
            all_top10_maccs.append(sims_maccs_sorted[:10].mean())

            valid.append(len(smiles))
            unique.append(len(set(smiles)))
        else:
            print("no smiles")
        # Skip this query, or append default values
            all_means.append(0.0)
            all_top1.append(0.0)
            all_top5.append(0.0)
            all_top10.append(0.0)
            all_means_maccs.append(0.0)
            all_top1_maccs.append(0.0)
            all_top5_maccs.append(0.0)
            all_top10_maccs.append(0.0)
            
            valid.append(0)
            unique.append(0)
            maxsim_smiles_list.append(None)
            continue
            

    results_df = pd.DataFrame({
    "query_smiles": query_smiles[:len(all_means)],
    "mean_tanimoto": all_means,
    "top1_tanimoto": all_top1,
    "top5_tanimoto": all_top5,
    "top10_tanimoto": all_top10,
    "mean_maccs": all_means_maccs,
    "top1_maccs": all_top1_maccs,
    "top5_maccs": all_top5_maccs,
    "top10_maccs": all_top10_maccs,
    "validity": valid,
    "uniqueness": unique,
    "maxsim_smiles": maxsim_smiles_list})

    results_df["smiles_match"] = (results_df["query_smiles"] == results_df["maxsim_smiles"]).astype(int)
    # Save to CSV
    results_df.to_csv("/hpfs/userws/mqawag/output/results/canopus_test.csv", index=False)
    print("✅ Results saved")
    print("Reconstruction_success = ", len(results_df[results_df["smiles_match"]==1]))

if __name__ == "__main__":
    main()