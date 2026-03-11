import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import torch
import os
from modules.cond_lit_model import CondFlowMolBERTLitModule
from utils.metrics import decode_tokens_to_smiles
from utils.sample import cond_generate_mols
from configs import *
from utils.functions import canonicalize
from multiprocessing import Process
from tqdm import tqdm
from collections import Counter

steps = 128

morgan_gen1 = GetMorganGenerator(radius=2, fpSize=2048)

def fast_smiles_to_fps(smiles_list, radius=2, fp_size=2048):
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=fp_size)
    fps_list = []
    for smi in smiles_list:
        if not isinstance(smi, str) or smi.strip() == "" or smi.lower() == "none":
            fps_list.append(None); continue
        mol = Chem.MolFromSmiles(smi)
        fps_list.append(morgan_gen.GetFingerprint(mol) if mol else None)
    if all(fp is None for fp in fps_list):
        return None, None
    fps_np = []
    for fp in fps_list:
        arr = np.zeros((fp_size,), dtype=np.float32)
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, arr)
        fps_np.append(arr)
    return np.array(fps_np), fps_list

def compute_tanimoto_to_reference(smiles_list, reference_smiles):
    ref_mol = Chem.MolFromSmiles(reference_smiles)
    if ref_mol is None:
        return np.zeros(len(smiles_list))
    ref_fp = morgan_gen1.GetFingerprint(ref_mol)
    _, fps_list = fast_smiles_to_fps(smiles_list)
    if fps_list is None:
        return np.zeros(len(smiles_list))
    return np.array([DataStructs.TanimotoSimilarity(fp, ref_fp) if fp else 0.0 for fp in fps_list])


def process_chunk(chunk_df, gpu_id, chunk_id, checkpoint_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")
    print(f"[GPU {gpu_id}] Starting chunk {chunk_id} ({len(chunk_df)} samples)")

    model = CondFlowMolBERTLitModule.load_from_checkpoint(checkpoint_path)
    cfm = model.model
    cfm.eval().to(device)

    query_smiles = chunk_df.canon_smiles
    fps_np = np.stack(chunk_df["cddd"].values).astype(np.float32)
    conds = torch.tensor(fps_np, dtype=torch.float32).to(device)

    results = []
    for i in tqdm(range(len(query_smiles)), desc=f"GPU {gpu_id} - Chunk {chunk_id}"):
        cond = conds[i]
        samples = cond_generate_mols(
            cfm, cond=cond, source_distribution='uniform',
            num_samples=100, steps=steps, device=device,
            temperature=1, guidance_scale=1.5,
        )
        _, smiles = decode_tokens_to_smiles(samples, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
        smiles = [canonicalize(s) for s in smiles if s]
        smiles = [s for s in smiles if s is not None]
        smiles_ordered = [item for item, _ in Counter(smiles).most_common()]

        query_smi = canonicalize(query_smiles.iloc[i])

        if smiles:
            sims = compute_tanimoto_to_reference(smiles, query_smi)
            sims_1 = compute_tanimoto_to_reference([smiles_ordered[0]], query_smi)
            sims_10 = compute_tanimoto_to_reference(smiles_ordered[:10], query_smi)
            result = {
                "query_smiles": query_smi,
                "mean_tanimoto": float(sims.mean()),
                "top1_tanimoto": float(np.sort(sims)[::-1][0]),
                "validity": len(smiles),
                "uniqueness": len(set(smiles)),
                "maxsim_smiles": smiles[int(np.argmax(sims))],
                "sim_top1": float(sims_1[0]) if len(sims_1) > 0 else 0.0,
                "sim_top10": float(np.max(sims_10)) if len(sims_10) > 0 else 0.0,
                "maxsim_smiles_1": smiles_ordered[0],
                "maxsim_smiles_10": smiles_ordered[int(np.argmax(sims_10))] if sims_10.size > 0 else None,
                "acc_top1": 1 if query_smi == smiles_ordered[0] else 0,
                "acc_top10": 1 if query_smi in smiles_ordered[:10] else 0,
                "mces_top1": -1,
                "mces_top10": -1,
            }
        else:
            result = {
                "query_smiles": query_smi, "mean_tanimoto": 0.0, "top1_tanimoto": 0.0,
                "validity": 0, "uniqueness": 0, "maxsim_smiles": None,
                "sim_top1": 0.0, "sim_top10": 0.0, "maxsim_smiles_1": None,
                "maxsim_smiles_10": None, "acc_top1": 0, "acc_top10": 0,
                "mces_top1": 0, "mces_top10": 0,
            }
        results.append(result)

    out_path = f"output/results/canopus_chunk_{chunk_id}_{steps}steps.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"[GPU {gpu_id}] Saved to {out_path}")


def main():
    input_parquet = "output/conditions/canopus_test_cddd.parquet"
    checkpoint_path = "checkpoints/MSFlow/Decoder/MSFlow_cddds.ckpt"
    os.makedirs("output/results", exist_ok=True)

    df_test = pd.read_parquet(input_parquet)
    num_gpus = 1
    chunks = np.array_split(df_test, num_gpus)

    processes = []
    for gpu_id, chunk_df in enumerate(chunks):
        p = Process(target=process_chunk, args=(chunk_df, gpu_id, gpu_id, checkpoint_path))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    all_results = [pd.read_csv(f"output/results/canopus_chunk_{i}_{steps}steps.csv") for i in range(num_gpus)]
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(f"output/results/canopus_benchmark_{steps}steps.csv", index=False)

    print(f"\n=== CANOPUS Benchmark Results ===")
    print(f"Total samples: {len(final_df)}")
    print(f"Acc@1:  {final_df['acc_top1'].mean():.4f}")
    print(f"Acc@10: {final_df['acc_top10'].mean():.4f}")
    print(f"Tanimoto Top1:  {final_df['sim_top1'].mean():.4f}")
    print(f"Tanimoto Top10: {final_df['sim_top10'].mean():.4f}")
    print(f"Mean Tanimoto:  {final_df['mean_tanimoto'].mean():.4f}")
    print(f"Validity (avg valid/100): {final_df['validity'].mean() / 100:.4f}")


if __name__ == "__main__":
    main()
