import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

# === Load SMILES & Molecules ===
df = pd.read_parquet("/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_safe_parsed.parquet")
df = df[df.seq_len<=72]
df = df.iloc[:100000,:]
df['mol'] = df['SMILES_standard'].apply(lambda smi: Chem.MolFromSmiles(smi))
df_valid = df[df['mol'].notnull()].copy()
print(f"\n✅ Calculating descriptors table for : {df_valid.shape[0]} molecules.")


# === STEP 1: Robust Descriptor Calculation with Try-Except per Molecule ===
def calc_chunk(mols_chunk):

    calc = Calculator(descriptors, ignore_3D=True)
    results = []
    smiles_list=[]
    for mol in mols_chunk:
        try:
            desc = calc.pandas([mol])  # single molecule at a time
            results.append(desc)
            smiles_list.append(Chem.MolToSmiles(mol))
        except Exception as e:
            try:
                smi = Chem.MolToSmiles(mol)
            except:
                smi = "<invalid>"
            print(f"❌ Error computing descriptors for molecule: {smi}\n    {type(e).__name__}: {e}")
    if results:
        df = pd.concat(results, ignore_index=True)
        df.insert(0, "SMILES_standard", smiles_list)
        return df
    else:
        return pd.DataFrame()

# === Run in Parallel ===
mols = df_valid['mol'].tolist()
num_cpus = 16
# chunks = np.array_split(mols, num_cpus)
chunk_size = 1000 # or 1000 if memory allows
chunks = [mols[i:i + chunk_size] for i in range(0, len(mols), chunk_size)]
print("Number of chunks: ", len(chunks))

descriptor_dfs = []

with ProcessPoolExecutor(max_workers=num_cpus) as executor:
    futures = [executor.submit(calc_chunk, list(chunk)) for chunk in chunks]
    print("length of futures:", len(futures))
    for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating descriptors"):
        try:
           descriptor_dfs.append(future.result())
        except Exception as e:
            print(f"❌ Chunk failed with error: {type(e).__name__}: {e}")
descriptor_df = pd.concat(descriptor_dfs, ignore_index=True)

# Apply row filtering
desc_array = descriptor_df.drop(columns=['SMILES_standard']).to_numpy(dtype=np.float64)
row_mask = np.isfinite(desc_array).all(axis=1)

# Cleaned final dataframe with SMILES + descriptors
final_df = descriptor_df.loc[row_mask].reset_index(drop=True)

# Save
output_path = "/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_mordred_descriptors.parquet"
final_df.to_parquet(output_path, index=False)

print(f"\n✅ Final descriptor table saved: {final_df.shape[1] - 1} descriptors × {final_df.shape[0]} molecules + SMILES.")
