import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

# Load your data
df = pd.read_parquet("/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_safe_parsed.parquet")
df['mol'] = df['SMILES_standard'].apply(lambda smi: Chem.MolFromSmiles(smi))
df_valid = df[df['mol'].notnull()].copy()

calc = Calculator(descriptors, ignore_3D=True)

def calc_chunk(mols_chunk):
    return calc.pandas(mols_chunk)

# Split molecules list into chunks for parallel processing
mols = df_valid['mol'].tolist()
num_cpus = 16
chunks = np.array_split(mols, num_cpus)

descriptor_dfs = []

with ProcessPoolExecutor(max_workers=num_cpus) as executor:
    futures = [executor.submit(calc_chunk, chunk) for chunk in chunks]

    # Wrap with tqdm for progress bar
    for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating descriptors"):
        descriptor_dfs.append(future.result())

# Concatenate all results
descriptor_df = pd.concat(descriptor_dfs, ignore_index=True)

# Combine with original dataframe
result = pd.concat([df_valid.reset_index(drop=True), descriptor_df.reset_index(drop=True)], axis=1)

result.to_parquet("/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_mordred_descriptors.parquet", index=False)

print(f"✅ Computed {descriptor_df.shape[1]} descriptors for {descriptor_df.shape[0]} molecules.")
