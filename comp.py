import logging
from typing import List, Dict, Collection, Optional, Union
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors
from rdkit.Chem import Lipinski
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import swifter
from rdkit.Chem.rdMolDescriptors import CalcNumAliphaticRings
# Mute RDKit logging
RDLogger.logger().setLevel(RDLogger.CRITICAL)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def get_mol(smiles_str: Collection[str]) -> List[Optional[Chem.Mol]]:

    if smiles_str is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles_str)
    except Exception:
        mol = None
    return mol

def is_valid(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None

def is_sanitize_valid(mol: Chem.Mol) -> bool:
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False
    

def tokenize_and_pad_smiles_safe(smiles: str, max_len: int = 72) -> list[str]:
    from safe import encode, split, SAFEFragmentationError

    try:
        encoded = encode(smiles)
        tokens = split(encoded)
        padded = tokens[:max_len] + [''] * (max_len - len(tokens))
        return padded
    except Exception as e:  # Can also catch SAFEFragmentationError specifically
        # Return a special token list or empty padding if encoding fails
        return ['<ERR>'] + [''] * (max_len - 1)

def count_fragments(tokens: list[str]) -> int:
    return tokens.count('.') + 1 

def compute_properties_for_single(smiles: str) -> Dict[str, Union[float, bool]]:
    m = get_mol(smiles) # Get single mol
    return {
        "SMILES": smiles,
        "Valid": is_valid(smiles),
        "Sanitize": is_sanitize_valid(m),
        "num_atoms": m.GetNumAtoms() if m is not None else float('nan'),
        "QED": QED.qed(m) if m is not None else float('nan'),
        "MW": Descriptors.MolWt(m) if m is not None else float('nan'),
        "logP": Descriptors.MolLogP(m) if m is not None else float('nan'),
        "TPSA": Descriptors.TPSA(m) if m is not None else float('nan'),
        "NumHeavyAtoms": m.GetNumHeavyAtoms() if m is not None else float('nan'),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(m) if m is not None else float('nan'),
        "NumAromaticRings": Descriptors.NumAromaticRings(m) if m is not None else float('nan'),
        "NumAliphaticRings": CalcNumAliphaticRings(m) if m is not None else float('nan'),
        "num_H_donors": Lipinski.NumHDonors(m) if m is not None else float('nan'),
        "num_h_acceptors": Lipinski.NumHAcceptors(m) if m is not None else float('nan'),
    }

def compute_chem_properties(smiles_list: List[str]) -> Dict[str, List[Union[float, bool]]]:
    results = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        for res in tqdm(executor.map(compute_properties_for_single, smiles_list), total=len(smiles_list)):
            results.append(res)
    print("Aggregating results across CPUs ...")
    aggregated = {key: [res[key] for res in results] for key in results[0].keys()}
    return aggregated

def save_properties_to_dataframe(
    smiles_list: List[str], 
    properties_dict: Dict[str, List[Union[float, bool]]],
    filename: str = "df_with_chem_props.parquet"
) -> pd.DataFrame:
    df = pd.DataFrame(properties_dict)
    df.insert(0, "SMILES", smiles_list)
    df.to_parquet(filename)
    return df

# def main(input_csv_path: str, output_path: str = "df_with_chem_props.parquet"):
#     # Load DataFrame
#     df = pd.read_parquet(input_csv_path)
#     smiles_list = df['SMILES_standard'].tolist()
#     print("Computing ...")
#     # Compute properties
#     properties = compute_chem_properties(smiles_list)
#     print("Computing is over ...")
#     # Save new dataframe with properties
#     save_properties_to_dataframe(smiles_list, properties, filename=output_path)

#     print(f"Computed properties saved to {output_path}")



def main(input_csv_path: str, output_path: str = "df_with_chem_props.parquet"):
    df = pd.read_parquet(input_csv_path)
    print("Computing properties with swifter ...")
    
    df_props = df['smiles'].swifter.apply(compute_properties_for_single)
    properties_df = pd.json_normalize(df_props)
    
    df = pd.concat([df, properties_df], axis=1)
    
    df.to_parquet(output_path)
    print(f"Computed properties saved to {output_path}")

if __name__ == "__main__":
    # input_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_safe_parsed.parquet'
    # output_path ='/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_with_chem_props.parquet'
    input_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/df5k_generated4.parquet'
    output_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/df5k_generated4_with_props.parquet'

    main(input_path,output_path)