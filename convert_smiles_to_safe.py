import pandas as pd
import safe  
import json
from tqdm import tqdm
from joblib import Parallel, delayed
# from utils import *
import ast
# from configs import * 
from utils.functions import canonicalize


SPECIAL_TOKENS = ['MASK', 'PAD']        
MASK, PAD = SPECIAL_TOKENS

# --- Load existing vocab ---
vocab_path = "/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/vocab.json"
with open(vocab_path, "r") as f:
    vocab_data = json.load(f)

TOK2ID = vocab_data["tok2id"]
MAX_LEN = 128

# Ensure IDs are ints (JSON may have converted them to strings)
TOK2ID = {str(k): int(v) for k, v in TOK2ID.items()}


# --- Functions ---
def encode_row(s):
    """Encode a SMILES string into SAFE + tokens."""
    try:
        s = canonicalize(s)
        encoded = safe.encode(s, ignore_stereo=True)
        tokens = list(safe.split(encoded))
        return encoded, tokens, len(tokens)
    except Exception:
        # if SAFE fails, skip it
        return None

def encode(tokens: list[str], TOK2ID, MAX_LEN) -> list[int]:
    """Convert tokens to IDs, skip sample if any token not in vocab."""
    if any(t not in TOK2ID for t in tokens):
        return None
    if(len(tokens)<= MAX_LEN):
        return [TOK2ID[t] for t in tokens] + [TOK2ID[PAD]] * (MAX_LEN - len(tokens))
    else: 
        return None


# --- Read Data ---
# path = "/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_12_chem_props_canon_smiles.parquet"
# df = pd.read_parquet(path)
path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/ms_data/fingerprints_from_pretrained_encoder/canopus/canopus_test.parquet'
df = pd.read_parquet(path)

# --- Parallel Tokenization ---
results = Parallel(n_jobs=-1)(
    delayed(encode_row)(s) for s in tqdm(df['smiles'])
)

# drop rows where SAFE encoding failed
results = [r for r in results if r is not None]
df = df.iloc[:len(results)].copy()  # align indices
df[['SAFE', 'safe_tokens', 'seq_len']] = pd.DataFrame(results, index=df.index)

# --- Parallel Encoding of Tokens tok->id ---
encoded_results = Parallel(n_jobs=-1)(
    delayed(encode)(tokens, TOK2ID, MAX_LEN) for tokens in tqdm(df['safe_tokens'])
)

# keep only rows with valid encodings (skip unknown-token cases)
valid_mask = [res is not None for res in encoded_results]
df = df[valid_mask].copy()
df['encoded'] = [res for res in encoded_results if res is not None]


# df.to_parquet(
#     "/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_12props_canon_safe_encoding.parquet"
# )
df.to_parquet(
    '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/ms_data/fingerprints_from_pretrained_encoder/canopus/canopus_test_safe_encoded128.parquet'
)
