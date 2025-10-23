import pandas as pd
import safe  
import json
from tqdm import tqdm
from joblib import Parallel, delayed
# from utils import *
import ast
# from configs import * 
from utils.functions import canonicalize
import swifter

SPECIAL_TOKENS = ['MASK', 'PAD']        
MASK, PAD = SPECIAL_TOKENS

# --- Load existing vocab ---
vocab_path = "/home/mqawag/projects/morflow2.0/vocab173.json"
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
    except safe.SAFEFragmentationError:
        # if SAFE fails, skip it
        return None
    except safe.SAFEEncodeError:
        # if SAFE fails, skip it
        return None
    except safe.SAFEDecodeError:
        # if SAFE fails, skip it
        return None
    except:
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

path = '/home/mqawag/projects/data/combined_data.parquet'
df = pd.read_parquet(path)

df['results'] = df['canon_smiles'].swifter.apply(encode_row)
df = df[df['results'].notnull()].copy()
df[['SAFE', 'safe_tokens', 'seq_len']] = pd.DataFrame(df['results'].tolist(), index=df.index)
df['encoded'] = df['safe_tokens'].swifter.apply(lambda tokens: encode(tokens,TOK2ID, MAX_LEN))
df = df[df['encoded'].notnull()].copy()

df.to_parquet(
    '/home/mqawag/projects/data/combined_data_128_encodedd.parquet'
)
