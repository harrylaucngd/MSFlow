import random, numpy as np, torch
import json

# data_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_12props_canon_safe_encoding.parquet'
# data_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_12props_canon_safe_encoding_filtered_ms.parquet'
data_path = '/home/mqawag/projects/data/combined_data_128_encoded.parquet'
vocab_path = '/home/mqawag/projects/morflow2.0/vocab173.json'
output_path = '/home/mqawag/projects/data/checkpoints/'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
SPECIAL_TOKENS = ['MASK', 'PAD']        
MASK, PAD = SPECIAL_TOKENS
with open(vocab_path, "rb") as f:
    vocab = json.load(f)
TOK2ID, ID2TOK = vocab['tok2id'], vocab['id2tok']
TOK2ID = {k: int(v) for k, v in TOK2ID.items()} 
ID2TOK = {int(k): v for k, v in ID2TOK.items()}

vocab_size = len(TOK2ID)   ##173 tokens

batch_size = 256  
MAX_LEN = 128         