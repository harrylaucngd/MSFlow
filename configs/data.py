import random, numpy as np, torch
import json
# data_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_safe_parsed.parquet'
# data_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_with_chem_props.parquet'
data_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_12_chem_props.parquet'
# data_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_main_wconditions.parquet'
vocab_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/vocab.json'
output_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/checkpoints/'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
SPECIAL_TOKENS = ['[MASK]', '[PAD]']        
MASK, PAD = SPECIAL_TOKENS
with open(vocab_path, "rb") as f:
    vocab = json.load(f)
TOK2ID, ID2TOK = vocab['tok2id'], vocab['id2tok']
TOK2ID = {k: int(v) for k, v in TOK2ID.items()} 
ID2TOK = {int(k): v for k, v in ID2TOK.items()}
# Fix TOK2ID
TOK2ID['[PAD]'] = TOK2ID.pop('')
# Fix ID2TOK
ID2TOK[1] = '[PAD]'
vocab_size = len(TOK2ID)   ##173 tokens

batch_size = 1024  #2048 for all exps, except n_layers=14
MAX_LEN = 72           #1601 orig the longest seq