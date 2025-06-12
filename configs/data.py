import random, numpy as np, torch
data_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/df_safe_parsed.parquet'
vocab_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/vocab.json'
output_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/checkpoints/'
SPECIAL_TOKENS = ['[MASK]', '']        
MASK, PAD = SPECIAL_TOKENS
batch_size = 2048
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
MAX_LEN = 128 #1601 orig the longest seq