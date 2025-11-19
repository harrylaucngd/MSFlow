import random, numpy as np, torch
import json

# data_path = '/hpfs/userws/mqawag/output/data/pretrain/combined_training_data.parquet' #cddd
data_path = '/hpfs/userws/mqawag/output/data/msg_canopus_ms_sum_safe.pkl'
vocab_path = '/home/mqawag/projects/morflow2.0/vocab173.json'
output_path = '/hpfs/userws/mqawag/output/checkpoints/'
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