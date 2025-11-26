from modules.lit_model import FlowMolBERTLitModule
from configs import *
from utils.metrics import compute_smiles_metrics,decode_tokens_to_smiles
from utils.sample import generate_mols
import safe
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
checkpoint_path = '/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/checkpoints/best/MixturePathGeneralizedKL()_L=72_uniform_layers=12_dim=768_best-validity-epoch=112-validity=0.9350-v1.ckpt'
model = FlowMolBERTLitModule.load_from_checkpoint(checkpoint_path)
dfm = model.model
dfm.eval()
import sys
from tqdm import tqdm 
# Open the file in write mode
with open('outputs.txt', 'w') as file:
    # Save the current stdout (console) to restore later
    original_stdout = sys.stdout
    # Redirect stdout to the file
    sys.stdout = file

    # Print statements will be written to the file
   
    val =[]
    div = []
    uni = []
    for step in tqdm([2,5,10,20,40,60,80,100,128,256,512]):
        samples = generate_mols(dfm, num_samples=1000,steps=step, device = 'cuda',temperature=1.0)
        total_samples = len(samples)
        _, smiles = decode_tokens_to_smiles(samples, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
        metrics = compute_smiles_metrics(total_samples=total_samples, decoded_smiles=smiles)
        val.append(metrics['validity'])
        div.append(metrics['diversity'])
        uni.append(metrics['uniqueness'])
    print(val)
    print(div)
    # Restore stdout to the console
    sys.stdout = original_stdout

# This will print to the console again
print("This will be printed on the console.")
