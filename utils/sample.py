from sampling.samplers import sample_flow_custom, sample_flow
from configs.lit_model import *
from configs.data import *
def generate_mols(model, num_samples = 1000,path = path, seq_len=MAX_LEN,source_distribution='masked',mask_token_id=0, device = 'cuda'):
    # samples = sample_flow_custom(model=model,vocab_size=173,seq_len=128,num_samples=num_samples,source_distribution='masked',mask_token_id=0,device='cuda')
    samples = sample_flow(num_samples=num_samples,model=model,path =path, vocab_size=173,seq_len=seq_len,source_distribution=source_distribution,mask_token_id=mask_token_id,device=device)
    return samples