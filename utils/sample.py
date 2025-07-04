from sampling.samplers import  sample_flow
from sampling.cond_samplers import sample_flow_cond
from configs.lit_model import *
from configs.data import *

def generate_mols(model, num_samples = 1000,steps = 100,path = path, seq_len=MAX_LEN,source_distribution='uniform',mask_token_id=0, device = 'cuda',temperature = temperature):
    # samples = sample_flow_custom(model=model,vocab_size=173,seq_len=128,num_samples=num_samples,source_distribution='masked',mask_token_id=0,device='cuda')
    samples = sample_flow(num_samples=num_samples,model=model,steps = steps, path =path, vocab_size=173,seq_len=seq_len,source_distribution=source_distribution,mask_token_id=mask_token_id,device=device,temperature=temperature)
    return samples


def cond_generate_mols(model, cond, guidance_scale=1.0, num_samples = 1000, steps = 100,path = path, seq_len=MAX_LEN,source_distribution='uniform',mask_token_id=0, device = 'cuda',temperature = temperature):
    # samples = sample_flow_custom(model=model,vocab_size=173,seq_len=128,num_samples=num_samples,source_distribution='masked',mask_token_id=0,device='cuda')
    samples = sample_flow_cond(num_samples=num_samples,model=model,cond = cond,guidance_scale=guidance_scale, steps = steps, path =path, vocab_size=173,seq_len=seq_len,source_distribution=source_distribution,mask_token_id=mask_token_id,device=device,temperature=temperature)
    return samples