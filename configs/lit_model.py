import torch
model_name = 'mfm'
d_model=128  # must be divisble by num of heads
n_layers=12
n_heads=8
max_steps = 20000 #100 epochs = 34800 steps
epochs = 50
batch_size = 4096
lr = 5e-4
warmup_ratio = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
