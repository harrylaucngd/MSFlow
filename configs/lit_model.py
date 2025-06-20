import torch
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from torch.nn import CrossEntropyLoss

model_name = 'dfm'
source = 'uniform' #masked
d_model=767  # must be divisble by num of heads, +1 for time embedding
vocab = 173
n_layers=12
n_heads=12
mlp = 1024
max_steps = 63000 #100 epochs = 52900 steps
lr = 8e-4
warmup_ratio = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))
loss = MixturePathGeneralizedKL(path=path)
# loss = CrossEntropyLoss()