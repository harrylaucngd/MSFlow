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
max_steps = 65000 #100 epochs = 52900 steps
lr = 5e-4
warmup_ratio = 0.07
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scheduler=PolynomialConvexScheduler(n=2.0)
path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=2.0))
# loss = MixturePathGeneralizedKL(path=path)
weighted = True
if(weighted):
    loss = CrossEntropyLoss(reduction='none')
else:
    loss = CrossEntropyLoss()