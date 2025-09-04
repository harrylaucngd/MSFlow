import torch
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from torch.nn import CrossEntropyLoss

model_name = 'dfm'
source = 'uniform' #masked
d_model=767  # must be divisble by num of heads, +1 for time embedding
vocab = 173
COND_DIM = 1024 #1449 #11 #12 with sa_score
n_layers=12
n_heads=12
mlp = 2048
max_steps = 30000 #was 30k for 11 conds of props 
lr = 1e-3 #5e-4
warmup_ratio = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scheduler=PolynomialConvexScheduler(n=1.0)
path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))
# loss = MixturePathGeneralizedKL(path=path)
weighted = False
if(weighted):
    loss = CrossEntropyLoss(reduction='none')
else:
    loss = CrossEntropyLoss()
temperature = 1
uncond_prob = 0
T_STEPS = 100
