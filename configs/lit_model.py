import torch
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from torch.nn import CrossEntropyLoss
from configs.data import MAX_LEN

model_name = 'dfm'
source = 'uniform' #masked
d_model= 1535 # must be divisble by num of heads, +1 for time embedding
vocab = 173
COND_DIM = 4096 #1449 #12 with sa_score
n_layers=12
n_heads=12
mlp = 2048
dropout = 0.4
max_len  = MAX_LEN

max_steps =  1000 #ft msg # 120000 
lr = 1e-6 #5e-4
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
uncond_prob = 0.1
T_STEPS = 100
