from modules.cond_lit_model import CondFlowMolBERTLitModule
import torch
from flow_matching.path import MixtureDiscreteProbPath
from utils.sample import cond_generate_mols
from utils.metrics import decode_tokens_to_smiles
from configs import *
import os
from flow_matching.path.scheduler import PolynomialConvexScheduler
torch.serialization.add_safe_globals([MixtureDiscreteProbPath, PolynomialConvexScheduler, torch.nn.modules.loss.CrossEntropyLoss])
cfm_module = CondFlowMolBERTLitModule.load_from_checkpoint()
model = cfm_module.model
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)