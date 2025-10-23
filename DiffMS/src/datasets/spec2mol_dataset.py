import os
import pathlib
from typing import Any, Sequence
import random

import numpy as np
import pandas as pd
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import Descriptors
import torch
import torch.nn.functional as F
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

from src.mist.data import datasets, splitter, featurizers
# from src.datasets.abstract_dataset import  MolecularDataModule
# import src.utils as utils
from src.mist.data.datasets import get_paired_loader_graph
# from src.datasets.abstract_dataset import ATOM_TO_VALENCY, ATOM_TO_WEIGHT
ATOM_TO_VALENCY = {
    'H': 1,
    'He': 0,
    'Li': 1,
    'Be': 2,
    'B': 3,
    'C': 4,
    'N': 3,
    'O': 2,
    'F': 1,
    'Ne': 0,
    'Na': 1,
    'Mg': 2,
    'Al': 3,
    'Si': 4,
    'P': 3,
    'S': 2,
    'Cl': 1,
    'Ar': 0,
    'K': 1,
    'Ca': 2,
    'Sc': 3,
    'Ti': 4,
    'V': 5,
    'Cr': 2,
    'Mn': 7,
    'Fe': 2,
    'Co': 3,
    'Ni': 2,
    'Cu': 2,
    'Zn': 2,
    'Ga': 3,
    'Ge': 4,
    'As': 3,
    'Se': 2,
    'Br': 1,
    'Kr': 0,
    'Rb': 1,
    'Sr': 2,
    'Y': 3,
    'Zr': 2,
    'Nb': 2,
    'Mo': 2,
    'Tc': 6,
    'Ru': 2,
    'Rh': 3,
    'Pd': 2,
    'Ag': 1,
    'Cd': 1,
    'In': 1,
    'Sn': 2,
    'Sb': 3,
    'Te': 2,
    'I': 1,
    'Xe': 0,
    'Cs': 1,
    'Ba': 2,
    'La': 3,
    'Ce': 3,
    'Pr': 3,
    'Nd': 3,
    'Pm': 3,
    'Sm': 2,
    'Eu': 2,
    'Gd': 3,
    'Tb': 3,
    'Dy': 3,
    'Ho': 3,
    'Er': 3,
    'Tm': 2,
    'Yb': 2,
    'Lu': 3,
    'Hf': 4,
    'Ta': 3,
    'W': 2,
    'Re': 1,
    'Os': 2,
    'Ir': 1,
    'Pt': 1,
    'Au': 1,
    'Hg': 1,
    'Tl': 1,
    'Pb': 2,
    'Bi': 3,
    'Po': 2,
    'At': 1,
    'Rn': 0,
    'Fr': 1,
    'Ra': 2,
    'Ac': 3,
    'Th': 4,
    'Pa': 5,
    'U': 2,
}

def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]

atom_decoder = ['C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H']
valency = [ATOM_TO_VALENCY.get(atom, 0) for atom in atom_decoder]


class Spec2MolDataModule():
    def __init__(self, cfg):
        self.remove_h = False
        self.datadir = cfg.dataset.datadir
        self.filter_dataset = cfg.dataset.filter
        self.train_smiles = []
        
        data_splitter = splitter.PresetSpectraSplitter(split_file=cfg.dataset.split_file)

        paired_featurizer = featurizers.PairedFeaturizer(
            spec_featurizer=featurizers.PeakFormula(cfg.dataset.subform_folder),
            mol_featurizer=featurizers.RawMolFeaturizer(),  #featurizers.FingerprintFeaturizer(fp_names=['morgan4096'])
            graph_featurizer=None,
        )

        spectra_mol_pairs = datasets.get_paired_spectra(labels_file=cfg.dataset.labels_file, spec_folder=cfg.dataset.spec_folder)
        spectra_mol_pairs = list(zip(*spectra_mol_pairs))

        # Redefine splitter s.t. this splits three times and remove subsetting
        split_name, (train, val, test) = data_splitter.get_splits(spectra_mol_pairs)

        # randomly shuffle test set with fixed seed
        random.seed(42)
        random.shuffle(test)

        self.ms_datasets = {'train': datasets.SpectraMolDataset(spectra_mol_list=train, featurizer=paired_featurizer,),
                    'val': datasets.SpectraMolDataset(spectra_mol_list=val, featurizer=paired_featurizer),
                    'test': datasets.SpectraMolDataset(spectra_mol_list=test, featurizer=paired_featurizer,)}
        
        # Save references for dataloaders
        self.train_dataset = self.ms_datasets["train"]
        self.val_dataset = self.ms_datasets["val"]
        self.test_dataset = self.ms_datasets["test"]
        self.batch_size = 32
        self.eval_batch_size = 1

    def train_dataloader(self) -> DataLoader:
        return get_paired_loader_graph(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return get_paired_loader_graph(self.val_dataset, shuffle=False, batch_size=self.eval_batch_size)
    
    def test_dataloader(self) -> DataLoader:
        return get_paired_loader_graph(self.test_dataset, shuffle=False, batch_size=self.eval_batch_size)
    
#     def valency_count(self, max_n_nodes):
#         valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

#         # No bond, single bond, double bond, triple bond, aromatic bond
#         multiplier = torch.tensor([0, 1, 2, 3, 1.5])

#         for batch in self.train_dataloader():
#             data = batch['graph']
#             n = data.x.shape[0]

#             for atom in range(n):
#                 edges = data.edge_attr[data.edge_index[0] == atom]
#                 edges_total = edges.sum(dim=0)
#                 valency = (edges_total * multiplier).sum()
#                 valencies[valency.long().item()] += 1
#         valencies = valencies / valencies.sum()
#         return valencies
    
#     def node_counts(self, max_nodes_possible=150):
#         all_counts = torch.zeros(max_nodes_possible)
#         for loader in [self.train_dataloader(), self.val_dataloader()]:
#             for batch in loader:
#                 data = batch['graph']
#                 unique, counts = torch.unique(data.batch, return_counts=True)
#                 for count in counts:
#                     all_counts[count] += 1
#         max_index = max(all_counts.nonzero())
#         all_counts = all_counts[:max_index + 1]
#         all_counts = all_counts / all_counts.sum()
#         return all_counts

#     def node_types(self):
#         num_classes = None
#         for batch in self.train_dataloader():
#             data = batch['graph']
#             num_classes = data.x.shape[1]
#             break

#         counts = torch.zeros(num_classes)

#         for i, batch in enumerate(self.train_dataloader()):
#             data = batch['graph']
#             counts += data.x.sum(dim=0)

#         counts = counts / counts.sum()
#         return counts
    
#     def edge_counts(self):
#         num_classes = None
#         for batch in self.train_dataloader():
#             data = batch['graph']
#             num_classes = data.edge_attr.shape[1]
#             break

#         d = torch.zeros(num_classes, dtype=torch.float)

#         for i, batch in enumerate(self.train_dataloader()):
#             data = batch['graph']
#             unique, counts = torch.unique(data.batch, return_counts=True)

#             all_pairs = 0
#             for count in counts:
#                 all_pairs += count * (count - 1)

#             num_edges = data.edge_index.shape[1]
#             num_non_edges = all_pairs - num_edges

#             edge_types = data.edge_attr.sum(dim=0)
#             assert num_non_edges >= 0
#             d[0] += num_non_edges
#             d[1:] += edge_types[1:]

#         d = d / d.sum()
#         return d


# class Spec2MolDatasetInfos(AbstractDatasetInfos):
#     def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
#         self.name = 'canopus'
#         self.input_dims = None
#         self.output_dims = None
#         self.remove_h = False

#         self.atom_decoder = atom_decoder
#         self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
#         self.atom_weights = {i: ATOM_TO_WEIGHT.get(atom, 0) for i, atom in enumerate(self.atom_decoder)}
#         self.valencies = valency
#         self.num_atom_types = len(self.atom_decoder)
#         self.max_weight = max(self.atom_weights.values())
#         self._root_path = os.path.join(cfg.general.parent_dir, cfg.dataset.datadir)

#         meta_files = dict(n_nodes=f'{self._root_path}/n_counts.txt',
#                           node_types=f'{self._root_path}/atom_types.txt',
#                           edge_types=f'{self._root_path}/edge_types.txt',
#                           valency_distribution=f'{self._root_path}/valencies.txt')
        
#         if cfg.dataset.stats_dir:
#             meta_read = dict(n_nodes=f'{self._root_path}/n_counts.txt',
#                           node_types=f'{cfg.dataset.stats_dir}/atom_types.txt',
#                           edge_types=f'{cfg.dataset.stats_dir}/edge_types.txt',
#                           valency_distribution=f'{self._root_path}/valencies.txt')
#         else:
#             meta_read = dict(n_nodes=f'{self._root_path}/n_counts.txt',
#                           node_types=f'{self._root_path}/atom_types.txt',
#                           edge_types=f'{self._root_path}/edge_types.txt',
#                           valency_distribution=f'{self._root_path}/valencies.txt')

#         self.n_nodes = None
#         self.node_types = None
#         self.edge_types = None
#         self.valency_distribution = None

#         if meta is None:
#             meta = dict(n_nodes=None, node_types=None, edge_types=None, valency_distribution=None)
#         assert set(meta.keys()) == set(meta_files.keys())

#         for k, v in meta_read.items():
#             if (k not in meta or meta[k] is None) and os.path.exists(v):
#                 meta[k] = torch.tensor(np.loadtxt(v))
#                 setattr(self, k, meta[k])

#         self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None

#         if recompute_statistics or self.n_nodes is None:
#             self.n_nodes = datamodule.node_counts()
#             print("Distribution of number of nodes", self.n_nodes)
#             np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
#             self.max_n_nodes = len(self.n_nodes) - 1
#         if recompute_statistics or self.node_types is None:
#             self.node_types = datamodule.node_types()                                     # There are no node types
#             print("Distribution of node types", self.node_types)
#             np.savetxt(meta_files["node_types"], self.node_types.numpy())

#         if recompute_statistics or self.edge_types is None:
#             self.edge_types = datamodule.edge_counts()
#             print("Distribution of edge types", self.edge_types)
#             np.savetxt(meta_files["edge_types"], self.edge_types.numpy())
#         if recompute_statistics or self.valency_distribution is None:
#             valencies = datamodule.valency_count(self.max_n_nodes)
#             print("Distribution of the valencies", valencies)
#             np.savetxt(meta_files["valency_distribution"], valencies.numpy())
#             self.valency_distribution = valencies
        
#         self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

#     def compute_input_output_dims(self, datamodule, extra_features, domain_features):
#         example_batch = next(iter(datamodule.train_dataloader()))['graph']
#         ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
#                                              example_batch.batch)
#         example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

#         self.input_dims = {'X': example_batch['x'].size(1),
#                            'E': example_batch['edge_attr'].size(1),
#                            'y': example_batch['y'].size(1) + 1}      # + 1 due to time conditioning

#         ex_extra_feat = extra_features(example_data)
#         self.input_dims['X'] += ex_extra_feat.X.size(-1)
#         self.input_dims['E'] += ex_extra_feat.E.size(-1)
#         self.input_dims['y'] += ex_extra_feat.y.size(-1)

#         ex_extra_molecular_feat = domain_features(example_data)
#         self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
#         self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
#         self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

#         self.output_dims = {'X': example_batch['x'].size(1),
#                             'E': example_batch['edge_attr'].size(1),
#                             'y': example_batch['y'].size(1)}