import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph
from rdkit import Chem
import networkx as nx
import numpy as np
import os
from Bio.PDB import PDBParser, Polypeptide
from typing import List, Dict, Tuple, Union

from holoprot.feat import KD_SCALE
from holoprot.feat.complex import (AtomProp, BondProp,
     ResidueProp, get_atom_features, get_bond_features, get_residue_features,
     get_contact_features, get_secondary_struct_features)
from holoprot.utils.tensor import create_pad_tensor

parser = PDBParser()
pp_builder = Polypeptide.PPBuilder()


class LigandMol:

    def build(self, lig_smi: str, target = None) -> Data:
        mol = Chem.MolFromSmiles(lig_smi)
        if mol is None:
            print(f"Mol is None {lig_smi}", flush=True)
            return None

        data = {}
        G = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        G = nx.convert_node_labels_to_integers(G)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()

        x = []
        for atom in mol.GetAtoms():
            x.append(get_atom_features(AtomProp(atom)))
        data['x'] = torch.tensor(x).float()

        edge_attr = []
        mess_idx, edge_dict = [[]], {}

        for a1, a2 in G.edges():
            bond = mol.GetBondBetweenAtoms(a1, a2)
            bond_feat = get_bond_features(BondProp(bond))
            edge_attr.append(bond_feat)
            edge_dict[(a1, a2)] = eid = len(edge_dict) + 1
            mess_idx.append([])

        data['edge_attr'] = torch.tensor(edge_attr).float()
        data['edge_index'] = edge_index

        for u, v in G.edges:
            eid = edge_dict[(u, v)]
            for w in G.predecessors(u):
                if w == v: continue
                mess_idx[eid].append(edge_dict[(w, u)])
        mess_idx = create_pad_tensor(mess_idx)
        data['mess_idx'] = mess_idx

        def finite_check(x):
            return torch.isfinite(x).all().item()

        data = Data.from_dict(data)
        checks = [finite_check(data.x), finite_check(data.edge_attr)]
        if target is not None:
            target = torch.tensor(target)
            if not len(target.shape):
                target = target.unsqueeze(0)
            data.y = target
            checks += [finite_check(data.y)]

        if not all(checks):
            print(f"Nan checks failed for ligand: {lig_smi}", flush=True)
            return None

        return data


class Complex:

    def __init__(self,
                 prot_builder,
                 **kwargs):
        self.prot_builder = prot_builder
        self.mol_builder = LigandMol()
    
    def __call__(self, pdb_file: str, lig_smi: str, 
                 target: Union[float, np.ndarray], **kwargs) -> Union[Data, Tuple[Data]]:
        return self.build(pdb_file=pdb_file, lig_smi=lig_smi, target=target, **kwargs)
    
    def build(self, pdb_file: str, lig_smi: str, target: Union[float, np.ndarray],
              **kwargs) -> Union[Data, Tuple[Data]]:
        ligand = self.mol_builder.build(lig_smi)
        if ligand is None:
            return None

        if 'build_prot' in kwargs and kwargs.get('build_prot'):
            prot_kwargs = {'pdb_file': pdb_file}
            if 'mesh_file' in kwargs:
                prot_kwargs['mesh_file'] = kwargs['mesh_file']
            if 'assignments_file' in kwargs:
                prot_kwargs['assignments_file'] = kwargs['assignments_file']

            protein = self.prot_builder(**prot_kwargs)
            if protein is None:
                return None
        
        if target is not None:
            y = torch.tensor(target)
            if not len(y.shape):
                y = y.unsqueeze(0)
            ligand.y = y

        if 'build_prot' in kwargs and kwargs.get('build_prot'):
            return (protein, ligand)
        return ligand