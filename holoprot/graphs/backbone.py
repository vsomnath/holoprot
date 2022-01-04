import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph
import numpy as np
import os
from Bio.PDB import PDBParser, Polypeptide
from typing import List, Dict, Union
import pickle

from holoprot.feat import KD_SCALE
from holoprot.feat.complex import (ResidueProp, get_residue_features,
     get_contact_features)

DSSP_BIN = os.environ['DSSP_BIN']
parser = PDBParser()
pp_builder = Polypeptide.PPBuilder()


class Backbone:

    def __init__(self, max_num_neighbors: int = 128, mode: str = 'ca',
                 radius: float = 12.0, sigma: float = 0.01, **kwargs):
        self.max_num_neighbors = max_num_neighbors
        self.radius = radius
        self.mode = mode
        self.sigma = sigma

    def get_coords(self, residues):
        if self.mode == 'ca':
            coords = np.array([res['CA'].get_coord() for res in residues])
        elif self.mode == 'com':
            coords = None
        else:
            raise ValueError(f'Coords cannot be generated for mode {self.mode}')
        return torch.from_numpy(coords)

    @staticmethod
    def add_node_feats(amino_data: Data, residues: List, dssp_dict: Dict) -> Data:
        x = []
        key_fn = lambda x: x.get_full_id()[2:]
        sas = np.array([dssp_dict[key_fn(res)][2] for res in residues])
        sas = sas / max(sas)

        for idx, res in enumerate(residues):
            ss = dssp_dict[res.get_full_id()[2:]][1]
            hydrophob = KD_SCALE[res.get_resname()] / 4.5
            res_prop = ResidueProp(residue=res, sas=sas[idx], sec=ss,
                                   hydrophobicity=hydrophob)
            x.append(get_residue_features(res_prop))

        x = torch.tensor(x, dtype=torch.float)
        amino_data.x = x

    @staticmethod
    def add_edge_feats(amino_data: Data, residues: List,
                       mode: str ='ca', sigma: float = 0.01) -> Data:
        edge_idx = amino_data.edge_index.numpy().T
        res_pair = [(residues[idx_a], residues[idx_b]) for idx_a, idx_b in edge_idx]
        edge_attr = []
        for elem in res_pair:
            edge_attr.append(get_contact_features(elem, mode=mode, sigma=sigma))
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        del edge_idx
        amino_data.edge_attr = edge_attr

    @staticmethod
    def primary_graph(amino_data: Data, residue_dict: Dict, model) -> Data:
        forward_edges = []

        for chain in model:
            residues_in_chain = list(chain.get_residues())
            for idx in range(len(residues_in_chain) - 1):
                prev_res, next_res = residues_in_chain[idx], residues_in_chain[idx+1]
                res_idx_a = residue_dict.get(prev_res.get_full_id()[2:], None)
                res_idx_b = residue_dict.get(next_res.get_full_id()[2:], None)

                if res_idx_a is not None and res_idx_b is not None:
                    forward_edges.append((res_idx_a, res_idx_b))

        forward_edges = torch.tensor(forward_edges).transpose(0, 1)
        reverse_edges = forward_edges[[1, 0], :]

        if amino_data.edge_index is None:
            amino_data.edge_index = torch.cat((forward_edges, reverse_edges), dim=1)
        else:
            amino_data.edge_index = torch.cat(
                        (amino_data.edge_index, forward_edges, reverse_edges), dim=1)
        return amino_data.coalesce()

    @staticmethod
    def secondary_graph(amino_data: Data, dssp_dict: Dict,
                         dssp_idx_to_res: Dict) -> Data:
        forward_edges = []
        for res_id in dssp_dict:
            dssp_info = dssp_dict[res_id]
            dssp_idx = dssp_info[5]
            nh_o_1_idx = dssp_info[6]
            nh_o_1_en = dssp_info[7]
            o_nh_1_idx = dssp_info[8]
            o_nh_1_en = dssp_info[9]
            nh_o_2_idx = dssp_info[10]
            nh_o_2_en = dssp_info[11]
            o_nh_2_idx = dssp_info[12]
            o_nh_2_en = dssp_info[13]

            if nh_o_1_idx and nh_o_1_en:
                other_res_dssp_idx = dssp_idx + nh_o_1_idx
                if dssp_idx in dssp_idx_to_res and other_res_dssp_idx in dssp_idx_to_res:
                    _, other_res_idx = dssp_idx_to_res[other_res_dssp_idx]
                    _, res_idx = dssp_idx_to_res[dssp_idx]
                    forward_edges.append((res_idx, other_res_idx))

            if o_nh_1_idx and o_nh_1_en:
                other_res_dssp_idx = dssp_idx + o_nh_1_idx
                if dssp_idx in dssp_idx_to_res and other_res_dssp_idx in dssp_idx_to_res:
                    _, other_res_idx = dssp_idx_to_res[other_res_dssp_idx]
                    _, res_idx = dssp_idx_to_res[dssp_idx]
                    forward_edges.append((res_idx, other_res_idx))

            if nh_o_2_idx and nh_o_2_en:
                other_res_dssp_idx = dssp_idx + nh_o_2_idx
                if dssp_idx in dssp_idx_to_res and other_res_dssp_idx in dssp_idx_to_res:
                    _, other_res_idx = dssp_idx_to_res[other_res_dssp_idx]
                    _, res_idx = dssp_idx_to_res[dssp_idx]
                    forward_edges.append((res_idx, other_res_idx))

            if o_nh_2_idx and o_nh_2_en:
                other_res_dssp_idx = dssp_idx + o_nh_2_idx
                if dssp_idx in dssp_idx_to_res and other_res_dssp_idx in dssp_idx_to_res:
                    _, other_res_idx = dssp_idx_to_res[other_res_dssp_idx]
                    _, res_idx = dssp_idx_to_res[dssp_idx]
                    forward_edges.append((res_idx, other_res_idx))

        forward_edges = torch.tensor(forward_edges).transpose(0, 1)
        reverse_edges = forward_edges[[1, 0], :]

        if amino_data.edge_index is None:
            amino_data.edge_index = torch.cat((forward_edges, reverse_edges), dim=1)
        else:
            amino_data.edge_index = torch.cat(
                        (amino_data.edge_index, forward_edges, reverse_edges), dim=1)
        return amino_data.coalesce()

    @staticmethod
    def tertiary_graph(amino_data: Data,
                       radius: float = 12.0, max_num_neighbors: int = 128):
        radius_graph = RadiusGraph(r=radius, max_num_neighbors=max_num_neighbors)
        return radius_graph(amino_data)

    def __call__(self, pdb_file: str, target: Union[torch.Tensor, np.ndarray] = None, **kwargs) -> Data:
        return self.build(pdb_file=pdb_file, target=target, **kwargs)

    def build(self, pdb_file: str, target: Union[torch.Tensor, np.ndarray] = None, **kwargs) -> Data:
        pdb_file = os.path.abspath(pdb_file)
        pdb_base = pdb_file.split("/")[-1]
        pdb_id = ".".join(pdb_base.split(".")[:-1])
        if "fixed" in pdb_file:
            pdb_id = "_".join(pdb_id.split("_")[:-1])
        dssp_file = "/".join(pdb_file.split("/")[:-1] + [f"{pdb_id}.dssp"])

        if not os.path.exists(dssp_file):
            print(f"{pdb_id}: dssp file not found. Returning None")
            return None

        with open(dssp_file, "rb") as f:
            dssp_dict = pickle.load(f)

        struct = parser.get_structure(pdb_id, file=pdb_file)
        polypeptides = pp_builder.build_peptides(struct)

        residues = [res for pp in polypeptides for res in pp]
        # Select only residues that have secondary structure information
        residues = [res for res in residues if res.get_full_id()[2:] in dssp_dict]
        residue_dict = {res.get_full_id()[2:]: idx for idx, res in enumerate(residues)}

        dssp_idx_to_res = {}
        for res_id in residue_dict:
            dssp_idx = dssp_dict[res_id][5]
            res_idx = residue_dict[res_id]
            dssp_idx_to_res[dssp_idx] = (res_id, res_idx)

        coords = self.get_coords(residues)
        amino_data = Data(pos=coords)

        # Primary, secondary and tertiary structures
        amino_data = Backbone.primary_graph(amino_data, residue_dict, struct[0])
        amino_data = Backbone.secondary_graph(amino_data, dssp_dict, dssp_idx_to_res)
        amino_data = Backbone.tertiary_graph(amino_data, radius=self.radius, max_num_neighbors=self.max_num_neighbors)

        # Add node and edge features
        Backbone.add_node_feats(amino_data, residues, dssp_dict)
        Backbone.add_edge_feats(amino_data, residues, sigma=self.sigma,
                                    mode=self.mode)

        def finite_check(x):
            return torch.isfinite(x).all().item()

        checks = [finite_check(amino_data.x), finite_check(amino_data.edge_attr)]
        if not all(checks):
            print(f"Nan checks failed for protein: {pdb_file}", flush=True)
            return None

        if target is not None:
            target = torch.tensor(target)
            if not len(target.shape):
                target = target.unsqueeze(0)
            amino_data.y = target
            if not finite_check(target):
                print(f"Invalid y value. {pdb_file}", flush=True)
                return None

        if 'return_res' in kwargs and kwargs['return_res']:
            return amino_data, residues
        return amino_data
