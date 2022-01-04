import os
from numpy.core.fromnumeric import amin
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils.subgraph import subgraph as subgraph_util
from scipy.spatial import KDTree
from typing import List, Dict, Union

from holoprot.data.base import HierData, SurfaceToPatch
from holoprot.utils.surface import get_surface
from holoprot.utils.tensor import create_pad_tensor
from holoprot.graphs import Backbone, Surface, Patch

MSMS_BIN = os.environ['MSMS_BIN']

def get_resid_from_name(name: str):
    """Converts name of node in mesh to residue_id."""
    entities = name.split("_")
    chain_id = entities[0]

    res_pos = int(entities[1])
    insertion = entities[2]

    if insertion == "x":
        insertion = " "

    res_id = (" ", res_pos, insertion)
    return (chain_id, res_id)


class Surface2Backbone:

    def __init__(self,
                 max_num_neighbors: int = 128,
                 mode: str = 'ca',
                 radius: float = 12.0,
                 sigma: float = 0.01,
                 msms_bin: str = MSMS_BIN):
        self.backbone_builder = Backbone(max_num_neighbors=max_num_neighbors,
                                         mode=mode, radius=radius, sigma=sigma)
        self.surface_builder = Surface(sigma=sigma, msms_bin=msms_bin)
        self.msms_bin = msms_bin

    @staticmethod
    def get_hier_map(names: List[str], resid_to_idx: Dict) -> torch.Tensor:
        """
        resid_to_idx is a dictionary mapping residue id to its idx. Each node in
        the surface graph maps to a residue, which is captured in names.
        get_hier_map returns this mapping of idxs of nodes from the surface
        mapping to corresponding residue indices.
        """
        node_to_resid = [get_resid_from_name(name) for name in names]
        hier_mappings = [[] for i in range(len(resid_to_idx))]
        for node_idx, resid in enumerate(node_to_resid):
            if resid in resid_to_idx:
                residx = resid_to_idx[resid]
                hier_mappings[residx].append(node_idx+1) #+1 so that the padded elements are 0
        mapping_lens = [len(res) for res in hier_mappings]
        hier_mappings = create_pad_tensor(hier_mappings)
        return hier_mappings

    def __call__(self, pdb_file: str, target: Union[np.ndarray, torch.Tensor] = None, **kwargs) -> Data:
        return self.build(pdb_file=pdb_file, target=target, **kwargs)

    def build(self, pdb_file: str, 
              target: Union[float, np.ndarray] = None, **kwargs) -> Data:
        import pymesh
        if 'mesh_file' not in kwargs:
            raise ValueError('mesh file not found.')
        mesh_file = kwargs['mesh_file']
        vertices, faces, normals, names, _ = get_surface(pdb_file=pdb_file,
                                                         msms_bin=self.msms_bin)

        tree = KDTree(data=vertices)
        new_mesh = pymesh.load_mesh(mesh_file)
        closest_idxs = tree.query(new_mesh.vertices)[1]
        node_names = [names[idx] for idx in closest_idxs]

        hier_data = HierData()
        surface_data = self.surface_builder(pdb_file=pdb_file, mesh_file=mesh_file)
        amino_data, residues = self.backbone_builder(pdb_file=pdb_file, return_res=True)
        resid_to_idx = {res.get_full_id()[2:]: idx
                        for idx, res in enumerate(residues)}
        mappings = Surface2Backbone.get_hier_map(node_names, resid_to_idx)

        hier_data.surface = surface_data
        hier_data.backbone = amino_data
        hier_data.mapping = mappings

        def finite_check(x):
            return torch.isfinite(x).all().item()

        checks = [hier_data.backbone is not None, hier_data.surface is not None,
                  finite_check(hier_data.mapping)]
        if target is not None:
            target = torch.tensor(target)
            if not len(target.shape):
                target = target.unsqueeze(0)
            hier_data.y = target
            checks += [finite_check(hier_data.y)]

        if not all(checks):
            print(f"Nan checks failed for hierarchical protein: {pdb_file}", flush=True)
            return None

        return hier_data


class Patch2Backbone:

    def __init__(self,
                 max_num_neighbors: int = 128,
                 mode: str = 'ca',
                 radius: float = 12.0,
                 sigma: float = 0.01,
                 msms_bin: str = MSMS_BIN):
        self.msms_bin = msms_bin
        self.backbone_builder = Backbone(max_num_neighbors=max_num_neighbors,
                                         mode=mode, radius=radius, sigma=sigma)
        self.patch_builder = Patch(msms_bin=msms_bin)
        self.surface_builder = Surface(sigma=sigma, msms_bin=msms_bin)

    def build_patch(self, hier_data: Data,
                    pdb_file: str, mesh_file: str,
                    assignments_file: str) -> Data:
        """Runs a call to the surface builder."""
        patch_data, patch_membership = self.patch_builder.build(mesh_file=mesh_file, pdb_file=pdb_file,
                                            assignments_file=assignments_file)
        hier_data.patch = patch_data
        return hier_data, patch_membership

    def build_backbone(self, hier_data: Data, pdb_file: str) -> Data:
        """Runs a call to the backbone builder."""
        amino_data, residues = self.backbone_builder.build(pdb_file, return_res=True)
        hier_data.backbone = amino_data
        return hier_data, residues

    @staticmethod
    def get_hier_map(node_names, resid_to_idx, patch_membership):
        node_to_resid = [get_resid_from_name(name) for name in node_names]
        hier_mappings = [[] for i in range(len(patch_membership))]
        for idx, (patch_id, patch_members) in enumerate(patch_membership.items()):
            resid_members = [node_to_resid[member] for member in patch_members]
            res_idxs = set()
            for resid in resid_members:
                if resid in resid_to_idx:
                    res_idxs.add(resid_to_idx[resid] + 1)
            hier_mappings[idx].extend(res_idxs)
        hier_mappings = create_pad_tensor(hier_mappings)
        assert hier_mappings.shape[0] == len(patch_membership)
        assert hier_mappings.shape[1] -1 <= len(resid_to_idx)
        return hier_mappings

    def __call__(self, pdb_file: str, target: Union[np.ndarray, float] = None, **kwargs) -> Data:
        return self.build(pdb_file=pdb_file, target=target, **kwargs)

    def build(self, pdb_file: str, target: Union[np.ndarray, float] = None, **kwargs) -> Data:
        if 'mesh_file' not in kwargs:
            raise ValueError(f"mesh_file is missing.")
        if 'assignments_file' not in kwargs:
            raise ValueError(f"assignments file is missing.")

        vertices, faces, normals, names, _ = get_surface(pdb_file=pdb_file,
                                                         msms_bin=self.msms_bin)
        import pymesh
        tree = KDTree(data=vertices)
        mesh_file = kwargs['mesh_file']
        assignments_file = kwargs['assignments_file']
        new_mesh = pymesh.load_mesh(mesh_file)
        closest_idxs = tree.query(new_mesh.vertices)[1]
        node_names = [names[idx] for idx in closest_idxs]

        hier_data = HierData()
        surface_data = self.surface_builder(pdb_file=pdb_file, mesh_file=mesh_file)
        patch_data, patch_membership = self.patch_builder(pdb_file=pdb_file, mesh_file=mesh_file, 
                                                          assignments_file=assignments_file)
        n_patches = len(patch_membership)
        amino_data, residues = self.backbone_builder(pdb_file=pdb_file, return_res=True)
        resid_to_idx = {res.get_full_id()[2:]: idx
                        for idx, res in enumerate(residues)}
        patch_members = [torch.tensor(members) for idx, members in patch_membership.items()]

        # Extract subgraph of surface nodes that share the same patch label
        # We use that to run an MPN over the surface subgraph, and pass it to the
        subgraphs = [subgraph_util(members, edge_index=surface_data.edge_index,
                      edge_attr=surface_data.edge_attr, num_nodes=surface_data.x.shape[0])
                    for members in patch_members]

        surface_to_patch = SurfaceToPatch(n_patches=n_patches)
        node_membership = [] # Captures membership of surface nodes to patch

        for idx, subgraph in enumerate(subgraphs):
            edge_index, edge_attr = subgraph

            if surface_to_patch.x is None:
                surface_to_patch.x = surface_data.x[patch_members[idx]]
            else:
                surface_to_patch.x = torch.cat([surface_to_patch.x, surface_data.x[patch_members[idx]]], dim=0)

            if surface_to_patch.edge_index is None:
                surface_to_patch.edge_index = edge_index
            else:
                surface_to_patch.edge_index = torch.cat([surface_to_patch.edge_index, edge_index], dim=1)

            if surface_to_patch.edge_attr is None:
                surface_to_patch.edge_attr = edge_attr
            else:
                surface_to_patch.edge_attr = torch.cat([surface_to_patch.edge_attr, edge_attr], dim=0)

            node_membership.extend([idx] * len(patch_members[idx]))

        node_membership = torch.tensor(node_membership).long()
        surface_to_patch.node_membership = node_membership
        mappings = Patch2Backbone.get_hier_map(node_names, resid_to_idx, patch_membership)
        hier_data.patch = patch_data
        hier_data.backbone = amino_data
        hier_data.mapping = mappings
        hier_data.surface_to_patch = surface_to_patch

        def finite_check(x):
            return torch.isfinite(x).all().item()

        checks = [hier_data.backbone is not None, hier_data.patch is not None,
                  finite_check(hier_data.mapping)]
        if target is not None:
            target = torch.tensor(target)
            if not len(target.shape):
                target = target.unsqueeze(0)
            hier_data.y = target
            checks += [finite_check(hier_data.y)]

        if not all(checks):
            print(f"Nan checks failed for hierarchical protein: {pdb_file}", flush=True)
            return None

        return hier_data
