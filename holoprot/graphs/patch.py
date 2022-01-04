import os
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Union

from holoprot.utils.surface import get_surface, compute_normal
from holoprot.feat.surface import compute_surface_features
from holoprot.utils.mesh import (get_edge_points, set_edge_lengths,
            dihedral_angle, symmetric_ratios, symmetric_opposite_angles,
            compute_face_normals_and_areas)

MSMS_BIN = os.environ['MSMS_BIN']


def load_mesh_from_file(mesh_file: str):
    import pymesh
    mesh = pymesh.load_mesh(mesh_file)
    normals = compute_normal(mesh.vertices, mesh.faces)
    n1 = normals[:, 0]
    n2 = normals[:, 1]
    n3 = normals[:, 2]

    mesh.add_attribute("vertex_nx")
    mesh.set_attribute("vertex_nx", n1)
    mesh.add_attribute("vertex_ny")
    mesh.set_attribute("vertex_ny", n2)
    mesh.add_attribute("vertex_nz")
    mesh.set_attribute("vertex_nz", n3)
    return mesh


# def load_model_from_file(model_file: str) -> torch.nn.Module:
#     loaded = torch.load(model_file, map_location='cpu')
#     model = PatchPointCloud(**loaded['saveables'], device='cpu')
#     model.load_state_dict(loaded['state'])
#     model.to('cpu')
#     model.eval()
#     return model


def build_gemm(data: Data) -> Data:
    edge_nb = []
    sides = []
    edge_to_idx = dict()
    edges_count = 0
    nb_count = []
    edges = []
    ve = [[] for _ in data.vertices]

    for face_id, face in enumerate(data.faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)

        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge_to_idx:
                edge_to_idx[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                ve[edge[0]].append(edges_count)
                ve[edge[1]].append(edges_count)
                nb_count.append(0)
                edges_count += 1

        for idx, edge in enumerate(faces_edges):
            edge_key = edge_to_idx[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge_to_idx[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge_to_idx[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2

        for idx, edge in enumerate(faces_edges):
            edge_key = edge_to_idx[edge]
            sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge_to_idx[faces_edges[(idx + 1) % 3]]] - 1
            sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge_to_idx[faces_edges[(idx + 2) % 3]]] - 2

    data.edges = np.array(edges, dtype=np.int32)
    data.ve = ve
    data.gemm_edges = np.array(edge_nb, dtype=np.int32)
    data.sides = np.array(sides, dtype=np.int32)
    data.edges_count = edges_count
    data.edge_to_idx = edge_to_idx
    return data


def remove_non_manifolds(data: Data) -> Data:
    edges_set = set()
    mask = np.ones(len(data.faces), dtype=bool)
    _, face_areas = compute_face_normals_and_areas(data.vertices, data.faces)

    for face_id, face in enumerate(data.faces):
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        faces_edges = []
        is_manifold = False
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                faces_edges.append(cur_edge)
        if is_manifold:
            mask[face_id] = False
        else:
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)
    data.faces = data.faces[mask]
    return data


def compute_edge_features(data: Data) -> Data:
    edge_feats = []
    edge_points = get_edge_points(data)
    set_edge_lengths(data, edge_points)
    with np.errstate(divide='raise'):
        try:
            for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios]:
                feature = extractor(data, edge_points)
                edge_feats.append(feature)
            edge_feats = np.concatenate(edge_feats, axis=0).T
            return edge_feats.tolist()
        except Exception as e:
            print(e)
            raise ValueError('bad features')


def compute_patch_members(patch_labels: np.ndarray,
                          edge_index: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    patch_labels_i = patch_labels[edge_index[:, 0]]
    patch_labels_j = patch_labels[edge_index[:, 1]]

    edges = []
    diff_label_idxs = np.flatnonzero(patch_labels_i != patch_labels_j)
    if len(diff_label_idxs) == 0:
        raise ValueError("All edges have same labelled points")

    patch_edge_index = np.stack([patch_labels_i[diff_label_idxs],
                                 patch_labels_j[diff_label_idxs]], axis=1)
    assert len(patch_edge_index) == len(diff_label_idxs)
    assert patch_edge_index.shape[-1] == 2
    patch_edge_index = np.sort(patch_edge_index, axis=1)
    patch_edge_index = np.unique(patch_edge_index, axis=0)

    reverse_edges = patch_edge_index[:, [1, 0]]
    patch_edge_index = np.concatenate([patch_edge_index, reverse_edges], axis=0)
    assert len(patch_edge_index) == 2 * len(reverse_edges)
    assert patch_edge_index.shape[-1] == 2

    patch_membership = {}
    for label in np.unique(patch_labels):
        patch_membership[label] = np.flatnonzero(patch_labels == label)
    return patch_edge_index, patch_membership


def compute_wass_dist(patch_feats_i: np.ndarray, patch_feats_j: np.ndarray):
    import ot
    dists = []
    n_feat = patch_feats_i.shape[-1]
    for idx in range(n_feat):
        feat_i = patch_feats_i[:, idx].reshape(-1, 1)
        feat_j = patch_feats_j[:, idx].reshape(-1, 1)
        dists.append(ot.bregman.empirical_sinkhorn2(feat_i, feat_j, reg=1.0))
    return np.asarray(dists).flatten()


def compute_patch_edge_feats(patch_edge_index: np.ndarray,
                             patch_membership: Dict[int, np.ndarray],
                             node_feats: np.ndarray) -> np.ndarray:
    edge_feats = []
    for i, j in patch_edge_index:
        patch_members_i = patch_membership[i]
        patch_members_j = patch_membership[j]

        patch_feats_i = node_feats[patch_members_i]
        patch_feats_j = node_feats[patch_members_j]

        wass_dist = compute_wass_dist(patch_feats_i, patch_feats_j)
        edge_feats.append(wass_dist)

    edge_feats = np.array(edge_feats)
    return edge_feats


def compute_patch_node_feats(patch_membership: Dict[int, np.ndarray],
                             node_feats: np.ndarray) -> np.ndarray:
    patch_node_feats = []
    for node_idx, patch_members in patch_membership.items():
        patch_feats = node_feats[patch_members]
        feats = (np.min(patch_feats), np.max(patch_feats),
                 np.mean(patch_feats), np.std(patch_feats))
        patch_node_feats.append(feats)
    patch_node_feats = np.asarray(patch_node_feats)
    return patch_node_feats


class Patch:

    def __init__(self, msms_bin: str = MSMS_BIN):
        self.msms_bin = msms_bin

    def __call__(self, pdb_file: str, target: Union[float, np.ndarray] = None, **kwargs) -> Data:
        return self.build(pdb_file=pdb_file, target=target, **kwargs)

    def build(self, pdb_file: str, target: Union[float, np.ndarray] = None, **kwargs) -> Data:
        if 'mesh_file' not in kwargs:
            raise ValueError("mesh file not found")
        if 'assignments_file' not in kwargs:
            raise ValueError("assignment file not found.")
        mesh_file = kwargs['mesh_file']
        assignments_file = kwargs['assignments_file']

        if not os.path.exists(mesh_file):
            raise ValueError(f"{mesh_file} not found. Please prepare a mesh.")
        
        pdb_path = os.path.abspath(pdb_file)
        pdb_name = pdb_path.split("/")[-1]
        pdb_id = pdb_name.split(".")[0]
        if "_fixed" in pdb_id:
            pdb_id = "_".join(elem for elem in pdb_id.split("_")[:-1])

        import pymesh
        mesh = load_mesh_from_file(mesh_file)
        surface = get_surface(pdb_file=pdb_file, msms_bin=self.msms_bin)

        tmp_data = Data(vertices=mesh.vertices, faces=mesh.faces)
        tmp_data = remove_non_manifolds(tmp_data)
        tmp_data = build_gemm(tmp_data)

        # Build a new mesh after removing any non-manifolds, and use this for
        # node features
        tmp_mesh = pymesh.form_mesh(tmp_data.vertices, tmp_data.faces)
        normals = compute_normal(tmp_mesh.vertices, tmp_mesh.faces)
        n1 = normals[:, 0]
        n2 = normals[:, 1]
        n3 = normals[:, 2]

        tmp_mesh.add_attribute("vertex_nx")
        tmp_mesh.set_attribute("vertex_nx", n1)
        tmp_mesh.add_attribute("vertex_ny")
        tmp_mesh.set_attribute("vertex_ny", n2)
        tmp_mesh.add_attribute("vertex_nz")
        tmp_mesh.set_attribute("vertex_nz", n3)

        node_feats = compute_surface_features(pdb_file=pdb_file,
                                              surface=surface,
                                              mesh=tmp_mesh, pdb_id=pdb_id)
        if node_feats is None:
            raise ValueError('Node feats is None')

        node_feats = np.stack(node_feats, axis=-1)
        G = nx.Graph()
        n = len(tmp_data.vertices)
        G.add_nodes_from(np.arange(n))

        f = np.asarray(tmp_data.faces, dtype = int)
        rowi = np.concatenate([f[:,0], f[:,0], f[:,1]], axis = 0)
        rowj = np.concatenate([f[:,1], f[:,2], f[:,2]], axis = 0)
        edges = np.stack([rowi, rowj]).T
        G.add_edges_from(edges)

        vertices = np.asarray(tmp_data.vertices.tolist())
        faces = np.asarray(tmp_data.faces.tolist())
        G = G.to_directed()
        edge_index = np.asarray(list(G.edges), dtype=np.int32)

        patch_labels = torch.load(assignments_file)
        patch_edge_index, patch_membership = compute_patch_members(patch_labels, edge_index)
        patch_edge_feats = compute_patch_edge_feats(patch_edge_index,
                                                    patch_membership, node_feats)
        patch_node_feats = compute_patch_node_feats(patch_membership, node_feats)

        assert len(patch_node_feats) >= np.max(patch_edge_index)

        x = torch.tensor(patch_node_feats).float()
        patch_edge_index = torch.tensor(patch_edge_index).long().t().contiguous()
        patch_edge_attr = torch.tensor(patch_edge_feats).float()

        patch_data = Data(pos=torch.tensor(vertices, dtype=torch.float),
                          face=torch.tensor(faces, dtype=torch.long),x=x,
                          edge_index=patch_edge_index,
                          edge_attr=patch_edge_attr)
        return patch_data, patch_membership
