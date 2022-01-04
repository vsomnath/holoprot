import os
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from typing import List, Dict, Union

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

def compute_dist(pos_i: np.ndarray, pos_j: np.ndarray, sigma: float = 0.01) -> float:
    y = pos_i - pos_j
    dist = np.exp(-np.sum(y**2) / sigma**2)
    return dist

def compute_angle(norm_i: np.ndarray, norm_j: np.ndarray) -> float:
    angle = np.arccos(norm_i.dot(norm_j))
    return angle / (2 * np.pi)

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

def compute_edge_features(data: Data) -> List:
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

class Surface:

    def __init__(self, sigma: float = 18.0, msms_bin: str = MSMS_BIN):
        self.sigma = sigma
        self.msms_bin = MSMS_BIN

    def __call__(self, pdb_file: str, target: Union[float, np.ndarray] = None, **kwargs):
        return self.build(pdb_file=pdb_file, target=target, **kwargs)

    def build(self, pdb_file: str, target: Union[float, np.ndarray] = None, **kwargs):
        import pymesh
        if 'mesh_file' not in kwargs:
            raise ValueError("mesh file not found.")
        mesh_file = kwargs['mesh_file']
        if not os.path.exists(mesh_file):
            raise ValueError(f"{mesh_file} not found. Please prepare a mesh.")

        pdb_path = os.path.abspath(pdb_file)
        pdb_name = pdb_path.split("/")[-1]
        pdb_id = pdb_name.split(".")[0]
        if "_fixed" in pdb_id:
            pdb_id = "_".join(elem for elem in pdb_id.split("_")[:-1])

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
                                              mesh=tmp_mesh, 
                                              pdb_id=pdb_id)
        if node_feats is None:
            raise ValueError('Node feats is None')

        node_feats = np.stack(node_feats, axis=-1)
        edge_feats = compute_edge_features(tmp_data)

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

        for (idx_i, idx_j) in G.edges():
            edge_idx = tmp_data.edge_to_idx[(idx_i, idx_j)]
            dist = compute_dist(vertices[idx_i], vertices[idx_j], sigma=self.sigma)
            angle = compute_angle(normals[idx_i], normals[idx_j])
            edge_feats[edge_idx].extend([dist, angle])
            G[idx_i][idx_j]['elem'] = edge_feats[edge_idx]

        G = G.to_directed()
        edge_index = torch.tensor(list(G.edges)).t().contiguous()

        edge_attr = []
        for _, _, feat in G.edges(data='elem'):
            edge_attr.append(feat)
        edge_attr = torch.tensor(edge_attr)
        surface_data = Data(pos=torch.tensor(vertices, dtype=torch.float),
                          face=torch.tensor(faces, dtype=torch.long),
                          edge_index=edge_index.long(),
                          edge_attr=edge_attr.float(),
                          x=torch.tensor(node_feats, dtype=torch.float))
        def finite_check(x):
            return torch.isfinite(x).all().item()

        del vertices, faces, normals
        checks = [finite_check(surface_data.x), finite_check(surface_data.edge_attr)]
        if not all(checks):
            print(f"Nan checks failed for protein patch: {pdb_file}", flush=True)
            return None

        if target is not None:
            target = torch.tensor(target)
            if not len(target.shape):
                target = target.unsqueeze(0)
            surface_data.y = target
            if not finite_check(target):
                print(f"Invalid y value. {pdb_file}", flush=True)
                return None

        return surface_data
