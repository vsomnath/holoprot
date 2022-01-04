"""
Utility functions for computing residue depth.
"""

import numpy as np
from Bio.PDB.Residue import Residue


def get_res_depth(residue: Residue, vertices: np.ndarray) -> np.ndarray:
    """
    Computes depth (distance from surface) for each residue, computed as an
    average of the depths of each atom in the residue.

    Parameters
    ----------
    residue: Residue
        residue whose c-alpha atom we compute depth for
    vertices: np.ndarray,

    Returns
    -------
    d: np.ndarray,
        Minimum distance of the c-alpha atom and the vertices
    """
    d_list = [min_dist(atom.get_coord(), vertices) for atom in residue]
    d = sum(d_list)
    d = d / len(d_list)
    return d


def get_ca_depth(residue: Residue, vertices: np.ndarray) -> np.ndarray:
    """Computes depth (distance from surface) for C-alpha atoms in each residue.

    Parameters
    ----------
    residue: Residue
        residue whose c-alpha atom we compute depth for
    vertices: np.ndarray,
        Vertex coordinates of the surface.

    Returns
    -------
    dist: np.ndarray,
        Minimum distance of the c-alpha atom and the vertices
    """
    if not residue.has_id("CA"):
        return None
    ca = residue["CA"]
    coord = ca.get_coord()
    dist = min_dist(coord, vertices)
    return dist


def min_dist(coord: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Computes minimum distance from coord to surface.

    Parameters
    ----------
    coord: np.ndarray,
        Coordinate vector for point
    vertices: np.ndarray,
        Surface vertices coordinates

    Returns
    -------
    minimum distance of coordinate to vertices.
    """
    d = vertices - coord
    d2 = np.sum(d * d, 1)
    return np.sqrt(min(d2))
