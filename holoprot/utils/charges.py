"""
Utility functions for computing hydrogen bonds and electrostatics on protein surface.
"""

import numpy as np
from Bio.PDB import NeighborSearch, calc_angle, calc_dihedral
from Bio.PDB import Vector
from Bio.PDB.Residue import Residue
import numpy as np
from sklearn.neighbors import KDTree

from holoprot.utils import POLAR_HYDROGENS, ACCEPTOR_ANGLES, HBOND_STD_DEV
from holoprot.utils import ACCEPTOR_PLANES, RADII, DONOR_ATOMS


def compute_hbond_helper(atom_name: str, res: Residue, v: np.ndarray) -> float:
    # Check if it is a polar hydrogen.
    if is_polar_hydrogen(atom_name, res.get_resname()):
        donor_atom_name = DONOR_ATOMS[atom_name]
        a = res[donor_atom_name].get_coord()  # N/O
        b = res[atom_name].get_coord()  # H
        # Donor-H is always 180.0 degrees, = pi
        angle_deviation = compute_angle_deviation(a, b, v, np.pi)
        angle_penalty = compute_angle_penalty(angle_deviation)
        return 1.0 * angle_penalty
    # Check if it is an acceptor oxygen or nitrogen
    elif is_acceptor_atom(atom_name, res):
        acceptor_atom = res[atom_name]
        b = acceptor_atom.get_coord()
        a = res[ACCEPTOR_ANGLES[atom_name]].get_coord()
        # 120 degress for acceptor
        angle_deviation = compute_angle_deviation(a, b, v, 2 * np.pi / 3)
        # TODO: This should not be 120 for all atoms, i.e. for HIS it should be
        #       ~125.0
        angle_penalty = compute_angle_penalty(angle_deviation)
        plane_penalty = 1.0
        if atom_name in ACCEPTOR_PLANES:
            try:
                d = res[ACCEPTOR_PLANES[atom_name]].get_coord()
            except:
                return 0.0
            plane_deviation = compute_plane_deviation(d, a, b, v)
            plane_penalty = compute_angle_penalty(plane_deviation)
        return -1.0 * angle_penalty * plane_penalty
        # Compute the
    return 0.0


# Compute the absolute value of the deviation from theta
def compute_angle_deviation(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                            theta: float) -> float:
    """
    Computes the absolute angle deviation from theta. a, b, c form the three
    points that define the angle.

    Parameters
    ----------
    a: np.ndarray,
        Coordinate vector of the first point
    b: np.ndarray,
        Coordinate vector of the second point
    c: np.ndarray,
        Coordinate vector of the third point
    theta: float,
        Angle to compute deviation with respect to.

    Returns
    -------
    (float) absolute deviation of the angle formed by a, b, c with theta
    """
    return abs(calc_angle(Vector(a), Vector(b), Vector(c)) - theta)


def compute_plane_deviation(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                            d: np.ndarray) -> float:
    """
    Computes the absolute angle deviation from theta. a, b, c form the three
    points that define the angle.

    Parameters
    ----------
    a: np.ndarray,
        Coordinate vector of the first point
    b: np.ndarray,
        Coordinate vector of the second point
    c: np.ndarray,
        Coordinate vector of the third point
    d: np.ndarray,
        Coordinate vector of the fourth point

    Returns
    -------
    (float) absolute deviation of the angle formed by a, b, c with theta
    """
    dih = calc_dihedral(Vector(a), Vector(b), Vector(c), Vector(d))
    dev1 = abs(dih)
    dev2 = np.pi - abs(dih)
    return min(dev1, dev2)


# angle_deviation from ideal value. TODO: do a more data-based solution
def compute_angle_penalty(angle_deviation: float) -> float:
    # Standard deviation: HBOND_STD_DEV
    return max(0.0, 1.0 - (angle_deviation / (HBOND_STD_DEV))**2)


def is_polar_hydrogen(atom_name: str, res_name: str) -> bool:
    """Check if the atom in a given residue has polar hydrogens.

    Parameters
    ----------
    atom_name: str,
        Name of the atom
    res_name: str,
        Residue the
    """
    if atom_name in POLAR_HYDROGENS[res_name]:
        return True
    else:
        return False


def is_acceptor_atom(atom_name: str, res: Residue) -> bool:
    if atom_name.startswith("O"):
        return True
    else:
        if res.get_resname() == "HIS":
            if atom_name == "ND1" and "HD1" not in res:
                return True
            if atom_name == "NE2" and "HE2" not in res:
                return True
    return False


# Compute the list of backbone C=O:H-N that are satisfied. These will be ignored.
def compute_satisfied_CO_HN(atoms):
    ns = NeighborSearch(atoms)
    satisfied_CO = set()
    satisfied_HN = set()
    for atom1 in atoms:
        res1 = atom1.get_parent()
        if atom1.get_id() == "O":
            neigh_atoms = ns.search(atom1.get_coord(), 2.5, level="A")
            for atom2 in neigh_atoms:
                if atom2.get_id() == "H":
                    res2 = atom2.get_parent()
                    # Ensure they belong to different residues.
                    if res2.get_id() != res1.get_id():
                        # Compute the angle N-H:O, ideal value is 180 (but in
                        # helices it is typically 160) 180 +-30 = pi
                        angle_N_H_O_dev = compute_angle_deviation(
                            res2["N"].get_coord(),
                            atom2.get_coord(),
                            atom1.get_coord(),
                            np.pi,
                        )
                        # Compute angle H:O=C, ideal value is ~160 +- 20 = 8*pi/9
                        angle_H_O_C_dev = compute_angle_deviation(
                            atom2.get_coord(),
                            atom1.get_coord(),
                            res1["C"].get_coord(),
                            8 * np.pi / 9,
                        )
                        ## Allowed deviations: 30 degrees (pi/6) and 20 degrees
                        #       (pi/9)
                        if (angle_N_H_O_dev - np.pi / 6 < 0 and
                                angle_H_O_C_dev - np.pi / 9 < 0.0):
                            satisfied_CO.add(res1.get_id())
                            satisfied_HN.add(res2.get_id())
    return satisfied_CO, satisfied_HN


def normalize_electrostatics(in_elec: np.ndarray) -> np.ndarray:
    """
    Normalizing charges on the surface, by clipping to upper and lower thresholds
    and converting all values to a -1/1 scale.

    Parameters
    ----------
    in_elec: np.ndarray,
        Input charges for all surface vertices

    Returns
    -------
    elec: np.ndarray,
        Normalized surface vertex charges
    """
    elec = np.copy(in_elec)
    upper_threshold = 3
    lower_threshold = -3
    elec[elec < lower_threshold] = lower_threshold
    elec[elec > upper_threshold] = upper_threshold
    elec = elec - lower_threshold
    elec = elec / (upper_threshold - lower_threshold)
    elec = 2 * elec - 1
    return elec
