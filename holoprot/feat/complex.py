"""
Functions to compute features for ligand and protein backbone graphs.
"""
import numpy as np
from rdkit import Chem
from typing import List, Union, Set, Any
from Bio.PDB.Residue import Residue
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from typing import List, Tuple, Dict

from holoprot.feat import SECONDARY_STRUCTS, AMINO_ACIDS, ATOM_LIST
from holoprot.feat import IMP_VALENCE, EXP_VALENCE, DEGREES, ATOM_FDIM
from holoprot.feat import BOND_FDIM, BOND_TYPES

idxfunc = lambda a: a.GetAtomMapNum() - 1
bond_idx_fn = lambda a, b, mol: mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx()).GetIdx()

class ResidueProp(object):
    """Wrapper class that holds all attributes of a protein residue."""

    def __init__(self,
                 residue: Residue,
                 sec: str,
                 sas: float,
                 hydrophobicity: float,
                 res_depth: float = None,
                 ca_depth: float = None) -> None:
        """
        Parameters
        ----------
        residue: Residue,
            Instance of the Bio.PDB.Residue.Residue
        sec: str,
            Single letter indicating the secondary structure. Refer SECONDARY_STRUCTS
            above for possible codes.
        sas: float,
            Solvent accessible surface area (TODO: (vsomnath): Normalize?)
        res_depth: float,
            Depth of residue, calculated as average depth of all atoms
        ca_depth: float,
            Depth of Calpha atom of the residue
        """
        self.name = residue.get_resname()
        self.sec = sec
        self.sas = sas
        self.res_depth = res_depth
        self.ca_depth = ca_depth
        self.hydrophobicity = hydrophobicity  # Hydrophobicity using the Kyte-Doolittle scale


class AtomProp(object):
    """Wrapper class that holds all properties of an atom."""

    def __init__(self, atom: Chem.Atom) -> None:
        """
        Parameters
        ----------
        atom: Chem.Atom,
            Instance of rdkit.Chem.Atom
        """
        self.symbol = atom.GetSymbol()
        self.degree = atom.GetDegree()
        self.exp_valence = atom.GetExplicitValence()
        self.imp_valence = atom.GetImplicitValence()
        self.is_aromatic = atom.GetIsAromatic()


class BondProp(object):
    """Wrapper class that holds all properties of a bond."""

    def __init__(self, bond: Chem.Bond) -> None:
        """
        Parameters
        ----------
        bond: Chem.Bond,
            Instance of rdkit.Chem.Bond
        """
        self.bond_type = bond.GetBondType()
        self.is_conj = bond.GetIsConjugated()
        self.is_ring = bond.IsInRing()


def onek_encoding_unk(x: Any, allowable_set: Union[List, Set]) -> List:
    """Converts x to one hot encoding.

    Parameters
    ----------
    x: Any,
        An element of any type
    allowable_set: Union[List, Set]
        Allowable element collection

    Returns
    -------
    list, indicating the one hot encoding of x in allowable_set
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))


def get_atom_features(atom_prop: AtomProp, **kwargs) -> np.ndarray:
    """
    Get atom features. The atom features computed

    Parameters
    ----------
    atom: Chem.Atom,
        Atom object from RDKit

    Returns
    -------
    atom_features: np.ndarray,
        Array of atom features
    """
    if atom_prop == "*":
        return np.array([0] * ATOM_FDIM)
    atom_features = np.array(
        onek_encoding_unk(atom_prop.symbol, ATOM_LIST) +
        onek_encoding_unk(atom_prop.degree, DEGREES) +
        onek_encoding_unk(atom_prop.exp_valence, EXP_VALENCE) +
        onek_encoding_unk(atom_prop.imp_valence, IMP_VALENCE) +
        [float(atom_prop.is_aromatic)])
    return atom_features


def get_bond_features(bond_prop: BondProp, **kwargs) -> np.ndarray:
    """
    Get bond features. Features computed are a one hot encoding of the bond type,
    its aromaticity and ring membership.

    Parameters
    ----------
    bond: Chem.Bond,
        bond object

    Returns
    -------
    bond_features: np.ndarray,
        Array of bond features
    """
    if bond_prop == "*":
        return np.array([0] * BOND_FDIM)
    bt = bond_prop.bond_type
    bond_features = [float(bt == bond_type) for bond_type in BOND_TYPES[1:]]
    bond_features.extend([float(bond_prop.is_conj), float(bond_prop.is_ring)])
    bond_features = np.array(bond_features, dtype=np.float32)
    return bond_features


def get_residue_features(residue_prop: ResidueProp,
                         use_depth: bool = False,
                         **kwargs) -> np.ndarray:
    """Get residue features.

    Parameters
    ----------
    residue_prop: ResidueProp
        Instance of the ResidueProp class that captures properties of a residue

    Returns
    -------
    res_features: np.ndarray,
        Array of residue features
    """
    if residue_prop == "*":
        if use_depth:
            return np.array(
                [0] * (len(AMINO_ACIDS) + len(SECONDARY_STRUCTS) + 4))
        else:
            return np.array(
                [0] * (len(AMINO_ACIDS) + len(SECONDARY_STRUCTS) + 2))
    res_features = onek_encoding_unk(residue_prop.name, AMINO_ACIDS) + \
                   onek_encoding_unk(residue_prop.sec.upper(), SECONDARY_STRUCTS) + \
                   [residue_prop.sas, residue_prop.hydrophobicity]
    if use_depth:
        res_features.extend([residue_prop.res_depth, residue_prop.ca_depth])
    res_features = np.array(res_features)
    return res_features


def compute_normal(residue: Residue) -> np.ndarray:
    """
    Compute the normal vector for a given residue. The normal vector is estimated
    as the cross product of the vectors formed by the difference of Calpha, C and
    O coordinates. The normal vector is length normalized to get a unit vector.

    Parameters
    ----------
    residue: Residue,
        Residue for which we want to compute normal

    Returns
    -------
    normal: np.ndarray,
        The normal vector for the residue
    """
    x_ca = residue['CA'].get_coord()
    x_c = residue['C'].get_coord()
    x_o = residue['O'].get_coord()
    x_oc = x_o - x_c
    x_cac = x_ca - x_c
    normal = np.cross(x_oc, x_cac)
    normal /= (np.sqrt(np.sum(normal**2) + 1e-8))  # Normalize by length
    return normal


def compute_angle(residue_pair: Tuple[Residue]) -> float:
    """
    Compute the angle between two residues. The angle is estimated as the cosine
    inverse of the dot product between the normal vectors of the two residues. The
    angle is normalized by dividing it by 2 * \pi

    Parameters
    ----------
    residue_pair: Tuple[Residue]
        The pair of residues between which we want to estimate the angle.

    Returns
    -------
    normalized angle (float) between the residues
    """
    res_i, res_j = residue_pair
    norm_i = compute_normal(res_i)
    norm_j = compute_normal(res_j)

    angle = np.arccos(norm_i.dot(norm_j))
    return angle / (2 * np.pi)


def get_contact_features(residue_pair: Tuple[Residue],
                         mode: str = 'ca',
                         sigma: float = 0.01,
                         **kwargs) -> np.ndarray:
    """
    Gets contact features. The features computed are the RBF kernel over the
    distance between residues and the angle between the residues. The RBF kernel's
    width is modulated by the parameter `sigma`, and the mode to compute distance
    is controlled by `mode` argument.

    Parameters
    ----------
    residue_pair: Tuple[Residue],
        pass
    mode: str, (default ca)
        Compute distance between two residues. Allowed options are
        `ca` (distance between calpha atoms) and `com` (distance between
        center of masses of residues)
    sigma: float
        Width of the gaussian kernel over the contact.

    Returns
    -------
    edge_features: np.ndarray,
        Features of the contact between residues
    """
    if residue_pair == "*":
        return np.array([0, 0])

    res_i, res_j = residue_pair
    if mode == 'ca':
        coord_i = res_i['CA'].get_coord()
        coord_j = res_j['CA'].get_coord()

    elif mode == 'com':
        coord_i = np.mean(
            [atom.get_coord() for atom in res_i.get_list()], axis=0)
        coord_j = np.mean(
            [atom.get_coord() for atom in res_j.get_list()], axis=0)

    else:
        raise ValueError(
            f"Computing distance with mode {mode} is not supported.")

    y = coord_i - coord_j
    dist = np.exp(-np.sum(y**2) / sigma**2)
    angle = compute_angle(residue_pair)

    edge_features = np.array([dist, angle])
    return edge_features


def get_secondary_struct_features(pdb_file: str,
                                  dssp_bin: str = 'dssp') -> Dict[str, List]:
    """Compute secondary structure features for the protein using DSSP.

    Parameters
    ----------
    pdb_file: str,
        PDB file for the protein.
    dssp_bin: str,
        Path to the DSSP binary executable

    Returns
    -------
    dssp_dict: Dict[str, List]
        Dictionary containing the secondary structure features for residues
    """
    dssp_dict = dssp_dict_from_pdb_file(pdb_file, DSSP=dssp_bin)[0]
    return dssp_dict