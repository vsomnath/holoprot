import argparse
import numpy as np
import os
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from Bio.PDB import PDBParser, Selection

parser = PDBParser()

# RADII for atoms in explicit case.
RADII = {}
RADII["N"] = "1.540000"
RADII["N"] = "1.540000"
RADII["O"] = "1.400000"
RADII["C"] = "1.740000"
RADII["H"] = "1.200000"
RADII["S"] = "1.800000"
RADII["P"] = "1.800000"
RADII["Z"] = "1.39"
RADII["X"] = "0.770000"  ## RADII of CB or CA in disembodied case.
# This  polar hydrogen's names correspond to that of the program Reduce.
POLAR_HYDROGENS = {}
POLAR_HYDROGENS["ALA"] = ["H"]
POLAR_HYDROGENS["GLY"] = ["H"]
POLAR_HYDROGENS["SER"] = ["H", "HG"]
POLAR_HYDROGENS["THR"] = ["H", "HG1"]
POLAR_HYDROGENS["LEU"] = ["H"]
POLAR_HYDROGENS["ILE"] = ["H"]
POLAR_HYDROGENS["VAL"] = ["H"]
POLAR_HYDROGENS["ASN"] = ["H", "HD21", "HD22"]
POLAR_HYDROGENS["GLN"] = ["H", "HE21", "HE22"]
POLAR_HYDROGENS["ARG"] = ["H", "HH11", "HH12", "HH21", "HH22", "HE"]
POLAR_HYDROGENS["HIS"] = ["H", "HD1", "HE2"]
POLAR_HYDROGENS["TRP"] = ["H", "HE1"]
POLAR_HYDROGENS["PHE"] = ["H"]
POLAR_HYDROGENS["TYR"] = ["H", "HH"]
POLAR_HYDROGENS["GLU"] = ["H"]
POLAR_HYDROGENS["ASP"] = ["H"]
POLAR_HYDROGENS["LYS"] = ["H", "HZ1", "HZ2", "HZ3"]
POLAR_HYDROGENS["PRO"] = []
POLAR_HYDROGENS["CYS"] = ["H"]
POLAR_HYDROGENS["MET"] = ["H"]

HBOND_STD_DEV = np.pi / 3

# Dictionary from an acceptor atom to its directly bonded atom on which to
# compute the angle.
ACCEPTOR_ANGLES = {}
ACCEPTOR_ANGLES["O"] = "C"
ACCEPTOR_ANGLES["O1"] = "C"
ACCEPTOR_ANGLES["O2"] = "C"
ACCEPTOR_ANGLES["OXT"] = "C"
ACCEPTOR_ANGLES["OT1"] = "C"
ACCEPTOR_ANGLES["OT2"] = "C"
# Dictionary from acceptor atom to a third atom on which to compute the plane.
ACCEPTOR_PLANES = {}
ACCEPTOR_PLANES["O"] = "CA"
# Dictionary from an H atom to its donor atom.
DONOR_ATOMS = {}
DONOR_ATOMS["H"] = "N"
# Hydrogen bond information.
# ARG
# ARG NHX
# Angle: NH1, HH1X, point and NH2, HH2X, point 180 degrees.
# RADII from HH: RADII[H]
# ARG NE
# Angle: ~ 120 NE, HE, point, 180 degrees
DONOR_ATOMS["HH11"] = "NH1"
DONOR_ATOMS["HH12"] = "NH1"
DONOR_ATOMS["HH21"] = "NH2"
DONOR_ATOMS["HH22"] = "NH2"
DONOR_ATOMS["HE"] = "NE"

# ASN
# Angle ND2,HD2X: 180
# Plane: CG,ND2,OD1
# Angle CG-OD1-X: 120
DONOR_ATOMS["HD21"] = "ND2"
DONOR_ATOMS["HD22"] = "ND2"
# ASN Acceptor
ACCEPTOR_ANGLES["OD1"] = "CG"
ACCEPTOR_PLANES["OD1"] = "CB"

# ASP
# Plane: CB-CG-OD1
# Angle CG-ODX-point: 120
ACCEPTOR_ANGLES["OD2"] = "CG"
ACCEPTOR_PLANES["OD2"] = "CB"

# GLU
# PLANE: CD-OE1-OE2
# ANGLE: CD-OEX: 120
# GLN
# PLANE: CD-OE1-NE2
# Angle NE2,HE2X: 180
# ANGLE: CD-OE1: 120
DONOR_ATOMS["HE21"] = "NE2"
DONOR_ATOMS["HE22"] = "NE2"
ACCEPTOR_ANGLES["OE1"] = "CD"
ACCEPTOR_ANGLES["OE2"] = "CD"
ACCEPTOR_PLANES["OE1"] = "CG"
ACCEPTOR_PLANES["OE2"] = "CG"

# HIS Acceptors: ND1, NE2
# Plane ND1-CE1-NE2
# Angle: ND1-CE1 : 125.5
# Angle: NE2-CE1 : 125.5
ACCEPTOR_ANGLES["ND1"] = "CE1"
ACCEPTOR_ANGLES["NE2"] = "CE1"
ACCEPTOR_PLANES["ND1"] = "NE2"
ACCEPTOR_PLANES["NE2"] = "ND1"

# HIS Donors: ND1, NE2
# Angle ND1-HD1 : 180
# Angle NE2-HE2 : 180
DONOR_ATOMS["HD1"] = "ND1"
DONOR_ATOMS["HE2"] = "NE2"

# TRP Donor: NE1-HE1
# Angle NE1-HE1 : 180
DONOR_ATOMS["HE1"] = "NE1"

# LYS Donor NZ-HZX
# Angle NZ-HZX : 180
DONOR_ATOMS["HZ1"] = "NZ"
DONOR_ATOMS["HZ2"] = "NZ"
DONOR_ATOMS["HZ3"] = "NZ"

# TYR acceptor OH
# Plane: CE1-CZ-OH
# Angle: CZ-OH 120
ACCEPTOR_ANGLES["OH"] = "CZ"
ACCEPTOR_PLANES["OH"] = "CE1"

# TYR donor: OH-HH
# Angle: OH-HH 180
DONOR_ATOMS["HH"] = "OH"
ACCEPTOR_PLANES["OH"] = "CE1"

# SER acceptor:
# Angle CB-OG-X: 120
ACCEPTOR_ANGLES["OG"] = "CB"

# SER donor:
# Angle: OG-HG-X: 180
DONOR_ATOMS["HG"] = "OG"

# THR acceptor:
# Angle: CB-OG1-X: 120
ACCEPTOR_ANGLES["OG1"] = "CB"

# THR donor:
# Angle: OG1-HG1-X: 180
DONOR_ATOMS["HG1"] = "OG1"


def str2bool(v: str) -> bool:
    """Converts str to bool.

    Parameters
    ----------
    v: str,
        String element

    Returns
    -------
    boolean version of v
    """
    v = v.lower()
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")

def get_global_agg(agg_type: str = 'mean'):
    if agg_type == 'mean':
        return global_mean_pool
    elif agg_type == 'sum':
        return global_add_pool
    elif agg_type == 'max':
        return global_max_pool
    else:
        raise ValueError(f"Aggregation of type {agg_type} is not supported.")

def get_residues(pdb_file):
    global parser
    pdb_file = os.path.abspath(pdb_file)
    base_file = pdb_file.split("/")[-1]  # Remove any full path prefixes
    pdb_id = base_file.split(".")[0]

    struct = parser.get_structure(file=pdb_file, id=pdb_id)
    residues = Selection.unfold_entities(struct, "R")
    # Remove heteroatoms
    residues = [res for res in residues if res.get_full_id()[3][0] == " "]
    return residues
