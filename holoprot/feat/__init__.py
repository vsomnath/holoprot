from rdkit import Chem

#Kyte-Doolittle scale for hydrophobicity
KD_SCALE = {}
KD_SCALE["ILE"] = 4.5
KD_SCALE["VAL"] = 4.2
KD_SCALE["LEU"] = 3.8
KD_SCALE["PHE"] = 2.8
KD_SCALE["CYS"] = 2.5
KD_SCALE["MET"] = 1.9
KD_SCALE["ALA"] = 1.8
KD_SCALE["GLY"] = -0.4
KD_SCALE["THR"] = -0.7
KD_SCALE["SER"] = -0.8
KD_SCALE["TRP"] = -0.9
KD_SCALE["TYR"] = -1.3
KD_SCALE["PRO"] = -1.6
KD_SCALE["HIS"] = -3.2
KD_SCALE["GLU"] = -3.5
KD_SCALE["GLN"] = -3.5
KD_SCALE["ASP"] = -3.5
KD_SCALE["ASN"] = -3.5
KD_SCALE["LYS"] = -3.9
KD_SCALE["ARG"] = -4.5

# Symbols for different atoms
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', \
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', \
    'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', \
    'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', \
    'Ce','Gd','Ga','Cs', '*', 'unk']

AMINO_ACIDS = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET',
    'ASN', 'PYL', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'SEC', 'VAL', 'TRP', 'TYR',
    'unk'
]

SECONDARY_STRUCTS = ['H', 'G', 'I', 'E', 'B', 'T', 'C', 'unk']

MAX_NB = 10
DEGREES = list(range(MAX_NB))
EXP_VALENCE = [1, 2, 3, 4, 5, 6]
IMP_VALENCE = [0, 1, 2, 3, 4, 5]

BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, \
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

BOND_FLOATS = [0.0, 1.0, 2.0, 3.0, 1.5]

ATOM_FDIM = len(ATOM_LIST) + len(DEGREES) + len(EXP_VALENCE) + len(
    IMP_VALENCE) + 1
BOND_FDIM = 6
CONTACT_FDIM = 2
SURFACE_NODE_FDIM = 4
SURFACE_EDGE_FDIM = 5
PATCH_NODE_FDIM = 4
PATCH_EDGE_FDIM = 4
