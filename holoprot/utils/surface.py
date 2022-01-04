"""
Utilities for preparing and computing features on molecular surfaces.
"""
import os
import numpy as np
from numpy.matlib import repmat
from Bio.PDB import PDBParser, Selection
from subprocess import Popen, PIPE
from typing import Tuple, List, Dict

from holoprot.utils import RADII, POLAR_HYDROGENS

eps = 1e-6
Surface = Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, str]]


def read_msms(
        file_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Read surface constituents from output files generated using MSMS.

    Parameters
    ----------
    file_root: str,
        Root name used in saving different output files from running MSMS.

    Returns
    -------
    vertices:

    """
    vertfile = open(file_root + ".vert")
    meshdata = (vertfile.read().rstrip()).split("\n")
    vertfile.close()

    # Read number of vertices.
    count = {}
    header = meshdata[2].split()
    count["vertices"] = int(header[0])
    ## Data Structures
    vertices = np.zeros((count["vertices"], 3))
    normalv = np.zeros((count["vertices"], 3))
    atom_id = [""] * count["vertices"]
    res_id = [""] * count["vertices"]
    for i in range(3, len(meshdata)):
        fields = meshdata[i].split()
        vi = i - 3
        vertices[vi][0] = float(fields[0])
        vertices[vi][1] = float(fields[1])
        vertices[vi][2] = float(fields[2])
        normalv[vi][0] = float(fields[3])
        normalv[vi][1] = float(fields[4])
        normalv[vi][2] = float(fields[5])
        atom_id[vi] = fields[7]
        res_id[vi] = fields[9]
        count["vertices"] -= 1

    # Read faces.
    facefile = open(file_root + ".face")
    meshdata = (facefile.read().rstrip()).split("\n")
    facefile.close()

    # Read number of vertices.
    header = meshdata[2].split()
    count["faces"] = int(header[0])
    faces = np.zeros((count["faces"], 3), dtype=int)
    normalf = np.zeros((count["faces"], 3))

    for i in range(3, len(meshdata)):
        fi = i - 3
        fields = meshdata[i].split()
        faces[fi][0] = int(fields[0]) - 1
        faces[fi][1] = int(fields[1]) - 1
        faces[fi][2] = int(fields[2]) - 1
        count["faces"] -= 1

    assert count["vertices"] == 0
    assert count["faces"] == 0

    return vertices, faces, normalv, res_id


def output_pdb_as_xyzrn(pdb_file: str) -> None:
    """Converts a .pdb file to a .xyzrn file.

    Parameters
    ----------
    pdb_file: str,
        PDB File to convert
    """
    pdb_path = os.path.abspath(pdb_file)
    xyzrn_file = pdb_path.split(".")[0] + ".xyzrn"

    base_file = pdb_path.split("/")[-1]  # Remove any full path prefixes
    pdb_id = base_file.split(".")[0]

    parser = PDBParser()
    struct = parser.get_structure(id=pdb_id, file=pdb_file)
    outfile = open(xyzrn_file, "w")

    for atom in struct.get_atoms():
        name = atom.get_name()
        residue = atom.get_parent()
        # Ignore hetatms.
        if residue.get_id()[0] != " ":
            continue
        resname = residue.get_resname()
        reskey = residue.get_id()[1]
        chain = residue.get_parent().get_id()
        atomtype = name[0]

        color = "Green"
        coords = None
        if atomtype in RADII and resname in POLAR_HYDROGENS:
            if atomtype == "O":
                color = "Red"
            if atomtype == "N":
                color = "Blue"
            if atomtype == "H":
                if name in POLAR_HYDROGENS[resname]:
                    color = "Blue"  # Polar hydrogens
            coords = "{:.06f} {:.06f} {:.06f}".format(atom.get_coord()[0],
                                                      atom.get_coord()[1],
                                                      atom.get_coord()[2])
            insertion = "x"
            if residue.get_id()[2] != " ":
                insertion = residue.get_id()[2]
            full_id = "{}_{:d}_{}_{}_{}_{}".format(
                chain,
                residue.get_id()[1], insertion, resname, name, color)
        if coords is not None:
            outfile.write(coords + " " + RADII[atomtype] + " 1 " + full_id +
                          "\n")


def get_surface(pdb_file: str, density: float = 0.5,
                msms_bin: str = None, remove_files: bool = True):
    """
    Wrapper function that calls the MSMS executable to build the protein surface.

    Parameters
    ----------
    pdb_file: str,
        PDB file to extract surface from
    msms_bin: str,
        Path to the MSMSBIN file
    remove_files: bool, (default True)
        Whether to remove the intermediate output files
    """
    pdb_path = os.path.abspath(pdb_file)
    file_base = pdb_path.split(".")[0]
    xyzrn_file = pdb_path.split(".")[0] + ".xyzrn"
    output_pdb_as_xyzrn(pdb_file)

    # Now run MSMS on xyzrn file
    FNULL = open(os.devnull, 'w')
    args = [msms_bin, "-density", f"{density}", "-hdensity", "3.0", "-probe",\
                    "1.5", "-if", xyzrn_file, "-of", file_base, "-af", file_base]
    #print msms_bin+" "+`args`
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    vertices, faces, normals, names = read_msms(file_base)
    areas = {}
    ses_file = open(file_base + ".area")
    next(ses_file)  # ignore header line
    for line in ses_file:
        fields = line.split()
        areas[fields[3]] = fields[1]

    files_to_remove = [
        xyzrn_file, file_base + ".vert", file_base + ".area",
        file_base + ".face"
    ]

    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
    return vertices, faces, normals, names, areas


def prepare_mesh(vertices: np.ndarray,
                 faces: np.ndarray,
                 normals: np.ndarray = None,
                 resolution: float = 1.0,
                 apply_fixes: bool = False):
    """
    Prepare the mesh surface given vertices and faces. Optionally, compute
    normals and apply fixes to mesh.

    Parameters
    ----------
    vertices: np.ndarray,
        Surface vertices
    faces: np.ndarray,
        Triangular faces on the mesh
    normals: np.ndarray, default None,
        Normals for each vertex
    apply_fixes: bool, default False,
        Optional application of fixes to mesh. Check fix_mesh for details on fixes.

    Returns
    -------
    mesh: Mesh,
        Pymesh.Mesh.Mesh instance
    """
    import pymesh
    mesh = pymesh.form_mesh(vertices, faces)
    if apply_fixes:
        mesh = fix_mesh(mesh, resolution=resolution)

    if apply_fixes or normals is None:
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


def fix_mesh(mesh, resolution: float = 1.0):
    """
    Applies a predefined set of fixes to the mesh, and converts it to a
    specified resolution. These fixes include removing duplicated vertices wihin
    a certain threshold, removing degenerate triangles, splitting longer edges to
    a given target length, and collapsing shorter edges.

    Parameters
    ----------
    mesh: Mesh,
        Pymesh.Mesh.Mesh object
    resolution: float,
        Maximum size of edge in the mesh

    Returns
    -------
    mesh: Mesh,
        Pymesh.Mesh.Mesh object with all fixes applied
    """
    import pymesh
    target_len = resolution
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)

    count = 0
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(
            mesh, target_len, preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        #print("#v: {}".format(num_vertices));
        count += 1
        if count > 10: break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)

    return mesh


def compute_normal(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertex = vertices.T
    face = faces.T
    nface = np.size(face, 1)
    nvert = np.size(vertex, 1)
    normal = np.zeros((3, nvert))
    # unit normals to the faces
    normalf = crossp(
        vertex[:, face[1, :]] - vertex[:, face[0, :]],
        vertex[:, face[2, :]] - vertex[:, face[0, :]],
    )
    sum_squares = np.sum(normalf**2, 0)
    d = np.sqrt(sum_squares)
    d[d < eps] = 1
    normalf = normalf / repmat(d, 3, 1)
    # unit normal to the vertex
    normal = np.zeros((3, nvert))
    for i in np.arange(0, nface):
        f = face[:, i]
        for j in np.arange(3):
            normal[:, f[j]] = normal[:, f[j]] + normalf[:, i]

    # normalize
    d = np.sqrt(np.sum(normal**2, 0))
    d[d < eps] = 1
    normal = normal / repmat(d, 3, 1)
    # enforce that the normal are outward
    vertex_means = np.mean(vertex, 0)
    v = vertex - repmat(vertex_means, 3, 1)
    s = np.sum(np.multiply(v, normal), 1)
    if np.sum(s > 0) < np.sum(s < 0):
        # flip
        normal = -normal
        normalf = -normalf
    return normal.T


def crossp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.zeros((x.shape))
    z[0, :] = np.multiply(x[1, :], y[2, :]) - np.multiply(x[2, :], y[1, :])
    z[1, :] = np.multiply(x[2, :], y[0, :]) - np.multiply(x[0, :], y[2, :])
    z[2, :] = np.multiply(x[0, :], y[1, :]) - np.multiply(x[1, :], y[0, :])
    return z

def save_mesh(save_file: str, vertices: np.ndarray, faces: np.ndarray,
              normals: np.ndarray = None, charges: np.ndarray = None,
              hbonds: np.ndarray = None, hydrophob: np.ndarray = None,
              normalize_charges: bool = False):
    """Saves mesh in .ply format."""
    import pymesh
    mesh = pymesh.form_mesh(vertices, faces)
    if normals is not None:
        n1 = normals[:, 0]
        n2 = normals[:, 1]
        n3 = normals[:, 2]
        mesh.add_attribute("vertex_nx")
        mesh.set_attribute("vertex_nx", n1)
        mesh.add_attribute("vertex_ny")
        mesh.set_attribute("vertex_ny", n2)
        mesh.add_attribute("vertex_nz")
        mesh.set_attribute("vertex_nz", n3)
    if charges is not None:
        mesh.add_attribute("charge")
        if normalize_charges:
            charges = charges / 10
        mesh.set_attribute("charge", charges)
    if hbonds is not None:
        mesh.add_attribute("hbonds")
        mesh.set_attribute("hbonds", hbonds)
    if hydrophob is not None:
        mesh.add_attribute("vertex_hphob")
        mesh.set_attribute("vertex_hphob", hydrophob)
    pymesh.save_mesh(
        save_file, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True
    )
