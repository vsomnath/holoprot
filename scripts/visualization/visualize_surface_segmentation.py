

# imports
import os

import numpy as np
import torch
import pymesh
import pyvista as pv
from pyvtk import PolyData, PointData, CellData, Scalars, Vectors, VtkData
import holoprot
from holoprot.utils.surface import get_surface, compute_normal
from holoprot.feat.surface import compute_surface_features


ROOT_DIR = os.environ['PROT']
MSMS_BIN = os.environ['MSMS_BIN']


def save_vtk(fname, xyz, triangles=None, values=None, vectors=None,
             triangle_values=None):
    """Saves a point cloud or triangle mesh as a .vtk file.

    Files can be opened with Paraview or displayed using the PyVista library.
    Args:
        fname (string): filename.
        xyz (Tensor): (N,3) point cloud or vertices.
        triangles (integer Tensor, optional): (T,3) mesh connectivity.
        values (Tensor, optional): (N,D) values, supported by the vertices.
        vectors (Tensor, optional): (N,3) vectors, supported by the vertices.
        triangle_values (Tensor, optional): (T,D) values, supported by triangles.
    """

    # encode the points/vertices as a VTK structure:
    if triangles is None:
        # point cloud
        structure = PolyData(points=xyz)
    else:
        # surface mesh
        structure = PolyData(points=xyz, polygons=triangles)

    data = [structure]
    pointdata, celldata = [], []

    # point values - one channel per column of the `values` array
    if values is not None:
        values = values
        if len(values.shape) == 1:
            values = values[:, None]
        features = values.T
        pointdata += [
            Scalars(f,
                    name=f"features_{i:02d}") for i, f in enumerate(features)
        ]

    # point vectors - one vector per point
    if vectors is not None:
        pointdata += [Vectors(vectors, name="vectors")]

    # store in the VTK object:
    if pointdata != []:
        pointdata = PointData(*pointdata)
        data.append(pointdata)

    # triangle values - one channel per column of the `triangle_values` array
    if triangle_values is not None:
        triangle_values = triangle_values
        if len(triangle_values.shape) == 1:
            triangle_values = triangle_values[:, None]
        features = triangle_values.T
        celldata += [
            Scalars(f,
                    name=f"features_{i:02d}") for i, f in enumerate(features)
        ]

        celldata = CellData(*celldata)
        data.append(celldata)

    #  write to hard drive
    vtk = VtkData(*data)
    vtk.tofile(fname)


def load_mesh_from_file(mesh_file: str):
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


def get_assignments(pdb):
    # set data path
    packagedir = holoprot.__path__[0]
    filename = os.path.join(ROOT_DIR,
                            'datasets/assignments/pdbbind',
                            pdb.upper() + '.pth')
    # load file
    assigment = torch.load(filename)

    return assigment


def get_mesh(pdb):
    # set data path
    packagedir = holoprot.__path__[0]
    filename = os.path.join(ROOT_DIR,
                            'datasets/surface_mesh/processed/pdbbind',
                            pdb + '.obj')

    # load file
    mesh = load_mesh_from_file(filename)
    return mesh


def get_pdb_file(pdb):
    # set data path
    packagedir = holoprot.__path__[0]
    filename = os.path.join(ROOT_DIR,
                            f'datasets/raw/pdbbind/pdb_files/{pdb}',
                            pdb + '.pdb')
    return filename


def get_surface_features(pdb, feature):
    # get mesh
    mesh = get_mesh(pdb)

    # get surface
    pdb_file = get_pdb_file(pdb)
    surface = get_surface(pdb_file, msms_bin=MSMS_BIN)

    # compute features of surface
    features = compute_surface_features(surface, pdb_file, mesh)

    if feature == 'hydrophob':
        return features[2]
    elif feature == 'shape':
        return features[0]
    elif feature == 'charge':
        return features[3]
    else:
        raise NotImplementedError


def create_mesh(mesh):
    save_vtk('tmp.vtk', mesh.vertices, mesh.faces.astype(int))
    mesh = pv.PolyData('tmp.vtk')
    return mesh


def main(args):
    # get surface and assignments
    mesh = get_mesh(args.pdb)
    if args.patch:
        cas = get_assignments(args.pdb)
    if not args.patch:
        cas = get_surface_features(args.pdb, args.feature)

    # create mesh
    mesh = create_mesh(mesh)

    # plot surface with feature visualization
    pl = pv.Plotter()
    pl.add_mesh(mesh, scalars=cas, cmap='RdBu')
    pl.background_color = 'white'
    pl.camera_position = 'xy'
    pl.show(screenshot=args.out + '.jpeg')


if __name__ == '__main__':
    import argparse

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str,
                        help='PDB identifies.')
    parser.add_argument('--out', type=str, default="test",
                        help='Filename of output.')
    parser.add_argument('--patch', action="store_true",
                        help='Visualization of patches.')
    parser.add_argument('--feature', type=str,
                        help='Visualization of surface feature.')
    args = parser.parse_args()

    main(args)
