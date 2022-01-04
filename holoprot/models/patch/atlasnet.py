import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from typing import List, Dict, Union
import numpy as np

from holoprot.layers.patch.patches import Mapping2DtoND, get_template
from holoprot.layers.patch.diff_props import DiffGeomProps

class AtlasNet(nn.Module):

    def __init__(self,
                 n_primitives: int,
                 template: str,
                 mapping_config: Dict,
                 remove_bnorm: bool = False,
                 diff_toggles: Dict = None,
                 surface_recon: bool = False,
                 device: str = 'cpu',
                 **kwargs):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        Author : Thibault Groueix 01.11.2019
        :param opt:
        """
        super(AtlasNet, self).__init__(**kwargs)
        self.n_primitives = n_primitives
        self.remove_bnorm = remove_bnorm
        self.template = template
        self.mapping_config = mapping_config
        self.diff_toggles = diff_toggles
        self.surface_recon = surface_recon
        self.device = device
        self._build_components()

    def _build_components(self):
        # Define number of points per primitives
        # self.points_primitive = self.n_points // self.n_primitives
        # self.points_primitive_eval = self.n_points_eval // self.n_primitives

        if self.remove_bnorm:
            torch.nn.BatchNorm1d = Identity
            print("Replacing all batchnorms by identities.", flush=True)

        # Initialize templates
        self.template = [get_template(self.template, device=self.device)
                         for i in range(0, self.n_primitives)]

        # Intialize deformation networks
        self.decoder = nn.ModuleList([Mapping2DtoND(**self.mapping_config)
                                      for i in range(0, self.n_primitives)])

        if self.diff_toggles is not None:
            self.dgp = DiffGeomProps(normals=self.diff_toggles.get('normals', True),
                                     curv_mean=self.diff_toggles.get('curv_mean', False),
                                     curv_gauss=self.diff_toggles.get('curv_gauss', False),
                                     fff=self.diff_toggles.get('fff', True),
                                     device=self.device)


    def forward(self, latent_vector, input_shape):
        """
        Deform points from self.template using the embedding latent_vector
        :param latent_vector: an opt.bottleneck size vector encoding a 3D shape or an image. size : batch, bottleneck
        :return: A deformed pointcloud os size : batch, nb_prim, num_point, 3
        """
        if self.training:
            points_primitive = input_shape[-1] // self.n_primitives
            input_points = [self.template[i].get_random_points(
                torch.Size((latent_vector.size(0), self.template[i].dim, points_primitive)),
                latent_vector.device) for i in range(self.n_primitives)]
        else:
            points_primitive = input_shape[-1] // self.n_primitives
            input_points = [self.template[i].get_regular_points(points_primitive,
                                                                device=latent_vector.device)
                            for i in range(self.n_primitives)]

        # Deform each patch
        output_points = torch.cat([self.decoder[i](input_points[i],
                                   latent_vector.unsqueeze(2)).unsqueeze(1) for i in
                                   range(0, self.n_primitives)], dim=1)
        output_points = output_points.contiguous()

        geom_props = {"computed": False}
        if self.training and self.diff_toggles is not None:
            if self.surface_recon:
                geom_props = self.dgp(output_points[:, :, :, :3], input_points)
            else:
                geom_props = self.dgp(output_points, input_points)
            geom_props["computed"] = True

        # Return the deformed pointcloud
        return output_points, geom_props  # batch, nb_prim, num_point, 3

    def generate_mesh(self, latent_vector, input_shape):
        assert latent_vector.size(0)==1, "input should have batch size 1!"
        import pymesh
        input_points = [self.template[i].get_regular_points(self.points_primitive, latent_vector.device)
                        for i in range(self.n_primitives)]
        input_points = [input_points[i] for i in range(self.n_primitives)]

        # Deform each patch
        output_points = [self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).squeeze() for i in
                         range(0, self.n_primitives)]

        output_meshes = [pymesh.form_mesh(vertices=output_points[i].transpose(1, 0).contiguous().cpu().numpy(),
                                          faces=self.template[i].mesh.faces)
                         for i in range(self.n_primitives)]

        # Deform return the deformed pointcloud
        mesh = pymesh.merge_meshes(output_meshes)

        return mesh

    def generate_patch_labels(self, latent_vector, x):
        assert latent_vector.size(0) == 1, "input should have batch size 1!"
        input_shape = x.shape
        points_primitive = input_shape[-1] // self.n_primitives

        patch_points = [self.template[i].get_regular_points(points_primitive, latent_vector.device)
                        for i in range(self.n_primitives)]
        patch_labels = [[i] * points_primitive for i in range(self.n_primitives)]
        patch_labels = np.asarray(patch_labels).flatten()
        patch_points = [patch_points[i] for i in range(self.n_primitives)]

        output_points = [self.decoder[i](patch_points[i], latent_vector.unsqueeze(2)).squeeze() for i in
                         range(0, self.n_primitives)]

        output_points = torch.cat(output_points, dim=1).transpose(1, 0).contiguous()

        if latent_vector.device == 'cpu':
            input_points = x.transpose(1, 2).contiguous().squeeze(0).numpy()
            output_points = output_points.numpy()

        else:
            input_points = x.transpose(1, 2).contiguous().squeeze(0).cpu().numpy()
            output_points = output_points.cpu().numpy()

        from scipy.spatial import KDTree
        tree = KDTree(data=output_points)
        closest_point_idxs = tree.query(input_points)[1]

        input_assignments = patch_labels[closest_point_idxs]
        assert len(input_assignments) == len(input_points)
        return input_assignments, input_points
