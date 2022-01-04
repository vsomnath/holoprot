import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union
import numpy as np

from holoprot.models.patch.atlasnet import AtlasNet

class PatchPointCloud(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self,
                 config: Dict,
                 toggles: Dict,
                 device: str = 'cpu',
                 **kwargs):
        super(PatchPointCloud, self).__init__(**kwargs)
        self.config = config
        self.toggles = toggles if toggles is not None else {}
        self.device = device
        self._build_components()

    def _build_components(self):
        config = self.config
        self.encoder = PointNet(latent_dim=config['bottleneck_size'],
                                input_dim=config['input_dim'])
        self.decoder = AtlasNet(n_primitives=config['n_primitives'],
                                template=config['template'],
                                mapping_config=config['mconfig'],
                                diff_toggles=self.toggles.get('diff_toggles', None),
                                remove_bnorm=self.toggles.get('remove_bnorm', False),
                                device=self.device)
        self.apply(weights_init)  # initialization of the weights

    def get_saveables(self) -> Dict:
        """
        Get the configuration used for model construction. Used for restoring models.

        Returns
        -------
        saveables: Dict,
            Dictionary with the model configuration
        """
        saveables = {}
        saveables['config'] = self.config
        saveables['toggles'] = None if self.toggles == {} else self.toggles
        return saveables

    def forward(self, surface_data):
        inputs = torch.transpose(surface_data.pos.unsqueeze(0), 2, 1).contiguous()
        if self.toggles.get('surface_recon', False):
            feats = torch.transpose(surface_data.x.unsqueeze(0), 2, 1).contiguous()
            inputs = torch.cat([inputs, feats], dim=1)
        inputs = inputs.to(self.device)
        return self.decoder(self.encoder(inputs), inputs.shape)

    @staticmethod
    def fuse_primitives(x_prim):
        bs = x_prim.size(0)
        out_dim = x_prim.shape[-1]
        x_recon = x_prim.transpose(2, 3).contiguous()
        x_recon = x_recon.view(bs, -1, out_dim).contiguous()
        return x_recon

    @staticmethod
    def conformal_regularization(geom_props):
        """conformal regularisation. author : Theo Deprelle """
        # compute the jacobian matrix J of the 2D to 3D mapping

        # ===========================================================================
        fff = geom_props["fff"]
        m_11, m_21, m_22 = fff[..., 0], fff[..., 1], fff[..., 2]
        # ===========================================================================
        # compute the optimal scalling coef
        # ===========================================================================
        coef = (m_11 + m_22) / (m_11 ** 2 + m_22 ** 2 + 2 * m_21 ** 2 + 1e-9)
        # ===========================================================================
        # compute the conformal loss over the element of M
        # ===========================================================================
        loss = (coef * m_11 - 1) ** 2 + (coef * m_22 - 1) ** 2 + 2 * coef * m_21 ** 2
        # ===========================================================================
        return torch.mean(loss)

    def train_step(self, surface_data):
        if self.training and not hasattr(self, 'dist_chamfer_3d'):
            import holoprot.layers.chamfer.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
            import holoprot.layers.chamfer.chamfer4D.dist_chamfer_4D as dist_chamfer_4D
            print(flush=True)
            self.dist_chamfer_3d = dist_chamfer_3D.chamfer_3DDist()
            self.dist_chamfer_4d = dist_chamfer_4D.chamfer_4DDist()

        metrics = {}
        x_prim, geom_props = self(surface_data)
        x_points = x_prim[:, :, :3]
        xp_recon = PatchPointCloud.fuse_primitives(x_points)

        points = torch.transpose(surface_data.pos.unsqueeze(0), 2, 1).contiguous()
        p_inCham = points.view(points.size(0), -1, 3).contiguous().to(self.device)
        pdist1, pdist2, pidx1, pidx2 = self.dist_chamfer_3d(p_inCham, xp_recon)  # mean over points
        loss = torch.mean(pdist1) + torch.mean(pdist2)  # mean over points
        metrics['chamfer_loss_points'] = loss.item()

        if self.toggles.get('surface_recon', False):
            x_surface = x_prim[:, :, 3:]
            xs_recon = PatchPointCloud.fuse_primitives(x_surface)

            surface_feat = torch.transpose(surface_data.x.unsqueeze(0), 2, 1).contiguous()
            s_inCham = surface_feat.view(surface_feat.size(0), -1, 4).contiguous().to(self.device)
            sdist1, sdist2, sidx1, sidx2 = self.dist_chamfer_4d(s_inCham, xs_recon)
            surface_loss = torch.mean(sdist1) + torch.mean(sdist2)
            metrics['chamfer_loss_surface'] = surface_loss.item()
            loss = loss + self.config['surface_fac'] * surface_loss

        if geom_props["computed"]:
            conform_loss = PatchPointCloud.conformal_regularization(geom_props)
            metrics['conform_loss'] = conform_loss.item()
            loss = loss + self.config['conform_fac'] * conform_loss

        metrics['loss'] = loss.item()
        return loss, metrics

    def eval_step(self, eval_data = None):
        eval_metrics = {}
        if eval_data is None:
            return eval_metrics

        eval_loss = []
        self.eval()
        with torch.no_grad():
            for idx, inputs in enumerate(eval_data):
                if inputs is None:
                    continue
                x_prim, geom_props = self(inputs)
                x_points = x_prim[:, :, :3]
                xp_recon = PatchPointCloud.fuse_primitives(x_points)

                points = torch.transpose(inputs.pos.unsqueeze(0), 2, 1).contiguous()
                p_inCham = points.view(points.size(0), -1, 3).contiguous().to(self.device)
                pdist1, pdist2, pidx1, pidx2 = self.dist_chamfer_3d(p_inCham, xp_recon)  # mean over points
                loss = torch.mean(pdist1) + torch.mean(pdist2)  # mean over points

                if self.toggles.get('surface_recon', False):
                    x_surface = x_prim[:, :, 3:]
                    xs_recon = PatchPointCloud.fuse_primitives(x_surface)

                    surface_feat = torch.transpose(inputs.x.unsqueeze(0), 2, 1).contiguous()
                    s_inCham = surface_feat.view(surface_feat.size(0), -1, 4).contiguous().to(self.device)
                    sdist1, sdist2, sidx1, sidx2 = self.dist_chamfer_4d(s_inCham, xs_recon)
                    surface_loss = torch.mean(sdist1) + torch.mean(sdist2)
                    loss = loss + self.config['surface_fac'] * surface_loss

                if geom_props["computed"]:
                    conform_loss = PatchPointCloud.conformal_regularization(geom_props)
                    loss = loss + self.config['conform_fac'] * conform_loss

                eval_loss.append(loss.item())

        eval_metrics['loss'] = np.round(np.mean(eval_loss), 4)
        self.train()
        return eval_metrics

    def generate_patch_assignments(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            patch_assignments, input_points = self.decoder.generate_patch_labels(self.encoder(x), x)
            unique_vals = np.unique(patch_assignments)
            label_to_idx = {label: idx for idx, label in enumerate(unique_vals)}
            patch_assignments = np.asarray([label_to_idx[elem]
                                            for elem in patch_assignments], dtype=np.float)
            return patch_assignments, input_points

    def generate_mesh(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            return self.decoder.generate_mesh(self.encoder(x), x.shape)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class PointNet(nn.Module):
    def __init__(self,
                 latent_dim: int = 1024,
                 input_dim: int = 3,
                 **kwargs):
        """
        PointNet Encoder
        See : PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
                Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
        """
        super(PointNet, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._build_components()

    def _build_components(self):
        self.conv1 = torch.nn.Conv1d(self.input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_dim, 1)
        self.lin1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.lin2 = nn.Linear(self.latent_dim, self.latent_dim)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.latent_dim)
        self.bn4 = torch.nn.BatchNorm1d(self.latent_dim)
        self.bn5 = torch.nn.BatchNorm1d(self.latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, self.latent_dim)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x
