"""
Model for predicting binding affinity.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Union
import numpy as np
import copy
from torch_geometric.nn import global_add_pool, global_mean_pool
import gc

from holoprot.layers import WLNConv, LSTMConv, GRUConv, WLNResConv, ProtMPN
from holoprot.data.base import ComplexData
from holoprot.utils.tensor import build_mlp


class AffinityPred(nn.Module):
    """Model to predict the binding affinity between ligand and protein."""

    def __init__(self,
                 config: Dict,
                 toggles: Dict,
                 metrics: Dict = None,
                 device: str = 'cpu',
                 **kwargs) -> None:
        """
        Parameters
        ----------
        config: Dict,
            Configuration for layers in model
        toggles: Dict, default None
            Optional toggles for the model. Useful for ablation studies
        encoder_name: str,
            Name of the encoder used. Allows message passing in directed or
            undirected format
        device: str,
            Device to run the model on.
        """
        super(AffinityPred, self).__init__(**kwargs)
        self.config = config
        self.toggles = toggles if toggles is not None else {}
        self.metrics = metrics
        self.device = device

        self._build_layers()

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

    def _make_backward_compatible(self, config, toggles):
        # This is currently just to ensure backward compatability with previously
        # trained models. Once the results become better, this function will be
        # deprecated.
        bconfig = config['bconfig']
        sconfig = config.get('sconfig', None)

        jk_pool = None
        if self.toggles.get("use_concat", False):
            jk_pool = "concat"

        def infer_prot_mode(graph_type):
            if graph_type == 'surface':
                return graph_type
            elif graph_type == 'amino':
                return 'backbone'
            elif graph_type == 'rev_hier_complex':
                return 'surface2backbone'
            elif graph_type == 'hier_complex':
                return 'backbone2surface'

        prot_mode = infer_prot_mode(config['graph_type'])
        if prot_mode == 'backbone':
            mpn_config = copy.deepcopy(bconfig)
            mpn_config['dropout_p'] = config['dropout_mpn']
            mpn_config['jk_pool'] = jk_pool
        elif prot_mode == 'surface':
            mpn_config = copy.deepcopy(sconfig)
            mpn_config['dropout_p'] = config['dropout_mpn']
            mpn_config['jk_pool'] = jk_pool
        else:
            assert bconfig is not None, "bconfig cannot be None"
            assert sconfig is not None, "sconfig cannot be None"
            bconfig['dropout_p'] = config['dropout_mpn']
            sconfig['dropout_p'] = config['dropout_mpn']
            bconfig['jk_pool'] = jk_pool
            sconfig['jk_pool'] = jk_pool
            mpn_config = {'bconfig': copy.deepcopy(bconfig),
                          'sconfig': copy.deepcopy(sconfig),
                          'dropout_mlp': config['dropout_mlp']}
        return mpn_config, prot_mode

    def _add_missing_attrs(self, config, mpn_config):
        if 'activation' not in config:
            config['activation'] = 'relu'
        if 'activation' not in mpn_config:
            mpn_config['activation'] = 'relu'

        if 'bconfig' in mpn_config:
            bconfig = mpn_config['bconfig']
            if 'activation' not in bconfig:
                bconfig['activation'] = 'relu'
            mpn_config['bconfig'] = bconfig

        if 'sconfig' in mpn_config:
            sconfig = mpn_config['sconfig']
            if 'activation' not in sconfig:
                sconfig['activation'] = 'relu'
            mpn_config['sconfig'] = sconfig

        return config, mpn_config

    def _build_layers(self):
        config = self.config
        toggles = self.toggles
        mpn_config = config.get("mpn_config", None)
        prot_mode = config.get("prot_mode", None)

        if mpn_config is None:
            mpn_config, prot_mode = self._make_backward_compatible(config, toggles)

        if "jk_pool" not in config: # This is also for compatability reasons as a newly added feature
            config['jk_pool'] = None
            if toggles.get("use_concat", False):
                config['jk_pool'] = 'concat'
            else:
                config['jk_pool'] = None

        config, mpn_config = self._add_missing_attrs(config, mpn_config)

        if not self.toggles.get("ligand_only", False):
            self.prot_mpn = ProtMPN(mpn_config=mpn_config,
                                    encoder=config['encoder'],
                                    prot_mode=prot_mode,
                                    graph_pool=mpn_config.get("graph_pool", "sum_pool"),
                                    use_attn=self.toggles.get("use_attn", False),
                                    use_mpn_in_patch=self.toggles.get("use_mpn_in_patch", False))

        if 'wln' in config['encoder']:
            WLN_ENCODERS = {'wln': WLNConv, 'wlnres': WLNResConv}
            wln_class = WLN_ENCODERS.get(config['encoder'])
            self.lig_mpn = wln_class(node_fdim=config['atom_fdim'],
                                     edge_fdim=config['bond_fdim'],
                                     depth=config['lig_depth'],
                                     hsize=config['lig_hsize'],
                                     dropout=config['dropout_mpn'],
                                     activation=config.get("activation", "relu"),
                                     jk_pool=config.get("jk_pool", None)) #Check this part

        elif config['encoder'] in ['gru', 'lstm']:
            raise NotImplementedError("RNN encoders don't work with code yet.")
        else:
            raise ValueError(f"Encoder of type {config['encoder']} not supported.")

        in_dim = config['lig_hsize']
        if config.get("jk_pool", None) is "concat":
            in_dim *= config['lig_depth']

        if not self.toggles.get("ligand_only", None):
            multiplier = 1

            if prot_mode in ['backbone', 'surface']:
                if config.get("jk_pool", None) == "concat":
                    multiplier = mpn_config['depth']
                in_dim += (multiplier * mpn_config['hsize'])

            elif prot_mode == 'surface2backbone' or prot_mode == "patch2backbone":
                bconfig = mpn_config['bconfig']
                sconfig = mpn_config['sconfig']

                if config.get("jk_pool", None) == "concat":
                    multiplier = bconfig['depth']
                in_dim += (multiplier * bconfig['hsize'])

            elif prot_mode == "backbone2patch":
                bconfig = mpn_config['bconfig']
                sconfig = mpn_config['sconfig']

                if config.get("jk_pool", None) == "concat":
                    multiplier = sconfig['depth']
                in_dim += (multiplier * sconfig['hsize'])

        self.activity_mlp = build_mlp(
            in_dim=in_dim,
            h_dim=config['hsize'],
            out_dim=1,
            dropout_p=config['dropout_mlp'],
            activation=config['activation'])

        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, complex_data: ComplexData) -> torch.Tensor:
        complex_data = complex_data.to(self.device)
        prot = complex_data.protein
        lig = complex_data.ligand
        lig_node_emb = self.lig_mpn(lig.x, lig.edge_index, lig.edge_attr)

        if self.config.get("graph_pool", "sum_pool") == "sum_pool":
            lig_emb = global_add_pool(lig_node_emb, lig.batch)
        elif self.config.get("graph_pool", "sum_pool") == "mean_pool":
            lig_emb = global_mean_pool(lig_node_emb, lig.batch)

        _, prot_emb = self.prot_mpn(prot)
        assert prot_emb.size(0) == lig_emb.size(0)
        complex_vec = torch.cat([prot_emb, lig_emb], dim=-1)
        return self.activity_mlp(complex_vec)

    def train_step(self, complex_data: ComplexData) -> Tuple[torch.Tensor, Dict[str, float]]:
        activity_pred = self(complex_data)
        loss = self.loss_fn(activity_pred.squeeze(-1), complex_data.y).mean()
        metrics = {'loss': loss.item()}
        return loss, metrics

    def eval_step(self, eval_data: None):
        eval_metrics = {}
        if eval_data is None:
            return eval_metrics

        eval_pred, eval_labels, eval_loss = [], [], []

        self.eval()
        with torch.no_grad():
            for idx, inputs in enumerate(eval_data):
                if inputs is None:
                    continue
                label = inputs.y
                try:
                    activity_pred = self(inputs)
                    loss = self.loss_fn(activity_pred.squeeze(-1), inputs.y).mean()
                    if loss is not None:
                        assert torch.isfinite(loss).all()
                    eval_pred.append(activity_pred.item())
                    eval_labels.append(label.item())
                    eval_loss.append(loss.item())

                except Exception as e:
                    print(f"Exception: {e}", flush=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

        eval_labels = np.array(eval_labels).flatten()
        eval_pred = np.array(eval_pred).flatten()
        for metric, metric_fn in self.metrics.items():
            eval_metrics[metric] = np.round(metric_fn(eval_labels, eval_pred), 4)

        eval_metrics['loss'] = np.round(np.mean(eval_loss), 4)
        self.train()
        return eval_metrics

    def predict(self, complex_data: ComplexData) -> torch.Tensor:
        with torch.no_grad():
            activity_pred = self(complex_data)
            return activity_pred
