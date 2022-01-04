import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Union

from holoprot.layers import ProtMPN
from holoprot.utils.tensor import build_mlp


class ProtClassifier(nn.Module):

    def __init__(self,
                 config: Dict,
                 toggles: Dict,
                 metrics: Dict = None,
                 class_weights = None,
                 device: str = 'cpu', **kwargs):
        super(ProtClassifier, self).__init__(**kwargs)
        self.config = config
        self.toggles = toggles if toggles is not None else {}
        self.metrics = metrics
        self.device = device
        self._build_layers(class_weights=class_weights)

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

    def _build_layers(self, class_weights=None):
        config = self.config
        toggles = self.toggles
        mpn_config = config.get("mpn_config", None)
        prot_mode = config.get("prot_mode", None)

        if mpn_config is None:
            raise ValueError("mpn_config cannot be None. Backward compatability is deprecated")

        # This is also for compatability reasons as a newly added feature
        if "jk_pool" not in config:
            config['jk_pool'] = None
            if toggles.get("use_concat", False):
                config['jk_pool'] = 'concat'
            else:
                config['jk_pool'] = None

        config, mpn_config = self._add_missing_attrs(config, mpn_config)

        self.mpn = ProtMPN(mpn_config=mpn_config,
                           encoder=config['encoder'],
                           prot_mode=prot_mode,
                           graph_pool=mpn_config.get("graph_pool", "sum_pool"),
                           use_attn=self.toggles.get("use_attn", False),
                           use_mpn_in_patch=self.toggles.get("use_mpn_in_patch", False))
        multiplier = 1
        in_dim = 0

        if prot_mode in ['backbone', 'surface']:
            if mpn_config.get('jk_pool', None) == "concat":
                multiplier = mpn_config['depth']
            in_dim += (multiplier * mpn_config['hsize'])

        elif prot_mode == 'surface2backbone':
            bconfig = mpn_config['bconfig']
            sconfig = mpn_config['sconfig']

            if bconfig.get('jk_pool', None) == "concat":
                multiplier = bconfig['depth']
            in_dim += (multiplier * bconfig['hsize'])

        else:
            bconfig = mpn_config['bconfig']
            sconfig = mpn_config['sconfig']

            if sconfig.get('jk_pool', None) == "concat":
                multiplier = sconfig['depth']
            in_dim += (multiplier * sconfig['hsize'])

        self.classification_mlp = build_mlp(in_dim=in_dim,
                                        h_dim=config['hsize'],
                                        out_dim=config['n_classes'],
                                        dropout_p=config['dropout_mlp'],
                                        activation=config['activation'])
        if class_weights is not None:
            class_weights = torch.tensor(class_weights).view(-1)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

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

    def forward(self, data: Data) -> torch.Tensor:
        data = data.to(self.device)
        _, prot_emb = self.mpn(data)
        pred_logits = self.classification_mlp(prot_emb)
        return pred_logits

    def train_step(self, data: Data) -> torch.Tensor:
        pred_logits = self(data)
        loss = self.loss_fn(pred_logits, data.y).mean()
        accuracy = (torch.argmax(pred_logits, dim=-1) == data.y).float().mean()
        metrics = {'loss': loss.item(), 'accuracy': accuracy.item()}
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
                    eval_pred.append(np.nan)
                    eval_labels.append(np.nan)
                    continue
                label = inputs.y
                try:
                    pred_logits = self(inputs)
                    loss = self.loss_fn(pred_logits, inputs.y).mean()
                    if loss is not None:
                        assert torch.isfinite(loss).all()
                    pred_pred = torch.argmax(pred_logits, dim=-1)
                    eval_pred.append(pred_pred.item())
                    eval_labels.append(label.item())
                    eval_loss.append(loss.item())

                except Exception as e:
                    print(f"Exception: {e}", flush=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

        for metric, metric_fn in self.metrics.items():
            eval_metrics[metric] = np.round(metric_fn(eval_labels, eval_pred), 4)

        eval_metrics['loss'] = np.round(np.mean(eval_loss), 4)
        self.train()
        return eval_metrics

    def predict(self, data: Data) -> torch.Tensor:
        with torch.no_grad():
            pred_logits = self(data)
            return torch.argmax(pred_logits, dim=-1)
