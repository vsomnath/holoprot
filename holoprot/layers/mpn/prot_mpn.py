import torch
import torch.nn as nn
from typing import Dict
from torch_geometric.nn import global_add_pool, global_mean_pool

from holoprot.layers import mpn_layer_from_config
from holoprot.utils.tensor import build_mlp, index_select_ND

class ProtMPN(nn.Module):

    def __init__(self,
                 mpn_config: Dict,
                 encoder: str,
                 prot_mode: str = 'backbone',
                 use_attn: bool = False,
                 graph_pool: str = 'sum_pool',
                 use_mpn_in_patch: bool = False,
                **kwargs):
        super(ProtMPN, self).__init__(**kwargs)
        self.encoder = encoder
        self.mpn_config = mpn_config
        self.prot_mode = prot_mode
        self.use_attn = use_attn
        self.graph_pool = graph_pool
        self.use_mpn_in_patch = use_mpn_in_patch
        self._build_components()

    def _build_components(self):
        if self.prot_mode in ['backbone', 'surface']:
            self.mpn = mpn_layer_from_config(self.mpn_config, self.encoder)

        elif self.prot_mode == 'backbone2patch':
            bconfig = self.mpn_config['bconfig']
            sconfig = self.mpn_config['sconfig']

            self.backbone_mpn = mpn_layer_from_config(bconfig, self.encoder)
            self.patch_mpn = mpn_layer_from_config(sconfig, self.encoder)

            multiplier = 1
            if bconfig.get("jk_pool", None) == "concat":
                multiplier = bconfig['depth']
            in_dim = multiplier * bconfig['hsize'] + sconfig['node_fdim']
            self.patch_mlp = build_mlp(in_dim=in_dim, h_dim=sconfig['hsize'],
                                       out_dim=sconfig['hsize'],
                                       dropout_p=self.mpn_config['dropout_mlp'],
                                       activation=sconfig['activation'])
            if self.use_attn:
                indim = bconfig['hsize']
                outdim = sconfig['hsize']
                if bconfig.get("jk_pool", None) == "concat":
                    indim *= bconfig['depth']
                if sconfig.get("jk_pool", None) == "concat":
                    outdim *= sconfig['depth']
                self.W_a = nn.Parameter(torch.Tensor(indim, outdim))
                nn.init.xavier_normal_(self.W_a)

        elif self.prot_mode == 'backbone2surface':
            bconfig = self.mpn_config['bconfig'] #Backbone / backbone config
            sconfig = self.mpn_config['sconfig']

            self.backbone_mpn = mpn_layer_from_config(bconfig, self.encoder)
            self.surface_mpn = mpn_layer_from_config(sconfig, self.encoder)

            multiplier = 1
            if bconfig.get("jk_pool", None) == "concat":
                multiplier = bconfig['depth']
            in_dim = multiplier * bconfig['hsize'] + sconfig['node_fdim']
            self.surface_mlp = build_mlp(in_dim=in_dim, h_dim=sconfig['hsize'],
                                          out_dim=sconfig['hsize'],
                                          dropout_p=self.mpn_config['dropout_mlp'],
                                          activation=sconfig['activation'])
            if self.use_attn:
                indim = bconfig['hsize']
                outdim = sconfig['hsize']
                if bconfig.get("jk_pool", None) == "concat":
                    indim *= bconfig['depth']
                if sconfig.get("jk_pool", None) == "concat":
                    outdim *= sconfig['depth']
                self.W_a = nn.Parameter(torch.Tensor(indim, outdim))
                nn.init.xavier_normal_(self.W_a)

        elif self.prot_mode == 'surface2backbone':
            bconfig = self.mpn_config['bconfig']
            sconfig = self.mpn_config['sconfig']

            self.backbone_mpn = mpn_layer_from_config(bconfig, self.encoder)
            self.surface_mpn = mpn_layer_from_config(sconfig, self.encoder)

            multiplier = 1
            if sconfig.get("jk_pool", None) == "concat":
                multiplier = sconfig['depth']
            in_dim = multiplier * sconfig['hsize'] + bconfig['node_fdim']
            self.backbone_mlp = build_mlp(in_dim=in_dim, h_dim=bconfig['hsize'],
                                          out_dim=bconfig['hsize'],
                                          dropout_p=self.mpn_config['dropout_mlp'],
                                          activation=bconfig['activation'])

            if self.use_attn:
                indim = sconfig['hsize']
                outdim = bconfig['hsize']
                if sconfig.get("jk_pool", None) == "concat":
                    indim *= sconfig['depth']
                if bconfig.get("jk_pool", None) == "concat":
                    outdim *= bconfig['depth']
                self.W_a = nn.Parameter(torch.Tensor(indim, outdim))
                nn.init.xavier_normal_(self.W_a)

        elif self.prot_mode == 'patch2backbone':
            bconfig = self.mpn_config['bconfig']
            sconfig = self.mpn_config['sconfig']

            if self.use_mpn_in_patch:
                multiplier = 1
                sp_config = self.mpn_config['sp_config']
                self.surface_to_patch_mpn = mpn_layer_from_config(sp_config, self.encoder)
                if sp_config.get("jk_pool", None) == "concat":
                    multiplier = sp_config['depth']
                in_dim = multiplier * sp_config['hsize'] + sconfig['node_fdim']
                self.surface_to_patch_mlp = build_mlp(in_dim=in_dim,
                                                      h_dim=sconfig['hsize'],
                                                      out_dim=sconfig['hsize'],
                                                      dropout_p=self.mpn_config['dropout_mlp'],
                                                      activation=sp_config['activation'])

            self.backbone_mpn = mpn_layer_from_config(bconfig, self.encoder)
            self.patch_mpn = mpn_layer_from_config(sconfig, self.encoder)

            multiplier = 1
            if sconfig.get("jk_pool", None) == "concat":
                multiplier = sconfig['depth']
            in_dim = multiplier * sconfig['hsize'] + bconfig['node_fdim']
            self.backbone_mlp = build_mlp(in_dim=in_dim, h_dim=bconfig['hsize'],
                                          out_dim=bconfig['hsize'],
                                          dropout_p=self.mpn_config['dropout_mlp'],
                                          activation=bconfig['activation'])

            if self.use_attn:
                indim = sconfig['hsize']
                outdim = bconfig['hsize']
                if sconfig.get("jk_pool", None) == "concat":
                    indim *= sconfig['depth']
                if bconfig.get("jk_pool", None) == "concat":
                    outdim *= bconfig['depth']
                self.W_a = nn.Parameter(torch.Tensor(indim, outdim))
                nn.init.xavier_normal_(self.W_a)

        else:
            raise ValueError(f"{self.prot_mode} is currently not supported.")

    def run_component_mpn(self, data):
        component = getattr(data, self.prot_mode)
        inputs = (component.x, component.edge_index, component.edge_attr)
        if self.encoder in ['gru', 'lstm']:
            inputs += (components.mess_idx,)
        node_emb = self.mpn(*inputs)
        return node_emb

    def run_mpn_in_patch(self, data):
        node_emb = self.surface_to_patch_mpn(data.x, data.edge_index, data.edge_attr)
        patch_emb = global_mean_pool(node_emb, data.patch_members)
        return patch_emb

    def run_full_mpn(self, data):
        order = self.prot_mode.split("2")
        bottom_layer, top_layer = order

        bottom_graph = getattr(data, bottom_layer)
        top_graph = getattr(data, top_layer)
        hier_mapping = data.mapping

        x = bottom_graph.x
        # This is a bit hacky for now since we are just prototyping
        if self.prot_mode == "patch2backbone" and self.use_mpn_in_patch:
            patch_emb = self.run_mpn_in_patch(data.surface_to_patch)
            x = self.surface_to_patch_mlp(torch.cat([x, patch_emb], dim=-1))

        bottom_mpn_inputs = (x, bottom_graph.edge_index, bottom_graph.edge_attr)
        if self.encoder in ['gru', 'lstm']:
            bottom_mpn_inputs += (bottom_graph.mess_idx,)

        bottom_layer_mpn = getattr(self, bottom_layer + "_mpn")
        hnode_bottom = bottom_layer_mpn(*bottom_mpn_inputs)
        hnode_bottom = torch.cat([hnode_bottom.new_zeros(1, hnode_bottom.shape[-1]),
                                  hnode_bottom], dim=0)

        hier_nei_emb = index_select_ND(hnode_bottom, index=hier_mapping, dim=0)
        hier_num_nei = (hier_mapping != 0).sum(dim=1, keepdim=True)
        hier_nei_agg = hier_nei_emb.sum(dim=1) / (hier_num_nei + 1e-8)
        msg = f"hier_nei_agg={hier_nei_agg.shape}, top_graph.x_shape={top_graph.x.shape}"
        assert len(hier_nei_agg) == len(top_graph.x), msg

        x = torch.cat([top_graph.x, hier_nei_agg], dim=-1)
        mlp_layer = getattr(self, top_layer + "_mlp")
        x = mlp_layer(x)

        top_mpn_inputs = (x, top_graph.edge_index, top_graph.edge_attr)
        if self.encoder in ['gru', 'lstm']:
            top_inputs += (top_graph.mess_idx,)
        top_layer_mpn = getattr(self, top_layer + "_mpn")
        hnode_top = top_layer_mpn(*top_mpn_inputs)

        if self.use_attn:
            hier_nei_proj = torch.tensordot(hier_nei_emb, self.W_a, dims=1)
            hnode_top_exp = hnode_top.unsqueeze(1).expand(*hier_nei_proj.shape)

            att_logits = (hnode_top_exp * hier_nei_proj).sum(dim=-1)
            att_logits[hier_mapping == 0] = float('-inf')

            att_logits[hier_num_nei.expand(hier_mapping.shape) == 0] = 0.0
            att_weights = torch.softmax(att_logits, dim=1)
            hnode_top_att = (att_weights.unsqueeze(-1) * hier_nei_proj).sum(dim=1)
            hnode_top = torch.where(hier_num_nei != 0, hnode_top_att, hnode_top)

        return (hnode_top, hnode_bottom)

    def forward(self, data):
        if self.prot_mode in ['backbone', 'surface']:
            node_emb = self.run_component_mpn(data)
        else:
            node_emb, _ = self.run_full_mpn(data)

        top_component = self.prot_mode.split("2")[-1] # Aggregate node embeddings of last layer
        if self.graph_pool == "sum_pool":
            graph_emb = global_add_pool(node_emb, getattr(data, top_component).batch)
        elif self.graph_pool == "mean_pool":
            graph_emb = global_mean_pool(node_emb, getattr(data, top_component).batch)
        return node_emb, graph_emb


if __name__ == "__main__":
    bconfig = {'node_fdim': 33, 'edge_fdim': 2, 'hsize': 15, 'depth': 2, 'dropout_p': 0.1}
    sconfig = {'node_fdim': 4, 'edge_fdim': 7, 'hsize': 10, 'depth': 2, 'dropout_p': 0.2}

    mpn_config = {'bconfig': bconfig, 'sconfig': sconfig, 'dropout_mlp': 0.3}
    prot_mpn = ProtMPN(mpn_config=mpn_config, encoder='wln', prot_mode='surface2backbone', use_concat=True)

    data_dict = torch.load("./datasets/processed/pdbbind/rev_hier_complex/4NRA.pth")
    prot_data = data_dict['prot']
    lig_data = data_dict['4nra_ligand']

    from holoprot.data.base_dataset import ProtBatch, ComplexBatch
    from holoprot.graphs.complex import ComplexData

    data = ComplexData(prot_data, lig_data)
    batch = ComplexBatch.from_data_list([data, data], prot_mode='surface2backbone')
    print(batch)
    #node, graph = prot_mpn(batch)
    #print(node.shape, graph.shape)
