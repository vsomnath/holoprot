import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import Tuple, List, Dict

from holoprot.utils import get_global_agg
from holoprot.utils.tensor import index_select_ND

class GRUConv(MessagePassing):

    def __init__(self,
                 node_fdim: int,
                 edge_fdim: int,
                 hsize: int, depth: int,
                 rnn_agg: str = 'mean',
                 dropout: float = 0.2): #TODO: Add activation function here
        super(GRUConv, self).__init__(aggr='mean')
        self.node_fdim = node_fdim
        self.input_size = node_fdim + edge_fdim
        self.hsize = hsize
        self.depth = depth
        self.rnn_agg = rnn_agg
        self.dropout_p = dropout
        self._build_components()

    def _build_components(self):
        """Build layer components."""
        self.W_z = nn.Linear(self.input_size + self.hsize, self.hsize)
        self.W_r = nn.Linear(self.input_size, self.hsize, bias=False)
        self.U_r = nn.Linear(self.hsize, self.hsize)
        self.W_h = nn.Linear(self.input_size + self.hsize, self.hsize)

        self.dropouts, self.bns = [], []
        for i in range(self.depth):
            self.dropouts.append(nn.Dropout(p=self.dropout_p))
        self.dropouts = nn.ModuleList(self.dropouts)
        self.node_mlp = nn.Linear(self.node_fdim + self.hsize, self.hsize)

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Tensor, bgraph: Tensor, x_batch: Tensor = None) -> Tensor:
        hnode = self.propagate(edge_index, x=x, edge_attr=edge_attr, bgraph=bgraph)
        if x_batch is None:
            return hnode

        if self.global_agg is not None:
            return hnode, self.global_agg(hnode, x_batch)
        else:
            return hnode, None

    def message(self, x_j: Tensor, edge_attr: Tensor, bgraph: Tensor) -> Tensor:
        hmess = x_j.new_zeros(1 + x_j.size(0), self.hsize)
        fmess = torch.cat([x_j, edge_attr], dim=-1)
        fmess = torch.cat([fmess.new_zeros(1, fmess.shape[-1]), fmess], dim=0)
        assert hmess.shape[0] == fmess.shape[0]

        mask = hmess.new_ones(hmess.size(0), 1)
        mask[0] = 0

        for i in range(self.depth):
            hmess_nei = index_select_ND(hmess, 0, bgraph)
            hmess = self.GRU(fmess, hmess_nei)
            hmess = hmess * mask
            hmess = self.dropouts[i](hmess)
        return hmess[mask.bool().squeeze(-1)] # TODO (vsomnath): This is ugly

    def get_hidden_state(self, h):
        return h

    def update(self, inputs: Tensor, x: Tensor) -> Tensor:
        return self.node_mlp(torch.cat([x, inputs], dim=-1))

    def GRU(self, x, h_nei):
        if self.rnn_agg == 'mean':
            agg_h = h_nei.mean(dim=1)
        elif self.rnn_agg == 'sum':
            agg_h = h_nei.sum(dim=1)
        else:
            raise ValueError()
        z_input = torch.cat([x, agg_h], dim=1)
        z = torch.sigmoid(self.W_z(z_input))

        r_1 = self.W_r(x).view(-1, 1, self.hsize)
        r_2 = self.U_r(h_nei)
        r = torch.sigmoid(r_1 + r_2)
        gated_h = r * h_nei

        if self.rnn_agg == 'mean':
            gated_agg_h = gated_h.mean(dim=1)
        elif self.rnn_agg == 'sum':
            gated_agg_h = gated_h.sum(dim=1)
        else:
            raise ValueError()
        h_input = torch.cat([x, gated_agg_h], dim=1)
        pre_h = torch.tanh(self.W_h(h_input))
        new_h = (1.0 - z) * agg_h + z * pre_h
        return new_h

class LSTMConv(MessagePassing):

    def __init__(self, node_fdim: int,
                 edge_fdim: int,
                 hsize: int, depth: int,
                 rnn_agg: str = 'mean',
                 dropout: float = 0.2):
        super(LSTMConv, self).__init__(aggr='mean')
        self.node_fdim = node_fdim
        self.input_size = node_fdim + edge_fdim
        self.hsize = hsize
        self.depth = depth
        self.rnn_agg = rnn_agg
        self.dropout_p = dropout
        self._build_components()

    def _build_components(self):
        """Build layer components."""
        self.W_i = nn.Sequential(
            nn.Linear(self.input_size + self.hsize, self.hsize), nn.Sigmoid())
        self.W_o = nn.Sequential(
            nn.Linear(self.input_size + self.hsize, self.hsize), nn.Sigmoid())
        self.W_f = nn.Sequential(
            nn.Linear(self.input_size + self.hsize, self.hsize), nn.Sigmoid())
        self.W = nn.Sequential(
            nn.Linear(self.input_size + self.hsize, self.hsize), nn.Tanh())

        self.dropouts = []
        for i in range(self.depth):
            self.dropouts.append(nn.Dropout(p=self.dropout_p))
        self.dropouts = nn.ModuleList(self.dropouts)
        self.node_mlp = nn.Linear(self.node_fdim + self.hsize, self.hsize)

    def get_hidden_state(self,
                         h: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Gets the hidden state.

        Parameters
        ----------
        h: Tuple[torch.Tensor, torch.Tensor],
            Hidden state tuple of the LSTM
        """
        return h[0]

    def LSTM(self, x: torch.Tensor, h_nei: torch.Tensor,
             c_nei: torch.Tensor) -> torch.Tensor:
        """Implements the LSTM gating equations.

        Parameters
        ----------
        x: torch.Tensor,
            Input tensor
        h_nei: torch.Tensor,
            Hidden states of the neighbors
        c_nei: torch.Tensor,
            Memory state of the neighbors
        """
        if self.rnn_agg == 'mean':
            agg_h = h_nei.mean(dim=1)
        elif self.rnn_agg == 'sum':
            agg_h = h_nei.sum(dim=1)
        else:
            raise ValueError()
        x_expand = x.unsqueeze(1).expand(-1, h_nei.size(1), -1)
        i = self.W_i(torch.cat([x, agg_h], dim=-1))
        o = self.W_o(torch.cat([x, agg_h], dim=-1))
        f = self.W_f(torch.cat([x_expand, h_nei], dim=-1))
        u = self.W(torch.cat([x, agg_h], dim=-1))
        c = i * u + (f * c_nei).sum(dim=1)
        h = o * torch.tanh(c)
        return h, c

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Tensor, bgraph: Tensor, x_batch: Tensor = None) -> Tensor:
        hnode = self.propagate(edge_index, x=x, edge_attr=edge_attr, bgraph=bgraph)
        if x_batch is None:
            return hnode

        if self.global_agg is not None:
            return hnode, self.global_agg(hnode, x_batch)
        else:
            return hnode, None

    def message(self, x_j: Tensor, edge_attr: Tensor, bgraph: Tensor) -> Tensor:
        hmess = x_j.new_zeros(1 + x_j.size(0), self.hsize)
        cmess = x_j.new_zeros(1 + x_j.size(0), self.hsize)

        fmess = torch.cat([x_j, edge_attr], dim=-1)
        fmess = torch.cat([fmess.new_zeros(1, fmess.shape[-1]), fmess], dim=0)
        assert hmess.shape[0] == fmess.shape[0]

        mask = hmess.new_ones(hmess.size(0), 1)
        mask[0] = 0

        for i in range(self.depth):
            hmess_nei = index_select_ND(hmess, 0, bgraph)
            cmess_nei = index_select_ND(cmess, 0, bgraph)
            hmess, cmess = self.LSTM(fmess, hmess_nei, cmess_nei)
            hmess = hmess * mask
            cmess = cmess * mask
            hmess, cmess = self.dropouts[i](hmess), self.bns[i](cmess)
        return hmess[mask.bool().squeeze(-1)] # TODO (vsomnath): This is ugly

    def update(self, inputs: Tensor, x: Tensor) -> Tensor:
        return self.node_mlp(torch.cat([x, inputs], dim=-1))


if __name__ == "__main__":
    from holoprot.graphs.complex import MolBuilder

    builder = MolBuilder(mpnn='gru')
    data = builder.build("CCC")

    layer_a = LSTMConv(node_fdim=data.x.shape[-1],
                    input_size=data.x.shape[-1] + data.edge_attr.shape[-1],
                    hsize=10, depth=3, rnn_agg='mean')
    layer_b = GRUConv(node_fdim=data.x.shape[-1],
                    input_size=data.x.shape[-1] + data.edge_attr.shape[-1],
                    hsize=10, depth=3, rnn_agg='mean')
    out_a, _ = layer_a(data.x, data.edge_index, data.edge_attr, data.mess_idx)
    out_b, _ = layer_b(data.x, data.edge_index, data.edge_attr, data.mess_idx)

    # print(out_a.shape, out_b.shape)
