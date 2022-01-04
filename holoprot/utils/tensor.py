import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Any, Optional, Union


def zip_tensors(tup_list):
    arr0, arr1, arr2 = zip(*tup_list)
    if type(arr2[0]) is int:
        arr0 = torch.stack(arr0, dim=0)
        arr1 = torch.tensor(arr1, dtype=torch.long)
        arr2 = torch.tensor(arr2, dtype=torch.long)
    else:
        arr0 = torch.cat(arr0, dim=0)
        arr1 = [x for a in arr1 for x in a]
        arr1 = torch.tensor(arr1, dtype=torch.long)
        arr2 = torch.cat(arr2, dim=0)
    return arr0, arr1, arr2


def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist]) + 1
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.tensor(alist, dtype=torch.long)


def index_scatter(sub_data, all_data, index):
    d0, d1 = all_data.size()
    buf = torch.zeros_like(all_data).scatter_(0,
                                              index.repeat(d1, 1).t(), sub_data)
    mask = torch.ones(d0, device=all_data.device).scatter_(0, index, 0)
    return all_data * mask.unsqueeze(-1) + buf


def create_scope_tensor(scope: List[Tuple[int, int]],
                        return_rev: bool = True,
                        device: str = 'cpu') -> Tuple[torch.Tensor]:
    """
    :return: A tensor that ind selects into flat to produce batch. A tensor that does the reverse.
    """
    max_num_bonds = max([num for ind, num in scope])
    sel, rev = [], []
    for i, entry in enumerate(scope):
        start, num = entry
        sel.append([start + ind for ind in range(num)])
        sel[-1].extend([0] * (max_num_bonds - num))
        if return_rev:
            rev.extend([i * max_num_bonds + ind for ind in range(num)])
    if return_rev:
        return torch.from_numpy(
            np.array(sel)).long().to(device), torch.from_numpy(
                np.array(rev)).long().to(device)
    else:
        return torch.from_numpy(np.array(sel)).long().to(device)


def flat_to_batch(source: torch.Tensor, scope: torch.Tensor) -> torch.Tensor:
    """
    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param scope: A tensor of shape (batch_size, max_num_bonds) expressing bond indices for each mol/row.
    :return: A tensor of shape (batch, max_num_bonds, hidden_size) containing the message features.
    """
    final_size = (scope.shape[0], -1,
                  source.shape[1])  # batch x max_num_bonds x hidden_size
    ret = source.index_select(dim=0, index=scope.view(-1))
    return ret.view(final_size)


def batch_to_flat(source: torch.Tensor, scope: torch.Tensor) -> torch.Tensor:
    """
    :param source: A tensor of shape (batch, max_num_bonds, hidden_size).
    :param scope: A tensor of shape (batch_size, max_num_bonds) expressing bond indices for each mol/row.
    :return: A tensor of shape (num_bonds, hidden_size) with mols concatenated.
    """
    hidden_size = source.shape[-1]
    ret = source.reshape(-1, hidden_size)  # (batch*max_num_bonds) x hidden_size
    ret = ret.index_select(dim=0, index=scope)

    return ret


def stack_pad_tensor(tensor_list):
    max_len = max([t.size(0) for t in tensor_list])
    for i, tensor in enumerate(tensor_list):
        pad_len = max_len - tensor.size(0)
        tensor_list[i] = F.pad(tensor, (0, 0, 0, pad_len))
    return torch.stack(tensor_list, dim=0)


def get_pair(tensor):
    """Gets paired combination.

    Given a tensor of shape (x0, x1, x2), this gives you a tensor of shape
    (x0, x1, x1, x2) where the pairs of elements in the x1 dim.
    """
    return tensor.unsqueeze(1) + tensor.unsqueeze(-2)


def index_select_ND(input, dim, index):
    """Wrapper around torch.index_select for non 1D index tensors."""
    input_shape = input.shape
    target_shape = index.shape + input_shape[dim + 1:]
    out_tensor = input.index_select(dim, index.view(-1))
    return out_tensor.view(target_shape)

def build_mlp(in_dim: int,
              h_dim: Union[int, List],
              out_dim: int = None,
              dropout_p: float = 0.2,
              activation: str = 'relu') -> nn.Sequential:
    """Builds an MLP.
    Parameters
    ----------
    in_dim: int,
        Input dimension of the MLP
    h_dim: int,
        Hidden layer dimension of the MLP
    out_dim: int, default None
        Output size of the MLP. If None, a Linear layer is returned, with ReLU
    dropout_p: float, default 0.2,
        Dropout probability
    """
    if isinstance(h_dim, int):
        h_dim = [h_dim]

    sizes = [in_dim] + h_dim
    mlp_size_tuple = list(zip(*(sizes[:-1], sizes[1:])))

    if isinstance(dropout_p, float):
        dropout_p = [dropout_p] * len(mlp_size_tuple)

    layers = []

    for idx, (prev_size, next_size) in enumerate(mlp_size_tuple):
        layers.append(nn.Linear(prev_size, next_size))
        if activation == 'relu':
            layers.append(nn.LeakyReLU())
        elif activation == 'lrelu':
            layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_p[idx]))

    if out_dim is not None:
        layers.append(nn.Linear(sizes[-1], out_dim))

    return nn.Sequential(*layers)
