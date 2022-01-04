from torch_geometric.data import Data, Batch
import torch
from typing import Iterable

from holoprot.data.base import ComplexData, HierData

class ProtBatch(Data):
    """
    ProtBatch is a wrapper class, used to create a batch for a 
    list of HierData objects. Components of HierData such as surface 
    or backbone can already be batched using torch_geometric.data.Batch.
    """

    @classmethod
    def from_data_list(cls, 
                       data_list: Iterable[HierData], 
                       prot_mode: str = 'backbone'):
        batch = cls()
        components = prot_mode.split("2") # Tells us the components of the graph
        # Options: [backbone2patch, patch2backbone, surface2backbone, backbone, surface]

        # The current version of pytorch_geometric does not support exclude_keys yet
        # Hence attributes need to be deleted
        if 'surface' in components or 'patch' in components:
            if hasattr(data_list[0], 'surface'):
                [delattr(data.surface, 'face') for data in data_list if hasattr(data.surface, 'face')]
            elif hasattr(data_list[0], 'patch'):
                [delattr(data.patch, 'face') for data in data_list if hasattr(data.patch, 'face')]

        for component in components:
            if hasattr(data_list[0], component):
                # These are all pytorch_geometric.Data objects with x, edge_index and edge_attr
                # We can use pytorch_geometric batching for these things
                component_batch = Batch.from_data_list([getattr(data, component)
                                                   for data in data_list])
                setattr(batch, component, component_batch)

        if hasattr(data_list[0], 'surface_to_patch'):
            batch.surface_to_patch = Batch.from_data_list([data.surface_to_patch for data in data_list])

        if hasattr(data_list[0], 'y') and getattr(data_list[0], 'y') is not None:
            batch.y = torch.tensor([data.y for data in data_list]).view(-1)

        if len(components) > 1 and hasattr(data_list[0], 'mapping'):
            # Mapping is the only special case
            lowest_attr = components[0]
            get_num_nodes = lambda x: getattr(x, lowest_attr).num_nodes

            num_nodes, in_dim, out_dim = 0, 0, 0
            mappings = []

            in_dim, out_dim = 0, 0

            for data in data_list:
                mapping = data.mapping
                mask = (mapping == 0) # Will set the mask to zero after update
                mapping = mapping + num_nodes
                mapping[mask] = 0

                in_dim += mapping.size(0)
                out_dim = max(out_dim, mapping.size(1))

                mappings.append(mapping)
                num_nodes += get_num_nodes(data)

            final_mapping = torch.zeros((in_dim, out_dim))
            start = 0
            for mapping in mappings:
                final_mapping[start: start + mapping.size(0), :mapping.size(1)] = mapping
                start += mapping.size(0)
            batch.mapping = final_mapping.long()

        return batch

    def to(self, device: str = 'cpu'):
        if hasattr(self, 'backbone'):
            self.backbone = self.backbone.to(device)
        if hasattr(self, 'surface'):
            self.surface = self.surface.to(device)
        if hasattr(self, 'patch'):
            self.patch = self.patch.to(device)
        if hasattr(self, 'mapping'):
            self.mapping = self.mapping.to(device)
        if hasattr(self, 'surface_to_patch'):
            self.surface_to_patch = self.surface_to_patch.to(device)
        if hasattr(self, 'y') and self.y is not None:
            self.y = self.y.to(device)
        return self

    def __repr__(self):
        reprs = []
        if hasattr(self, 'backbone'):
            reprs.append(f"Backbone={repr(self.backbone).strip()}")
        if hasattr(self, 'surface'):
            reprs.append(f"Surface={repr(self.surface).strip()}")
        if hasattr(self, 'patch'):
            reprs.append(f"Patch={repr(self.patch).strip()}")
        if hasattr(self, 'mapping'):
            reprs.append(f"Mapping={self.mapping.shape}".strip())
        if hasattr(self, 'surface_to_patch'):
            reprs.append(f"Surface2Patch={repr(self.surface_to_patch).strip()}")
        return ", ".join(reprs)


class ComplexBatch(Data):
    """
    ComplexBatch is a wrapper class, used to create a batch for a 
    list of ComplexData objects. ComplexData.protein is batched using
    ProtBatch while ComplexData.ligand is batched using torch_geometric batching.
    """

    @classmethod
    def from_data_list(cls, 
                       data_list: Iterable[ComplexData], 
                       prot_mode: str = 'backbone'):
        batch = cls()
        batch.protein = ProtBatch.from_data_list([getattr(data, 'protein')
                                                for data in data_list], prot_mode=prot_mode)
        [delattr(data.ligand, 'mess_idx') for data in data_list
         if hasattr(data.ligand, 'mess_idx')]
        batch.ligand = Batch.from_data_list([data.ligand for data in data_list])
        if hasattr(data_list[0], 'y') and getattr(data_list[0], 'y') is not None:
            batch.y = torch.tensor([data.y for data in data_list]).view(-1)
        return batch

    def to(self, device: str):
        self.protein = self.protein.to(device)
        self.ligand = self.ligand.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self

    def __repr__(self):
        reprs = [f"Protein=({repr(self.protein).strip()})",
                 f"Ligand=({repr(self.ligand).strip()})"]
        if self.y is not None:
            reprs.append(f"Target={self.y.shape}")
        return ", ".join(reprs)