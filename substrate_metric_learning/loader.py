from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter


from ipdb import set_trace as st

class Collater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        # elem = batch[0]
        # if isinstance(elem, BaseData):
        #     return Batch.from_data_list(
        #         batch,
        #         follow_batch=self.follow_batch,
        #         exclude_keys=self.exclude_keys,
        #     )
        # elif isinstance(elem, torch.Tensor):
        #     return default_collate(batch)
        # elif isinstance(elem, TensorFrame):
        #     return torch_frame.cat(batch, dim=0)
        # elif isinstance(elem, float):
        #     return torch.tensor(batch, dtype=torch.float)
        # elif isinstance(elem, int):
        #     return torch.tensor(batch)
        # elif isinstance(elem, str):
        #     return batch
        # elif isinstance(elem, Mapping):
        #     return {key: self([data[key] for data in batch]) for key in elem}
        # elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        #     return type(elem)(*(self(s) for s in zip(*batch)))
        # elif isinstance(elem, Sequence) and not isinstance(elem, str):
        #     return [self(s) for s in zip(*batch)]

        # raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")
        
        # Determine the batch size (number of graphs)
        batch_size = len(batch)
        
        # Extract the number of nodes and features per node from the first graph in the batch
        num_nodes = max([g.x.size(0) for g in batch])
        node_features = batch[0].x.size(1)
        
        # Extract the number of edge features from the first graph in the batch
        edge_features = batch[0].edge_attr.size(1)
        
        # Initialize tensors for nodes, edges, adjacency matrices, and masks
        nodes = torch.zeros((batch_size, num_nodes, node_features))
        edges = torch.zeros((batch_size, num_nodes, num_nodes, edge_features))
        adj_mat = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.int)
        mask = torch.ones((batch_size, num_nodes), dtype=torch.bool)
        y = torch.zeros((batch_size))
        
        # Iterate through the batch and populate the tensors
        for i, data in enumerate(batch):
            y[i] = data.y
            nodes[i, :data.x.size(0), :] = data.x
            mask[i, data.x.size(0):] = False
            adj_mat[i][data.edge_index[0], data.edge_index[1]] = 1
            if len(data.edge_index) > 0:
                for j in range(data.edge_index.size(1)):
                    src = data.edge_index[0, j]
                    dst = data.edge_index[1, j]
                    edges[i, src, dst] = data.edge_attr[j]
       
        return nodes, edges, adj_mat, mask, y


class GTDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )

class SC_Collater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        # elem = batch[0]
        # if isinstance(elem, BaseData):
        #     return Batch.from_data_list(
        #         batch,
        #         follow_batch=self.follow_batch,
        #         exclude_keys=self.exclude_keys,
        #     )
        # elif isinstance(elem, torch.Tensor):
        #     return default_collate(batch)
        # elif isinstance(elem, TensorFrame):
        #     return torch_frame.cat(batch, dim=0)
        # elif isinstance(elem, float):
        #     return torch.tensor(batch, dtype=torch.float)
        # elif isinstance(elem, int):
        #     return torch.tensor(batch)
        # elif isinstance(elem, str):
        #     return batch
        # elif isinstance(elem, Mapping):
        #     return {key: self([data[key] for data in batch]) for key in elem}
        # elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        #     return type(elem)(*(self(s) for s in zip(*batch)))
        # elif isinstance(elem, Sequence) and not isinstance(elem, str):
        #     return [self(s) for s in zip(*batch)]

        # raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")
        
        # Determine the batch size (number of graphs)
        batch_size = len(batch)
        
        # Extract the number of nodes and features per node from the first graph in the batch
        num_nodes = max([g.x.size(0) for g in batch])
        node_features = batch[0].x.size(1)
        
        # Extract the number of edge features from the first graph in the batch
        edge_features = batch[0].edge_attr.size(1)
        
        # Initialize tensors for nodes, edges, adjacency matrices, and masks
        nodes = torch.zeros((batch_size, num_nodes, node_features))
        edges = torch.zeros((batch_size, num_nodes, num_nodes, edge_features))
        adj_mat = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.int)
        mask = torch.ones((batch_size, num_nodes), dtype=torch.bool)
        values = torch.zeros((batch_size))
        labels = torch.zeros((batch_size))
        atm_idx = torch.zeros((batch_size), dtype=torch.int64)
        atm_cls = torch.zeros((batch_size), dtype=torch.int64)
        smiles_list = []
        
        # Iterate through the batch and populate the tensors
        for i, data in enumerate(batch):
            values[i] = data.y
            labels[i] = data.scope_id
            atm_idx[i] = data.atm_idx.item()
            atm_cls[i] = data.atm_cls.item()
            smiles_list.append(data.smiles)
            nodes[i, :data.x.size(0), :] = data.x
            mask[i, data.x.size(0):] = False
            adj_mat[i][data.edge_index[0], data.edge_index[1]] = 1
            if len(data.edge_index) > 0:
                for j in range(data.edge_index.size(1)):
                    src = data.edge_index[0, j]
                    dst = data.edge_index[1, j]
                    edges[i, src, dst] = data.edge_attr[j]
       
        return nodes, edges, adj_mat, mask, values, labels, atm_idx, smiles_list, atm_cls 

class SC_GTDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=SC_Collater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )


