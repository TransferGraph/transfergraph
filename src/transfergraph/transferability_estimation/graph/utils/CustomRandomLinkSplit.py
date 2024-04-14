import random
import warnings
from copy import copy
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType
from torch_geometric.utils.num_nodes import maybe_num_nodes


# from torch_geometric.utils import negative_sampling

def sample(population: int, k: int, device=None) -> Tensor:
    if population <= k:
        return torch.arange(population, device=device)
    else:
        return torch.tensor(random.sample(range(population), k), device=device)


def negative_sampling(negative_edge_index, num_neg_samples):
    neg_pairs = negative_edge_index[np.random.choice(negative_edge_index.size()[0], num_neg_samples)]
    return torch.t(neg_pairs)


def negative_sampling_ori(
        edge_index: Tensor,
        num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
        num_neg_samples: Optional[int] = None,
        method: str = "sparse",
        force_undirected: bool = False
) -> Tensor:
    r"""Samples random negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)
        num_neg_samples (int, optional): The (approximate) number of negative
            samples to return.
            If set to :obj:`None`, will try to return a negative edge for every
            positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:

        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> negative_sampling(edge_index)
        tensor([[3, 0, 0, 3],
                [2, 3, 2, 1]])

        >>> # For bipartite graph
        >>> negative_sampling(edge_index, num_nodes=(3, 4))
        tensor([[0, 2, 2, 1],
                [2, 2, 1, 3]])
    """
    assert method in ['sparse', 'dense']

    size = num_nodes
    bipartite = isinstance(size, (tuple, list))
    size = maybe_num_nodes(edge_index) if size is None else size
    size = (size, size) if not bipartite else size
    force_undirected = False if bipartite else force_undirected

    idx, population = edge_index_to_vector(
        edge_index, size, bipartite,
        force_undirected
    )

    if idx.numel() >= population:
        return edge_index.new_empty((2, 0))

    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)
    if force_undirected:
        num_neg_samples = num_neg_samples // 2

    prob = 1. - idx.numel() / population  # Probability to sample a negative.
    sample_size = int(1.1 * num_neg_samples / prob)  # (Over)-sample size.

    neg_idx = None
    if method == 'dense':
        # The dense version creates a mask of shape `population` to check for
        # invalid samples.
        mask = idx.new_ones(population, dtype=torch.bool)
        mask[idx] = False
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size, idx.device)
            rnd = rnd[mask[rnd]]  # Filter true negatives.
            neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
            mask[neg_idx] = False

    else:  # 'sparse'
        # The sparse version checks for invalid samples via `np.isin`.
        idx = idx.to('cpu')
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size, device='cpu')
            mask = np.isin(rnd, idx)
            if neg_idx is not None:
                mask |= np.isin(rnd, neg_idx.to('cpu'))
            mask = torch.from_numpy(mask).to(torch.bool)
            rnd = rnd[~mask].to(edge_index.device)
            neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break

    return vector_to_edge_index(neg_idx, size, bipartite, force_undirected)


def edge_index_to_vector(
        edge_index: Tensor,
        size: Tuple[int, int],
        bipartite: bool,
        force_undirected: bool = False,
) -> Tuple[Tensor, int]:
    row, col = edge_index
    if bipartite:  # No need to account for self-loops.
        idx = (row * size[1]).add_(col)
        population = size[0] * size[1]
        return idx, population
    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]
        # We only operate on the upper triangular matrix:
        mask = row < col
        row, col = row[mask], col[mask]
        offset = torch.arange(1, num_nodes, device=row.device).cumsum(0)[row]
        idx = row.mul_(num_nodes).add_(col).sub_(offset)
        population = (num_nodes * (num_nodes + 1)) // 2 - num_nodes
        return idx, population
    else:
        assert size[0] == size[1]
        num_nodes = size[0]
        # We remove self-loops as we do not want to take them into account
        # when sampling negative values.
        mask = row != col
        row, col = row[mask], col[mask]
        col[row < col] -= 1
        idx = row.mul_(num_nodes - 1).add_(col)
        population = num_nodes * num_nodes - num_nodes
        return idx, population


def vector_to_edge_index(
        idx: Tensor, size: Tuple[int, int], bipartite: bool,
        force_undirected: bool = False
) -> Tensor:
    if bipartite:  # No need to account for self-loops.
        row = idx.div(size[1], rounding_mode='floor')
        col = idx % size[1]
        return torch.stack([row, col], dim=0)

    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        offset = torch.arange(1, num_nodes, device=idx.device).cumsum(0)
        end = torch.arange(
            num_nodes, num_nodes * num_nodes, num_nodes,
            device=idx.device
        )
        row = torch.bucketize(idx, end.sub_(offset), right=True)
        col = offset[row].add_(idx) % num_nodes
        return torch.stack([torch.cat([row, col]), torch.cat([col, row])], 0)

    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        row = idx.div(num_nodes - 1, rounding_mode='floor')
        col = idx % (num_nodes - 1)
        col[row <= col] += 1
        return torch.stack([row, col], dim=0)


# @functional_transform('random_link_split')
class RandomLinkSplit(BaseTransform):
    r"""Performs an edge-level random split into training, validation and test
    sets of a :class:`~torch_geometric.data.Data` or a
    :class:`~torch_geometric.data.HeteroData` object
    (functional name: :obj:`random_link_split`).
    The split is performed such that the training split does not include edges
    in validation and test splits; and the validation split does not include
    edges in the test split.

    .. code-block::

        from torch_geometric.transforms import RandomLinkSplit

        transform = RandomLinkSplit(is_undirected=True)
        train_data, val_data, test_data = transform(data)

    Args:
        num_val (int or float, optional): The number of validation edges.
            If set to a floating-point value in :math:`[0, 1]`, it represents
            the ratio of edges to include in the validation set.
            (default: :obj:`0.1`)
        num_test (int or float, optional): The number of test edges.
            If set to a floating-point value in :math:`[0, 1]`, it represents
            the ratio of edges to include in the test set.
            (default: :obj:`0.2`)
        is_undirected (bool): If set to :obj:`True`, the graph is assumed to be
            undirected, and positive and negative samples will not leak
            (reverse) edge connectivity across different splits. Note that this
            only affects the graph split, label data will not be returned
            undirected.
            (default: :obj:`False`)
        key (str, optional): The name of the attribute holding
            ground-truth labels.
            If :obj:`data[key]` does not exist, it will be automatically
            created and represents a binary classification task
            (:obj:`1` = edge, :obj:`0` = no edge).
            If :obj:`data[key]` exists, it has to be a categorical label from
            :obj:`0` to :obj:`num_classes - 1`.
            After negative sampling, label :obj:`0` represents negative edges,
            and labels :obj:`1` to :obj:`num_classes` represent the labels of
            positive edges. (default: :obj:`"edge_label"`)
        split_labels (bool, optional): If set to :obj:`True`, will split
            positive and negative labels and save them in distinct attributes
            :obj:`"pos_edge_label"` and :obj:`"neg_edge_label"`, respectively.
            (default: :obj:`False`)
        add_negative_train_samples (bool, optional): Whether to add negative
            training samples for link prediction.
            If the model already performs negative sampling, then the option
            should be set to :obj:`False`.
            Otherwise, the added negative samples will be the same across
            training iterations unless negative sampling is performed again.
            (default: :obj:`True`)
        neg_sampling_ratio (float, optional): The ratio of sampled negative
            edges to the number of positive edges. (default: :obj:`1.0`)
        disjoint_train_ratio (int or float, optional): If set to a value
            greater than :obj:`0.0`, training edges will not be shared for
            message passing and supervision. Instead,
            :obj:`disjoint_train_ratio` edges are used as ground-truth labels
            for supervision during training. (default: :obj:`0.0`)
        edge_types (Tuple[EdgeType] or List[EdgeType], optional): The edge
            types used for performing edge-level splitting in case of
            operating on :class:`~torch_geometric.data.HeteroData` objects.
            (default: :obj:`None`)
        rev_edge_types (Tuple[EdgeType] or List[Tuple[EdgeType]], optional):
            The reverse edge types of :obj:`edge_types` in case of operating
            on :class:`~torch_geometric.data.HeteroData` objects.
            This will ensure that edges of the reverse direction will be
            split accordingly to prevent any data leakage.
            Can be :obj:`None` in case no reverse connection exists.
            (default: :obj:`None`)
    """

    def __init__(
            self,
            num_val: Union[int, float] = 0.1,
            num_test: Union[int, float] = 0.2,
            is_undirected: bool = False,
            key: str = 'edge_label',
            split_labels: bool = False,
            add_negative_train_samples: bool = True,
            neg_sampling_ratio: float = 1.0,
            negative_pairs=[],
            disjoint_train_ratio: Union[int, float] = 0.0,
            edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
            rev_edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
            custom_negative_sampling=True,
    ):
        if isinstance(edge_types, list):
            if rev_edge_types is None:
                rev_edge_types = [None] * len(edge_types)

            assert isinstance(rev_edge_types, list)
            assert len(edge_types) == len(rev_edge_types)
        self.custom_negative_sampling = custom_negative_sampling
        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = is_undirected
        self.key = key
        self.split_labels = split_labels
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio
        self.negative_pairs = negative_pairs
        self.disjoint_train_ratio = disjoint_train_ratio
        self.edge_types = edge_types
        self.rev_edge_types = rev_edge_types

    def __call__(
            self,
            data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        edge_types = self.edge_types
        rev_edge_types = self.rev_edge_types

        train_data, val_data, test_data = copy(data), copy(data), copy(data)

        if isinstance(data, HeteroData):
            if edge_types is None:
                raise ValueError(
                    "The 'RandomLinkSplit' transform expects 'edge_types' to "
                    "be specified when operating on 'HeteroData' objects"
                )

            if not isinstance(edge_types, list):
                edge_types = [edge_types]
                rev_edge_types = [rev_edge_types]

            stores = [data[edge_type] for edge_type in edge_types]
            train_stores = [train_data[edge_type] for edge_type in edge_types]
            val_stores = [val_data[edge_type] for edge_type in edge_types]
            test_stores = [test_data[edge_type] for edge_type in edge_types]
        else:
            rev_edge_types = [None]
            stores = [data._store]
            train_stores = [train_data._store]
            val_stores = [val_data._store]
            test_stores = [test_data._store]

        for item in zip(
                stores, train_stores, val_stores, test_stores,
                rev_edge_types
        ):
            store, train_store, val_store, test_store, rev_edge_type = item

            is_undirected = self.is_undirected
            is_undirected &= not store.is_bipartite()
            is_undirected &= (rev_edge_type is None
                              or store._key == data[rev_edge_type]._key)

            edge_index = store.edge_index
            if is_undirected:
                mask = edge_index[0] <= edge_index[1]
                perm = mask.nonzero(as_tuple=False).view(-1)
                perm = perm[torch.randperm(perm.size(0), device=perm.device)]
            else:
                device = edge_index.device
                perm = torch.randperm(edge_index.size(1), device=device)

            num_val = self.num_val
            if isinstance(num_val, float):
                num_val = int(num_val * perm.numel())
            num_test = self.num_test
            if isinstance(num_test, float):
                num_test = int(num_test * perm.numel())

            num_train = perm.numel() - num_val - num_test

            if num_train <= 0:
                raise ValueError("Insufficient number of edges for training")

            train_edges = perm[:num_train]
            val_edges = perm[num_train:num_train + num_val]
            test_edges = perm[num_train + num_val:]
            train_val_edges = perm[:num_train + num_val]

            num_disjoint = self.disjoint_train_ratio
            if isinstance(num_disjoint, float):
                num_disjoint = int(num_disjoint * train_edges.numel())
            if num_train - num_disjoint <= 0:
                raise ValueError("Insufficient number of edges for training")

            # Create data splits:
            self._split(
                train_store, train_edges[num_disjoint:], is_undirected,
                rev_edge_type
            )
            self._split(val_store, train_edges, is_undirected, rev_edge_type)
            self._split(
                test_store, train_val_edges, is_undirected,
                rev_edge_type
            )

            # Create negative samples:
            num_neg_train = 0
            if self.add_negative_train_samples:
                if num_disjoint > 0:
                    num_neg_train = int(num_disjoint * self.neg_sampling_ratio)
                else:
                    num_neg_train = int(num_train * self.neg_sampling_ratio)
            num_neg_val = int(num_val * self.neg_sampling_ratio)
            num_neg_test = int(num_test * self.neg_sampling_ratio)

            num_neg = num_neg_train + num_neg_val + num_neg_test

            size = store.size()
            if store._key is None or store._key[0] == store._key[-1]:
                size = size[0]

            if not self.custom_negative_sampling:
                neg_edge_index = negative_sampling_ori(
                    edge_index, size,
                    num_neg_samples=num_neg,
                    method='sparse'
                )
            else:
                neg_edge_index = negative_sampling(self.negative_pairs, num_neg)
            # print('==========')
            # print(f'neg_edge_index: {neg_edge_index}')

            # Adjust ratio if not enough negative edges exist
            if neg_edge_index.size(1) < num_neg:
                num_neg_found = neg_edge_index.size(1)
                ratio = num_neg_found / num_neg
                warnings.warn(
                    f"There are not enough negative edges to satisfy "
                    "the provided sampling ratio. The ratio will be "
                    f"adjusted to {ratio:.2f}."
                )
                num_neg_train = int((num_neg_train / num_neg) * num_neg_found)
                num_neg_val = int((num_neg_val / num_neg) * num_neg_found)
                num_neg_test = num_neg_found - num_neg_train - num_neg_val

            # Create labels:
            if num_disjoint > 0:
                train_edges = train_edges[:num_disjoint]
            self._create_label(
                store,
                train_edges,
                neg_edge_index[:, num_neg_val + num_neg_test:],
                out=train_store,
            )
            self._create_label(
                store,
                val_edges,
                neg_edge_index[:, :num_neg_val],
                out=val_store,
            )
            self._create_label(
                store,
                test_edges,
                neg_edge_index[:, num_neg_val:num_neg_val + num_neg_test],
                out=test_store,
            )

        return train_data, val_data, test_data

    def _split(
            self,
            store: EdgeStorage,
            index: Tensor,
            is_undirected: bool,
            rev_edge_type: EdgeType,
    ) -> EdgeStorage:

        edge_attrs = {key for key in store.keys() if store.is_edge_attr(key)}
        for key, value in store.items():
            if key == 'edge_index':
                continue

            if key in edge_attrs:
                value = value[index]
                if is_undirected:
                    value = torch.cat([value, value], dim=0)
                store[key] = value

        edge_index = store.edge_index[:, index]
        if is_undirected:
            edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)
        store.edge_index = edge_index

        if rev_edge_type is not None:
            rev_store = store._parent()[rev_edge_type]
            for key in rev_store.keys():
                if key not in store:
                    del rev_store[key]  # We delete all outdated attributes.
                elif key == 'edge_index':
                    rev_store.edge_index = store.edge_index.flip([0])
                else:
                    rev_store[key] = store[key]

        return store

    def _create_label(
            self,
            store: EdgeStorage,
            index: Tensor,
            neg_edge_index: Tensor,
            out: EdgeStorage,
    ) -> EdgeStorage:

        edge_index = store.edge_index[:, index]

        if hasattr(store, self.key):
            edge_label = store[self.key]
            edge_label = edge_label[index]
            # Increment labels by one. Note that there is no need to increment
            # in case no negative edges are added.
            if neg_edge_index.numel() > 0:
                assert edge_label.dtype == torch.long
                assert edge_label.size(0) == edge_index.size(1)
                edge_label.add_(1)
            if hasattr(out, self.key):
                delattr(out, self.key)
        else:
            edge_label = torch.ones(index.numel(), device=index.device)

        if neg_edge_index.numel() > 0:
            neg_edge_label = edge_label.new_zeros(
                (neg_edge_index.size(1),) +
                edge_label.size()[1:]
            )

        if self.split_labels:
            out[f'pos_{self.key}'] = edge_label
            out[f'pos_{self.key}_index'] = edge_index
            if neg_edge_index.numel() > 0:
                out[f'neg_{self.key}'] = neg_edge_label
                out[f'neg_{self.key}_index'] = neg_edge_index

        else:
            if neg_edge_index.numel() > 0:
                edge_label = torch.cat([edge_label, neg_edge_label], dim=0)
                edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)
            out[self.key] = edge_label
            out[f'{self.key}_index'] = edge_index

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_val={self.num_val}, '
                f'num_test={self.num_test})')
