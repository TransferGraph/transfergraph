import random
from typing import Optional, Tuple

import numpy as np

random.seed(0)
import time
import sys

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

# from pecanpy.experimental import Node2vecPlusPlus
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from pecanpy import pecanpy

###DEFAULT HYPER PARAMS###
HPARAM_P = 1
HPARAM_DIM = 64  # 128
##########################

# from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_CLUSTER
WITH_PYG_LIB = False  # True
WITH_TORCH_CLUSTER = True  # False
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

# from torch_geometric.utils.sparse import index2ptr


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f"Device: '{device}'")


class Classifier(torch.nn.Module):
    def forward(self, model, edge_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_0 = model[edge_index[0]]
        edge_feat_1 = model[edge_index[1]]
        # edge_feat_1= x[edge_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_0 * edge_feat_1).sum(dim=-1)


class N2VPlusModel():
    def __init__(
            self,
            edge_index,
            embedding_dim=64,
            walk_length=6,  # 5,
            context_size=5,
            walks_per_node=5,
            num_negative_samples=1,
            negative_pairs=[],
            extend=True,
            epochs=50,
    ):
        self.epochs = epochs
        self.base = Node2VecPlus(
            edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,  # 20,
            context_size=context_size,
            walks_per_node=walks_per_node,  # 10,
            num_negative_samples=num_negative_samples,
            negative_pairs=negative_pairs,
            p=q,
            q=1,
            sparse=False,
        )

    def forward(self, data):
        pred = self.classifier(
            self.base(),
            data
        )
        return pred


class N2VModel(torch.nn.Module):
    def __init__(
            self,
            edge_index,
            edge_attr,
            node_IDs,
            embedding_dim=32,  # 128 64 32
            walk_length=10,  # 5,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=1,
            negative_pairs=[],
            p=1,
            q=1,
            sparse=False,  # True
            epochs=15,
            extend=False,
    ):
        super().__init__()
        self.epochs = epochs
        self.base = Node2Vec(
            edge_index,
            edge_attr,
            node_IDs,
            embedding_dim=embedding_dim,
            walk_length=walk_length,  # 20,
            context_size=context_size,
            walks_per_node=walks_per_node,  # 10,
            num_negative_samples=num_negative_samples,
            negative_pairs=negative_pairs,
            p=1,
            q=1,
            sparse=True,
            extend=extend,
        ).to(device)
        self.classifier = Classifier()

    def forward(self, data):
        pred = self.classifier(
            self.base(),
            data
        )
        self.x_dict = self.base()
        return pred

    @torch.no_grad()
    def test(self):
        self.base.eval()
        z = self.base()
        # acc = model.test(z[data.train_mask], data.y[data.train_mask],
        #                  z[data.test_mask], data.y[data.test_mask],
        #                  max_iter=150)
        # return acc

    def train(self):
        start = time.time()
        num_workers = 0 if sys.platform.startswith('win') else 4
        loader = self.base.loader(
            batch_size=64, shuffle=True,
            num_workers=0
        )
        optimizer = torch.optim.SparseAdam(list(self.base.parameters()), lr=0.01)

        for epoch in range(1, 1 + self.epochs):
            self.base.train()
            total_loss = 0
            count = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                if count == 0:
                    count += 1
                loss = self.base.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss = total_loss / len(loader)
            # acc = test()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')  # , Acc: {acc:.4f}')
        train_time = time.time() - start
        return loss, train_time


def index2ptr(index: Tensor, size: int) -> Tensor:
    return torch._convert_indices_from_coo_to_csr(
        index, size, out_int32=index.dtype == torch.int32
    )


class Node2Vec(torch.nn.Module):
    r"""The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.

    .. note::

        For an example of using Node2Vec, see `examples/node2vec.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        node2vec.py>`_.

    Args:
        edge_index (torch.Tensor): The edge indices.
        embedding_dim (int): The size of each embedding vector.
        walk_length (int): The walk length.
        context_size (int): The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            weight matrix will be sparse. (default: :obj:`False`)
    """

    def __init__(
            self,
            edge_index: Tensor,
            edge_attr,
            node_IDs,
            embedding_dim: int,
            walk_length: int,
            context_size: int,
            walks_per_node: int = 1,
            p: float = 1.0,
            q: float = 1.0,
            num_negative_samples: int = 1,
            negative_pairs: Optional[Tensor] = None,
            num_nodes: Optional[int] = None,
            workers=1,
            sparse: bool = False,
            extend: bool = False,
    ):
        super().__init__()

        if WITH_PYG_LIB and p == 1.0 and q == 1.0:
            self.random_walk_fn = torch.ops.pyg.random_walk
        elif WITH_TORCH_CLUSTER:
            self.random_walk_fn = torch.ops.torch_cluster.random_walk
        else:
            if p == 1.0 and q == 1.0:
                raise ImportError(
                    f"'{self.__class__.__name__}' "
                    f"requires either the 'pyg-lib' or "
                    f"'torch-cluster' package"
                )
            else:
                raise ImportError(
                    f"'{self.__class__.__name__}' "
                    f"requires the 'torch-cluster' package"
                )

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)
        # self.node_IDs = node_IDs

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        # print(f'row: {row.numpy()}')
        # print(f'col: {col.numpy()}')
        # r,c = edge_index.cpu()
        # print(f'row: {r.numpy()}')
        # print(f'col: {c.numpy()}')
        # print(f'edge_index: {list(edge_index.numpy())}')
        print(f'\nmax_row: {max(edge_index[0])}, max_col: {max(edge_index[1])}')
        print(f'min_row: {min(edge_index[0])}, min_col: {min(edge_index[1])}')
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col
        # print(edge_index)
        # print(f'self.rowptr {self.rowptr}')
        # print(self.col)

        self.EPS = 1e-15
        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.negative_pairs = negative_pairs

        self.edge_index = edge_index
        self.edge_attr = edge_attr

        self.workers = workers
        self.extend = extend

        ##### Node2Vec+ 
        self.embedding = Embedding(
            self.num_nodes, embedding_dim,
            sparse=sparse
        )

        if extend:
            self.positive_pairs = self.get_node2vecplus_walks()
            # self.embedding.weight.data.copy_(w2v_weights)

        self.reset_parameters()

    def get_node2vecplus_walks(self):
        print('\n ----------')
        adj_mat = to_scipy_sparse_matrix(self.edge_index, edge_attr=self.edge_attr).todense()
        print(f'\nadj_mat.shape: {adj_mat.shape}')
        # print(f'adj_mat: {adj_mat}')
        # print(f'IDs: {len(IDs)}')
        # adj_mat, IDs = np.load(network_fp).values()
        self.IDs = range(adj_mat.shape[0])
        g = pecanpy.DenseOTF.from_mat(adj_mat, self.IDs, extend=True)
        positive_walks = g.simulate_walks(num_walks=self.walks_per_node, walk_length=self.walk_length)
        return positive_walks

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()

    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        """Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.embedding.weight
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs) -> DataLoader:
        return DataLoader(
            range(self.num_nodes), collate_fn=self.sample,
            **kwargs
        )

    @torch.jit.export
    def pos_sample_ori(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(
            self.rowptr, self.col, batch,
            self.walk_length, self.p, self.q
        )
        if not isinstance(rw, Tensor):
            rw = rw[0]
        # print('\n---')
        # print(f'batch.shape: {batch.shape}')
        # print(f'rw.shape: {rw.shape}')
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        # print(f'len(walks): {len(walks)}')
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def pos_sample_plus(self, batch):
        # print(f'extend: {self.extend}')
        # walks = [[walk for walk in self.positive_pairs if walk[0] == node] for node in batch.numpy()]
        walks = []
        for node in batch.numpy():
            extended_walks = []
            for walk in self.positive_pairs:
                if walk[0] == node:
                    size = len(walk)
                    # print(size,walk)
                    if size < self.context_size:
                        extended_walk = walk
                        extended_walk.extend([walk[i % size] for i in range(self.context_size - size)])
                    else:
                        extended_walk = walk[:self.context_size]
                    # print(f'filtered_walk:{extended_walk}')
                    extended_walks.append(extended_walk)
            walks.append(torch.as_tensor(random.sample(extended_walks, self.walks_per_node)))
        # batch = batch.repeat(self.walks_per_node)
        # print(f'\n walks.size:{len(walks)}, walk length: {walks[0].shape} ')
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample_ori(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        # print(f'batch size: {batch.size()}')
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        # print(f'batch size after repeat: {batch.size()}')
        try:
            neg_pairs = self.negative_pairs[np.isin(self.negative_pairs.numpy()[:, 0], batch.numpy())]
            # print(f'neg_pairs: {neg_pairs}')
            neg_pairs = neg_pairs[np.random.choice(neg_pairs.size()[0], batch.size(0))]
        except Exception as e:
            print(e)
            neg_pairs = torch.randint(0, self.num_nodes, size=(batch.size(0), 2))
        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length - 2))
        rw = torch.cat([neg_pairs[:, 0].view(-1, 1), rw, neg_pairs[:, 1].view(-1, 1)], dim=1)
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        # print(f'self.extend: {self.extend}')
        if self.extend:
            pos_sample = self.pos_sample_plus(batch)
        else:
            pos_sample = self.pos_sample_ori(batch)
        # print(f'pos_sample: {pos_sample}')
        neg_sample = self.neg_sample(batch)
        # print(f'neg_sample: {neg_sample}')
        return pos_sample, neg_sample

    @torch.jit.export
    def loss_e2e(self):
        pass

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        # print()
        # print(f'pos_rw: {pos_rw.shape}, neg_rw: {neg_rw.shape}')
        # print(f'start: {start.shape}, rest: {rest.shape}')
        # print(f'pos_rw.size(0): {pos_rw.size(0)}, rest.view(-1)): {rest.view(-1).shape}')
        # print(f'unique rest.view(-1): {torch.unique(rest.view(-1))}')

        h_start = self.embedding(start).view(
            pos_rw.size(0), 1,
            self.embedding_dim
        )
        h_rest = self.embedding(rest.view(-1)).view(
            pos_rw.size(0), -1,
            self.embedding_dim
        )
        # print(f'\nh_start.shape: {h_start.shape}')
        # print(f'\nh_rest.shape: {h_rest.shape}')
        # print(f'\nh_start * h_rest: {(h_start * h_rest).shape}')

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(
            neg_rw.size(0), 1,
            self.embedding_dim
        )
        h_rest = self.embedding(rest.view(-1)).view(
            neg_rw.size(0), -1,
            self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss

    def test(
            self,
            train_z: Tensor,
            train_y: Tensor,
            test_z: Tensor,
            test_y: Tensor,
            solver: str = 'lbfgs',
            multi_class: str = 'auto',
            *args,
            **kwargs,
    ) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            solver=solver, multi_class=multi_class, *args,
            **kwargs
        ).fit(
            train_z.detach().cpu().numpy(),
            train_y.detach().cpu().numpy()
        )
        return clf.score(
            test_z.detach().cpu().numpy(),
            test_y.detach().cpu().numpy()
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')
