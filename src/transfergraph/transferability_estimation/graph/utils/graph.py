import os

import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.data import HeteroData

from .CustomRandomLinkSplit import RandomLinkSplit

os.system('unset LD_LIBRARY_PATH')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class HGraph:
    def __init__(
            self,
            gnn_method,
            max_dataset_idx,
            model_idx,
            unique_model_id,
            model_features,
            unique_dataset_id,
            dataset_features,
            edge_index_accu_model_to_dataset,
            edge_attr_accu_model_to_dataset,
            edge_index_dataset_to_dataset,
            edge_attr_dataset_to_dataset,
            edge_index_tran_model_to_dataset,
            edge_attr_tran_model_to_dataset,
            negative_pairs,
            contain_data_similarity=True,
            contain_dataset_feature=False,
            contain_model_feature=False,
            custom_negative_sampling=False
    ):
        self.custom_negative_sampling = custom_negative_sampling
        self.model_idx = model_idx

        self.data = HeteroData()
        # Save node indices:
        # all edges related to datasets
        # edge_index = torch.cat((edge_index_accu_model_to_dataset[1],edge_index_tran_model_to_dataset[1],edge_index_dataset_to_dataset[0],edge_index_dataset_to_dataset[1]))
        # self.data["dataset"].num_nodes = len(torch.unique(edge_index)) #torch.arange(len(unique_dataset_id))
        self.data['dataset'].node_id = torch.arange(len(unique_dataset_id))  # len(torch.unique(edge_index)))
        # print(f"\nself.data['dataset'].node_id: {self.data['dataset'].node_id}")

        # all edges related to modelse
        # edge_index = torch.cat((edge_index_accu_model_to_dataset,edge_index_tran_model_to_dataset),1)
        # self.data["model"].num_nodes = len(torch.unique(edge_index[0]))#len(unique_model_id) # torch.unique(edge_index[0]) #

        self.data['model'].node_id = torch.arange(len(unique_model_id))  # len(torch.unique(edge_index[0])))#+self.data["dataset"].num_nodes
        # self.data['model'].node_id = torch.from_numpy(self.model_idx).to(torch.int64) #torch.arange(self.data.max_dataset_idx + 1)
        # print(f"\nself.data['model'].node_id: {self.data['model'].node_id}")

        data_feature_shape = list(dataset_features.values())[0].shape[0]
        features = []
        for i in range(len(unique_dataset_id)):
            features.append(dataset_features[unique_dataset_id[unique_dataset_id['mappedID'] == i]['dataset'].values[0]])

        if contain_model_feature:
            self.data["model"].x = torch.from_numpy(model_features).to(torch.float)
        else:
            self.data['model'].x = torch.rand((len(unique_model_id), data_feature_shape))

        # self.data["dataset"].x = dataset_features
        # self.data["dataset"].x = torch.from_numpy(dataset_features).to(torch.float)
        # datset_features = np.around(np.random.random_sample((len(dataset_features), 128))+0.00001,3)
        # dataset_features = np.random.randint(0,1,size=(len(dataset_features),20))
        if contain_dataset_feature:
            self.data['dataset'].x = torch.from_numpy(np.vstack(features)).to(torch.float)
        else:
            self.data['dataset'].x = torch.rand((len(unique_dataset_id), data_feature_shape))

        # print()
        # print('self.data["dataset"].x.shape')
        # print('========')
        # print(self.data["dataset"].x.dtype)
        # print(self.data["dataset"].x.shape)

        # self.data["model"].x = model_features
        # if 'homo' not in gnn_method:
        #     edge_index_accu_model_to_dataset[0] -= max_dataset_idx
        if 'without_accuracy' in gnn_method:
            print('\n without_accuary in gnn_method')
            self.label_type = ["model", "transfer_to", "dataset"]
        elif 'without_accuracy' not in gnn_method:
            print('\n without_accuary not in gnn_method')
            self.data["model", "trained_on", "dataset"].edge_index = edge_index_accu_model_to_dataset  # TODO
            if edge_attr_accu_model_to_dataset != None:
                self.data['model', 'trained_on', 'dataset'].edge_attr = edge_attr_accu_model_to_dataset  # TODO
            if 'trained_on_transfer' in gnn_method:
                self.label_type = ["model", "transfer_to", "dataset"]
            else:
                self.label_type = ["model", "trained_on", "dataset"]

        if contain_data_similarity:
            self.data["dataset", "similar_to", "dataset"].edge_index = edge_index_dataset_to_dataset  # TODO
        if edge_attr_dataset_to_dataset != None:
            self.data['dataset', 'similar_to', 'dataset'].edge_attr = edge_attr_dataset_to_dataset  # TODO

        if 'without_transfer' not in gnn_method:
            # if 'homo' not in gnn_method:
            #     edge_index_tran_model_to_dataset[0] -= max_dataset_idx
            self.data["model", "transfer_to", "dataset"].edge_index = edge_index_tran_model_to_dataset  # TODO
            self.data["model", "transfer_to", "dataset"].edge_attr = edge_attr_tran_model_to_dataset  # TODO

        self.negative_pairs = negative_pairs
        print()
        # print(f'\nedge_index_accu_model_to_dataset: {edge_index_accu_model_to_dataset}')
        # print(f'\nedge_index_tran_model_to_dataset: {edge_index_tran_model_to_dataset}')
        # print(f'-- max node index: {torch.max(edge_index_accu_model_to_dataset),0}, {torch.max(edge_index_tran_model_to_dataset),0},{torch.max(edge_index_dataset_to_dataset),0}')

        # print(self.data.metadata())

        self.transform()
        # self.split()
        self._print()

    def transform(self):
        self.data = T.ToUndirected()(self.data)
        # self.data = T.AddSelfLoops()(self.data)
        # self.data = T.NormalizeFeatures()(self.data)

    def _print(self):
        print()
        print("Data:")
        print("==============")
        print(self.data)
        print(self.data.metadata())
        print("self.data['dataset'].num_nodes")
        print(self.data['dataset'].num_nodes)
        print("self.data['model'].num_nodes")
        print(self.data['model'].num_nodes)
        # num_edges = self.data["model", "trained_on", "dataset"].num_edges
        # print(f'self.data["model", "trained_on", "dataset"].num_edges: {num_edges}')
        # num_edges = self.data["dataset", "similar_to", "dataset"].num_edges
        # print(f'self.data["dataset", "similar_to", "dataset"].num_edges: {num_edges}')
        # print("self.data['dataset'].x")
        # print(self.data['dataset'].x)
        # print(self.data["model", "trained_on", "dataset"].edge_label_index)
        # print(self.data["model", "trained_on", "dataset"].edge_label)
        print()

    def split(self, num_val=0.1, num_test=0.2):  # num_val=0.1,num_test=0.2):
        s, r, t = self.label_type
        print(f'\n label_type: {self.label_type}')
        transform = RandomLinkSplit(  # T.Compose([T.ToUndirected(),
            # transform = T.RandomLinkSplit(
            num_val=num_val,  # TODO
            num_test=num_test,  # TODO
            disjoint_train_ratio=0.3,  # TODO
            neg_sampling_ratio=2.0,  # TODO
            add_negative_train_samples=True,  # TODO
            negative_pairs=self.negative_pairs,
            # edge_types=("model", "trained_on", "dataset"),
            edge_types=(s, r, t),
            is_undirected=True,
            # rev_edge_types = self.data.metadata()[1]
            # rev_edge_types=("dataset", "rev_trained_on", "model"), 
            custom_negative_sampling=self.custom_negative_sampling
        )
        # ])
        # self.train_data, self.val_data, self.test_data = # self.val_data, 
        return transform(self.data)


class LineGraph():
    def __init__(
            self,
            edge_index_accu_model_to_dataset,
            edge_index_tran_model_to_dataset,
            edge_index_dataset_to_dataset,
            without_transfer=True,
            # max_model_id
    ):
        if without_transfer:
            edge_index = edge_index_accu_model_to_dataset
        else:
            edge_index = torch.cat((edge_index_accu_model_to_dataset, edge_index_tran_model_to_dataset), 1)
        edge_index = torch.cat((edge_index, edge_index_dataset_to_dataset), 1).numpy()
        edges = list(map(tuple, edge_index))
        G = nx.Graph()
        # print(f'\n edges: {edges}')
        G.add_edges_from(edges)
        self.graph = G


class Graph():
    def __init__(
            self,
            node_ID,
            edge_index_accu_model_to_dataset,
            edge_attr_accu_model_to_dataset,
            edge_index_tran_model_to_dataset,
            edge_attr_tran_model_to_dataset,
            edge_index_dataset_to_dataset,
            edge_attr_dataset_to_dataset,
            without_transfer=False,
            without_accuracy=False,
            # max_model_id
    ):

        # max_model_id = int(torch.max(edge_index_model_to_dataset[0,:]).item()) + 1
        # rename dataset index name
        # edge_index_model_to_dataset[1,:] += max_model_id + 1
        # edge_index_dataset_to_dataset += max_model_id + 1
        # print('max_model_id', torch.max(edge_index_model_to_dataset[1,:]))

        if without_transfer:
            edge_index = edge_index_accu_model_to_dataset
        elif without_accuracy:
            edge_index = edge_index_tran_model_to_dataset
        else:
            edge_index = torch.cat((edge_index_accu_model_to_dataset, edge_index_tran_model_to_dataset), 1)

        edge_index = torch.cat((edge_index, edge_index_dataset_to_dataset), 1).type(torch.int64)

        if without_transfer:
            edge_attr = edge_attr_accu_model_to_dataset
        elif without_accuracy:
            edge_attr = edge_attr_tran_model_to_dataset
        else:
            edge_attr = torch.cat((edge_attr_accu_model_to_dataset, edge_attr_tran_model_to_dataset))
        edge_attr = torch.cat((edge_attr, edge_attr_dataset_to_dataset))

        # convert it to undirected graph
        # from torch_geometric.utils import to_undirected
        # edge_index, edge_attr = to_undirected(edge_index,edge_attr)
        self.data = Data(edge_index=edge_index, edge_attr=edge_attr)
        self.data.node_id = node_ID
        # print(f'self.data.node_id: {self.data.node_id}')
        # import torch_geometric.transforms as T
        # self.data = T.ToUndirected()(data)
        try:
            print()
            print(f'----- Graph Properties -----')
            print(self.data)
            # print(self.data.edge_index)
            print(
                f'-- number of accuracy & transferability edge: {edge_index_accu_model_to_dataset.shape}, {edge_index_tran_model_to_dataset.shape}, {edge_index_dataset_to_dataset.shape}'
            )
            # print(f'-- max accu index: {torch.max(edge_index_accu_model_to_dataset),0}')
            # print(f'-- max tran index  {torch.max(edge_index_tran_model_to_dataset),0}')
            # print(f'-- max dataset index: {torch.max(edge_index_dataset_to_dataset),0}')
            print(f'-- number of nodes: {self.data.num_nodes}')
            print(f' number of edges: {self.data.num_edges}')
            print(f'-- data is directed(): {self.data.is_directed()}')
        except Exception as e:
            print(e)
