import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphSAGE, SAGEConv, GATConv, GraphConv, GCNConv, HGTConv, to_hetero, Linear, HeteroConv


# cuda issue


# os.system('unset LD_LIBRARY_PATH')
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class EebedGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        # self.conv1 = SAGEConv(-1, hidden_channels)
        # self.conv1 = SAGEConv((-1, -1), hidden_channels)
        # self.conv2 = SAGEConv(-1, hidden_channels)
        # self.conv1 = GCNConv(-1, hidden_channels) 
        # self.conv1 = GATConv((-1, -1), hidden_channels,add_self_loops=False)
        # self.lin1 = Linear(-1, hidden_channels)
        # self.conv2 = GATConv((-1, -1), hidden_channels,add_self_loops=False)
        # self.lin2 = Linear(-1, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:  #: Tensor
        # print(f'x.shape: {x.shape}')
        # Define a 2-layer GNN computation graph.
        # Use a *single* `ReLU` non-linearity in-between.
        x = F.relu(self.conv1(x, edge_index))  # + self.lin1(x)
        x = self.conv2(x, edge_index)  # + self.lin2(x)
        return x
        # raise NotImplementedError


class SAGENN(torch.nn.Module):
    def __init__(self, hidden_channels, add_regression_layer=True):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.add_regression_layer = add_regression_layer
        self.hidden_channels = hidden_channels

    def forward(self, x, edge_index) -> Tensor:  #: Tensor
        x = F.relu(self.conv1(x, edge_index))  # + self.lin1(x)
        x = self.conv2(x, edge_index)  # + self.lin2(x)
        return x


class GraphNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GraphConv(-1, hidden_channels, add_self_loops=False)
        self.conv2 = GraphConv(-1, hidden_channels, add_self_loops=False)

    def forward(self, x, edge_index) -> Tensor:  #: Tensor
        x = F.relu(self.conv1(x, edge_index))  # + self.lin1(x)
        x = self.conv2(x, edge_index)  # + self.lin2(x)
        return x


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ('model', 'trained_on', 'dataset'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('dataset', 'similar_to', 'dataset'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('dataset', 'rev_trained_on', 'model'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('model', 'transfer_to', 'dataset'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                }, aggr='sum'
            )
            self.convs.append(conv)

        # self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        # print(f'\n-- HeteroGNN.x_dict: {x_dict.keys()}')
        # print(f"\n-- x_dict.model: {x_dict['model'].shape}")
        # print(f"\n-- x_dict.model: {x_dict['dataset'].shape}")
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # print(f'\n-- after_convs.x_dict: {x_dict.keys()}')
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        # print(x_dict)
        # return self.lin(x_dict['author'])
        return x_dict


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(-1, hidden_channels, add_self_loops=False)

    def forward(self, x, edge_index) -> Tensor:  #: Tensor
        x = F.relu(self.conv1(x, edge_index))  # + self.lin1(x)
        x = self.conv2(x, edge_index)  # + self.lin2(x)
        return x


class HeteroModel(torch.nn.Module):
    def __init__(
            self,
            metadata,
            num_dataset_nodes,
            num_model_nodes,
            dim_model_feature,
            dim_dataset_feature,
            contain_model_feature,
            contain_dataset_feature,
            embed_model_feature,
            embed_dataset_feature,
            gnn_method,
            label_type,
            hidden_channels,
            node_types=None
    ):
        super().__init__()
        self.contain_model_feature = contain_model_feature
        self.contain_dataset_feature = contain_dataset_feature
        self.embed_model_feature = embed_model_feature
        self.embed_dataset_feature = embed_dataset_feature
        self.gnn_method = gnn_method
        self.s, self.r, self.t = label_type

        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        # print()
        # print('=========')
        # print(f'data["dataset"].x.shape: {data["dataset"].x.shape[0]},{data["dataset"].x.shape[1]}')
        if self.contain_dataset_feature:
            self.dataset_lin = torch.nn.Linear(dim_dataset_feature, hidden_channels)  # data["dataset"].x.shape[1]
        if self.contain_model_feature:
            self.model_lin = torch.nn.Linear(dim_model_feature, hidden_channels)  # data["dataset"].x.shape[1]
        # print()
        # print("== self.dataset_lin.state_dict()['weight'].shape")
        # print(self.dataset_lin.state_dict()['weight'].shape)

        self.model_emb = torch.nn.Embedding(num_model_nodes, hidden_channels)  # data["model"].num_nodes
        self.dataset_emb = torch.nn.Embedding(num_dataset_nodes, hidden_channels)  # data["dataset"].num_nodes

        ## Instantiate homogeneous GNN:
        print('\nself.gnn_method: {self.gnn_method}')
        if 'SAGEConv' in self.gnn_method:
            self.gnn = SAGENN(hidden_channels)
        elif 'GATConv' in self.gnn_method:
            self.gnn = GAT(hidden_channels)
        elif 'HGTConv' in self.gnn_method:
            self.gnn = HGT(hidden_channels, node_types, metadata)
        elif 'GraphConv' in self.gnn_method:
            self.gnn = GraphNN(hidden_channels)
        elif 'GCNConv' in self.gnn_method:
            self.gnn = GCN(hidden_channels)
        elif 'HeteroGNN' in self.gnn_method:
            self.gnn = HeteroGNN(hidden_channels)
        elif 'GraphSAGE' in self.gnn_method:
            self.gnn = GraphSAGE(hidden_channels, hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        if ('HGTConv' not in self.gnn_method) and ('HeteroGNN' not in self.gnn_method) and ('homo' not in self.gnn_method):
            self.gnn = to_hetero(self.gnn, metadata=metadata)  # ,aggr='sum')

        self.classifier = HeteroClassifier()
        self.flag = True

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {}
        if self.contain_model_feature:
            if self.embed_model_feature:
                # any meaning?
                # print('===============')
                # print(f'data["model"].node_id.shape: {data["model"].node_id.shape}')
                # print(f'data["model"].x.shape: {data["model"].x.shape}')
                try:
                    model_emb = self.model_lin(data['model'].x)  # + self.model_emb(data["model"].node_id)
                except:
                    x = data["model"].x
                    x = torch.index_select(x, 0, data['model'].node_id)
                    model_emb = self.model_lin(x) + self.model_emb(data["model"].node_id)
                    del x
            else:
                model_emb = data['model'].x
            # x_dict['model'] = model_emb
        else:
            if self.embed_model_feature:
                model_emb = self.model_emb(data["model"].node_id)

        if self.contain_dataset_feature:
            if self.embed_dataset_feature:
                dataset_emb = self.dataset_lin(data["dataset"].x)  # + self.dataset_emb(data["dataset"].node_id)
            else:
                dataset_emb = data['dataset'].x

        else:
            if self.embed_dataset_feature:
                dataset_emb = self.dataset_emb(data["dataset"].node_id)
        x_dict = {
            'model': model_emb,
            'dataset': dataset_emb
        }
        if self.flag:
            # print(f'data["dataset"].x:{data["dataset"].x.shape}')
            # print(f'data["dataset"].node_id:{data["dataset"].node_id.shape}')
            print('\n============')
            print(f'data.edge_attr_dict: {data.edge_attr_dict.keys()}')
            # print(f'data.edge_index_dict: {data.edge_index_dict}')
            self.flag = False

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        # self.x_dict = self.gnn(x_dict,data.edge_index_dict)#(x_dict, data.edge_index_dict)
        if self.gnn_method == 'GATConv':
            contain_attr = True
            if not contain_attr:
                data.edge_attr_dict = None
            self.x_dict = self.gnn(x_dict, data.edge_index_dict, data.edge_attr_dict)  # (x_dict, data.edge_index_dict)
        elif self.gnn_method == 'GraphConv' or self.gnn_method == 'GCNConv':
            self.x_dict = self.gnn(x_dict, data.edge_index_dict)  # (x_dict, data.edge_index_dict)
        else:
            self.x_dict = self.gnn(x_dict, data.edge_index_dict)  # (x_dict, data.edge_index_dict)

        pred = self.classifier(
            self.x_dict["model"],
            self.x_dict["dataset"],
            data[self.s, self.r, self.t].edge_label_index,
        )
        return pred


class HomoModel(torch.nn.Module):
    def __init__(
            self,
            metadata,
            gnn_method,
            hidden_channels,
            node_types=None
    ):
        super().__init__()
        self.gnn_method = gnn_method

        # Instantiate homogeneous GNN:
        if 'SAGEConv' in self.gnn_method:
            self.gnn = SAGENN(hidden_channels)
        elif 'GATConv' in self.gnn_method:
            self.gnn = GAT(hidden_channels)
        elif 'HGTConv' in self.gnn_method:
            self.gnn = HGT(hidden_channels, node_types, metadata)
        elif 'GraphConv' in self.gnn_method:
            self.gnn = GraphNN(hidden_channels)
        elif 'GCNConv' in self.gnn_method:
            self.gnn = GCN(hidden_channels)
        elif 'HeteroGNN' in self.gnn_method:
            self.gnn = HeteroGNN(hidden_channels)
        # elif 'GraphSAGE' in self.gnn_method:
        #     self.gnn = GraphSAGE(hidden_channels,hidden_channels,num_layers=2)

        if 'lr' in self.gnn_method or 'rf' in self.gnn_method:
            self.classifier = HomoClassifier()
        elif 'e2e' in self.gnn_method:
            self.classifier = HomoRegression(hidden_channels)
        self.flag = True

    def forward(self, data) -> Tensor:

        if self.gnn_method == 'GATConv':
            contain_attr = True
            if not contain_attr:
                data.edge_attr_dict = None
            self.x_dict = self.gnn(data.x, data.edge_index, data.edge_attr)  # (x_dict, data.edge_index_dict)
        # elif self.gnn_method == 'GraphConv': # or self.gnn_method == 'GCNConv':
        #     self.x_dict = self.gnn(data.x,data.edge_index)#(x_dict, data.edge_index_dict)
        else:
            self.x_dict = self.gnn(data.x, data.edge_index)  # (x_dict, data.edge_index_dict)

        pred = self.classifier(
            self.x_dict,
            data.edge_label_index,
        )
        return pred


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, node_types, metadata, num_heads=2, num_layers=2):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                hidden_channels, hidden_channels, metadata,
                num_heads, group='sum'
            )
            self.convs.append(conv)

        # self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict
        # return self.lin(x_dict['author'])


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin2 = Linear(-1, hidden_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr) + self.lin2(x)
        return x


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class HeteroClassifier(torch.nn.Module):
    def forward(self, x_model: Tensor, x_dataset: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_model = x_model[edge_label_index[0]]
        edge_feat_dataset = x_dataset[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        # return (edge_feat_model * edge_feat_dataset).sum(dim=-1)
        return torch.sigmoid((edge_feat_model * edge_feat_dataset).sum(dim=-1))


class HomoClassifier(torch.nn.Module):
    def forward(self, x, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_model = x[edge_label_index[0]]
        edge_feat_dataset = x[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_model * edge_feat_dataset).sum(dim=-1)
        # return torch.sigmoid((edge_feat_model * edge_feat_dataset).sum(dim=-1))


class HomoRegression(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(HomoRegression, self).__init__()
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        # print(f'\nedge_label_index.shape:{edge_label_index.shape}')
        edge_feat_model = x[edge_label_index[0]]
        edge_feat_dataset = x[edge_label_index[1]]

        x = self.fc1(torch.cat([edge_feat_model, edge_feat_dataset], dim=1))
        x = F.relu(x)
        x = self.fc2(x)
        output = torch.flatten(torch.sigmoid(x))
        return output

        # Apply dot-product to get a prediction per supervision edge:
        # return (edge_feat_model * edge_feat_dataset).sum(dim=-1)
