# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# os.environ['TORCH'] = torch.__version__
# print(torch.__version__)
import tqdm
from torch_geometric.utils import degree

from .utils.graph import Graph
from .utils.node2vec import N2VModel
from .utils.node2vec_w2v import N2V_W2VModel

# from torch_geometric.nn import Node2Vec

# import torch_cluster
# !pip install torch_cluster -f -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f"Device: '{device}'")

SAVE_GRAPH = False


def get_graph(args, data_dict, setting_dict):
    edge_index_accu_model_to_dataset = data_dict['edge_index_accu_model_to_dataset']
    edge_attr_accu_model_to_dataset = data_dict['edge_attr_accu_model_to_dataset']
    edge_index_tran_model_to_dataset = data_dict['edge_index_tran_model_to_dataset']
    edge_attr_tran_model_to_dataset = data_dict['edge_attr_tran_model_to_dataset']

    print(f'\nedge_index_accu_model_to_dataset: {edge_index_accu_model_to_dataset.shape}')
    # print(torch.unique(edge_index_accu_model_to_dataset))
    print(f'\nedge_index_tran_model_to_dataset: {edge_index_tran_model_to_dataset.shape}')
    # print(torch.unique(edge_index_tran_model_to_dataset))

    edge_index_dataset_to_dataset = data_dict['edge_index_dataset_to_dataset']
    edge_attr_dataset_to_dataset = data_dict['edge_attr_dataset_to_dataset']
    # negative_pairs = data_dict['negative_pairs']
    # node_IDs = data_dict['node_ID']
    ## Construct a graph
    without_accuracy = False
    without_transfer = False
    if 'without_transfer' in args.gnn_method:
        without_transfer = True
    elif 'without_accuracy' in args.gnn_method:
        without_accuracy = True

    graph = Graph(
        data_dict['node_ID'],
        edge_index_accu_model_to_dataset,
        edge_attr_accu_model_to_dataset,
        edge_index_tran_model_to_dataset,
        edge_attr_tran_model_to_dataset,
        edge_index_dataset_to_dataset,
        edge_attr_dataset_to_dataset,
        without_transfer=without_transfer,
        without_accuracy=without_accuracy,
        # max_model_id = data_dict['max_model_idx']
    )
    data = graph.data

    ### save thd graph
    setting_dict.pop('gnn_method')
    config_name = ','.join([('{0}={1}'.format(k[:14], str(v)[:5])) for k, v in setting_dict.items()])
    if SAVE_GRAPH:
        if 'without_transfer' in args.gnn_method:
            gnn = 'walk_graph_without_transfer'
        elif 'without_accuracy' in args.gnn_method:
            gnn = 'walk_graph_without_accuracy'
        else:
            gnn = 'walk_graph'

        if 'without_transfer' in args.gnn_method:
            config_name = 'without_transfer_' + config_name
        if 'without_accuracy' in args.gnn_method:
            config_name = 'without_accuracy_' + config_name
        _dir = os.path.join('./saved_graph', f"{args.test_dataset.replace('/', '_')}", gnn)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # if not os.path.exists(os.path.join(_dir,config_name+'.pt')):
        torch.save(data, os.path.join(_dir, config_name + '.pt'))

    print(f'------- data: ----------')
    print(data)

    return data


############################
def node2vec_train(args, df_perf, data_dict, evaluation_dict, setting_dict, batch_size, extend=False):
    data = get_graph(args, data_dict, setting_dict)
    avg_degree = torch.mean(degree(data.edge_index[0])).numpy()  # dtype=torch.long
    test_dataset = args.test_dataset.replace('/', '_')

    if not os.path.exists(f'./baselines/{test_dataset}/degree.csv'):
        df_degree = pd.DataFrame(columns=['degree', 'ratio'])
    else:
        df_degree = pd.read_csv(f'./baselines/{test_dataset}/degree.csv', index_col=0)
    df_degree = pd.concat([df_degree, pd.DataFrame({'degree': avg_degree, 'ratio': args.finetune_ratio}, index=[0])], ignore_index=True)
    print(f'\n --- saving degree to ./baselines/{test_dataset}/degree.csv')
    if not os.path.exists(f'./baselines/{test_dataset}'):
        os.makedirs(f'./baselines/{test_dataset}')
    df_degree.to_csv(f'./baselines/{test_dataset}/degree.csv')

    # Training settings
    epochs = 50
    evaluation_dict['epochs'] = epochs

    if 'w2v' in args.gnn_method:
        model = N2V_W2VModel(
            args.gnn_method,
            data.edge_index, data.edge_attr,
            num_walks=10, walk_length=80,
            hidden_channels=args.hidden_channels
        )

        from utils.node2vec_w2v import EdgeLabelDataset
        training_set = EdgeLabelDataset(args, data_dict['finetune_records'], data_dict['unique_model_id'], data_dict['unique_dataset_id'])
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size)

        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)  # ,capturable=True)
        L_fn = nn.MSELoss()

        # Loop over epochs
        epochs = 30
        start = time.time()
        total_loss = 0
        total_examples = 0
        for epoch in range(epochs):
            # Training
            for local_batch, local_labels in tqdm.tqdm(training_generator):
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                pred = model(torch.transpose(local_batch, 0, 1))
                loss = L_fn(pred, local_labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            # if total_loss < 1: break
            print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
            # if (total_loss / total_examples) < 0.1: break
        train_time = time.time() - start
        loss = round(total_loss / total_examples, 4)
        # return model,round(total_loss/total_examples,4), train_time

    else:
        model = N2VModel(
            data.edge_index,
            data.edge_attr,
            data_dict['node_ID'],
            embedding_dim=args.hidden_channels,
            negative_pairs=data_dict['negative_pairs'],
            epochs=epochs,
            extend=extend
        )
        loss, train_time = model.train()

    evaluation_dict['loss'] = loss
    evaluation_dict['train_time'] = train_time

    # save
    dataset_index = data_dict['test_dataset_idx']  # + data_dict['max_model_idx'] + 1
    print('dataset_index', dataset_index)

    # dataset_index = np.repeat(dataset_index,len(data_dict['unique_model_id']))
    dataset_index = np.repeat(dataset_index, len(data_dict['model_idx']))  # len(edge_index_accu_model_to_dataset[0,:]))
    print(f"\nlen(model_index): {len(data_dict['model_idx'])}'")
    print(f"\data_dict['model_idx']:{np.unique(data_dict['model_idx'])}")
    # edge_index = torch.stack([torch.from_numpy(data_dict['model_idx']).to(torch.int64),torch.from_numpy(dataset_index).to(torch.int64)],dim=0)
    edge_index = torch.stack(
        [torch.from_numpy(data_dict['model_idx']).to(torch.int64), torch.from_numpy(dataset_index).to(torch.int64)],
        dim=0
    )
    # data["model", "trained_on", "dataset"].edge_label_index = torch.stack([data['model'].node_id,torch.from_numpy(dataset_index).to(torch.int64)],dim=0)
    # dataset_emb = model.base(dataset_index)
    # print(f'\nedge_index: {edge_index}')
    # print(f"node_ID: {data_dict['node_ID']}")

    ###############
    ### Generate prediction for models on target dataset
    ###############
    from .utils._util import predict_model_for_dataset
    pred, x_embedding_dict = predict_model_for_dataset(model, edge_index, gnn_method='node2vec')

    if 'lr' in args.gnn_method or 'rf' in args.gnn_method:
        from .train_with_linear_regression import RegressionModel
        trainer = RegressionModel(
            args.test_dataset,
            finetune_ratio=args.finetune_ratio,
            method=args.gnn_method,
            hidden_channels=args.hidden_channels,
            dataset_embed_method=args.dataset_embed_method,
            reference_model=args.dataset_reference_model,
            task_type=args.task_type,
        )
        score, results = trainer.train(x_embedding_dict, data_dict)
    else:
        results = {}

    config_name = ','.join([('{0}={1}'.format(k[:14], str(v)[:5])) for k, v in setting_dict.items()])
    results, save_path = save_pred(args, pred, df_perf, data_dict, evaluation_dict, config_name, trainer.directory_experiments)
    return results, save_path


def save_pred(args, pred, df_perf, data_dict, evaluation_dict, config_name, directory_experiments):
    df_model = pd.DataFrame(data_dict['model_idx'], columns=['mappedID'])
    unique_model_id = data_dict['unique_model_id']
    unique_model_id.index = range(len(unique_model_id))
    df_results = df_model.merge(unique_model_id, how='inner', on='mappedID')
    df_results['score'] = pred

    # Save graph embedding distance results
    dir_path = os.path.join(directory_experiments, 'rank_final', f"{args.test_dataset.replace('/', '_')}", args.gnn_method)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    save_path = os.path.join(dir_path, config_name + '.csv')

    df_perf = pd.concat([df_perf, pd.DataFrame(evaluation_dict, index=[0])], ignore_index=True)
    print()
    print('======== save =======')
    # save the 
    df_perf.to_csv(args.path)

    return df_results, save_path
