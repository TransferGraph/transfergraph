# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import tqdm

from .utils._util import save_pred
from .utils.graph import Graph
from .utils.node2vec import N2VModel
from .utils.node2vec_w2v import N2V_W2VModel

logger = logging.getLogger(__name__)

device = 'cpu'
logger.info(f"Device: '{device}'")

SAVE_GRAPH = False


def get_graph(args, data_dict, setting_dict):
    edge_index_accu_model_to_dataset = data_dict['edge_index_accu_model_to_dataset']
    edge_attr_accu_model_to_dataset = data_dict['edge_attr_accu_model_to_dataset']
    edge_index_tran_model_to_dataset = data_dict['edge_index_tran_model_to_dataset']
    edge_attr_tran_model_to_dataset = data_dict['edge_attr_tran_model_to_dataset']

    logger.info(f'\nedge_index_accu_model_to_dataset: {edge_index_accu_model_to_dataset.shape}')
    logger.info(f'\nedge_index_tran_model_to_dataset: {edge_index_tran_model_to_dataset.shape}')

    edge_index_dataset_to_dataset = data_dict['edge_index_dataset_to_dataset']
    edge_attr_dataset_to_dataset = data_dict['edge_attr_dataset_to_dataset']

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
        torch.save(data, os.path.join(_dir, config_name + '.pt'))

    logger.info(f'------- data: ----------')
    logger.info(data)

    return data


############################
def node2vec_train(args, df_perf, data_dict, evaluation_dict, setting_dict, batch_size, extend=False):
    data = get_graph(args, data_dict, setting_dict)

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
            logger.info(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
            # if (total_loss / total_examples) < 0.1: break
        train_time = time.time() - start
        loss = round(total_loss / total_examples, 4)
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
    logger.info('dataset_index', dataset_index)

    # dataset_index = np.repeat(dataset_index,len(data_dict['unique_model_id']))
    dataset_index = np.repeat(dataset_index, len(data_dict['model_idx']))  # len(edge_index_accu_model_to_dataset[0,:]))
    logger.info(f"\nlen(model_index): {len(data_dict['model_idx'])}'")
    logger.info(f"\data_dict['model_idx']:{np.unique(data_dict['model_idx'])}")
    # edge_index = torch.stack([torch.from_numpy(data_dict['model_idx']).to(torch.int64),torch.from_numpy(dataset_index).to(torch.int64)],dim=0)
    edge_index = torch.stack(
        [torch.from_numpy(data_dict['model_idx']).to(torch.int64), torch.from_numpy(dataset_index).to(torch.int64)],
        dim=0
    )

    ###############
    ### Generate prediction for models on target dataset
    ###############
    from .utils._util import predict_model_for_dataset
    pred, x_embedding_dict = predict_model_for_dataset(model, edge_index, gnn_method='node2vec')

    if 'xgb' in args.gnn_method or 'lr' in args.gnn_method or 'rf' in args.gnn_method:
        from .train_with_linear_regression import RegressionModel
        logger.info("Using graph embeddings to learn prediction model")
        trainer = RegressionModel(
            args.test_dataset,
            finetune_ratio=args.finetune_ratio,
            method=args.gnn_method,
            hidden_channels=args.hidden_channels,
            dataset_embed_method=args.dataset_embed_method,
            reference_model=args.dataset_reference_model,
            task_type=args.task_type,
        )
        trainer.train(x_embedding_dict, data_dict)
    else:
        logger.info("Directly using graph prediction to rank models")
        save_pred(args, pred, data_dict)
