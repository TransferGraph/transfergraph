import time

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score
from torch_geometric.data import Data
from tqdm import tqdm

from .utils.CustomRandomLinkSplit import RandomLinkSplit
from .utils._util import *
from .utils.graph import HGraph

logger = logging.getLogger(__name__)

SAVE_GRAPH = False
SAVE_AND_BREAK = False
SAVE_EMBED = True


## Train
def train(model, train_data, graph_type='hetero', label_type=[], gnn_method='lr_node2vec', batch_size=4, epochs=5):
    start = time.time()
    s, r, t = label_type
    # model = model.to(device)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # ,capturable=True)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    if graph_type == 'hetero':
        train_loader = get_dataloader(train_data, label_type, batch_size=batch_size, is_train=True)
    elif graph_type == 'homo':
        train_loader = get_homo_dataloader(train_data, batch_size=batch_size, is_train=True)
    else:
        raise Exception(f"Unexpected graph type {graph_type}")

    if 'e2e' in gnn_method:
        L_fn = nn.MSELoss()
    elif 'xgb' in gnn_method or 'lr' in gnn_method or 'rf' in gnn_method:
        L_fn = F.binary_cross_entropy_with_logits
    else:
        raise Exception(f"Unexpected gnn_method {gnn_method}")

    number_of_training_step = epochs * len(train_loader)
    progress_bar = tqdm(total=number_of_training_step)
    for epoch in range(1, epochs + 1):
        total_loss = total_examples = 0
        for sampled_data in train_loader:
            optimizer.zero_grad()
            pred = model(sampled_data)
            if graph_type == 'hetero':
                ground_truth = sampled_data[s, r, t].edge_label
                loss = F.binary_cross_entropy(pred, ground_truth)
            elif graph_type == 'homo':
                ground_truth = sampled_data.edge_label
                loss = L_fn(pred, ground_truth)
            else:
                raise Exception(f"Unexpected graph type {graph_type}")
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            loss = f"{total_loss / total_examples:.4f}"
            progress_bar.set_postfix({'epoch': epoch, 'loss': loss})
            progress_bar.update(1)
    train_time = time.time() - start
    return model, round(total_loss / total_examples, 4), train_time


def validate(model, val_data, graph_type='hetero', label_type=[], batch_size=8):
    s, r, t = label_type
    preds = []
    ground_truths = []
    if graph_type == 'hetero':
        val_loader = get_dataloader(val_data, label_type, batch_size=batch_size, is_train=True)
    elif graph_type == 'homo':
        val_loader = get_homo_dataloader(val_data, batch_size=batch_size, is_train=True)

    for sampled_data in tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            if graph_type == 'hetero':
                ground_truths.append(sampled_data[s, r, t].edge_label)
            elif graph_type == 'homo':
                ground_truths.append(sampled_data.edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    logger.info(f'len(pred): {len(pred)}')  # , pred: {pred}')
    mask = pred > 0
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred, multi_class='ovr')
    pre = precision_score(ground_truth.astype(int), mask)
    logger.info(f"Validation AUC: {auc:.4f}, pred: {pre:.4f}")
    
    return auc


def custom_replace(data):
    edge_label = data.edge_label.clone()
    edge_label[data.edge_label == 2] = 1
    data.edge_label = edge_label
    return data


def get_edge_label(data):
    logger.info(f'======= homo graph processing ========')
    edge_label_index = data["model", "trained_on", "dataset"].edge_label_index
    edge_label = data["model", "trained_on", "dataset"].edge_label
    return edge_label_index, edge_label


def gnn_train(args, df_perf, data_dict, evaluation_dict, setting_dict, batch_size, custom_negative_sampling=False):
    unique_dataset_id = data_dict['unique_dataset_id']
    data_feat = data_dict['data_feat']
    unique_model_id = data_dict['unique_model_id']
    model_feat = data_dict['model_feat']
    model_idx = data_dict['model_idx']
    edge_index_accu_model_to_dataset = data_dict['edge_index_accu_model_to_dataset']
    edge_attr_accu_model_to_dataset = data_dict['edge_attr_accu_model_to_dataset']
    edge_index_tran_model_to_dataset = data_dict['edge_index_tran_model_to_dataset']
    edge_attr_tran_model_to_dataset = data_dict['edge_attr_tran_model_to_dataset']
    edge_index_dataset_to_dataset = data_dict['edge_index_dataset_to_dataset']
    edge_attr_dataset_to_dataset = data_dict['edge_attr_dataset_to_dataset']
    negative_pairs = data_dict['negative_pairs']
    max_dataset_idx = data_dict['max_dataset_idx']

    ## Construct a heterogeneous graph
    graph = HGraph(
        args.gnn_method,
        max_dataset_idx,
        model_idx,
        unique_model_id,
        model_feat,
        unique_dataset_id,
        data_feat,
        edge_index_accu_model_to_dataset,
        edge_attr_accu_model_to_dataset,
        edge_index_dataset_to_dataset,
        edge_attr_dataset_to_dataset,
        edge_index_tran_model_to_dataset,
        edge_attr_tran_model_to_dataset,
        negative_pairs,
        contain_data_similarity=args.contain_data_similarity,
        contain_dataset_feature=args.contain_dataset_feature,
        contain_model_feature=args.contain_model_feature,
        custom_negative_sampling=custom_negative_sampling
    )

    data = graph.data
    logger.info(f'== Begin splitting dataset ==')
    train_data, val_data, test_data = graph.split()

    ### !!! label_type ["model", "trained_on", "dataset"] or  ["model", "transfer", "dataset"]
    label_type = graph.label_type
    s, r, t = label_type

    unique_values = train_data[s, r, t].edge_label.unique(return_counts=True)
    if len(unique_values[0]) > 2 or 2 in train_data[s, r, t].edge_label:
        logger.info(f'=== 2 in train_data')
        train_data = custom_replace(train_data)
    unique_values = val_data[s, r, t].edge_label.unique(return_counts=True)
    if len(unique_values[0]) > 2 or 2 in val_data[s, r, t].edge_label:
        logger.info(f'=== 2 in val_data')
        val_data = custom_replace(val_data)
    unique_values = test_data[s, r, t].edge_label.unique(return_counts=True)
    if len(unique_values[0]) > 2 or 2 in test_data[s, r, t].edge_label:
        logger.info(f'=== 2 in test_data')
        test_data = custom_replace(test_data)

    ### save thd graph
    setting_dict.pop('gnn_method')
    config_name = ','.join([('{0}={1}'.format(k[:14], str(v)[:5])) for k, v in setting_dict.items()])
    if SAVE_GRAPH:
        if 'without_transfer' in args.gnn_method:
            gnn = 'gnn_graph_without_transfer'
            config_name = 'without_transfer,' + config_name
        elif 'without_accuracy' in args.gnn_method:
            gnn = 'gnn_graph_without_accuracy'
            config_name = 'without_accuracy,' + config_name
        else:
            gnn = 'gnn_graph'
        if 'homo' in args.gnn_method: gnn = 'homo_' + gnn

        _dir = os.path.join('./saved_graph', f'{args.test_dataset}', gnn)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # if not os.path.exists(os.path.join(_dir,config_name+'.pt')):
        torch.save(data, os.path.join(_dir, config_name + '.pt'))
        if SAVE_AND_BREAK:
            return 0, ''

    ## Define GNN
    logger.info(f'== Load {args.gnn_method} ==')
    from .utils.gnn import HeteroModel, HomoModel
    num_dataset_nodes = data["dataset"].num_nodes
    num_model_nodes = data["model"].num_nodes
    evaluation_dict['num_model'] = num_model_nodes
    evaluation_dict['num_dataset'] = num_dataset_nodes
    if args.contain_dataset_feature:
        dim_dataset_feature = data["dataset"].x.shape[1]
    else:
        dim_dataset_feature = 1
    if args.contain_model_feature:
        dim_model_feature = data["model"].x.shape[1]
    else:
        dim_model_feature = 1
    metadata = data.metadata()

    if 'homo' in args.gnn_method:
        logger.info(f'\n== data{data}')
        data_hetero = data
        edge_lists = []
        for edge_name in metadata[1]:
            edge_index = data_hetero[edge_name].edge_index
            edge_lists.append(edge_index)

        node_x_lists = []
        dimension = 64  # 128
        for node_name in metadata[0]:
            try:
                dimension = data_hetero[node_name].x.shape[1]
                logger.info(f'dimension: {dimension}')
                node_x_lists.append(data_hetero[node_name].x)
            except:
                node_x_lists.append(torch.rand((data_hetero[node_name].num_nodes, dimension)))

        data = Data(x=torch.cat(node_x_lists, dim=0), edge_index=torch.cat(edge_lists, dim=1).type(torch.int64))
        data.node_id = torch.from_numpy(np.asarray(data_dict['node_ID']))
        logger.info(f'\n===== homo_data: {data}')

        transform = RandomLinkSplit(  # T.Compose([T.ToUndirected(),
            num_val=0.1,  # TODO
            num_test=0.2,  # TODO
            disjoint_train_ratio=0.3,  # TODO
            neg_sampling_ratio=2.0,  # TODO
            add_negative_train_samples=True,  # TODO
            negative_pairs=negative_pairs,
            is_undirected=True,
            # rev_edge_types = self.data.metadata()[1]
            # rev_edge_types=("dataset", "rev_trained_on", "model"), 
            custom_negative_sampling=custom_negative_sampling
        )
        train_data, val_data, test_data = transform(data)
        logger.info(f'data_homo: {data}')

        ## Validate (sub)graphs
        data.validate()
        train_data.validate()
        val_data.validate()
        test_data.validate()

        logger.info(f'\nnew train_data_homo.edge_label.shape: {train_data.edge_label.shape}')
        logger.info(f'train_data_homo.edge_label.shape: {train_data.edge_label.shape}')
        logger.info(f'train_data_homo.edge_label.unique count: {train_data.edge_label.unique(return_counts=True)}')

        ################# Edge Label ##############
        logger.info(f'\n\ntrain_data_homo.edge_label_index: {train_data.edge_label_index}')

        for _data in [val_data, test_data]:
            unique_values = _data.edge_label.unique(return_counts=True)
            logger.info(f'\n_data_homo.edge_label.unique count: {unique_values}')
            # logger.info(val_data)
            if len(unique_values[0]) > 2 or 2 in _data.edge_label:
                logger.info(f'=== 2 in _data')
                _data = custom_replace(_data)

        ############ Get regression label
        if 'e2e' in args.gnn_method:
            from .utils._util import set_labels
            ft_records = data_dict['finetune_records']
            unique_model_id = data_dict['unique_model_id']
            unique_dataset_id = data_dict['unique_dataset_id']
            max_dataset_idx = data_dict['max_dataset_idx']
            for i, _data in enumerate([train_data, val_data, test_data]):
                edge_labels, edge_label_index = set_labels(
                    _data,
                    ft_records,
                    args.accu_pos_thres,
                    args.accu_neg_thres,
                    max_dataset_idx,
                    unique_model_id,
                    unique_dataset_id,
                    seed=i
                )
                _data.edge_label = torch.Tensor(edge_labels)
                _data.edge_label_index = torch.Tensor(edge_label_index)

        ########## Defining model
        model = HomoModel(
            metadata,
            args.gnn_method,
            hidden_channels=args.hidden_channels,
        )
    else:
        model = HeteroModel(
            metadata,
            num_dataset_nodes,
            num_model_nodes,
            dim_model_feature,
            dim_dataset_feature,
            args.contain_model_feature,
            args.contain_dataset_feature,
            args.embed_model_feature,
            args.embed_dataset_feature,
            args.gnn_method,
            label_type,
            hidden_channels=args.hidden_channels,
            node_types=data.node_types
        )
    logger.info('== Begin training network ==')
    # with torch.no_grad():  # Initialize lazy modules.
    #     out = model(data.x_dict, data.edge_index_dict)
    ## Train model
    if args.gnn_method == 'SAGEConv':
        if args.contain_model_feature:
            epochs = 50
        else:
            epochs = 50
    elif args.gnn_method == 'GATConv':
        epochs = 50
    else:
        epochs = 50

    if 'homo' in args.gnn_method:
        graph_type = 'homo'
    else:
        graph_type = 'hetero'

    model, loss, train_time = train(
        model,
        train_data,
        graph_type,
        label_type,
        gnn_method=args.gnn_method,
        batch_size=batch_size,
        epochs=epochs
    )

    evaluation_dict['epochs'] = epochs
    evaluation_dict['loss'] = loss
    evaluation_dict['train_time'] = train_time
    evaluation_dict['hidden_channels'] = args.hidden_channels

    ## Validate model on validation set
    val_AUC = validate(model, val_data, graph_type, label_type, batch_size=batch_size)
    evaluation_dict['val_AUC'] = val_AUC
    ## Evaluate model on test set
    test_AUC = validate(model, test_data, graph_type, label_type, batch_size=batch_size)
    evaluation_dict['test_AUC'] = test_AUC

    logger.info('========== predict model nodes for the test dataset ========')
    logger.info(unique_dataset_id[unique_dataset_id['dataset'] == args.test_dataset])

    dataset_index = data_dict['test_dataset_idx']
    dataset_index = np.repeat(dataset_index, len(data_dict['model_idx']))

    if graph_type == 'hetero':
        # data['model'].node_id
        data[s, r, t].edge_label_index = torch.stack(
            [
                torch.from_numpy(data_dict['model_idx']).to(torch.int64), torch.from_numpy(dataset_index).to(torch.int64)], dim=0
        )
    elif graph_type == 'homo':
        data.edge_label_index = torch.stack(
            [torch.from_numpy(data_dict['model_idx']).to(torch.int64), torch.from_numpy(dataset_index).to(torch.int64)],
            dim=0
        )

    pred, x_embedding_dict = predict_model_for_dataset(model, data)

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
