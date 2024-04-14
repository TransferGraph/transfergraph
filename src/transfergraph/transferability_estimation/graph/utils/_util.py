import numpy as np
import pandas as pd
import torch

print(torch.__version__)
# We can make use of the `loader.LinkNeighborLoader` from PyG:
from torch_geometric.loader import LinkNeighborLoader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f"Device: '{device}'")

import scipy.spatial.distance as distance
# distance.cosine(h1, h2)

import sys

sys.path.append('/')


def get_variances(*embeddings, normalized=False):
    return [get_variance(e, normalized=normalized) for e in embeddings]


def get_variance(e, normalized=False):
    var = 1. / np.array(e)
    return var


def cosine(e0, e1):
    h1, h2 = get_scaled_hessian(e0, e1)
    # print(h1)
    # print(e0)
    # print(distance.cosine(h1, h2))
    return distance.cosine(h1, h2)


def correlation(e0, e1):
    v1, v2 = get_variances(e0, e1, normalized=False)
    print(len(v1), type(v1), len(v2))
    print(distance.correlation(v1, v2))
    return distance.correlation(v1, v2)


def get_hessian(e, normalized=False):
    hess = np.array(e)
    return hess


def get_hessians(*embeddings, normalized=False):
    return [get_hessian(e, normalized=normalized) for e in embeddings]


def get_scaled_hessian(e0, e1):
    # h0, h1 = get_hessians(e0, e1, normalized=False)
    h0 = np.array(e0)
    h1 = np.array(e1)
    return h0 / (h0 + h1 + 1e-8), h1 / (h0 + h1 + 1e-8)

def get_unique_node(col, name):
    unique_id = col.unique()
    unique_id = pd.DataFrame(
        data={
            name: unique_id,
            'mappedID': pd.RangeIndex(len(unique_id)),
        }
    )
    return unique_id


def merge(df1, df2, col_name):
    mapped_id = pd.merge(df1, df2, left_on=col_name, right_on=col_name, how='left')  # how='left
    mapped_id = torch.from_numpy(mapped_id['mappedID'].values)
    return mapped_id


# Dataloader
def get_dataloader(data, label_type, batch_size=8, is_train=False):
    print('== get dataloader ==')
    s, r, t = label_type
    # Define the validation seed edges:
    edge_label_index = data[s, r, t].edge_label_index
    # print(data)
    print(f'\nmax: {torch.max(edge_label_index)},  min: {torch.min(edge_label_index)}')
    # print(f'\nedge_label_index: {edge_label_index}')
    # print(f'== edge_label_indx.dtype: {edge_label_index.dtype}')
    edge_label = data[s, r, t].edge_label
    # print(f'edge_label:{edge_label}')
    if is_train:
        try:
            dataloader = LinkNeighborLoader(
                data=data,
                num_neighbors=[20, 10],
                neg_sampling_ratio=2.0,
                edge_label_index=((s, r, t), edge_label_index),
                # edge_label=edge_label,
                batch_size=batch_size,
                shuffle=True,
            )
        except Exception as e:
            print(e)
    else:
        try:
            dataloader = LinkNeighborLoader(
                data=data,
                num_neighbors=[20, 10],  # [8,4],
                edge_label_index=((s, r, t), edge_label_index),
                # edge_label=edge_label,
                batch_size=batch_size,
                shuffle=False,
            )
        except Exception as e:
            print(e)
    ## print a batch
    # batch = next(iter(dataloader))
    # print("Batch:", batch)
    # print("batch edge_index", batch["model", "trained_on", "dataset"].edge_index)
    # print("Labels:", batch["model", "trained_on", "dataset"].edge_label)
    # print("Batch indices:", batch["model", "trained_on", "dataset"].edge_label_index)

    # sampled_data = next(iter(val_loader))
    # print(f'\ndataloader: {dataloader}')
    return dataloader


def get_homo_dataloader(data, batch_size=8, is_train=False):
    print('== get dataloader ==')
    # Define the validation seed edges:
    edge_label_index = data.edge_label_index
    # print(f'== edge_label_indx.dtype: {edge_label_index.dtype}')
    edge_label = data.edge_label
    # print(f'edge_label:{np.sort(edge_label)}')
    if 2 in edge_label:
        print('\n === 2 in edge_label')
    if is_train:
        dataloader = LinkNeighborLoader(
            data=data,
            num_neighbors=[20, 10],  # [8,4],#
            neg_sampling_ratio=2.0,
            edge_label_index=edge_label_index,
            # edge_label=edge_label,
            batch_size=batch_size,
            shuffle=True,
        )
    else:
        dataloader = LinkNeighborLoader(
            data=data,
            num_neighbors=[20, 10],  # [8,4], #,
            edge_label_index=edge_label_index,
            # edge_label=edge_label,
            batch_size=batch_size,
            shuffle=False,
        )
    return dataloader


def extract_label_edges(i, j, edge_label_index, max_dataset_id, unique_model_id, unique_dataset_id):
    m0 = np.where(edge_label_index[i] <= max_dataset_id)
    m1 = np.where(edge_label_index[j] > max_dataset_id)
    index = list(set(m0[0]).intersection(set(m1[0])))
    # print('\n',index)
    unique_model_id.index = unique_model_id['mappedID']
    # print(f'\n','max_dataset_id',max_dataset_id)
    model_names = unique_model_id.loc[edge_label_index[j][index], 'model'].values
    # model_names = unique_model_id[unique_model_id['mappedID'].isin(edge_label_index[j][index])]['model'].values
    print(f'\n len(model_names): {len(model_names)}')

    unique_dataset_id.index = unique_dataset_id['mappedID']
    # print('\n',unique_model_id.index)
    dataset_names = unique_dataset_id.loc[edge_label_index[i][index], 'dataset'].values
    # dataset_names = unique_dataset_id[unique_dataset_id['mappedID'].isin(edge_label_index[i][index])]['dataset'].values
    print(f'\n len(dataset_names): {len(dataset_names)}')
    return index, model_names, dataset_names


def set_labels(_data, ft_records, accu_pos_thres, accu_neg_thres, max_dataset_idx, unique_model_id, unique_dataset_id, seed=0):
    indices = []
    models = []
    datasets = []
    edge_label_index = _data.edge_label_index.detach().numpy()

    index, _model_names, _dataset_names = extract_label_edges(0, 1, edge_label_index, max_dataset_idx, unique_model_id, unique_dataset_id)
    indices.extend(index)
    models.extend(_model_names)
    datasets.extend(_dataset_names)

    index, _model_names, _dataset_names = extract_label_edges(1, 0, edge_label_index, max_dataset_idx, unique_model_id, unique_dataset_id)
    indices.extend(index)
    models.extend(_model_names)
    datasets.extend(_dataset_names)

    # print('\n',ft_records.columns)
    df_ = pd.DataFrame({'model': models, 'dataset': datasets})
    # print(df_.head())
    df_ = df_.merge(ft_records, how='inner', on=['model', 'dataset'])
    num_pos_sample = len(indices)

    print(f'\nnum_pos_sample: {num_pos_sample}')

    neg_edge_index = get_ft_edges(
        ft_records, accu_pos_thres, accu_neg_thres,
        unique_model_id, unique_dataset_id,
        num_sample=num_pos_sample, edge_type='negative'
    )

    edge_labels = list(df_['accuracy'].values) + [0] * num_pos_sample
    edge_label_index = np.concatenate((edge_label_index[:, indices], neg_edge_index), axis=1)

    return edge_labels, edge_label_index


def get_ft_edges(args, ft_records, unique_model_id, unique_dataset_id, num_sample=0, seed=0, edge_type='positive'):
    #### negative edges
    if edge_type == 'positive':
        ft_records_neg = ft_records[ft_records['accuracy'] >= args.accu_pos_thres]
    elif edge_type == 'negative':
        ft_records_neg = ft_records[ft_records['accuracy'] <= args.accu_neg_thres]
    if num_sample != 0:
        ft_records_neg = ft_records_neg.sample(n=num_sample, random_state=seed, replace=True)
    _models = ft_records_neg['model']
    _datasets = ft_records_neg['dataset']
    unique_model_id.index = unique_model_id['model']
    unique_dataset_id.index = unique_dataset_id['dataset']
    neg_edge_index = np.asarray([unique_model_id.loc[_models, 'mappedID'], unique_dataset_id.loc[_datasets, 'mappedID']])
    return neg_edge_index


def predict_model_for_dataset(model, data, gnn_method='SageConv'):
    preds = []
    # ground_truths = []
    # map all the model nodes to the dataset node
    # model - dataset
    # print()
    # print('============')
    # print(f'dataset_index: {dataset_index}')
    # print(f"data['model'].num_nodes: {data['model'].num_nodes}")
    pred = []
    # idx = []
    with torch.no_grad():
        data.to(device)
        preds.append(model(data))
        # print(model.x_dict)
        # ground_truths.append(edge_label)
    # pred = torch.cat(preds, dim=0).cpu().numpy()
    pred = preds[0].cpu().numpy()
    # idx = idx.cpu().numpy()

    return pred, model.x_dict
    # ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    # auc = roc_auc_score(ground_truth, pred)
    # print()
    # print(f"Validation AUC: {auc:.4f}")
    # return auc
