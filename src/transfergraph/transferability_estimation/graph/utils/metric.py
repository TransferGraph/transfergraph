import os

import numpy as np
import pandas as pd
from scipy import stats

from transfergraph.transferability_estimation.graph.utils.ranking_metrics import apk

dataset_map = {'oxford_iiit_pet': 'pets',
               'oxford_flowers102': 'flowers'}


def get_records(test_dataset, directory_experiments):
    df_record = pd.read_csv(os.path.join(directory_experiments, 'records.csv'), index_col=0)[
        ['model', 'finetuned_dataset', 'eval_accuracy']]

    df_record = df_record[df_record['finetuned_dataset'] == test_dataset].sort_values(by=['eval_accuracy'], ascending=False)
    df_record_subset = df_record.drop_duplicates(subset=['model', 'finetuned_dataset'], keep='first')
    return df_record_subset


def map_common_models(results, df_record_subset):
    # print(results.head())

    # get model intersection
    model_result = results['model'].unique()
    model_record = df_record_subset['model'].unique()
    model_lists = set.intersection(set(model_result), set(model_record))
    # print(f'\nmodel_lists: {model_lists}')
    # print(df_record_subset)

    results = results[results['model'].isin(model_lists)].drop_duplicates(
        subset=['model'],
        keep='last'
    )  # (subset=['model', 'mappedID'], keep='last')
    ## Rank results by score
    results = results.sort_values(by=['score'], ascending=False)
    results.index = range(len(results))
    # print(f'\n --results: {results}')

    df_record_subset = df_record_subset[df_record_subset['model'].isin(model_lists)]

    return results, df_record_subset


def record_metric(method, test_dataset, setting_dict, results, directory_experiments):
    if test_dataset in dataset_map.keys():
        dataset = dataset_map[test_dataset]
    else:
        dataset = test_dataset

    gnn_method = setting_dict['gnn_method']
    df_record_subset = get_records(dataset, directory_experiments)

    df_results, df_record = map_common_models(results, df_record_subset)

    if method == 'correlation':
        corr = record_correlation_metric(gnn_method, df_results, df_record, test_dataset, directory_experiments)
        return corr, {}
    elif method == 'rank':
        rank, apks = record_rank_metric(gnn_method, df_results, df_record)
        return rank, apks


def record_correlation_metric(gnn_method, df_results, df_record, test_dataset, directory_experiments):
    df_join = df_results.set_index('model').join(df_record.set_index('model'))
    df_join['score'] = df_join['score'].replace([-np.inf, np.nan], -20)
    df_join['score'] = df_join['score'].replace([np.inf], 20)
    df_join['eval_accuracy'] = df_join['eval_accuracy'].replace([np.nan], 0)

    x = df_join['score']
    y = df_join['eval_accuracy']
    try:
        corr = correlation(x, y)
    except Exception as e:
        print(e)
        print(f'x: {x}')
        print(f'y: {y}')

    if 'LogME' in gnn_method:
        if not os.path.exists(os.path.join(directory_experiments, 'rank_final', test_dataset.replace('/', '_'), gnn_method)):
            os.makedirs(os.path.join(directory_experiments, 'rank_final', test_dataset.replace('/', '_'), gnn_method))
        df_results.to_csv(
            os.path.join(directory_experiments, 'rank_final', test_dataset.replace('/', '_'), gnn_method, 'results.csv')
        )

    return corr


def record_rank_metric(gnn_method, df_results, df_record):
    print(f'gnn_method: {gnn_method}')

    df_join = df_results.set_index('model').join(df_record.set_index('model'))
    # df_join.index = range(len(df_join))
    # print(df_join.head())

    df_join_sort = df_join.sort_values('score', ascending=False)
    # print('\n df_join')
    # print(df_join[['score','test_accuracy']].head())
    df_record_sort = df_join.sort_values('test_accuracy', ascending=False)
    # print('\n df_record_sort')
    # print(df_record_sort.head())

    accu_top_dict = {}
    apks_dict = {}
    for topK in [5, 10, 15, 20]:
        # avg accuracy
        df_join_sub = df_join_sort.iloc[:topK]
        # print(f'\n df_join_sub: {df_join_sub.head()}')
        accu_top_dict[topK] = df_join_sub['test_accuracy'].mean()
        # average precision
        gt_rank_index = df_record_sort.iloc[:topK].index.tolist()
        result_rank_index = df_join_sub.index.tolist()
        _apk, running_sum = apk(gt_rank_index, result_rank_index, topK)
        apks_dict[topK] = _apk
    return accu_top_dict, apks_dict


def rank(gnn_method, test_dataset, model_lists, gt_rank_index, result_rank_index):
    # compute metrics
    apks = []
    sums = []
    for topK in [5, 10, 15, 20, 30, 40, 50]:
        _apk, running_sum = apk(gt_rank_index, result_rank_index, topK)
        apks.append(_apk)
        sums.append(running_sum)
        print(f'--- Top {topK} --- apk: {_apk}')
    apks.extend(sums)
    apks.append(np.mean(apks))
    print(f'\n apks: {apks}')
    index_name = [f'apk_@_{topK}' for topK in [5, 10, 15, 20, 30, 40, 50]]
    index_name += [f'sum_{topK}' for topK in [5, 10, 15, 20, 30, 40, 50]]
    index_name += ['map', 'num_model']
    print(f'\n index_name: {index_name}')
    # path = os.path.join(dir,filename)
    metrics = {gnn_method: apks + [len(model_lists)]}

    # set output dir structure and return config list to delete
    # print(setting_dict)
    if 'LogME' not in gnn_method:
        setting_dict, dir = set_output_dir(os.path.join('rank_final', test_dataset.replace('/', '_')), setting_dict)
        setting_dict = {k: v for k, v in setting_dict.items() if v != ''}
        print(setting_dict)
        filename = 'metric,' + ','.join(['{0}={1}'.format(k, v) for k, v in setting_dict.items()]) + '.csv'
        filename = os.path.join(dir, filename)
        print(f'\n--- filename: {filename}')

    if not os.path.exists(filename):
        df = pd.DataFrame(metrics, index=index_name)
    else:
        df = pd.read_csv(filename, index_col=0)
        df[gnn_method] = metrics[gnn_method]
    df.to_csv(filename)


def set_output_dir(base_dir, setting_dict):
    ######### set output folders (tree structure)
    delete_config_list = ['gnn_method', 'contain_dataset_feature', 'contain_model_feature', 'dataset_embed_method']
    dir = base_dir
    for i, key in enumerate(delete_config_list[:]):
        print(f'\n--- {key}: {setting_dict[key]}')
        if i != 0:
            if setting_dict[key]:
                if isinstance(setting_dict[key], str):
                    path = setting_dict[key]
                else:
                    path = key
            else:
                path = f'not_{key}'
            dir = os.path.join(dir, path)
        setting_dict[key] = ''

    if not os.path.exists(dir):
        os.makedirs(dir)
    return setting_dict, dir


def correlation(x, y):  # correlation; rank
    corr = stats.pearsonr(x, y)[0]
    print(f'\ncorr: {corr}, {type(corr)}')
    if np.isnan(corr): corr = 'constant'
    return corr  # .statistics

