import argparse
import time

from transfergraph.config import get_directory_experiments
from transfergraph.transferability_estimation.graph.attributes import *

logger = logging.getLogger(__name__)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def djoin(ldict, req=''):
    return req + ' & '.join(
        [('{0} == "{1}"'.format(k, v)) if isinstance(v, str) else ('{0} == {1}'.format(k, v)) for k, v in ldict.items()]
    )


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info('======================= Begin New Session ==========================')
    directory_experiments = get_directory_experiments(args.task_type)
    directory_log = os.path.join(directory_experiments, 'log')
    args.time_start = time.time()

    if not os.path.exists(directory_log):
        os.makedirs(directory_log)

    path = os.path.join(directory_log, f'performance_score.csv')
    args.path = path

    if os.path.exists(path):
        df_perf = pd.read_csv(path, index_col=0)
    else:
        df_perf = pd.DataFrame(
            columns=[
                'contain_data_similarity',
                'dataset_edge_distance_method',

                'contain_dataset_feature',
                'embed_dataset_feature',
                'dataset_embed_method',
                'dataset_reference_model',

                'contain_model_feature',
                'embed_model_feature',
                'model_embed_method',
                'complete_model_features',
                'model_dataset_edge_method',

                'distance_thres',
                'top_pos_K',
                'top_neg_K',
                'accu_pos_thres',
                'accu_neg_thres',

                'gnn_method',
                'accuracy_thres',
                'finetune_ratio',
                'hidden_channels',
                'num_model',
                'num_dataset',
                'test_dataset',
                'train_time',
                'loss',
                'val_AUC',
                'test_AUC',
                'time_total',
            ]
        )
    setting_dict = {
        # 'contain_data_similarity':args.contain_data_similarity,
        'contain_dataset_feature': args.contain_dataset_feature,
        # 'embed_dataset_feature':args.embed_dataset_feature,
        'dataset_embed_method': args.dataset_embed_method.value,
        'contain_model_feature': args.contain_model_feature,
        'dataset_reference_model': args.dataset_reference_model,  # model_embed_method
        # 'embed_model_feature':args.embed_model_feature,
        'dataset_edge_distance_method': args.dataset_distance_method,
        'model_dataset_edge_method': args.model_dataset_edge_attribute,
        'gnn_method': args.gnn_method,
        # 'accuracy_thres':args.accuracy_thres,
        # 'complete_model_features':args.complete_model_features,
        'hidden_channels': args.hidden_channels,
        'top_pos_K': args.top_pos_K,
        'top_neg_K': args.top_neg_K,
        'accu_pos_thres': args.accu_pos_thres,
        'accu_neg_thres': args.accu_neg_thres,
        'distance_thres': args.distance_thres,
        'finetune_ratio': args.finetune_ratio,
    }
    if args.dataset_reference_model != 'resnet50':
        setting_dict['dataset_reference_model'] = args.dataset_reference_model

    logger.info('======= evaluation_dict ==========')
    evaluation_dict = setting_dict.copy()
    evaluation_dict['test_dataset'] = args.test_dataset

    for k, v in evaluation_dict.items():
        logger.info(f'{k}: {v.__str__()}')
    logger.info(f'gnn_method: {args.gnn_method}\n')

    ## Check executed
    query = ' & '.join(list(map(djoin, [evaluation_dict])))

    ## skip running because the performance exist
    if not df_perf.empty:
        df_tmp = df_perf.query(query)
        if not df_tmp.empty:
            logger.info('Already executed, skipping.')
            pass

    if args.gnn_method == 'lr':
        graph_attributes = GraphAttributes(args)
    elif args.dataset_embed_method == DatasetEmbeddingMethod.DOMAIN_SIMILARITY:
        graph_attributes = GraphAttributesWithDomainSimilarity(args)
    elif args.dataset_embed_method == DatasetEmbeddingMethod.TASK2VEC:
        graph_attributes = GraphAttributesWithTask2Vec(args)
    else:
        graph_attributes = GraphAttributes(args)

    data_dict = {
        'unique_dataset_id': graph_attributes.unique_dataset_id,
        'data_feat': graph_attributes.data_features,
        'unique_model_id': graph_attributes.unique_model_id,
        'model_feat': graph_attributes.model_features,
        'edge_index_accu_model_to_dataset': graph_attributes.edge_index_accu_model_to_dataset,
        'edge_attr_accu_model_to_dataset': graph_attributes.edge_attr_accu_model_to_dataset,
        'edge_index_tran_model_to_dataset': graph_attributes.edge_index_tran_model_to_dataset,
        'edge_attr_tran_model_to_dataset': graph_attributes.edge_attr_tran_model_to_dataset,
        'edge_index_dataset_to_dataset': graph_attributes.edge_index_dataset_to_dataset,
        'edge_attr_dataset_to_dataset': graph_attributes.edge_attr_dataset_to_dataset,
        'negative_pairs': graph_attributes.negative_pairs,
        'test_dataset_idx': graph_attributes.test_dataset_idx,
        'model_idx': graph_attributes.model_idx,
        'node_ID': graph_attributes.node_ID,
        'max_dataset_idx': graph_attributes.max_dataset_idx,
        'finetune_records': graph_attributes.finetune_records,
        'model_config': graph_attributes.model_config
    }

    batch_size = 16

    if 'node2vec+' in args.gnn_method:
        from transfergraph.transferability_estimation.graph.train_with_node2vec import node2vec_train
        node2vec_train(args, df_perf, data_dict, evaluation_dict, setting_dict, batch_size, extend=True)
    elif 'node2vec' in args.gnn_method:
        from transfergraph.transferability_estimation.graph.train_with_node2vec import node2vec_train
        node2vec_train(args, df_perf, data_dict, evaluation_dict, setting_dict, batch_size)
    elif 'Conv' in args.gnn_method:
        from transfergraph.transferability_estimation.graph.train_with_GNN import gnn_train
        gnn_train(args, df_perf, data_dict, evaluation_dict, setting_dict, batch_size, custom_negative_sampling=True)
    elif args.gnn_method == '""':
        from transfergraph.transferability_estimation.graph.utils.basic import get_basic_features
        get_basic_features(args.test_dataset, data_dict, setting_dict)
    elif 'lr' in args.gnn_method:
        from transfergraph.transferability_estimation.graph.train_with_linear_regression import lr_train
        lr_train(args, graph_attributes)
    else:
        raise Exception(f"Unexpected gnn_method: {args.gnn_method}")

    setting_dict['gnn_method'] = args.gnn_method

    unique_model_id = data_dict['unique_model_id']
    unique_model_id.index = range(len(unique_model_id))

    evaluation_dict['time_total'] = time.time() - args.time_start
    df_perf = pd.concat([df_perf, pd.DataFrame(evaluation_dict, index=[0])], ignore_index=True)
    logger.info(f'======== save log: {args.path} =======')
    # save the
    df_perf.to_csv(args.path)


if __name__ == '__main__':

    '''
    Configurations
    '''
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--contain_dataset_feature', default='True', type=str, help="Whether to apply selectivity on a model level")
    parser.add_argument('--contain_data_similarity', default='True', type=str, help="Whether to apply selectivity on a model level")
    parser.add_argument('--contain_model_feature', default='False', type=str, help="contain_model_feature")
    parser.add_argument('--embed_dataset_feature', default='True', type=str, help='embed_dataset_feature')
    parser.add_argument('--embed_model_feature', default='True', type=str, help="embed_model_feature")
    parser.add_argument('--complete_model_features', default='True', type=str)
    parser.add_argument('--dataset_reference_model', default='microsoft_resnet-50', type=str)
    parser.add_argument('--task_type', required=True, type=TaskType)
    parser.add_argument('--gnn_method', default='SAGEConv', type=str, help='contain_model_feature')
    parser.add_argument('--accuracy_thres', default=0.7, type=float, help='accuracy_thres')
    parser.add_argument('--finetune_ratio', default=1.0, type=float, help='finetune_ratio')
    parser.add_argument('--test_dataset', default='dmlab', type=str, help='remove all the edges from the dataset')
    parser.add_argument('--hidden_channels', default=128, type=int, help='hidden channels')  # 128
    parser.add_argument('--top_pos_K', default=0.5, type=float, help='hidden channels')
    parser.add_argument('--top_neg_K', default=0.5, type=float, help='hidden channels')
    parser.add_argument('--accu_pos_thres', default=0.6, type=float, help='hidden channels')
    parser.add_argument('--accu_neg_thres', default=0.2, type=float, help='hidden channels')
    parser.add_argument('--distance_thres', default=-1, type=float)
    parser.add_argument('--dataset_embed_method', default=DatasetEmbeddingMethod.DOMAIN_SIMILARITY, type=DatasetEmbeddingMethod)  # task2vec
    parser.add_argument('--dataset_distance_method', default='euclidean', type=str)  # correlation
    parser.add_argument('--model_dataset_edge_attribute', default='LogMe', type=str)  # correlation
    parser.add_argument("--peft_method", required=False, type=str, help="PEFT method to use.", choices=['lora'])

    args = parser.parse_args()

    if args.task_type == TaskType.SEQUENCE_CLASSIFICATION:
        args.dataset_reference_model = 'EleutherAI_gpt-neo-125m'
    elif args.task_type == TaskType.IMAGE_CLASSIFICATION:
        if args.dataset_embed_method == DatasetEmbeddingMethod.DOMAIN_SIMILARITY:
            args.dataset_reference_model = 'google_vit_base_patch16_224'
        elif args.dataset_embed_method == DatasetEmbeddingMethod.TASK2VEC:
            args.dataset_reference_model = 'resnet34'
        else:
            raise Exception(f'Unexpected embedding method {args.dataset_embed_method}')
    else:
        raise Exception(f'Unexpected task type {args.task_type}')

    args.contain_data_similarity = str2bool(args.contain_data_similarity)
    args.contain_model_feature = str2bool(args.contain_model_feature)
    args.contain_dataset_feature = str2bool(args.contain_dataset_feature)

    args.embed_model_feature = str2bool(args.embed_model_feature)
    args.embed_dataset_feature = str2bool(args.embed_dataset_feature)
    args.complete_model_features = str2bool(args.complete_model_features)

    main(args)
