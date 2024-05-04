import logging
import os
import time
from glob import glob

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

from transfergraph.config import get_root_path_string
from transfergraph.dataset.embed_utils import DatasetEmbeddingMethod
from transfergraph.dataset.task import TaskType

SAVE_FEATURE = True

logger = logging.getLogger(__name__)

class RegressionModel():
    def __init__(
            self,
            test_dataset,
            finetune_ratio=1.0,
            method='',
            hidden_channels=128,
            dataset_embed_method=DatasetEmbeddingMethod.DOMAIN_SIMILARITY,
            reference_model='resnet50',
            task_type=TaskType.SEQUENCE_CLASSIFICATION,
            peft_method=None,
    ):
        if reference_model == 'resnet34' or reference_model == 'google_vit_base_patch16_224' or reference_model == 'microsoft/resnet-50':
            base_dataset = 'imagenet'
        elif reference_model == 'Ahmed9275_Vit-Cifar100':
            base_dataset = 'cifar100'
        elif reference_model == 'johnnydevriese_vit_beans':
            base_dataset = 'beans'
        elif reference_model == 'EleutherAI_gpt-neo-125m':
            base_dataset = 'gpt'
        else:
            raise Exception(f'Unexpected reference model {reference_model}')

        if dataset_embed_method == DatasetEmbeddingMethod.TASK2VEC:
            dataset_correlation_file_name = f'corr_task2vec_{reference_model}_{base_dataset}.csv'
        elif dataset_embed_method == DatasetEmbeddingMethod.DOMAIN_SIMILARITY:
            dataset_correlation_file_name = f'corr_domain_similarity_{reference_model}_{base_dataset}.csv'
        else:
            raise Exception(f"Unexpected embedding method {dataset_embed_method}")

        self.selected_columns = ['architectures', 'accuracy',
                                 'model_type', 'number_of_parameters',
                                 'train_runtime',
                                 'dataset',
                                 'finetune_dataset', 'size', 'number_of_classes',
                                 'eval_accuracy'
                                 ]  # 'input_shape', 'elapsed_time', '#labels',
        dataset_map = {'oxford_iiit_pet': 'pets',
                       'oxford_flowers102': 'flowers'}
        if test_dataset in dataset_map.keys():
            self.test_dataset = dataset_map[test_dataset]
        else:
            self.test_dataset = test_dataset
        self.finetune_ratio = finetune_ratio
        self.method = method
        self.peft_method = peft_method
        self.dataset_correlation_file_name = dataset_correlation_file_name
        self.hidden_channels = hidden_channels
        self.task_type = task_type
        self.directory_experiments = os.path.join(get_root_path_string(), 'resources/experiments', self.task_type.value)

        if 'task2vec' in dataset_correlation_file_name:
            self.embed_addition = '_task2vec'
        else:
            self.embed_addition = ''

        if 'without_accuracy' in method:
            self.y_label = 'score'
        else:
            self.y_label = 'eval_accuracy'
        pass

    def feature_preprocess(self, embedding_dict={}, data_dict={}):
        df_model_config = pd.read_csv(os.path.join(self.directory_experiments, 'model_config_dataset.csv'))
        df_dataset_config = pd.read_csv(os.path.join(self.directory_experiments, 'target_dataset_features.csv'))

        df_finetune = pd.read_csv(os.path.join(self.directory_experiments, 'records.csv'), index_col=0)

        # Filter by fine-tuning method
        if 'peft_method' in df_finetune.columns:
            if self.peft_method is not None:
                df_finetune = df_finetune[df_finetune['peft_method'] == self.peft_method]
            else:
                df_finetune = df_finetune[pd.isna(df_finetune['peft_method'])]

            df_finetune = df_finetune.rename(columns={'finetuned_dataset': 'finetune_dataset'})

        # joining finetune records with model config (model)
        df_model = df_finetune.merge(df_model_config, how='inner', on='model')

        # joining finetune records with dataset config (dataset metadata)
        df_feature = df_model.merge(df_dataset_config, how='inner', on='finetune_dataset')

        if self.task_type == TaskType.SEQUENCE_CLASSIFICATION:
            df_feature = df_feature.dropna(subset=['eval_accuracy'])
            df_feature = fill_null_value(df_feature, columns=['size', 'number_of_classes'])

            if 'input_shape' in df_feature.columns:
                df_feature = df_feature.drop(columns=['input_shape'])
        elif self.task_type == TaskType.IMAGE_CLASSIFICATION:
            df_feature = fill_null_value(df_feature, columns=['eval_accuracy', 'number_of_parameters'])
        else:
            raise Exception(f'Unexpected task type {self.task_type.value}')

        df_feature = df_feature.dropna(subset=['model_type'])

        if 'normalize' in self.method:
            df_feature['eval_accuracy'] = df_feature[['finetune_dataset', 'eval_accuracy']].groupby('finetune_dataset').transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )

        if ('Conv' in self.method or 'node2vec' in self.method):
            if embedding_dict == {}:
                method = self.method
                if 'normalize' in method:
                    method = '_'.join(method.split('_')[:-1])
                if 'basic' in method or 'all' in method:
                    method = '_'.join(method.split('_')[:-1])
                # logger.info(method)
                if method[-8:] == 'distance':
                    method = '_'.join(method.split('_')[:-2])

                path_dataset_features = os.path.join(self.directory_experiments, f"features_final/{self.test_dataset.replace('/', '_')}")
                if not os.path.exists(path_dataset_features):
                    os.makedirs(path_dataset_features)
                if self.finetune_ratio == 1:
                    file = os.path.join(
                        f"{path_dataset_features}/features_{method.replace('rf_', 'lr_').replace('svm_', 'lr_').replace('xgb_', 'lr_')}_{self.hidden_channels}.csv"
                    )
                    logger.info(f'\nfile: {file}')
                    # file = os.path.join(f"../../features/{self.test_dataset}/features_{method}.csv")
                    # if not os.path.exists(file):
                    # file = os.path.join(f"../../features/{self.test_dataset}/features_{method.replace('lr','rf').replace('svm','rf')}.csv")
                else:
                    # file = os.path.join(f"../../features/{self.test_dataset}/features_{method}_{self.finetune_ratio}.csv")
                    # if not os.path.exists(file):
                    file = os.path.join(
                        f"{path_dataset_features}/features_{method.replace('rf_', 'lr_').replace('svm_', 'lr_').replace('xgb_', 'lr_')}_{self.hidden_channels}_{self.finetune_ratio}.csv"
                    )
                df_feature = pd.read_csv(file)  # index_col=0
                logger.info(f'\n1st df_feature.shape: {df_feature.shape}')
                # assert df_feature.shape[0] > 600

                if 'model.1' in df_feature.columns:
                    # df_feature = df_feature.rename(columns={'model.1':'model'})
                    df_feature = df_feature.drop(columns=['model.1'])
                # logger.info(df_feature.columns)

                df_feature.index = range(len(df_feature))

                df_feature = df_feature.dropna(subset=['model_type'])
                # if 'normalize' in self.method:
                #     df_feature['test_accuracy'] = df_feature[['finetune_dataset','test_accuracy']].groupby('finetune_dataset').transform(lambda x: (x - x.min()) / (x.max()-x.min()))

                # if 'basic' in self.method or 'all' in self.method or 'without_accuracy' in self.method:
                self.selected_columns += [col for col in df_feature.columns if 'm_f' in col or 'd_f' in col]
                # logger.info(list(df_feature.columns))

            if embedding_dict != {}:
                # assert data_dict == {}
                unique_model_id = data_dict['unique_model_id']
                unique_model_id.index = range(len(unique_model_id))
                df_feature = df_feature.merge(unique_model_id, how='inner', on='model')
                df_data_id = data_dict['unique_dataset_id'].rename({'dataset': 'finetune_dataset'}, axis='columns')
                df_feature = df_feature.merge(df_data_id, how='inner', on='finetune_dataset')

                logger.info(f'\n embedding_dict != null, len(df_feature):{len(df_feature)}')
                # assert len(df_feature) > 600

                # logger.info(f'\n df_feature.clumns: \n{df_feature.columns}')
                # logger.info(f'\ndf_feature: {df_feature.head()}')
                ### Capture the embeddings
                df = pd.DataFrame()
                model_emb = []
                dataset_emb = []
                for i, row in df_feature.iterrows():
                    model_id = row['mappedID_x']
                    dataset_id = row['mappedID_y']
                    if 'node2vec' in self.method:
                        model_emb.append(embedding_dict[model_id].detach().numpy())
                        dataset_emb.append(embedding_dict[dataset_id].detach().numpy())
                    else:
                        # logger.info(f'\nembedding_dict:{embedding_dict}')

                        model_emb.append(embedding_dict[model_id].numpy())
                        dataset_emb.append(embedding_dict[dataset_id].numpy())

                ## if 'all' in method, taking all the features into account
                if not 'all' in self.method:
                    self.selected_columns = ['finetune_dataset', self.y_label]

                columns = ['m_f' + str(i) for i in range(len(embedding_dict[model_id]))]
                df_ = pd.DataFrame(model_emb, columns=columns)
                df_feature = pd.concat([df_feature, df_], axis=1)
                self.selected_columns += columns

                columns = ['d_f' + str(i) for i in range(len(embedding_dict[dataset_id]))]
                df_ = pd.DataFrame(dataset_emb, columns=columns)
                df_feature = pd.concat([df_feature, df_], axis=1)
                self.selected_columns += columns

                df_feature = df_feature.dropna(subset=['m_f0', 'd_f0'])

        if 'logme' in self.method or 'without_accuracy' in self.method:  # or 'all' in self.method
            if 'score' not in df_feature.columns:
                model_list = df_feature['model'].unique()
                df_feature.index = range(len(df_feature))
                df_dataset_list = []
                for dataset in df_feature['finetune_dataset'].unique():
                    logme_path = os.path.join(self.directory_experiments, 'transferability_score_records.csv')
                    logme = pd.read_csv(logme_path)
                    df_logme = logme[logme['model'] != 'time']
                    df_logme = df_logme[df_logme['target_dataset'] == dataset]

                    # identify common models
                    df_logme = df_logme[df_logme['model'].isin(model_list)]
                    df_logme = df_logme.dropna(subset=['score'])
                    # normalize
                    df_logme['score'].replace([-np.inf, np.nan], -50, inplace=True)

                    score = df_logme['score']  # .astype('float64')
                    normalized_pred = (score - score.mean()) / score.std()
                    df_logme['score'] = normalized_pred

                    # df_feature.loc[df_logme['model'].values,'score'] = normalized_pred
                    df_dataset_list.append(
                        df_feature[df_feature['finetune_dataset'] == dataset].merge(df_logme, how='inner', on=['model'])
                    )  # ,'finetune_dataset']))
                df_feature = pd.concat(df_dataset_list)
            if 'score' not in self.selected_columns:
                self.selected_columns += ['score']

        if 'data_distance' in self.method or 'all' in self.method:
            corr_path = os.path.join(self.directory_experiments, self.dataset_correlation_file_name)
            df_corr = pd.read_csv(corr_path, index_col=0)

            columns = df_corr.columns
            df_corr['finetune_dataset'] = df_corr.index
            maps = {'oxford_iiit_pet': 'pets', 'svhn_cropped': 'svhn', 'oxford_flowers102': 'flowers',
                    'smallnorb': 'smallnorb_label_elevation'}
            for k, v in maps.items():
                df_corr = df_corr.replace(k, v)
            df_corr = pd.melt(df_corr, id_vars=['finetune_dataset'], var_name='dataset', value_vars=columns, value_name='distance')
            df_corr['distance'] = df_corr[['finetune_dataset', 'distance']].groupby('finetune_dataset').transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )

            if 'distance' in df_feature.columns:
                df_feature = df_feature.drop(columns=['distance'])

            df_feature = df_feature.merge(df_corr, how='outer', on=['finetune_dataset', 'dataset'])

            #### fill missing value with minimum value
            min_value = df_feature['distance'].min()
            df_feature['distance'].fillna(value=min_value, inplace=True)

            logger.info(f'\n2nd df_feature.shape: {df_feature.shape}')
            assert df_feature.shape[0] > 600

            self.selected_columns += ['distance']

        if 'feature' in self.method:
            logger.info(f'feature in self.method')
            records = {'finetune_dataset': [], 'model': [], 'feature': []}
            emb_files = glob(os.path.join('../../../../../', 'model_embed', 'embeddings') + '/*')
            for file in emb_files:
                components = file.split(',')
                array = np.reshape(np.load(file), (1, -1))
                array = normalize(array, norm='max').ravel()
                model = components[2].split('_')
                model_name = model[0] + '/' + '_'.join(model[1:])
                model_name = model_name.replace('.npy', '')

                records['finetune_dataset'].append(components[1])
                records['model'].append(model_name)
                records['feature'].append(array)
            columns = ['c' + str(i) for i in range(array.size)]
            self.selected_columns += columns
            df = pd.DataFrame.from_dict({k: v for k, v in records.items() if k != 'feature'})

            # Normalize features
            scaler = MinMaxScaler()
            features = records['feature']
            scaler.fit(features)
            featues = scaler.transform(features)
            df[columns] = featues
            df_feature = df_feature.merge(df, how='inner', on=['finetune_dataset', 'model'])

        df_feature.index = df_feature['model']
        if SAVE_FEATURE:
            logger.info(f'\n-----save feature')
            _dir = os.path.join(self.directory_experiments, 'features_final', self.test_dataset.replace('/', '_'))
            if not os.path.exists(_dir):
                os.makedirs(_dir)
            if self.finetune_ratio < 1:
                file = os.path.join(_dir, f'features_{self.method}{self.embed_addition}_{self.hidden_channels}_{self.finetune_ratio}.csv')
            else:
                file = os.path.join(_dir, f'features_{self.method}{self.embed_addition}_{self.hidden_channels}.csv')

            logger.info(f'3rd df_feature.shape: {df_feature.shape}')
            df_feature.to_csv(file)

        df_feature = df_feature[self.selected_columns]
        if 'score' in df_feature.columns:
            df_feature = df_feature.dropna(subset=['score'])
        if 'm_f0' in df_feature.columns:
            df_feature = df_feature.dropna(subset=['m_f0', 'd_f0'])
        logger.info(f'\n df_feature.len: {len(df_feature)}')

        nan_columns = df_feature.columns[df_feature.isna().any()].tolist()
        logger.info(f'nan_columns: {nan_columns}')
        df_feature = fill_null_value(df_feature, columns=nan_columns)

        self.df_feature = df_feature

    def split(self):
        df_train = self.df_feature[self.df_feature['finetune_dataset'] != self.test_dataset]

        ##### Sampling the finetune records given the ratio
        if self.finetune_ratio != 1:
            df_train = df_train.sample(frac=self.finetune_ratio, random_state=1)

        df_test = self.df_feature[self.df_feature['finetune_dataset'] == self.test_dataset]

        self.selected_columns.remove('finetune_dataset')

        categorical_columns = ['architectures', 'model_type', 'finetune_dataset', 'dataset']
        if set(categorical_columns) < set(list(df_train.columns)):
            df_train = encode(df_train, categorical_columns)
            df_test = encode(df_test, categorical_columns)

        return df_train, df_test

    def train(self, embedding_dict={}, data_dict={}):
        self.feature_preprocess(embedding_dict, data_dict)

        df_train, df_test = self.split()

        feature_columns = [col for col in self.selected_columns if col != self.y_label]
        X_train = df_train[feature_columns].values
        X_test = df_test[feature_columns].values
        y_train = df_train[self.y_label].values
        y_test = df_test[self.y_label].values

        logger.info(f'\n --- dataset: {self.test_dataset}')
        logger.info(f'\n --- shape of X_train: {X_train.shape}, X_test.shape: {X_test.shape}')
        assert X_train.shape[0] > 200

        if 'lr' in self.method:
            model = LinearRegression()
        elif 'rf' in self.method:
            model = RandomForestRegressor(
                n_estimators=100,
                max_features='sqrt',
                max_depth=5,
                random_state=18
            )
        elif 'svm' in self.method:
            model = svm.SVR()
        elif 'xgb' in self.method:
            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=5,
                eta=0.1,
                subsample=1,
                colsample_bytree=0.8,
                random_state=18
            )

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        y_pred = model.predict(X_test)

        df_results = pd.DataFrame({'model': df_test.index, 'score': y_pred})

        if embedding_dict == {}:
            dir_path = os.path.join(self.directory_experiments, 'rank_final', f"{self.test_dataset.replace('/', '_')}", self.method)
        else:
            dir_path = os.path.join(self.directory_experiments, 'rank_final', f"{self.test_dataset.replace('/', '_')}", self.method)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file = f'results{self.embed_addition}_{self.finetune_ratio}_{self.hidden_channels}_0.csv'

        df_results.to_csv(os.path.join(dir_path, file))

        return score, df_results


def fill_null_value(df, columns, method='mean', value=0):
    for col in columns:
        dtype = pd.api.types.infer_dtype(df[col])
        if dtype == 'floating' or dtype == 'integer':
            if method == 'mean':
                df[col] = df[col].fillna((df[col].mean()))
            else:
                df[col] = df[col].fillna((value))
        else:
            df[col].fillna((''), inplace=True)

    return df


def encode(df, columns):
    encoder = LabelEncoder()
    for col in columns:
        if df[col].dtypes != 'str':
            df[col] = df[col].replace(0, '')
        _df = pd.DataFrame(encoder.fit_transform(df[col]), columns=[col])
        _df.index = df.index
        df = df.drop(columns=[col])
        df = pd.concat([df, _df], axis=1)
    return df


if __name__ == '__main__':
    task_type = 'sequence_classification'
    path = os.path.join(get_root_path_string(), 'resources/experiments', task_type, 'log')
    performance_file = os.path.join(path, f'performance_rf_score.csv')
    logger.info(f'====== path: {path} ======')
    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(performance_file):
        df_perf = pd.read_csv(performance_file, index_col=0)
    else:
        df_perf = pd.DataFrame(
            columns=[
                'method',
                'finetune_dataset',
                'train_time',
                'score'
            ]
        )

    datasets = [
        # 'tweet_eval/sentiment',
        # 'tweet_eval/emotion',
        # 'rotten_tomatoes',
        # 'glue/cola',
        # 'tweet_eval/irony',
        # 'tweet_eval/hate',
        # 'tweet_eval/offensive',
        # 'ag_news',
        'glue/sst2',
        # 'smallnorb_label_elevation',
        # 'stanfordcars',
        # 'cifar100',
        # 'caltech101',
        # 'dtd',
        # 'oxford_flowers102',
        # 'oxford_iiit_pet',

        #  'diabetic_retinopathy_detection',
        #  'kitti',
        #  'svhn',
        #  'smallnorb_label_azimuth',
        #  'eurosat',
        #  'pets','flowers',
    ]

    # ratios = 0.3,0.5,0.7: lr_node2vec+ lr_homo_SAGEConv
    for method in [
        'lr_normalize', 'rf_normalize', 'svm_normalize',
        'svm_node2vec_normalize',
        'svm_node2vec+_normalize',
        'svm_data_distance_normalize',
        'svm_node2vec+_normalize',
        'svm_logme_data_distance_normalize',
        # 'svm_node2vec_basic_normalize',
        'svm_node2vec_all_normalize',
        # 'svm_node2vec+_basic_normalize',
        'svm_node2vec+_all_normalize',
        # 'svm_node2vec_without_accuracy_all_normalize',
        # 'svm_node2vec+_without_accuracy_all_normalize',
        # 'svm_node2vec_without_accuracy_basic_normalize',
        # 'svm_node2vec+_without_accuracy_basic_normalize',
        # 'svm_node2vec_without_accuracy_normalize',
        # 'svm_node2vec+_without_accuracy_normalize',
        # 'svm_homo_SAGEConv_trained_on_transfer',
        # 'svm_homo_SAGEConv_trained_on_transfer_basic',
        'xgb_normalize',
        'xgb_data_distance_normalize',
        'xgb_logme_data_distance_normalize',
        # 'xgb_node2vec_basic_normalize',
        # 'xgb_node2vec+_basic_normalize',
        # 'xgb_node2vec_without_accuracy_all_normalize',
        # 'xgb_node2vec+_without_accuracy_all_normalize',
        # 'xgb_node2vec_without_accuracy_basic_normalize',
        # 'xgb_node2vec+_without_accuracy_basic_normalize',
        # 'xgb_node2vec_without_accuracy_normalize',
        # 'xgb_node2vec+_without_accuracy_normalize',
        'xgb_node2vec_all_normalize',
        'xgb_node2vec+_all_normalize',
        'xgb_node2vec_normalize',
        'xgb_node2vec+_normalize',
        # 'xgb_node2vec_data_distance_normalize',
        # 'xgb_node2vec+_data_distance_normalize',

        'rf_data_distance_normalize',
        'rf_logme_data_distance_normalize',
        'rf_node2vec_normalize',
        'rf_node2vec+_normalize',
        # 'rf_node2vec_basic_normalize',
        # 'rf_node2vec+_basic_normalize',
        'rf_node2vec_all_normalize',
        'rf_node2vec+_all_normalize',
        # 'rf_node2vec_data_distance_normalize',
        # 'rf_node2vec+_data_distance_normalize',
        # 'rf_node2vec_without_accuracy_all_normalize',
        # 'rf_node2vec+_without_accuracy_all_normalize',
        # 'rf_node2vec_without_accuracy_basic_normalize',
        # 'rf_node2vec+_without_accuracy_basic_normalize',
        # 'rf_node2vec_without_accuracy_normalize',
        # 'rf_node2vec+_without_accuracy_normalize',

        'lr_data_distance_normalize',
        'lr_logme_data_distance_normalize',
        'lr_node2vec_normalize',
        'lr_node2vec+_normalize',
        'lr_node2vec_all_normalize',
        'lr_node2vec+_all_normalize',
        # 'lr_node2vec_basic_normalize',
        # 'lr_node2vec+_basic_normalize',
        # 'lr_node2vec_data_distance_normalize',
        # 'lr_node2vec+_data_distance_normalize',
        # 'lr_node2vec_without_accuracy_all_normalize',
        # 'lr_node2vec+_without_accuracy_all_normalize',
        # 'lr_node2vec_without_accuracy_basic_normalize',
        # 'lr_node2vec+_without_accuracy_basic_normalize',
        # 'lr_node2vec_without_accuracy_normalize',
        # 'lr_node2vec+_without_accuracy_normalize',

        'lr_homo_SAGEConv_normalize',
        'lr_homoGATConv_normalize',
        'rf_homo_SAGEConv_normalize',
        'rf_homoGATConv_normalize',
        'xgb_homo_SAGEConv_normalize',
        'xgb_homoGATConv_normalize',

        # 'lr_homo_SAGEConv_basic_normalize',
        # 'lr_homoGATConv_basic_normalize',
        # 'rf_homo_SAGEConv_basic_normalize',
        # 'rf_homoGATConv_basic_normalize',
        # 'xgb_homo_SAGEConv_basic_normalize',
        # 'xgb_homoGATConv_basic_normalize',

        # 'lr_homo_SAGEConv_without_accuracy_basic_normalize',
        # 'lr_homoGATConv_without_accuracy_basic_normalize',
        # 'rf_homo_SAGEConv_without_accuracy_basic_normalize',
        # 'rf_homoGATConv_without_accuracy_basic_normalize',
        # 'xgb_homo_SAGEConv_without_accuracy_basic_normalize',
        # 'xgb_homoGATConv_without_accuracy_basic_normalize',

        'rf_homoGCNConv_normalize',
        'xgb_homoGCNConv_normalize',
        'lr_homoGCNConv_normalize',
        # 'xgb_homoGCNConv_basic_normalize',
        # 'rf_homoGCNConv_basic_normalize',
        # 'lr_homoGCNConv_basic_normalize',
        'lr_homoGCNConv_all_normalize',
        'rf_homoGCNConv_all_normalize',
        'xgb_homoGCNConv_all_normalize',

        'lr_homo_SAGEConv_all_normalize',
        'lr_homoGATConv_all_normalize',
        'rf_homo_SAGEConv_all_normalize',
        'rf_homoGATConv_all_normalize',
        'xgb_homo_SAGEConv_all_normalize',
        'xgb_homoGATConv_all_normalize',

        # 'lr_homo_SAGEConv_without_accuracy_all_normalize',
        # 'lr_homoGATConv_without_accuracy_all_normalize',
        # 'rf_homo_SAGEConv_without_accuracy_all_normalize',
        # 'rf_homoGATConv_without_accuracy_all_normalize',
        # 'xgb_homo_SAGEConv_without_accuracy_all_normalize',
        # 'xgb_homoGATConv_without_accuracy_all_normalize',

    ]:

        for test_dataset in datasets:
            logger.info(f'\n\n======== test_dataset: {test_dataset}, method: {method} =============')
            for ratio in [1.0]:  #: 0.6, 0.8   0.3, 0.5, 0.7
                logger.info(f'\n -- ratio: {ratio}')
                start = time.time()
                df_list = []
                for hidden_channels in [128]:  # 32,64,
                    logger.info(f'\n -------- hidden_channels: {hidden_channels}')
                    trainer = RegressionModel(
                        test_dataset,
                        finetune_ratio=ratio,
                        method=method,
                        hidden_channels=hidden_channels,
                        dataset_embed_method=DatasetEmbeddingMethod.DOMAIN_SIMILARITY,  # '', #  task2vec
                        reference_model='EleutherAI_gpt-neo-125m',
                        task_type=TaskType.SEQUENCE_CLASSIFICATION
                    )
                    # try:
                    score, df_results = trainer.train()
                    # except Exception as e:
                    #     logger.info(e)
                    #     continue
                train_time = time.time() - start
                df_perf.loc[len(df_perf)] = [method, test_dataset, train_time, score]

        df_perf.to_csv(performance_file)
