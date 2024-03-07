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

SAVE_FEATURE = True


class RegressionModel():
    def __init__(
            self,
            test_dataset,
            finetune_ratio=1.0,
            method='',
            hidden_channels=128,
            dataset_embed_method='domain_similarity',
            reference_model='resnet50',
            task_type='sequence_classification',
    ):
        if dataset_embed_method == 'task2vec':
            corr_path = f'corr_task2vec_{reference_model}.csv'
        elif dataset_embed_method == 'domain_similarity':
            corr_path = f'corr_domain_similarity_{reference_model}.csv'
        self.selected_columns = ['architectures', 'accuracy',
                                 'model_type', 'number_of_parameters',
                                 'train_runtime',
                                 'dataset',
                                 'finetune_dataset', 'size', 'number_of_classes',
                                 'test_accuracy'
                                 ]  # 'input_shape', 'elapsed_time', '#labels',
        dataset_map = {'oxford_iiit_pet': 'pets',
                       'oxford_flowers102': 'flowers'}
        if test_dataset in dataset_map.keys():
            self.test_dataset = dataset_map[test_dataset]
        else:
            self.test_dataset = test_dataset
        self.finetune_ratio = finetune_ratio
        self.method = method
        self.corr_path = corr_path
        self.hidden_channels = hidden_channels
        self.task_type = task_type
        self.directory_experiments = os.path.join(get_root_path_string(), 'resources/experiments', self.task_type)

        if 'task2vec' in corr_path and 'task2vec' in dataset_embed_method:
            self.embed_addition = '_task2vec'
        else:
            self.embed_addition = ''

        if 'without_accuracy' in method:
            self.y_label = 'score'
        else:
            self.y_label = 'test_accuracy'
        pass

    def feature_preprocess(self, embedding_dict={}, data_dict={}):

        df_model_config = pd.read_csv(os.path.join(self.directory_experiments, 'model_config_dataset.csv'))
        # print(f'\n df_model_config.columns: {df_model_config.columns}')
        df_dataset_config = pd.read_csv(os.path.join(self.directory_experiments, 'target_dataset_features.csv'))
        # print(f'\n df_dataset_config.columns: {df_dataset_config.columns}')

        df_finetune = pd.read_csv(os.path.join(self.directory_experiments, 'records.csv'), index_col=0)
        df_finetune = df_finetune.rename(columns={'eval_accuracy': 'test_accuracy'})

        df_finetune = df_finetune.rename(columns={'finetuned_dataset': 'finetune_dataset'})
        if 'input_shape' in df_finetune.columns:
            df_finetune = df_finetune.drop(columns=['input_shape'])

        # joining finetune records with model config (model)
        df_model = df_finetune.merge(df_model_config, how='inner', on='model')

        # joining finetune records with dataset config (dataset metadata)
        df_feature = df_model.merge(df_dataset_config, how='inner', on='finetune_dataset')

        ## !! image - fill those with mean value
        # df_feature = fill_null_value(df_feature,columns=['test_accuracy','number_of_parameters'])
        ## !! text
        df_feature = df_feature.dropna(subset=['test_accuracy'])
        df_feature = fill_null_value(df_feature, columns=['size', 'number_of_classes'])

        df_feature = df_feature.dropna(subset=['model_type'])

        if 'normalize' in self.method:
            df_feature['test_accuracy'] = df_feature[['finetune_dataset', 'test_accuracy']].groupby('finetune_dataset').transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )

        if ('Conv' in self.method or 'node2vec' in self.method):
            if embedding_dict == {}:
                method = self.method
                if 'normalize' in method:
                    method = '_'.join(method.split('_')[:-1])
                if 'basic' in method or 'all' in method:
                    method = '_'.join(method.split('_')[:-1])
                # print(method)
                if method[-8:] == 'distance':
                    method = '_'.join(method.split('_')[:-2])

                path_dataset_features = os.path.join(self.directory_experiments, f"features_final/{self.test_dataset.replace('/', '_')}")
                if not os.path.exists(path_dataset_features):
                    os.makedirs(path_dataset_features)
                if self.finetune_ratio == 1:
                    file = os.path.join(
                        f"{path_dataset_features}/features_{method.replace('rf_', 'lr_').replace('svm_', 'lr_').replace('xgb_', 'lr_')}_{self.hidden_channels}.csv"
                    )
                    print(f'\nfile: {file}')
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
                print(f'\n1st df_feature.shape: {df_feature.shape}')
                # assert df_feature.shape[0] > 600

                if 'model.1' in df_feature.columns:
                    # df_feature = df_feature.rename(columns={'model.1':'model'})
                    df_feature = df_feature.drop(columns=['model.1'])
                # print(df_feature.columns)

                df_feature.index = range(len(df_feature))

                df_feature = df_feature.dropna(subset=['model_type'])
                # if 'normalize' in self.method:
                #     df_feature['test_accuracy'] = df_feature[['finetune_dataset','test_accuracy']].groupby('finetune_dataset').transform(lambda x: (x - x.min()) / (x.max()-x.min()))

                # if 'basic' in self.method or 'all' in self.method or 'without_accuracy' in self.method:
                self.selected_columns += [col for col in df_feature.columns if 'm_f' in col or 'd_f' in col]
                # print()
                # print(list(df_feature.columns))

            if embedding_dict != {}:
                # assert data_dict == {}
                unique_model_id = data_dict['unique_model_id']
                unique_model_id.index = range(len(unique_model_id))
                df_feature = df_feature.merge(unique_model_id, how='inner', on='model')
                df_data_id = data_dict['unique_dataset_id'].rename({'dataset': 'finetune_dataset'}, axis='columns')
                df_feature = df_feature.merge(df_data_id, how='inner', on='finetune_dataset')

                print(f'\n embedding_dict != null, len(df_feature):{len(df_feature)}')
                # assert len(df_feature) > 600

                # print(f'\n df_feature.clumns: \n{df_feature.columns}')
                # print(f'\ndf_feature: {df_feature.head()}')
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
                        # print(f'\nembedding_dict:{embedding_dict}')

                        model_emb.append(embedding_dict[model_id].numpy())
                        dataset_emb.append(embedding_dict[dataset_id].numpy())

                ## if 'all' in method, taking all the features into account
                if not 'all' in self.method:
                    self.selected_columns = ['finetune_dataset', self.y_label]

                columns = ['m_f' + str(i) for i in range(len(embedding_dict[model_id]))]
                df_ = pd.DataFrame(model_emb, columns=columns)
                # df_feature[columns] = model_emb
                df_feature = pd.concat([df_feature, df_], axis=1)
                self.selected_columns += columns

                columns = ['d_f' + str(i) for i in range(len(embedding_dict[dataset_id]))]
                df_ = pd.DataFrame(dataset_emb, columns=columns)
                # df_feature[columns] = dataset_emb
                df_feature = pd.concat([df_feature, df_], axis=1)
                self.selected_columns += columns

                # print(df_feature.head())
                # df_feature.to_csv('./methods/features.csv')

                df_feature = df_feature.dropna(subset=['m_f0', 'd_f0'])
                # print(df_feature.head())

        if 'logme' in self.method or 'without_accuracy' in self.method:  # or 'all' in self.method
            # print(df_feature.columns)
            # model_col_count = df_feature.columns.tolist().count('model')
            # if model_col_count > 1: df_feature = df_feature.drop(columns=['model'])

            if 'score' not in df_feature.columns:
                model_list = df_feature['model'].unique()
                # df_feature.index = df_feature['model']
                df_feature.index = range(len(df_feature))
                df_dataset_list = []
                for dataset in df_feature['finetune_dataset'].unique():
                    logme_path = os.path.join(self.directory_experiments, 'transferability_score_records.csv')
                    logme = pd.read_csv(logme_path)
                    df_logme = logme[logme['model'] != 'time']
                    df_logme = df_logme[df_logme['target_dataset'] == dataset]

                    # identify common models
                    # print('\n',df_feature['model'])
                    df_logme = df_logme[df_logme['model'].isin(model_list)]
                    df_logme = df_logme.dropna(subset=['score'])
                    # normalize
                    df_logme['score'].replace([-np.inf, np.nan], -50, inplace=True)

                    score = df_logme['score']  # .astype('float64')
                    # normalized_pred = (score-score.min())/(score.max()-score.min()) #df_score['score'].values/norm
                    normalized_pred = (score - score.mean()) / score.std()
                    df_logme['score'] = normalized_pred

                    # df_feature.loc[df_logme['model'].values,'score'] = normalized_pred
                    df_dataset_list.append(
                        df_feature[df_feature['finetune_dataset'] == dataset].merge(df_logme, how='inner', on=['model'])
                    )  # ,'finetune_dataset']))
                df_feature = pd.concat(df_dataset_list)
                # print(f'after logme: if score in columns: {list(df_feature.columns)}')
            if 'score' not in self.selected_columns:
                self.selected_columns += ['score']

        if 'data_distance' in self.method or 'all' in self.method:
            # if True:
            corr_path = os.path.join(self.directory_experiments, 'resources', self.corr_path)
            # print(f'\ncorr_path: {corr_path}')
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

            # print('df_feature')
            # print(df_feature.finetune_dataset.unique())
            # print(df_feature.dataset.unique())
            # print('df_corr')
            # print(df_corr.finetune_dataset.unique())
            # print(df_corr.dataset.unique())

            df_feature = df_feature.merge(df_corr, how='outer', on=['finetune_dataset', 'dataset'])
            # df_feature = df_feature.merge(df_corr,how='inner',on=['finetune_dataset','dataset'])
            #### fill missing value with minimum value
            min_value = df_feature['distance'].min()
            df_feature['distance'].fillna(value=min_value, inplace=True)

            # print(df_feature.head())
            print(f'\n2nd df_feature.shape: {df_feature.shape}')
            assert df_feature.shape[0] > 600

            self.selected_columns += ['distance']

        if 'feature' in self.method:
            print(f'feature in self.method')
            records = {'finetune_dataset': [], 'model': [], 'feature': []}
            emb_files = glob(os.path.join('../../../../../', 'model_embed', 'embeddings') + '/*')
            for file in emb_files:
                components = file.split(',')
                array = np.reshape(np.load(file), (1, -1))
                # print(f'array.shape: {array.shape}')
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
            # print(f'\ndf: {df.head()}')
            df_feature = df_feature.merge(df, how='inner', on=['finetune_dataset', 'model'])
            # print(f'\ndf_feature: {df_feature.head()}')

        # print(list(df_feature.columns))
        df_feature.index = df_feature['model']
        # print(f'\nSAVE_FEATURE: {SAVE_FEATURE}')
        if SAVE_FEATURE:
            # print(f'\n-----save feature')
            _dir = os.path.join(self.directory_experiments, 'features_final', self.test_dataset.replace('/', '_'))
            if not os.path.exists(_dir):
                os.makedirs(_dir)
            if self.finetune_ratio < 1:
                file = os.path.join(_dir, f'features_{self.method}{self.embed_addition}_{self.hidden_channels}_{self.finetune_ratio}.csv')
            else:
                file = os.path.join(_dir, f'features_{self.method}{self.embed_addition}_{self.hidden_channels}.csv')
            # if not os.path.exists(file):
            #     df_feature.to_csv(file)
            print(f'3rd df_feature.shape: {df_feature.shape}')
            # assert df_feature.shape[0] > 600
            df_feature.to_csv(file)

        # print(f'\n selected_columns: {self.selected_columns}')
        df_feature = df_feature[self.selected_columns]
        if 'score' in df_feature.columns:
            df_feature = df_feature.dropna(subset=['score'])
        if 'm_f0' in df_feature.columns:
            df_feature = df_feature.dropna(subset=['m_f0', 'd_f0'])
        print(f'\n df_feature.len: {len(df_feature)}')

        nan_columns = df_feature.columns[df_feature.isna().any()].tolist()
        print(f'nan_columns: {nan_columns}')
        df_feature = fill_null_value(df_feature, columns=nan_columns)
        # print(f'df_feature.columns:{df_feature.columns}')
        # print(df_feature[['finetune_dataset','score']].head())

        self.df_feature = df_feature

        #
        # print(f'\n nan_columns: {nan_columns}')
        # normal_columns = [col for col in df_feature.columns if 'm_f' not in col and 'd_f' not in col]
        # df_feature[normal_columns].to_csv('../../features/features.csv')

    def split(self):

        df_train = self.df_feature[self.df_feature['finetune_dataset'] != self.test_dataset]

        ##### Sampling the finetune records given the ratio
        if self.finetune_ratio != 1:
            df_train = df_train.sample(frac=self.finetune_ratio, random_state=1)

        # df_train = df_train.drop(columns='finetune_dataset')
        df_test = self.df_feature[self.df_feature['finetune_dataset'] == self.test_dataset]
        # print(f'\n - test_dataset: {self.test_dataset}')
        # print(df_test.head())

        # df_test = df_test.drop(columns='finetune_dataset')
        self.selected_columns.remove('finetune_dataset')

        categorical_columns = ['architectures', 'model_type', 'finetune_dataset', 'dataset']
        if set(categorical_columns) < set(list(df_train.columns)):
            df_train = encode(df_train, categorical_columns)
            df_test = encode(df_test, categorical_columns)

        if 0:
            df_train.to_csv(os.path.join(self.directory_experiments, 'train.csv'))
            df_test.to_csv(os.path.join(self.directory_experiments, 'test.csv'))

        return df_train, df_test

    def train(self, embedding_dict={}, data_dict={}, save_path='', run=0):
        # if 'Conv' in method:
        #     method = 'lr_' + method
        # print(embedding_dict)

        self.feature_preprocess(embedding_dict, data_dict)

        df_train, df_test = self.split()

        feature_columns = [col for col in self.selected_columns if col != self.y_label]
        X_train = df_train[feature_columns].values
        X_test = df_test[feature_columns].values
        y_train = df_train[self.y_label].values
        y_test = df_test[self.y_label].values

        print(f'\n --- dataset: {self.test_dataset}')
        print(f'\n --- shape of X_train: {X_train.shape}, X_test.shape: {X_test.shape}')
        # print(f'\n---y_train:{y_train[:20]}')
        assert X_train.shape[0] > 200
        # print(f'\n --- shape of Y_train: {y_train.shape}, y_test.shape: {y_test.shape}')
        # print(f'X_test:{X_test}')
        # ### Model
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
            )  # random_state=18,subsample=0.7,
            # early_stop = xgb.callback.EarlyStopping(
            #     rounds=2, metric_name='logloss', data_name='Validation_0', save_best=True
            # )
            # model = xgb.XGBClassifier(tree_method="hist") #callbacks=[early_stop] early_stopping_rounds=2

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        print(f'\n === score: {score}')
        y_pred = model.predict(X_test)
        # print(f'\n y_pred: {y_pred[:10]}')

        df_results = pd.DataFrame({'model': df_test.index, 'score': y_pred})

        if embedding_dict == {}:
            dir_path = os.path.join(self.directory_experiments, 'rank_final', f"{self.test_dataset.replace('/', '_')}", self.method)
        else:
            dir_path = os.path.join(self.directory_experiments, 'rank_final', f"{self.test_dataset.replace('/', '_')}", self.method)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if self.finetune_ratio > 1:
            file = f'results{self.embed_addition}_{self.hidden_channels}_{run}.csv'
        else:
            file = f'results{self.embed_addition}_{self.finetune_ratio}_{self.hidden_channels}_{run}.csv'

        print(f'\n -- os.path.join(dir_path,file): {os.path.join(dir_path, file)}')
        df_results.to_csv(os.path.join(dir_path, file))

        return score, df_results


def fill_null_value(df, columns, method='mean', value=0):
    for col in columns:
        print(f'fill null value on column: {col}')
        dtype = pd.api.types.infer_dtype(df[col])
        print(f'{col} dtype: {dtype}')
        if dtype == 'floating' or dtype == 'integer':
            if method == 'mean':
                df[col].fillna((df[col].mean()), inplace=True)
            else:
                df[col].fillna((value), inplace=True)
        else:
            print(f'data type: {pd.api.types.infer_dtype(df[col])}')
            # df[col] = df[col].astype(str)
            df[col].fillna((''), inplace=True)

    return df


def encode(df, columns):
    # convert features to one-hot encoding
    encoder = LabelEncoder()
    for col in columns:
        # print(f'\n encoder col:{col}')
        # print(df[col].dtypes)
        if df[col].dtypes != 'str':
            df[col] = df[col].replace(0, '')
            # df = df.dropna(subset=[col])
        _df = pd.DataFrame(encoder.fit_transform(df[col]), columns=[col])
        _df.index = df.index
        df = df.drop(columns=[col])
        df = pd.concat([df, _df], axis=1)
    return df


if __name__ == '__main__':
    task_type = 'sequence_classification'
    path = os.path.join(get_root_path_string(), 'resources/experiments', task_type, 'log')
    performance_file = os.path.join(path, f'performance_rf_score.csv')
    print(f'====== path: {path} ======')
    if os.path.exists(performance_file):
        df_perf = pd.read_csv(performance_file, index_col=0)
    else:
        os.makedirs(path)
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
            print(f'\n\n======== test_dataset: {test_dataset}, method: {method} =============')
            for ratio in [1.0]:  #: 0.6, 0.8   0.3, 0.5, 0.7
                print(f'\n -- ratio: {ratio}')
                start = time.time()
                df_list = []
                for hidden_channels in [128]:  # 32,64,
                    print(f'\n -------- hidden_channels: {hidden_channels}')
                    trainer = RegressionModel(
                        test_dataset,
                        finetune_ratio=ratio,
                        method=method,
                        hidden_channels=hidden_channels,
                        dataset_embed_method='domain_similarity',  # '', #  task2vec
                        reference_model='gpt2_gpt',
                        task_type=task_type
                    )
                    # try:
                    score, df_results = trainer.train()
                    # except Exception as e:
                    #     print(e)
                    #     continue
                train_time = time.time() - start
                df_perf.loc[len(df_perf)] = [method, test_dataset, train_time, score]

        df_perf.to_csv(performance_file)
