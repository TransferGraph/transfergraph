import argparse
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from gensim.models import Word2Vec
from pecanpy import pecanpy
from sklearn.linear_model import LogisticRegression
from torch_geometric.utils.convert import to_scipy_sparse_matrix

# N2V_OUTPUT_DIR = f"{RESULT_DIR}/gene_classification_n2v"
# N2VPLUS_OUTPUT_DIR = f"{RESULT_DIR}/gene_classification_n2vplus"
# N2VPLUSPLUS_OUTPUT_DIR = f"{RESULT_DIR}/gene_classification_n2vplusplus"
# N2V_TISSUE_OUTPUT_DIR = f"{RESULT_DIR}/tissue_gene_classification_n2v"
# N2VPLUS_TISSUE_OUTPUT_DIR = f"{RESULT_DIR}/tissue_gene_classification_n2vplus"
# N2VPLUSPLUS_TISSUE_OUTPUT_DIR = f"{RESULT_DIR}/tissue_gene_classification_n2vplusplus"
# LABEL_DIR = f"{DATA_DIR}/labels/gene_classification"

# check_dirs([
#     N2VPLUSPLUS_OUTPUT_DIR,
#     N2VPLUSPLUS_TISSUE_OUTPUT_DIR,
#     N2VPLUS_OUTPUT_DIR,
#     N2VPLUS_TISSUE_OUTPUT_DIR,
#     N2V_OUTPUT_DIR,
#     N2V_TISSUE_OUTPUT_DIR,
#     RESULT_DIR,
# ])

###DEFAULT HYPER PARAMS###
HPARAM_DIM = 128


##########################
class EdgeLabelDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, args, ft_records, unique_model_id, unique_dataset_id):
        'Initialization'
        self.ft_records = ft_records
        self.unique_model_id = unique_model_id
        self.unique_dataset_id = unique_dataset_id
        self.accu_pos_thres = args.accu_pos_thres
        self.accu_neg_thres = args.accu_neg_thres
        self.edge_labels, self.edge_index = self.get_node2vec_labels()

    def get_node2vec_labels(self):
        ft_records_pos = self.ft_records[
            (self.ft_records['accuracy'] >= self.accu_pos_thres) | (self.ft_records['accuracy'] <= self.accu_neg_thres)]
        # ft_records_neg = self.ft_records[self.ft_records['accuracy']<=self.accu_neg_thres]

        _models = ft_records_pos['model'].values.tolist()  # + ft_records_neg['model'].values.tolist()
        _datasets = ft_records_pos['dataset'].values.tolist()  # + ft_records_neg['dataset'].values.tolist()

        self.unique_model_id.index = self.unique_model_id['model']
        self.unique_dataset_id.index = self.unique_dataset_id['dataset']
        edge_index = np.asarray([self.unique_model_id.loc[_models, 'mappedID'], self.unique_dataset_id.loc[_datasets, 'mappedID']])
        edge_labels = list(ft_records_pos['accuracy'].values)  # + [0]*len(ft_records_neg)
        print(f'\n len of edge_labels: {len(edge_labels)}')
        print(f'\nedge_index: {edge_index.shape}')
        return edge_labels, edge_index

    def __len__(self):
        'Denotes the total number of samples'
        return self.edge_index.shape[1]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # print()
        # print(index)
        # print(self.edge_labels[index])
        X = torch.Tensor(self.edge_index[:, index]).long()
        y = torch.Tensor([self.edge_labels[index]])

        # ID = self.list_IDs[index]
        # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        # y = self.labels[ID]
        return X, y


class N2V_W2VModel(torch.nn.Module):
    def __init__(
            self, gnn_method, edge_index, edge_attr,
            num_walks=10, walk_length=80,
            context_size=10, workers=4,
            hidden_channels=128
    ):
        super().__init__()
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.hidden_channels = hidden_channels
        self.context_size = context_size
        self.workers = workers

        print('\n ----------')
        adj_mat = to_scipy_sparse_matrix(edge_index, edge_attr=edge_attr).todense()
        print(f'adj_mat.shape: {adj_mat.shape}')
        print(f'adj_mat: {adj_mat}')
        # print(f'IDs: {len(IDs)}')
        # adj_mat, IDs = np.load(network_fp).values()

        self.IDs = range(adj_mat.shape[0])
        if '+' in gnn_method:
            extend = True
        else:
            extend = False
        self.g = pecanpy.DenseOTF.from_mat(adj_mat, self.IDs, extend=extend)

        self.walks, self.x_dict = self.embed(edge_index, edge_attr)
        # print(f'\n X_emb: {self.X_emd}')
        # print(f'\n type(self.X_emd): {type(self.X_emd)}')

        from .gnn import HomoRegression
        self.classifier = HomoRegression(hidden_channels)
        pass

    def forward(self, edge_label_index):
        pred = self.classifier(
            self.x_dict,
            edge_label_index,
        )
        return pred

    def embed(self, edge_index, edge_attr=None):
        # simulate random walks and genearte embedings
        walks = self.g.simulate_walks(num_walks=self.num_walks, walk_length=self.walk_length)
        print(f'walks: {len(walks)}, {walks[0]}')

        w2v = Word2Vec(
            walks, vector_size=self.hidden_channels, window=self.context_size,
            min_count=0, sg=1, workers=self.workers, epochs=1
        )

        # sort embeddings by IDs
        IDmap = {j: i for i, j in enumerate(w2v.wv.index_to_key)}
        idx_ary = [IDmap[i] for i in self.IDs]
        X_emd = torch.tensor(w2v.wv.vectors[idx_ary])
        return walks, X_emd

    def train(self):
        start = time.time()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)  # ,capturable=True)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        if 'e2e' in self.args.gnn_method:
            L_fn = nn.MSELoss()
        # print("Capturing:", torch.cuda.is_current_stream_capturing())
        # torch.cuda.empty_cache()
        for epoch in range(1, self.epochs + 1):
            total_loss = total_examples = 0
            for sampled_data in tqdm.tqdm(dataloader):
                optimizer.zero_grad()
                # print(sampled_data['dataset'].x)
                # sampled_data.to(device)
                # print(sampled_data)
                # print(sampled_data['dataset'].x)
                pred = self.classifier(sampled_data)

                ground_truth = sampled_data.edge_label
                loss = L_fn(pred, ground_truth)
                # loss = F.cross_entropy(pred, ground_truth)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            # if total_loss < 1: break
            print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
            # if (total_loss / total_examples) < 0.1: break
        train_time = time.time() - start
        return model, round(total_loss / total_examples, 4), train_time


def _evaluate(X_emd, IDs, label_fp, random_state, df_info):
    # load labels and study-bias holdout splits
    y, train_idx, valid_idx, test_idx, label_ids, gene_ids = np.load(label_fp).values()
    y, gene_ids = align_gene_ids(IDs, y, train_idx, valid_idx, test_idx, gene_ids)
    train_valid_test_idx = train_idx, valid_idx, test_idx
    n_tasks = label_ids.size

    # train and evaluate predictions for each task
    score_lists = [], [], []
    for task_idx in range(n_tasks):
        mdl = LogisticRegression(penalty="l2", solver="liblinear", max_iter=500)
        mdl.fit(X_emd[train_idx], y[train_idx, task_idx])

        for score_list, idx in zip(score_lists, train_valid_test_idx):
            score_list.append(score_func(y[idx, task_idx], mdl.decision_function(X_emd[idx])))

    df = pd.DataFrame()
    df["Training score"], df["Validation score"], df["Testing score"] = score_lists
    df["Task"] = list(label_ids)
    for name, val in df_info.items():
        df[name] = val

    return df


def _get_method_name_and_dir(extend, extend_cts, task_name):
    if extend and extend_cts:
        raise ValueError("extend and extend_cts cannot be set together.")
    elif extend:
        method = "Node2vec+"
        method_abrv = "n2vplus"
        standard_output_dir = N2VPLUS_OUTPUT_DIR
        tissue_output_dir = N2VPLUS_TISSUE_OUTPUT_DIR
    elif extend_cts:
        method = "Node2vec++"
        method_abrv = "n2vplusplus"
        standard_output_dir = N2VPLUSPLUS_OUTPUT_DIR
        tissue_output_dir = N2VPLUSPLUS_TISSUE_OUTPUT_DIR
    else:
        method = "Node2vec"
        method_abrv = "n2v"
        standard_output_dir = N2V_OUTPUT_DIR
        tissue_output_dir = N2V_TISSUE_OUTPUT_DIR

    output_dir = standard_output_dir if task_name == "standard" else tissue_output_dir

    return method, method_abrv, output_dir


def evaluate(args):
    gene_universe = args.gene_universe
    network = args.network
    extend = args.extend
    extend_cts = args.extend_cts
    p = args.p
    q = args.q
    gamma = args.gamma
    random_state = args.random_state
    nooutput = args.nooutput
    task = args.task

    if task == "standard":
        datasets = ["GOBP", "DisGeNet"]
    elif task == "tissue":
        datasets = ["GOBP-tissue"]
    else:
        raise ValueError(f"Unknown task {task}")

    if args.test:
        NUM_THREADS = 128
    else:
        NUM_THREADS = 4

    try:
        numba.set_num_threads(NUM_THREADS)
    except ValueError:
        pass

    method, method_abrv, output_dir = _get_method_name_and_dir(extend, extend_cts, task)
    network_fp = get_network_fp(network)
    output_fn = f"{network}_{method_abrv}_{p=}_{q=}_{gamma=}.csv"

    # Generate embeddings
    t = time()
    X_emd, IDs = embed(network_fp, HPARAM_DIM, extend, extend_cts, p, q, NUM_THREADS, gamma)
    t = time() - t
    print(f"Took {int(t / 3600):02d}:{int(t / 60):02d}:{t % 60:05.02f} to generate embeddings using {method}")

    # Run evaluation on all datasets
    t = time()
    result_df_list = []
    for dataset in datasets:
        label_fp = f"{LABEL_DIR}/{gene_universe}_{dataset}_label_split.npz"

        df_info = {"Dataset": dataset, "Network": network, "Method": method,
                   "p": p, "q": q, "gamma": gamma}
        df = _evaluate(X_emd, IDs, label_fp, random_state, df_info)
        result_df_list.append(df)
    t = time() - t
    print(f"Took {int(t / 3600):02d}:{int(t / 60):02d}:{t % 60:05.02f} to evaluate")

    # combine results into a single dataframe
    result_df = pd.concat(result_df_list).sort_values("Task")

    # Print results summary (and save)
    print(result_df[["Training score", "Validation score", "Testing score"]].describe())
    if not nooutput:
        output_fp = f"{output_dir}/{output_fn}"
        result_df.to_csv(output_fp, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation on gene classification datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--gene_universe", required=True, help="Name of the gene universe")
    parser.add_argument("--network", required=True, help="Name of hierarchical cluster graph to use")
    parser.add_argument("--task", default="standard", help="'standard': GOBP, DisGeNet or 'tissue': GOBP-tissue")
    parser.add_argument("--p", required=True, type=float, help="return bias parameter p")
    parser.add_argument("--q", required=True, type=float, help="in-out bias parameter q")
    parser.add_argument("--gamma", type=float, default=0, help="Noisy edge threshold parameter")
    parser.add_argument("--nooutput", action="store_true", help="Disable results saving and print results to screen")
    parser.add_argument("--random_state", type=int, default=0, help="Random state used for generating random splits")
    parser.add_argument("--test", action="store_true", help="Toggle test mode, run with more workers")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--extend", action="store_true", help="Use node2vec+ if specified, otherwise use node2vec")
    group.add_argument("--extend_cts", action="store_true", help="Use node2vec++ if specified, otherwise use node2vec")

    args = parser.parse_args()
    print(args)

    return args


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
