import argparse
import pathlib

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from pecanpy import pecanpy
from pecanpy.experimental import Node2vecPlusPlus
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

HOME_DIR = pathlib.Path(__file__).absolute().parents[1]
DATA_DIR = HOME_DIR / "data"
RESULT_DIR = HOME_DIR / "result"
NETWORK_DIR = DATA_DIR / "networks"

LABEL_PATH_DICT = {
    "DisGeNet": "/mnt/research/compbio/krishnanlab/data/disease-gene_annotations/disgenet/disgenet_disease-genes_prop.gsea-min10-max600-ovlppt7-jacpt5.nonred.gmt",
    "GOBP": "/mnt/research/compbio/krishnanlab/data/functional_annotations/go/go_bp-genes_exp-ec_prop_pos-slim.gsea-min10-max200-ovlppt7-jacpt5.nonred.gmt",
    "GOBP-tissue": "/mnt/research/compbio/krishnanlab/data/functional_annotations/go/go_tissue/GOBP-tissue-subset.gmt",
}
PUBMED_COUNT_PATH = "/mnt/research/compbio/krishnanlab/data/pubmed/gene2pubmed_human_gene-counts.txt"

REPETITION = 10

W2V_NUMWALKS = 10
W2V_WALKLENGTH = 80
W2V_WINDOW = 10
W2V_EPOCHS = 1

OUTPUT_DIR = f"{RESULT_DIR}/realworld_networks"
NETWORK_DIR = f"{DATA_DIR}/networks/realworld"
LABEL_DIR = f"{DATA_DIR}/labels"

# check_dirs([RESULT_DIR, OUTPUT_DIR])

WEIGHTED_DICT = {"BlogCatalog": False, "Wikipedia": True}

###DEFAULT HYPER PARAMS###
HPARAM_P = 1
HPARAM_DIM = 128


##########################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation on real-world datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--network", required=True,
        help="Name of real-world network to use"
    )

    parser.add_argument(
        "--q", required=True, type=float,
        help="in-out bias parameter q"
    )

    parser.add_argument(
        "--extend", action="store_true",
        help="Use node2vec+ if specified, otherwise use node2vec"
    )

    parser.add_argument(
        "--nooutput", action='store_true',
        help="Disable output if specified, and print results to screen"
    )

    args = parser.parse_args()
    print(args)

    return args


def _evaluate(X_emd, IDs, label_fp):
    # load labels and convert to label matrix
    label_df = pd.read_csv(label_fp, sep='\t', header=None)
    IDmap = {j: i for i, j in enumerate(IDs)}
    task_id_map = {j: i for i, j in enumerate(label_df[1].unique())}
    n_tasks = len(task_id_map)

    node_idx_list = [IDmap[str(i)] for i in label_df[0]]
    task_idx_list = [task_id_map[i] for i in label_df[1]]

    y_mat = np.zeros((IDs.size, n_tasks), dtype=bool)
    y_mat[node_idx_list, task_idx_list] = True

    # initialize classifiaction model and split generator
    mdl = LogisticRegression(penalty='l2', solver='liblinear', max_iter=500)
    skf = StratifiedKFold(n_splits=2)

    # train and evaluate predictions for each task
    train_score_list, test_score_list = [], []
    for task_idx in range(n_tasks):
        y = y_mat[:, task_idx].copy()
        train_idx, test_idx = next(skf.split(X_emd, y))  # generate splits
        mdl.fit(X_emd[train_idx], y[train_idx])  # train model

        # evaluate performance
        y_pred = mdl.predict(X_emd)
        train_score_list.append(f1_score(y[train_idx], y_pred[train_idx]))
        test_score_list.append(f1_score(y[test_idx], y_pred[test_idx]))

    train_score = np.mean(train_score_list)
    test_score = np.mean(test_score_list)

    return train_score, test_score


def evaluate(args):
    network = args.network
    extend = args.extend
    q = args.q
    nooutput = args.nooutput

    method = 'Node2vec+' if extend else 'Node2vec'
    network_fp = f"{NETWORK_DIR}/{network}.edg"
    output_fp = f"{OUTPUT_DIR}/{network}_n2v{'plus' if extend else ''}_q={q}.csv"
    label_fp = f"{LABEL_DIR}/{network}.tsv"

    # run evaluation with repetitions on both tasks
    score_list = []
    for _ in range(1):  # REPETITION
        X_emd, IDs = embed(
            network_fp, HPARAM_DIM, extend, HPARAM_P,
            q, weighted=WEIGHTED_DICT[network]
        )
        score_list.append(_evaluate(X_emd, IDs, label_fp))
        print("DONE")

    result_df = pd.DataFrame()
    result_df['Training score'], result_df['Testing score'] = zip(*score_list)
    result_df['Network'], result_df['Method'], result_df['q'] = network, method, q

    # save or print results
    if nooutput:
        print(result_df)
    else:
        result_df.to_csv(output_fp, index=False)


def embed(network_fp, dim=128, extend=True, extend_cts=False, p=1, q=10, workers=1, gamma=0.1):
    # initialize DenseOTF graph
    adj_mat, IDs = np.load(network_fp).values()
    if extend_cts:
        g = Node2vecPlusPlus.from_mat(
            adj_mat, IDs, p=p, q=q, workers=workers,
            gamma=gamma
        )
    else:
        g = pecanpy.DenseOTF.from_mat(adj_mat, IDs, p=p, q=q, extend=extend)

    # simulate random walks and genearte embedings
    walks = g.simulate_walks(num_walks=W2V_NUMWALKS, walk_length=W2V_WALKLENGTH)
    w2v = Word2Vec(
        walks, vector_size=dim, window=W2V_WINDOW,
        min_count=0, sg=1, workers=workers, epochs=W2V_EPOCHS
    )

    # sort embeddings by IDs
    IDmap = {j: i for i, j in enumerate(w2v.wv.index_to_key)}
    idx_ary = [IDmap[i] for i in IDs]
    X_emd = w2v.wv.vectors[idx_ary]

    return X_emd, IDs


def main():
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
