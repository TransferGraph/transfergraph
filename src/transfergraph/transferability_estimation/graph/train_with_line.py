from __future__ import print_function

import sys

# sys.path.append(os.getcwd())
sys.path.append('../')

import numpy as np
from utils.line import Classifier
from utils.line import LINE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def evaluate_embeddings(X, Y, embeddings):
    # X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print(
        "Training classifier using {:.2f}% nodes...".format(
            tr_frac * 100
        )
    )
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(X, Y, embeddings, ):
    # X, Y = read_node_label('../data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def main(G):
    # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = LINE(G, embedding_size=128, order='second')
    model.train(batch_size=20, epochs=5, verbose=2)  # 1024
    embeddings = model.get_embeddings()

    X = list(G.nodes)
    Y = np.random.randint(3, size=(len(X), 1))
    # print(f'\n X: {X}, Y:{Y}')
    # print(type(Y[0]))
    evaluate_embeddings(X, Y, embeddings)
    # plot_embeddings(embeddings)


if __name__ == "__main__":
    G = nx.Graph()
    edges = list(map(tuple, np.random.randint(20, size=(2000, 2))))
    # print(f'\n edges: {edges}')
    G.add_edges_from(edges)
    print(f'\nG: {G}\n')
    main(G)
