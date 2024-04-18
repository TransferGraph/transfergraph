import numpy as np
from sklearn.metrics import silhouette_score


class MSC(object):
    def __init__(self, args):
        self.args = args

    def score(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels
        :return: TransRate score (how well f can fit y directly)
        """
        dist = str(self.args.method.split("-")[1])
        if dist == "l2":
            return silhouette_score(f, y, metric="l2")
        elif dist == "cos":
            return silhouette_score(f, y, metric="cosine")
        elif dist == "corr":
            return silhouette_score(f, y, metric="correlation")
