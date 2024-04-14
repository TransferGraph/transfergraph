import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# code for the Label-Feature Correlation (LFC) score in A linearized framework and a new benchmark for model selection for fine-tuning


class LFC(object):
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
            thetaF = -euclidean_distances(f)
        elif dist == "dot":
            thetaF = np.dot(f, f.T)
        elif dist == "cos":
            thetaF = cosine_similarity(f)
        elif dist == "corr":
            thetaF = np.corrcoef(f)
        thetaF -= np.mean(thetaF)
        lsm = (y[:, None] == y[None, :]).astype(np.float32) * 2 - 1  # label similariy matrix
        lsm -= np.mean(lsm)
        score = cosine_similarity(thetaF.reshape(1, -1), lsm.reshape(1, -1)).item()
        return score
