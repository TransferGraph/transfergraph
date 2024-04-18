import numpy as np
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class RSA(object):
    def __init__(self, args):
        self.args = args

    def get_lowertri(self, rdm):
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions, 1)]

    def score(self, feats1: np.ndarray, feats2: np.ndarray) -> float:
        """
        RSA score from https://github.com/dbolya/parc/blob/main/methods.py

        Args:
            feats1 (np.ndarray): _description_
            feats2 (np.ndarray): _description_

        Returns:
            float: _description_
        """
        feats1 = sklearn.preprocessing.StandardScaler().fit_transform(feats1)
        feats2 = sklearn.preprocessing.StandardScaler().fit_transform(feats2)

        dist = str(self.args.method.split("-")[1])

        if dist == 'corr':
            rdm1 = 1 - np.corrcoef(feats1)
            rdm2 = 1 - np.corrcoef(feats2)
        elif dist == 'cos':
            rdm1 = 1 - cosine_similarity(feats1)
            rdm2 = 1 - cosine_similarity(feats2)
        elif dist == "l2":
            rdm1 = euclidean_distances(feats1)
            rdm2 = euclidean_distances(feats2)
        elif dist == "dot":
            rdm1 = -np.dot(feats1, feats1.T)
            rdm2 = -np.dot(feats2, feats2.T)

        lt_rdm1 = self.get_lowertri(rdm1)
        lt_rdm2 = self.get_lowertri(rdm2)

        return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100
