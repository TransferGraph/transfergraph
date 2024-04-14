import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class DDS(object):
    def __init__(self, args):
        self.args = args

    def rdm(self, activations_value, dist):
        if dist == 'corr':
            RDM = 1 - np.corrcoef(activations_value)
        elif dist == 'cos':
            RDM = 1 - cosine_similarity(activations_value)
        elif dist == 'dot':
            RDM = -np.dot(activations_value, activations_value.T)
        elif dist == "l2":
            RDM = euclidean_distances(activations_value)
        return RDM

    def center_gram(self, gram, unbiased=False):
        if not np.allclose(gram, gram.T):
            raise ValueError('Input must be a symmetric matrix.')
        gram = gram.copy()

        if unbiased:
            n = gram.shape[0]
            np.fill_diagonal(gram, 0)
            means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            gram -= means[:, None]
            gram -= means[None, :]
            np.fill_diagonal(gram, 0)
        else:
            means = np.mean(gram, 0, dtype=np.float64)
            means -= np.mean(means) / 2
            gram -= means[:, None]
            gram -= means[None, :]

        return gram

    def cka(self, gram_x, gram_y, debiased=False, centered=True):
        if centered:
            gram_x = self.center_gram(gram_x, unbiased=debiased)
            gram_y = self.center_gram(gram_y, unbiased=debiased)

        scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

        normalization_x = np.linalg.norm(gram_x)
        normalization_y = np.linalg.norm(gram_y)

        return scaled_hsic / (normalization_x * normalization_y)

    def score(self, feats1: np.ndarray, feats2: np.ndarray) -> float:
        """
        DDS score from https://github.com/dbolya/parc/blob/main/methods.py

        Args:
            feats1 (np.ndarray): _description_
            feats2 (np.ndarray): _description_

        Returns:
            float: _description_
        """
        dist = str(self.args.method.split("-")[1])

        feats1 = sklearn.preprocessing.StandardScaler().fit_transform(feats1)
        feats2 = sklearn.preprocessing.StandardScaler().fit_transform(feats2)

        return self.cka(self.rdm(feats1, dist), self.rdm(feats2, dist), debiased=True, centered=True) * 100
