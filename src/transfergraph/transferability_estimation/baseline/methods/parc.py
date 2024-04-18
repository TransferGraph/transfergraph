import numpy as np
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from transfergraph.transferability_estimation.baseline.methods.utils import TransferabilityDistanceFunction


class PARC(object):
    def __init__(self, distance_function: TransferabilityDistanceFunction):
        self.distance_function = distance_function

    def get_lowertri(self, rdm):
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions, 1)]

    def score(self, features, labels):
        """PARC score from https://github.com/dbolya/parc/blob/main/methods.py

        Args:
            features (_type_): _description_
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        scaler = sklearn.preprocessing.StandardScaler()
        features = scaler.fit_transform(features)
        num_classes = len(set(labels))
        labels = np.eye(num_classes)[labels.astype(int)]

        if self.distance_function == TransferabilityDistanceFunction.CORRELATION:
            rdm1 = 1 - np.corrcoef(features)
            rdm2 = 1 - np.corrcoef(labels)
        elif self.distance_function == TransferabilityDistanceFunction.COSINE:
            rdm1 = 1 - cosine_similarity(features)
            rdm2 = 1 - cosine_similarity(labels)
        elif self.distance_function == TransferabilityDistanceFunction.EUCLIDIAN:
            rdm1 = euclidean_distances(features)
            rdm2 = euclidean_distances(labels)
        elif self.distance_function == TransferabilityDistanceFunction.DOT:
            rdm1 = -np.dot(features, features.T)
            rdm2 = -np.dot(labels, labels.T)
        else:
            raise Exception(f"Unexpected distance function {self.distance_function}")

        lt_rdm1 = self.get_lowertri(rdm1)
        lt_rdm2 = self.get_lowertri(rdm2)

        return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100
