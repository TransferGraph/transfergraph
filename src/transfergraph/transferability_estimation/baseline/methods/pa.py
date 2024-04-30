import numpy as np
from scipy.linalg import orthogonal_procrustes


class PA(object):
    def __init__(self, args):
        self.args = args

    def score(self, plm_features: np.ndarray, tgt_features: np.ndarray) -> float:
        """
        Procrustes analysis, a similarity test for two data sets in 'To Share or not to Share: Predicting Sets of Sources for Model Transfer Learning'

        Args:
            plm_features (np.ndarray): features extracted by pre-trained language model.
            tgt_features (np.ndarray): features extracted by target model.

        Shape:
            plm_features: (N, F), with number of samples N and feature dimension F.
            tgt_features: (N, F), with number of samples N and feature dimension F.

        Returns:
            score: float
        """
        W = orthogonal_procrustes(plm_features, plm_features)[0]
        I = np.eye(W.shape[1])

        return np.linalg.norm(W - I) * -1.
