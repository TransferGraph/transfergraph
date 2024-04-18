import numpy as np


class DSE(object):
    def __init__(self, args):
        self.args = args

    def l2(self, features_1, features_2):
        return -np.mean(np.linalg.norm(features_1 - features_2, axis=1))

    def dot(self, features_1, features_2):
        return np.mean(np.sum(features_1 * features_2, axis=1))

    def cosine(self, features_1, features_2):
        return np.mean(np.sum(features_1 * features_2, axis=1) / (np.linalg.norm(features_1, axis=1) * np.linalg.norm(features_2, axis=1)))

    def corr(self, features_1, features_2):
        features_1 -= np.mean(features_1, axis=1, keepdims=True)
        features_2 -= np.mean(features_2, axis=1, keepdims=True)
        return np.mean(np.sum(features_1 * features_2, axis=1) / (np.linalg.norm(features_1, axis=1) * np.linalg.norm(features_2, axis=1)))

    def score(self, plm_features: np.ndarray, tgt_features: np.ndarray) -> float:
        """
        Direct Similarity Estimation (DSE) with two versions:
        (1) 'MeanEmb' in 'Exploring and Predicting Transferability across NLP Tasks'
        (2) 'MeanSim' in 'CogTaskonomy: Cognitively Inspired Task Taxonomy Is Beneficial to Transfer Learning in NLP'

        Args:
            plm_features (np.ndarray): features extracted by pre-trained language model.
            tgt_features (np.ndarray): features extracted by target model.

        Shape:
            plm_features: (N, F), with number of samples N and feature dimension F.
            tgt_features: (N, F), with number of samples N and feature dimension F.

        Returns:
            score: float
        """
        dist = str(self.args.method.split("-")[1])

        if dist == "l2":
            return self.l2(plm_features, tgt_features)
        elif dist == "dot":
            return self.dot(plm_features, tgt_features)
        elif dist == "cos":
            return self.cosine(plm_features, tgt_features)
        elif dist == "corr":
            return self.corr(plm_features, tgt_features)
