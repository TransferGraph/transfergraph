import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class kNN(object):
    def __init__(self, args):
        self.args = args

    def score(
            self, train_features: np.ndarray, train_labels: np.ndarray,
            val_features: np.ndarray, val_labels: np.ndarray
    ):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels
        :return: TransRate score (how well f can fit y directly)
        """
        K = int(self.args.method.split("-")[1])
        dist = str(self.args.method.split("-")[2])
        if dist == "l2":
            model = KNeighborsClassifier(n_neighbors=K).fit(train_features, train_labels)
        elif dist == "cos":
            model = KNeighborsClassifier(n_neighbors=K, metric="cosine").fit(train_features, train_labels)
        elif dist == "corr":
            model = KNeighborsClassifier(n_neighbors=K, metric="correlation").fit(train_features, train_labels)
        return (model.predict(val_features) == val_labels).mean()
