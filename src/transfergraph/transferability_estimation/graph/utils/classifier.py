import numpy as np
from sklearn.metrics import f1_score, accuracy_score


class Classifier(object):

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = (clf)
        # self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X_dict, Y):
        # self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        # print(f'\nY: {Y}')
        # Y = self.binarizer.transform(Y)
        # print(f'Y: {Y}')
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        # top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        # Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        results['acc'] = accuracy_score(Y, Y_)
        print('-------------------')
        print(results)
        return results

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = np.random.get_state()

        training_size = int(train_precent * len(X))
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        # print(f'\nshuffle_indices:{shuffle_indices}')

        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        print(f'\n X_train: {X_train}')
        print(f'\nY_train: {Y_train}')
        self.train(X_train, Y_train, Y)
        np.random.set_state(state)
        return self.evaluate(X_test, Y_test)
