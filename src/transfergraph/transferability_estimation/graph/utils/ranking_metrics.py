from itertools import product

import numpy as np


def mapk(actual, predicted, k=0):
    return np.mean([apk(a, p, k) for a, p in product([actual], predicted)])


def apk(y_true, y_pred, k_max=0):
    # Check if all elements in lists are unique
    if len(set(y_true)) != len(y_true):
        raise ValueError("Values in y_true are not unique")

    if len(set(y_pred)) != len(y_pred):
        raise ValueError("Values in y_pred are not unique")

    if k_max != 0:
        y_pred = y_pred[:k_max]
        y_true = y_true[:k_max]

    correct_predictions = 0
    running_sum = 0

    for i, yp_item in enumerate(y_pred):

        k = i + 1  # our rank starts at 1

        if yp_item in y_true:
            correct_predictions += 1
            running_sum += correct_predictions / k

    return running_sum / len(y_true), correct_predictions
