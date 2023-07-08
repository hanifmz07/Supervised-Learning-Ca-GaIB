import numpy as np


def f1_score(y_true, y_pred):
    fp = tp = fn = 0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    categories = np.unique(y_true)
    for i in range(len(y_true)):
        if y_true[i] == categories[0] and y_pred[i] == categories[0]:
            tp += 1
        elif y_true[i] == categories[1] and y_pred[i] == categories[0]:
            fp += 1
        elif y_true[i] == categories[0] and y_pred[i] == categories[1]:
            fn += 1
    return (2 * tp) / (2 * tp + fp + fn)
