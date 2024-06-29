import numpy as np
from sklearn import metrics


def metric(y_true, y_pred, y_prob, empty=-1):
    '''用于分类的评估，仅用于单任务评估
    :param y_true: 1-D, e.g. [1, 0, 1, 1]
    :param y_pred: 1-D, e.g. [0, 0, 1, 1]
    :param y_prob: 1-D, e.g. [0.7, 0.5, 0.2, 0.7]
    :return:
    '''
    assert len(y_true) == len(y_pred) == len(y_prob)

    y_true, y_pred, y_prob = np.array(y_true).flatten(), np.array(y_pred).flatten(), np.array(y_prob).flatten()

    # filter empty data
    flag = y_true != empty
    y_true, y_pred, y_prob = y_true[flag], y_pred[flag], y_prob[flag]

    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_prob)
    precision_list, recall_list, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = metrics.auc(recall_list, precision_list)

    return {
        "accuracy": acc,
        "ROCAUC": auc,
        "AUPR": aupr,
    }

