import numpy as np


def sub_accuracy_score(y_true, y_pred):
    y_true = [''.join([str(yi) for yi in y]) for y in y_true]
    y_pred = [''.join([str(yi) for yi in y]) for y in y_pred]
    return sum([y_true[i] == y_pred[i] for i in range(len(y_true))])/len(y_true)


def accuracy_score(y_true, y_pred):
    if isinstance(y_true, list):
        y_true = np.array([np.array(y) for y in y_true])
    if isinstance(y_pred, list):
        y_pred = np.array([np.array(y) for y in y_pred])

    return np.mean(np.sum(y_true*y_pred, axis=1) / np.sum((y_true+y_pred) > 0, axis=1))


def recall_score(y_true, y_pred):
    if isinstance(y_true, list):
        y_true = np.array([np.array(y) for y in y_true])
    if isinstance(y_pred, list):
        y_pred = np.array([np.array(y) for y in y_pred])

    return np.mean(np.sum(y_true * y_pred, axis=1) / np.sum(y_true, axis=1))


def precision_score(y_true, y_pred):
    if isinstance(y_true, list):
        y_true = np.array([np.array(y) for y in y_true])
    if isinstance(y_pred, list):
        y_pred = np.array([np.array(y) for y in y_pred])

    recall = np.mean(np.sum(y_true * y_pred, axis=1) / np.sum(y_pred, axis=1))
    return recall


def f1_score(y_true, y_pred):
    if isinstance(y_true, list):
        y_true = np.array([np.array(y) for y in y_true])
    if isinstance(y_pred, list):
        y_pred = np.array([np.array(y) for y in y_pred])

    return np.mean(2 * np.sum(y_true * y_pred, axis=1) / (np.sum(y_true, 1) + np.sum(y_true, 1)))


def hamming_loss(y_true, y_pred):
    if isinstance(y_true, list):
        y_true = np.array([np.array(y) for y in y_true])
    if isinstance(y_pred, list):
        y_pred = np.array([np.array(y) for y in y_pred])

    n_sample, n_out = y_true.shape
    return np.mean(np.sum(np.bitwise_xor(y_true, y_pred), axis=1)/n_out)


def get_multi_label_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)
    sub_acc = sub_accuracy_score(y_true, y_pred)
    return acc, sub_acc, f1, precision, recall, hamming