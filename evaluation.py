# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import normalize


def cal_DET(features, labels):
    if features.shape[0] == features.shape[-1]:
        sim = features
    else:
        features = normalize(features)
        sim = np.dot(features, features.T)
    num = len(labels)
    sim = sim[~np.eye(num, dtype=np.bool)]
    bool = np.dot(labels, labels.T)
    bool = bool[~np.eye(num, dtype=np.bool)]

    fpr, tpr, thresholds = metrics.roc_curve(bool, sim)
    fnr = 1 - tpr

    eer = fpr[np.argmin(np.abs(fpr - fnr))]
    roc_auc = metrics.auc(fpr, tpr)
    return eer, fnr, tpr, roc_auc, thresholds


def plot_DET(data):
    plt.figure()
    lw = 2
    for eer, fpr, fnr, roc_auc in data:
        plt.semilogx(
            fpr,
            fnr,
            color=np.random.random(3),
            lw=lw,
            label='auc = {:.2f}, eer = {:.2f}%'.format(roc_auc, eer * 100))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('DET curve')
    plt.legend(loc="lower right")
    plt.show()
