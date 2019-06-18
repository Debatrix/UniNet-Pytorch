# coding=utf-8
from random import choices

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def select_to_show(features, labels, num=10):
    select = []
    while len(select) < num:
        select.append(choices(labels)[0])
    select_index = [idx for idx, label in enumerate(labels) if label in select]
    select_features = [features[idx] for idx in select_index]
    select_labels = [labels[idx] for idx in select_index]
    return select_features, select_labels


def decomposition(labels, features, c_score=None):
    labels_set = list(set(labels))

    features = PCA(n_components=2).fit_transform(features)
    # features = TSNE(n_components=2).fit_transform(features)

    silhouette_score = metrics.silhouette_score(
        features, labels, metric='euclidean')
    calinski_harabaz_score = metrics.calinski_harabaz_score(features, labels)
    print(silhouette_score, calinski_harabaz_score)
    if c_score == None:
        c_score = (silhouette_score, calinski_harabaz_score)

    for lid, label in enumerate(labels_set):
        c = np.random.random(3)
        idxs = [idx for idx, sl in enumerate(labels) if sl == label]
        feature = np.zeros((len(idxs), 2))
        for x, y in enumerate(idxs):
            feature[x, :] = features[y, :]
        plt.scatter(feature[:, 0], feature[:, 1], color=c, label=label)
    plt.legend(ncol=2)
    plt.title("Clustering Result\nsilhouette score:{:.2f}\ncalinski harabaz score:{:.2f}".format(c_score[0],
                                                                                                 c_score[1]))
    plt.show()


def cal_DET(features, labels):
    if features.shape[0] == features.shape[-1]:
        sim = features
    else:
        features = normalize(features)
        sim = np.dot(features, features.T)
    num = len(labels)
    sim = np.delete(np.reshape(sim, (1, -1)), [x * x for x in range(num)])
    bool = np.dot(labels, labels.T)
    bool = np.delete(np.reshape(bool, (1, -1)), [x * x for x in range(num)])

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
