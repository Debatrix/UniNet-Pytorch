# -*- coding: utf-8 -*-
"""
matching function for iris test
RenMin 20191024

Modified by Yunlong Wang, 2020.07.20
1. update OneHot function as labels are strings not intergers
2. labels = data_feature['labels']#.cuda(), labels not transferred to GPU
3. substitute self-programmed ROC curve function with sklearn.metrics 
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
from collections import OrderedDict


def BitExpand(features, bit=0):
    if bit != 0:
        W = features.shape[-1]
        if bit < 0:
            bit = W // 2
        left_part = features[..., :bit]
        right_part = features[..., W - bit:]
        features = torch.cat((right_part, features, left_part),
                             dim=-1).contiguous()
    return features


def conv_Hamming(feat1,
                 feat2,
                 mask1=None,
                 mask2=None,
                 shift_bits=10,
                 dtype=torch.float):
    H, W = feat2.shape[-2:]

    feat1 = feat1.to(dtype)
    feat2 = feat2.to(dtype)
    mask1 = mask1.to(dtype)
    mask2 = mask2.to(dtype)

    feat1 = BitExpand(feat1, shift_bits)
    feat1 = torch.nn.functional.unfold(feat1, (H, W)).unsqueeze(1)
    feat2 = torch.nn.functional.unfold(feat2, (H, W)).unsqueeze(0)

    if mask1 is None or mask2 is None:
        mask1 = torch.ones_like(feat1).to(dtype)
        mask2 = torch.ones_like(feat2).to(dtype)
    else:
        mask1 = BitExpand(mask1.to(dtype), shift_bits)
        mask1 = torch.nn.functional.unfold(mask1, (H, W)).unsqueeze(1)
        mask2 = torch.nn.functional.unfold(mask2.to(dtype),
                                           (H, W)).unsqueeze(0)

    mask = mask1 * mask2
    dist = (1 - (feat1 * feat2) - ((1 - feat1) * (1 - feat2))) * mask
    dist = (dist.sum(-2) / mask.sum(-2)).max(-1)[0]

    return dist


def conv_batch_Hamming(features, masks, batch_size=64, shift_bits=10):
    if masks is None:
        masks = torch.ones_like(features).to(features.dtype)
    sim = torch.zeros((features.shape[0], features.shape[0]))

    batch_num = features.shape[0] // batch_size
    batch_num = batch_num if features.shape[
        0] % batch_size == 0 else batch_num + 1

    for cols in tqdm(range(batch_num), ncols=80, ascii=True):
        for rows in range(batch_num):
            sim[cols * batch_size:(cols + 1) * batch_size,
                rows * batch_size:(rows + 1) * batch_size] = conv_Hamming(
                    features[cols * batch_size:(cols + 1) * batch_size],
                    features[rows * batch_size:(rows + 1) * batch_size],
                    masks[cols * batch_size:(cols + 1) * batch_size],
                    masks[rows * batch_size:(rows + 1) * batch_size],
                    shift_bits)
    return sim


################################################################################


def OneHot(x):
    # get one hot vectors, from x as a list of strings
    classes = tuple(set(x))
    n_class = len(classes)
    indx = [classes.index(sample) for sample in x]
    onehot = torch.eye(n_class).index_select(0, torch.tensor(indx))
    return onehot  # N X D


def RocCurve(scores, signals):
    FAR, TAR, T = roc_curve(signals, scores)
    FRR = 1 - TAR
    return FAR, FRR, T


def getEER(FAR, FRR, T):
    # get EER from roc curve
    FAR = np.array(FAR)
    FRR = np.array(FRR)
    T = np.array(T)
    gap = np.abs(FAR - FRR)
    index = np.where(gap == np.min(gap))
    EER = FRR[index][0]
    T_eer = T[index][0]
    return EER, T_eer


################################################################################


def MatchBinary(data_feature,
                shift_bits=10,
                batch_size=32,
                device='cuda:0',
                roc_res=10000):
    # get similarity scores and the signal of pairs for binary feature
    #
    # features: (N, H, W) or (N, 1, H, W) feature matrix, N is the number of samples
    # masks: (N, H, W) or (N, 1, H, W) feature matrix, N is the number of samples
    # labels:   N labels of the features

    features = data_feature['features'].to(device)
    masks = data_feature['masks'].to(device)
    labels = data_feature['labels']
    N = features.size(0)
    if len(features.shape) == 3:
        features = features.unsqueeze(1)
    elif len(features.shape) == 4:
        pass
    else:
        raise NotImplementedError

    if len(masks.shape) == 3:
        masks = masks.unsqueeze(1)
    elif len(masks.shape) == 4:
        pass
    else:
        raise NotImplementedError
    # -------------------------------------------------

    # Hamming distance is normalize to [0,1]
    with torch.no_grad():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        sim_mat = conv_batch_Hamming(features, masks, batch_size, shift_bits)

    one_hot = OneHot(labels)
    sig_mat = torch.mm(one_hot, one_hot.t())

    ind_keep = 1. - torch.eye(N)
    ind_keep = ind_keep.contiguous().view(-1)
    scores = sim_mat.contiguous().view(-1)
    signals = sig_mat.contiguous().view(-1)
    scores = scores[ind_keep > 0].numpy()
    signals = signals[ind_keep > 0].numpy()
    # -------------------------------------------------

    _, indices = torch.sort(sim_mat - 2.0 * torch.eye(N), descending=True)
    sig_mat_sort = torch.gather(sig_mat, 1, indices)
    sig_mat_rank1 = sig_mat_sort[..., :1]
    acc_rank1 = sig_mat_rank1.sum(1).norm(p=0) / N
    sig_mat_rank5 = sig_mat_sort[..., :5]
    acc_rank5 = sig_mat_rank5.sum(1).norm(p=0) / N
    sig_mat_rank10 = sig_mat_sort[..., :10]
    acc_rank10 = sig_mat_rank10.sum(1).norm(p=0) / N

    # -------------------------------------------------

    FAR, FRR, T = RocCurve(scores, signals)
    EER, T_eer = getEER(FAR, FRR, T)
    # --------------------------------------------------

    nrof_pos = signals.sum()
    nrof_neg = signals.size - nrof_pos

    acc_level = np.floor(np.log(nrof_neg) / np.log(10))

    FNMR_FMR = OrderedDict()

    for lv in range(int(acc_level)):
        fmr = pow(10, -1 * lv)
        idx = np.where(FAR <= fmr)[0]
        fnmr = FRR[idx[-1]]
        FNMR_FMR[fmr] = fnmr
    # --------------------------------------------------

    if FAR.size > roc_res:
        stride = FAR.size // roc_res + 1
        FAR = FAR[::stride]
        FRR = FRR[::stride]
        T = T[::stride]
    return FAR, FRR, T, EER, T_eer, FNMR_FMR, acc_rank1, acc_rank5, acc_rank10, sim_mat


if __name__ == "__main__":

    path = 'feature/feature_UniNet_ND_CASIA-Complex-CX3.pth'
    shift_bits = 10
    batch_size = 32
    device = 'cuda:1'

    feature_dict = torch.load(path)
    print('\nload data...')
    data_feature = {'features': [], 'masks': [], 'labels': []}
    for v in feature_dict.values():
        data_feature['features'].append(torch.tensor(v['template']))
        data_feature['labels'].append(v['label'])
        data_feature['masks'].append(torch.tensor(v['mask']))
    data_feature['features'] = torch.stack(data_feature['features'], 0)
    data_feature['masks'] = torch.stack(data_feature['masks'], 0)

    print('\nevaluate...')
    FAR, FRR, T, EER, T_eer, FNMR_FMR, acc_rank1, acc_rank5, acc_rank10, sim_mat = MatchBinary(
        data_feature, shift_bits, batch_size, device)
    DET_data = dict(FAR=FAR,
                    FRR=FRR,
                    T=T,
                    EER=EER,
                    T_eer=T_eer,
                    FNMR_FMR=FNMR_FMR,
                    acc_rank1=acc_rank1,
                    acc_rank5=acc_rank5,
                    acc_rank10=acc_rank10,
                    sim=sim_mat)

    torch.save(
        DET_data,
        'feature/evaluation_UniNet_{}_{}.pth'.format(*path.split('_')[-2:]))
    print('-' * 50)
    print('\nEER:{:.4f}%\nAcc: rank1 {:.4f}% rank5 {:.4f}% rank10 {:.4f}%'.
          format(EER * 100, acc_rank1 * 100, acc_rank5 * 100,
                 acc_rank10 * 100))
    print('-' * 50)
    for fmr, fnmr in FNMR_FMR.items():
        print('FNMR:{:.2f}%% @FMR:{:.2f}%%'.format(100.0 * fnmr, 100.0 * fmr))
    print('-' * 50)