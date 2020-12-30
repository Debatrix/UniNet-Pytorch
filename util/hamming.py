# %%
import tqdm
import time
import torch
import torch.nn.functional as F


# %%
def BitShift(features, bit=0):
    W = features.shape[-1]
    if bit > 0:
        left_part = features[..., :W - bit]
        right_part = features[..., W - bit:]
        features = torch.cat((right_part, left_part), dim=-1).contiguous()
    elif bit < 0:
        bit = -bit
        left_part = features[..., :bit]
        right_part = features[..., bit:]
        features = torch.cat((right_part, left_part), dim=-1).contiguous()
    return features


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


# %%
def cal_Hamming(feat1, feat2, mask1=None, mask2=None):
    if mask1 is None or mask2 is None:
        mask1 = torch.ones_like(feat1).to(torch.bool)
        mask2 = torch.ones_like(feat2).to(torch.bool)
    mask = torch.logical_and(mask1, mask2)
    dist = torch.logical_and(torch.logical_xor(feat1, feat2), mask).to(
        torch.float).sum() / mask.to(torch.float).sum()
    return dist


def cal_batch_Hamming(features, masks, shift_bits=10):
    if masks is None:
        masks = torch.ones_like(features).to(features.dtype)
    dist = torch.zeros(features.shape[0], features.shape[0])

    for x in range(features.shape[0]):
        for y in range(features.shape[0]):
            _dist = []
            for bit in range(-shift_bits, shift_bits + 1):
                _feature = BitShift(features[x], bit)
                _mask = BitShift(masks[x], bit)
                _dist.append(
                    cal_Hamming(_feature, features[y], _mask, masks[y]))
            dist[x][y] = torch.max(torch.tensor(_dist))
    return dist


# %%
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

    for cols in tqdm.tqdm(range(batch_num)):
        for rows in range(batch_num):
            sim[cols * batch_size:(cols + 1) * batch_size,
                rows * batch_size:(rows + 1) * batch_size] = conv_Hamming(
                    features[cols * batch_size:(cols + 1) * batch_size],
                    features[rows * batch_size:(rows + 1) * batch_size],
                    masks[cols * batch_size:(cols + 1) * batch_size],
                    masks[rows * batch_size:(rows + 1) * batch_size],
                    shift_bits)
    return sim


# %%
def _test(test=1, dataset=[8]):
    for data_num in dataset:
        t_xor, t_conv = 0, 0
        err = 0
        conv, xor = 0, 0
        for _ in range(test):
            torch.cuda.empty_cache()
            feat = (torch.rand(data_num, 1, 64, 512) > 0.5).cuda()
            mask = (torch.rand(data_num, 1, 64, 512) > 0.5).cuda()

            s1 = time.time()
            xor = cal_batch_Hamming(feat, None).cpu()
            s2 = time.time()
            with torch.no_grad():
                conv = conv_batch_Hamming(feat, None, 16).cpu()
            s3 = time.time()

            t_xor += s2 - s1
            t_conv += s3 - s2
            err += torch.abs(xor - conv).mean()
        print('\ndatanum:{} xor:{:.4f}s conv:{:.4f}s err:{:.2e}'.format(
            data_num, t_xor / test, t_conv / test, err))


if __name__ == "__main__":
    _test()

# %%
