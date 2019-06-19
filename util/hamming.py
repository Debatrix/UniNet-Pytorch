from itertools import combinations

import numpy as np
from tqdm import tqdm


def feature_binarization(feature, mask, t=0.6):
    bin_feature = feature > feature.mean()
    bin_mask = np.logical_not(np.logical_and(mask, np.abs(feature - feature.mean()) < t))
    return np.logical_and(bin_feature, bin_mask)


def shiftbits(template, noshifts):
    """
    Description:
        Shift the bit-wise iris patterns.

    Input:
        template    - The template to be shifted.
        noshifts    - The number of shift operators, positive for right
                      direction and negative for left direction.

    Output:
        templatenew    - The shifted template.
    """
    # Initialize
    templatenew = np.zeros(template.shape)
    width = template.shape[1]
    s = 2 * np.abs(noshifts)
    p = width - s

    # Shift
    if noshifts == 0:
        templatenew = template

    elif noshifts < 0:
        x = np.arange(p)
        templatenew[:, x] = template[:, s + x]
        x = np.arange(p, width)
        templatenew[:, x] = template[:, x - p]

    else:
        x = np.arange(s, width)
        templatenew[:, x] = template[:, x - s]
        x = np.arange(s)
        templatenew[:, x] = template[:, p + x]

    return templatenew


def cal_hamming_dist(template1, template2, mask1=None, mask2=None):
    """
        Description:
            Calculate the Hamming distance between two iris templates.

        Input:
            template1    - The first template.
            mask1        - The first noise mask.
            template2    - The second template.
            mask2        - The second noise mask.

        Output:
            hd            - The Hamming distance as a ratio.
        """
    # Initialize
    hd = np.nan

    # Shift template left and right, use the lowest Hamming distance
    if mask1 is None or mask2 is None:
        for shifts in range(-8, 9):
            template1s = shiftbits(template1, shifts)

            hd1 = np.logical_xor(template1s, template2).sum() / template1s.size

            if hd1 < hd or np.isnan(hd):
                hd = hd1
    else:
        for shifts in range(-8, 9):
            template1s = shiftbits(template1, shifts)
            mask1s = shiftbits(mask1, shifts)

            mask = np.logical_or(mask1s, mask2)
            nummaskbits = np.sum(mask == 1)
            totalbits = template1s.size - nummaskbits

            C = np.logical_xor(template1s, template2)
            C = np.logical_and(C, np.logical_not(mask))
            bitsdiff = np.sum(C == 1)

            if totalbits == 0:
                hd = np.nan
            else:
                hd1 = bitsdiff / totalbits
                if hd1 < hd or np.isnan(hd):
                    hd = hd1

    return hd


def get_hmdist_mat(features, masks):
    num_feature = features.shape[0]
    hm_dists = np.zeros((num_feature, num_feature))
    pairs = [x for x in combinations([y for y in range(num_feature)], 2)]
    for x, y in tqdm(pairs, ncols=75, ascii=True):
        hm_dists[x, y] = cal_hamming_dist(features[x, :, :], features[y, :, :], masks[x, :, :], masks[y, :, :])

    hm_dists = hm_dists + hm_dists.T

    return hm_dists
