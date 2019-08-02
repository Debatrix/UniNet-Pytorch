import os
import pickle
import warnings
from fnmatch import filter
from itertools import repeat
from multiprocessing import Pool, cpu_count
from os import listdir

import numpy as np
import scipy.io as sio

warnings.filterwarnings("ignore")


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

    # Return
    return templatenew


def calHammingDist(template1, mask1, template2, mask2):
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
    for shifts in range(-8, 9):
        template1s = shiftbits(template1, shifts)
        mask1s = shiftbits(mask1, shifts)

        mask = np.logical_or(mask1s, mask2)
        nummaskbits = np.sum(mask == 1)
        totalbits = template1s.size - nummaskbits

        C = np.logical_xor(template1s, template2)
        C = np.logical_and(C, mask)
        bitsdiff = np.sum(C == 1)

        if totalbits == 0:
            hd = np.nan
        else:
            hd1 = bitsdiff / totalbits
            if hd1 < hd or np.isnan(hd):
                hd = hd1

    # Return
    return hd


def matching(template_extr, mask_extr, temp_dir, threshold=0.38, cache=None):
    """
    Description:
        Match the extracted template with database.

    Input:
        template_extr    - Extracted template.
        mask_extr        - Extracted mask.
        threshold        - Threshold of distance.
        temp_dir         - Directory contains templates.
        cache            - Filename of cache file.

    Output:
        List of strings of matched files, 0 if not, -1 if no registered sample.
    """

    if cache is not None:
        if os.path.exists(cache):
            # Load Hamming distances
            with open(cache, 'rb') as f:
                mdict = pickle.load(f)
            hm_dists = mdict['hm_dists']
            filenames = mdict['filenames']
        else:
            # Get the number of accounts in the database
            n_files = len(filter(listdir(temp_dir), '*.mat'))
            if n_files == 0:
                return -1

            # Use all cores to calculate Hamming distances
            args = zip(
                sorted(listdir(temp_dir)),
                repeat(template_extr),
                repeat(mask_extr),
                repeat(temp_dir),
            )

            with Pool(processes=cpu_count()) as pools:
                result_list = pools.starmap(matchingPool, args)

            filenames = [result_list[i][0] for i in range(len(result_list))]
            hm_dists = np.array(
                [result_list[i][1] for i in range(len(result_list))])
            mdict = {'hm_dists': hm_dists, 'filenames': filenames}
            with open(cache, 'wb') as f:
                pickle.dump(mdict, f)
    else:
        # Get the number of accounts in the database
        n_files = len(filter(listdir(temp_dir), '*.mat'))
        if n_files == 0:
            return -1

        # Use all cores to calculate Hamming distances
        args = zip(
            sorted(listdir(temp_dir)),
            repeat(template_extr),
            repeat(mask_extr),
            repeat(temp_dir),
        )

        with Pool(processes=cpu_count()) as pools:
            result_list = pools.starmap(matchingPool, args)

        filenames = [result_list[i][0] for i in range(len(result_list))]
        hm_dists = np.array(
            [result_list[i][1] for i in range(len(result_list))])

    # Remove NaN elements
    ind_valid = np.where(hm_dists > 0)[0]
    hm_dists = hm_dists[ind_valid]
    filenames = [filenames[idx] for idx in ind_valid]

    # Threshold and give the result ID
    ind_thres = np.where(hm_dists <= threshold)[0]

    # Return
    if len(ind_thres) == 0:
        return 0
    else:
        hm_dists = hm_dists[ind_thres]
        filenames = [filenames[idx] for idx in ind_thres]
        ind_sort = np.argsort(hm_dists)
        return [filenames[idx] for idx in ind_sort]


def matchingPool(file_temp_name, template_extr, mask_extr, temp_dir):
    """
    Description:
        Perform matching session within a Pool of parallel computation

    Input:
        file_temp_name    - File name of the examining template
        template_extr    - Extracted template
        mask_extr        - Extracted mask of noise

    Output:
        hm_dist            - Hamming distance
    """
    # Load each account
    data_template = sio.loadmat(os.path.join(temp_dir, file_temp_name))
    template = data_template['template']
    mask = data_template['mask']

    # Calculate the Hamming distance
    hm_dist = calHammingDist(template_extr, mask_extr, template, mask)
    return (file_temp_name, hm_dist)
