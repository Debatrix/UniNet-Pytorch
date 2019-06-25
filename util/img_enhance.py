import cv2
import numpy as np
import bisect


def stretchlim(img, tol=(0.0, 0.99)):
    tol_low = tol[0]
    tol_high = tol[1]

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cdf = np.cumsum(hist) / np.sum(hist)
    ilow = np.where(cdf > tol_low)[0]
    ihigh = np.where(cdf >= tol_high)[0]
    th = (ilow, ihigh)
    return th


def imadjust(src, tol=1, vin=[0, 255], vout=(0, 255)):
    # https://stackoverflow.com/questions/39767612/what-is-the-equivalent-of-matlabs-imadjust-in-python/44529776
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2, 'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src, bins=list(range(256)), range=(0, 255))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, 256):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src - vin[0]
    vs[src < vin[0]] = 0
    vd = vs * scale + 0.5 + vout[0]
    vd[vd > vout[1]] = vout[1]
    dst = vd

    return dst


def adjust_iris(img, r_th=(0.05, 0.95)):
    th = stretchlim(img, (0.0, 0.99))
    rfl = img >= th[1]
    eps_shape = int(np.round(0.005 * img.size))
    rfl = cv2.morphologyEx(
        rfl, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (eps_shape, eps_shape)))
    rfl = cv2.dilate(rfl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    th = stretchlim(img * (1 - rfl), r_th)
    img = imadjust(img, th)
    return img


def enh_contrast(img):
    img = np.clip(img, stretchlim(img, (0.05, 0.95)))
    return cv2.equalizeHist(img)
