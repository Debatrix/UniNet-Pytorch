import cv2
import numpy as np


def stretchlim(img, tol=(0.0, 0.99)):
    tol_low = tol[0]
    tol_high = tol[1]

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cdf = np.cumsum(hist) / np.sum(hist)
    ilow = np.where(cdf > tol_low)[0]
    ihigh = np.where(cdf >= tol_high)[0]
    th = (ilow, ihigh)
    return th


def imadjust(img, th):
    img = np.clip(img, th[0], th[1])
    img = cv2.equalizeHist(img)
    return img


def adjust_iris(img, r_th=(0.05, 0.95)):
    th = stretchlim(img, (0.0, 0.99))
    rfl = img >= th[1]
    eps_shape = int(np.round(0.005 * img.size))
    rfl = cv2.morphologyEx(rfl, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (eps_shape, eps_shape)))
    rfl = cv2.dilate(rfl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    th = stretchlim(img * (1 - rfl), r_th)
    img = imadjust(img, th)
    return img


def enh_contrast(img):
    img = np.clip(img, stretchlim(img, (0.05, 0.95)))
    return cv2.equalizeHist(img)
