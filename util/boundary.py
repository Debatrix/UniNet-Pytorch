# #-----------------------------------------------------------------------------
# #  Import
# #-----------------------------------------------------------------------------
import numpy as np
from scipy import signal


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def searchInnerBound(img):
    """
    Description:
        Search for the inner boundary of the iris.

    Input:
        img		- The input iris image.

    Output:
        inner_y	- y-coordinate of the inner circle centre.
        inner_x	- x-coordinate of the inner circle centre.
        inner_r	- Radius of the inner circle.
    """

    # Integro-Differential operator coarse (jump-level precision)
    Y = img.shape[0]
    X = img.shape[1]
    sect = X / 4  # Width of the external margin for which search is excluded
    minrad = 10
    maxrad = sect * 0.8
    jump = 4  # Precision of the coarse search, in pixels

    # Hough Space (y,x,r)
    sz = np.array([np.floor((Y - 2 * sect) / jump),
                   np.floor((X - 2 * sect) / jump),
                   np.floor((maxrad - minrad) / jump)]).astype(int)

    # Resolution of the circular integration
    integrationprecision = 1
    angs = np.arange(0, 2 * np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(sz[1]),
                          np.arange(sz[0]),
                          np.arange(sz[2]))
    y = sect + y * jump
    x = sect + x * jump
    r = minrad + r * jump
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative R
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2] - 1), 0, 0)]

    # Blur
    sm = 3  # Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm, sm, sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y, x, r = np.unravel_index(indmax, hspdrs.shape)

    inner_y = sect + y * jump
    inner_x = sect + x * jump
    inner_r = minrad + (r - 1) * jump

    # Integro-Differential operator fine (pixel-level precision)
    integrationprecision = 0.1  # Resolution of the circular integration
    angs = np.arange(0, 2 * np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(jump * 2),
                          np.arange(jump * 2),
                          np.arange(jump * 2))
    y = inner_y - jump + y
    x = inner_x - jump + x
    r = inner_r - jump + r
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative R
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2] - 1), 0, 0)]

    # Bluring
    sm = 3  # Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm, sm, sm]), mode="same")
    indmax = np.argmax(hspdrs.ravel())
    y, x, r = np.unravel_index(indmax, hspdrs.shape)

    inner_y = inner_y - jump + y
    inner_x = inner_x - jump + x
    inner_r = inner_r - jump + r - 1

    return inner_y, inner_x, inner_r


# ------------------------------------------------------------------------------
def searchOuterBound(img, inner_y, inner_x, inner_r):
    """
    Description:
        Search for the outer boundary of the iris.

    Input:
        img		- The input iris image.
        inner_y	- The y-coordinate of the inner circle centre.
        inner_x	- The x-coordinate of the inner circle centre.
        inner_r	- The radius of the inner circle.

    Output:
        outer_y	- y-coordinate of the outer circle centre.
        outer_x	- x-coordinate of the outer circle centre.
        outer_r	- Radius of the outer circle.
    """
    # Maximum displacement 15# (Daugman 2004)
    maxdispl = np.round(inner_r * 0.15).astype(int)

    # 0.1 - 0.8 (Daugman 2004)
    minrad = np.round(inner_r / 0.8).astype(int)
    maxrad = np.round(inner_r / 0.3).astype(int)

    # # Hough Space (y,x,r)
    # hs = np.zeros([2*maxdispl, 2*maxdispl, maxrad-minrad])

    # Integration region, avoiding eyelids
    intreg = np.array([[2 / 6, 4 / 6], [8 / 6, 10 / 6]]) * np.pi

    # Resolution of the circular integration
    integrationprecision = 0.05
    angs = np.concatenate([np.arange(intreg[0, 0], intreg[0, 1], integrationprecision),
                           np.arange(intreg[1, 0], intreg[1, 1], integrationprecision)],
                          axis=0)
    x, y, r = np.meshgrid(np.arange(2 * maxdispl),
                          np.arange(2 * maxdispl),
                          np.arange(maxrad - minrad))
    y = inner_y - maxdispl + y
    x = inner_x - maxdispl + x
    r = minrad + r
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative R
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2] - 1), 0, 0)]

    # Blur
    sm = 7  # Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm, sm, sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y, x, r = np.unravel_index(indmax, hspdrs.shape)

    outer_y = inner_y - maxdispl + y + 1
    outer_x = inner_x - maxdispl + x + 1
    outer_r = minrad + r - 1

    return outer_y, outer_x, outer_r


# ------------------------------------------------------------------------------
def ContourIntegralCircular(imagen, y_0, x_0, r, angs):
    """
    Description:
        Performs contour (circular) integral.
        Use discrete Rie-mann approach.

    Input:
        imagen  - The input iris image.
        y_0     - The y-coordinate of the circle centre.
        x_0     - The x-coordinate of the circle centre.
        r       - The radius of the circle.
        angs    - The region of the circle considering clockwise 0-2pi.

    Output:
        hs      - Integral result.
    """
    # Get y, x
    y = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    x = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    for i in range(len(angs)):
        ang = angs[i]
        y[i, :, :, :] = np.round(y_0 - np.cos(ang) * r).astype(int)
        x[i, :, :, :] = np.round(x_0 + np.sin(ang) * r).astype(int)

    # Adapt y
    ind = np.where(y < 0)
    y[ind] = 0
    ind = np.where(y >= imagen.shape[0])
    y[ind] = imagen.shape[0] - 1

    # Adapt x
    ind = np.where(x < 0)
    x[ind] = 0
    ind = np.where(x >= imagen.shape[1])
    x[ind] = imagen.shape[1] - 1

    # Return
    hs = imagen[y, x]
    hs = np.sum(hs, axis=0)
    return hs.astype(float)
