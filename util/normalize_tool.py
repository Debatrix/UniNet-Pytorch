import os
from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np


def normalize(image, x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil,
              radpixels, angulardiv):
    """
	Description:
		Normalize iris region by unwraping the circular region into a rectangular
		block of constant dimensions.

	Input:
		image		- Input iris image.

		x_iris		- x-coords of the circle defining the iris boundary.
		y_iris		- y-coords of the circle defining the iris boundary.
		r_iris		- Radius of the circle defining the iris boundary.

		x_pupil		- x-coords of the circle defining the pupil boundary.
		y_pupil		- y-coords of the circle defining the pupil boundary.
		r_pupil		- Radius of the circle defining the pupil boundary.

		radpixels	- Radial resolution (vertical dimension).
		angulardiv	- Angular resolution (horizontal dimension).

	Output:
		polar_array	- Normalized form of the iris region.
		polar_noise	- Normalized form of the noise region.
	"""
    radiuspixels = radpixels + 2
    angledivisions = angulardiv - 1

    r = np.arange(radiuspixels)
    theta = np.linspace(0, 2 * np.pi, angledivisions + 1)

    # Calculate displacement of pupil center from the iris center
    ox = x_pupil - x_iris
    oy = y_pupil - y_iris

    if ox <= 0:
        sgn = -1
    elif ox > 0:
        sgn = 1

    if ox == 0 and oy > 0:
        sgn = 1

    a = np.ones(angledivisions + 1) * (ox ** 2 + oy ** 2)

    # Need to do something for ox = 0
    if ox == 0:
        phi = np.pi / 2
    else:
        phi = np.arctan(oy / ox)

    b = sgn * np.cos(np.pi - phi - theta)

    # Calculate radius around the iris as a function of the angle
    r = np.sqrt(a) * b + np.sqrt(a * b ** 2 - (a - r_iris ** 2))
    r = np.array([r - r_pupil])

    rmat = np.dot(np.ones([radiuspixels, 1]), r)

    rmat = rmat * np.dot(
        np.ones([angledivisions + 1, 1]),
        np.array([np.linspace(0, 1, radiuspixels)])).transpose()
    rmat = rmat + r_pupil

    # Exclude values at the boundary of the pupil iris border, and the iris scelra border
    # as these may not correspond to areas in the iris region and will introduce noise.
    # ie don't take the outside rings as iris data.
    rmat = rmat[1:radiuspixels - 1, :]

    # Calculate cartesian location of each data point around the circular iris region
    xcosmat = np.dot(np.ones([radiuspixels - 2, 1]), np.array([np.cos(theta)]))
    xsinmat = np.dot(np.ones([radiuspixels - 2, 1]), np.array([np.sin(theta)]))

    xo = rmat * xcosmat
    yo = rmat * xsinmat

    xo = x_pupil + xo
    xo = np.round(xo).astype(int)
    coords = np.where(xo >= image.shape[1])
    xo[coords] = image.shape[1] - 1
    coords = np.where(xo < 0)
    xo[coords] = 0

    yo = y_pupil - yo
    yo = np.round(yo).astype(int)
    coords = np.where(yo >= image.shape[0])
    yo[coords] = image.shape[0] - 1
    coords = np.where(yo < 0)
    yo[coords] = 0

    # Extract intensity values into the normalised polar representation through
    # interpolation
    # x,y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    # f = interpolate.interp2d(x, y, image, kind='linear')
    # polar_array = f(xo, yo)
    # polar_array = polar_array / 255

    polar_array = image[yo, xo]
    polar_array = polar_array / 255

    # Create noise array with location of NaNs in polar_array
    polar_noise = np.zeros(polar_array.shape)
    coords = np.where(np.isnan(polar_array))
    polar_noise[coords] = 1

    # Get rid of outling points in order to write out the circular pattern
    image[yo, xo] = 255

    # Get pixel coords for circle around iris
    x, y = circlecoords([x_iris, y_iris], r_iris, image.shape)
    image[y, x] = 255

    # Get pixel coords for circle around pupil
    xp, yp = circlecoords([x_pupil, y_pupil], r_pupil, image.shape)
    image[yp, xp] = 255

    # Replace NaNs before performing feature encoding
    coords = np.where((np.isnan(polar_array)))
    polar_array2 = polar_array
    polar_array2[coords] = 0.5
    avg = np.sum(polar_array2) / (polar_array.shape[0] * polar_array.shape[1])
    polar_array[coords] = avg

    return polar_array, polar_noise.astype(bool)


def circlecoords(c, r, imgsize, nsides=600):
    """
	Description:
		Find the coordss of a circle based on its centre and radius.

	Input:
		c   	- Centre of the circle.
		r  		- Radius of the circle.
		imgsize - Size of the image that the circle will be plotted onto.
		nsides 	- Number of sides of the convex-hull bodering the circle
				  (default as 600).

	Output:
		x,y     - Circle coordss.
	"""
    a = np.linspace(0, 2 * np.pi, 2 * nsides + 1)
    xd = np.round(r * np.cos(a) + c[0])
    yd = np.round(r * np.sin(a) + c[1])

    #  Get rid of values larger than image
    xd2 = xd
    coords = np.where(xd >= imgsize[1])
    xd2[coords[0]] = imgsize[1] - 1
    coords = np.where(xd < 0)
    xd2[coords[0]] = 0

    yd2 = yd
    coords = np.where(yd >= imgsize[0])
    yd2[coords[0]] = imgsize[0] - 1
    coords = np.where(yd < 0)
    yd2[coords[0]] = 0

    x = np.round(xd2).astype(int)
    y = np.round(yd2).astype(int)
    return x, y


def on_mouse1(event, x, y, flags, param):
    global img_s, points, win, coords
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point = (x, y)
        cv2.circle(img_s, point, 4, (255, 0, 255), -1)
        cv2.imshow(win, img_s)
        points.append(point)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 2:
            center, radius = cv2.minEnclosingCircle(np.array(points))
            coords = [int(center[0]), int(center[1]), int(radius)]
            cv2.circle(img_s, (coords[0], coords[1]), coords[2], (0, 255, 255),
                       2)
            cv2.circle(img_s, (coords[0], coords[1]), 4, (0, 255, 255), -1)
            cv2.imshow(win, img_s)


def on_mouse2(event, x, y, flags, param):
    global img_s, points, win, coords
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point = (x, y)
        cv2.circle(img_s, point, 4, (255, 0, 0), -1)
        cv2.imshow(win, img_s)
        points.append(point)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 2:
            center, radius = cv2.minEnclosingCircle(np.array(points))
            coords = [int(center[0]), int(center[1]), int(radius)]
            cv2.circle(img_s, (coords[0], coords[1]), coords[2], (0, 255, 0),
                       2)
            cv2.circle(img_s, (coords[0], coords[1]), 4, (0, 255, 0), -1)
            cv2.imshow(win, img_s)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default='./pic')
    parser.add_argument("-e", "--ext", type=str, default='bmp')
    parser.add_argument("-s", "--size", type=tuple, default=(70, 540))
    args = parser.parse_args()

    result_path = os.path.join(args.path, 'Result/NormIm')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    norm_result = []

    imglist = glob(os.path.join(args.path, '*.' + args.ext))

    for idx, img_path in enumerate(imglist, 1):
        img_name = os.path.split(img_path)[-1]
        global img_s, points, win, coords
        img = cv2.imread(img_path, 0)
        img1 = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR)
        coordinate = []
        k = None
        while k != ord('q'):
            coords = None
            win = '{}/{}-iris'.format(idx, len(imglist))
            img_s = img1.copy()
            points = []
            cv2.namedWindow(win)
            cv2.setMouseCallback(win, on_mouse1)
            cv2.imshow(win, img_s)
            k = cv2.waitKey(0)
        coordinate.append(coords)
        cv2.destroyAllWindows()
        k = None
        img1 = img_s.copy()
        while k != ord('q'):
            coords = None
            win = '{}/{}-pupil'.format(idx, len(imglist))
            points = []
            img_s = img1.copy()
            cv2.namedWindow(win)
            cv2.setMouseCallback(win, on_mouse2)
            cv2.imshow(win, img_s)
            k = cv2.waitKey(0)
        coordinate.append(coords)
        cv2.destroyAllWindows()

        try:
            coordinate = sorted(coordinate, key=lambda x: x[-1])
            x_iris, y_iris, r_iris = coordinate[1]
            x_pupil, y_pupil, r_pupil = coordinate[0]
            print('{}, 0 {} {} {} {} {} {}'.format(
                img_name, x_pupil, y_pupil, r_pupil, x_iris, y_iris, r_iris))
            norm, _ = normalize(img, x_iris, y_iris, r_iris, x_pupil, y_pupil,
                                r_pupil, args.size[0], args.size[1])
            cv2.imwrite(
                os.path.join(result_path, img_name),
                (norm * 255).astype(np.uint8))
            norm_result.append('{}, 0 {} {} {} {} {} {}'.format(
                img_name, x_pupil, y_pupil, r_pupil, x_iris, y_iris, r_iris))
        except:
            print('ERROR:{}'.format(img_path))
            norm_result.append('{}, -1 -1 -1 -1 -1 -1 -1'.format(img_name))

    with open(os.path.join(result_path, 'LocResult.txt'), 'w') as f:
        for line in norm_result:
            f.writelines(line + '\n')
