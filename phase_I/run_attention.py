import os
import glob
import argparse
import numpy as np
from scipy import signal as sg
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from phase_I.test_tfl_attention import test_find_tfl_lights


def get_kernel() -> np.array:
    kernel = np.array([[-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
                    [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1],
                    [-1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, -1, -1],
                    [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, -1],
                    [1,  1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                    [-1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
                    [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1],
                    [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1]], float)
    return kernel*(1/225)


def get_coordinate(c_image: np.ndarray, kernel: np.array) -> [list, list]:
    dim = 15
    filtered_image = ndimage.maximum_filter(sg.convolve2d(c_image, kernel, boundary='symm', mode='same'), size=dim)
    x, y = [], []
    height, width = filtered_image.shape

    for i in range(dim // 2, height, dim):

        for j in range(dim // 2, width, dim):

            if filtered_image[i, j] > 25:
                y.append(i)
                x.append(j)

    return x, y


def find_tfl_lights(c_image: np.ndarray, **kwargs) -> [list, list, list, list]:
    kernel = get_kernel()
    red_x, red_y = get_coordinate(c_image[:, :, 0], kernel)
    green_x, green_y = get_coordinate(c_image[:, :, 1], kernel)

    return red_x, red_y, green_x, green_y


def main(argv=None):
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = 'images'

    if args.dir is None:
        args.dir = default_base
    images_list = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in images_list:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(images_list):
        print("You should now see some images, with the ground truth marked on them. Close all to quit")

    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()

