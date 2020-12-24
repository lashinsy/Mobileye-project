import os
import glob
import argparse
import numpy as np
import json
from PIL import Image
from scipy import signal as sg
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


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


def show_image_and_gt(image: np.array, objs: list, fig_num=None) -> None:
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()

    if objs is not None:

        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])

        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path: str, json_path=None, fig_num=None) -> None:
    image = np.array(Image.open(image_path))

    if json_path is None:
        objects = None

    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
    show_image_and_gt(image, objects, fig_num)
    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan data in')
    args = parser.parse_args(argv)
    default_base = 'data'

    if args.dir is None:
        args.dir = default_base
    images_list = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in images_list:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(images_list):
        print("You should now see some data, with the ground truth marked on them. Close all to quit")

    else:
        print("Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()

