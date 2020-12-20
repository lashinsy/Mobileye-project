import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from phase_I import run_attention


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
    red_x, red_y, green_x, green_y = run_attention.find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)



