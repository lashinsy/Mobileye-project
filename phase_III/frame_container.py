import matplotlib.pyplot as plt
import numpy as np


class FrameContainer(object):
    def __init__(self, img_path):
        self.img = plt.imread(img_path)
        self.traffic_light = np.array
        self.traffic_lights_3d_location = []
        self.EM = np.ndarray
        self.corresponding_ind = []
        self.valid = []
