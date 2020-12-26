import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np


def show_image(image: np.array) -> None:
    plt.imshow(image)
    plt.show()


def darken_image(image: np.array) -> np.array:
    contrast = iaa.GammaContrast(gamma=2.0)
    contrast_image = contrast.augment_image(image)

    return contrast_image


def bright_image(image: np.array) -> np.array:
    contrast = iaa.GammaContrast(gamma=0.5)
    contrast_image = contrast.augment_image(image)

    return contrast_image


def noise_image(image: np.array) -> np.array:
    gaussian_noise = iaa.AdditiveGaussianNoise(10, 20)
    noised_image = gaussian_noise.augment_image(image)

    return noised_image

