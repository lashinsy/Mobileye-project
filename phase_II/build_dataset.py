import os
import numpy as np
from PIL import Image
from phase_I.run_attention import find_tfl_lights
from phase_II.data_augmentation import darken_image, bright_image, noise_image


def get_images_list(dir_name: str) -> list:
    path = f'data/labelIds/{dir_name}/'
    img_list = [file for file in os.listdir(path)]

    return img_list


def crop_image(image: np.array, x_coordinate: int, y_coordinate: int) -> np.array:
    return image.crop((x_coordinate - 41, y_coordinate - 41, x_coordinate + 40, y_coordinate + 40))


def write_image_to_binary_file(image: np.array, label: int, dir_name: str) -> None:
    with open(f"dataset/{dir_name}/data.bin", "wb") as file_name:
        image.tofile(file_name, sep="", format="%s")
        write_label_to_binary_file(label, dir_name)


def write_label_to_binary_file(label: int, dir_name: str) -> None:
    with open(f"dataset/{dir_name}/labels.bin", "ab") as file_name:
        file_name.write(label.to_bytes(1, byteorder='big', signed=False))


def crop_image_by_coordinates(dir_name, origin_img, coordinates, label):
    x_coordinates, y_coordinates = coordinates
    x = x_coordinates[int(len(x_coordinates) / 2)]
    y = y_coordinates[int(len(x_coordinates) / 2)]
    cropped_img = crop_image(origin_img, y, x)
    write_image_to_binary_file(np.array(cropped_img).astype(np.uint8), label, dir_name)

    if dir_name == 'train':
        write_image_to_binary_file(np.fliplr(np.array(cropped_img)).astype(np.uint8), label, dir_name)
        write_image_to_binary_file(darken_image(cropped_img).astype(np.uint8), label, dir_name)
        write_image_to_binary_file(bright_image(cropped_img).astype(np.uint8), label, dir_name)
        write_image_to_binary_file(noise_image(cropped_img).astype(np.uint8), label, dir_name)


def find_not_tfl(image: np.array, tfl_coordinates: np.ndarray) -> tuple:
    tfl_suspicious = find_tfl_lights(image)
    tfl_suspicious = (tfl_suspicious[0] + tfl_suspicious[2], tfl_suspicious[1] + tfl_suspicious[3])
    not_tfl = ([], [])

    for x, y in zip(tfl_coordinates[0], tfl_coordinates[1]):

        for i, j in zip(tfl_suspicious[0], tfl_suspicious[1]):

            if x != i or y != j and all([m != 19 for m in image[x - 41: x + 41][y - 41: y + 41]]):
                not_tfl[0].append(i)
                not_tfl[1].append(j)

    return not_tfl


def prepare_dataset(dir_name: str) -> None:
    img_list = get_images_list(dir_name)

    for img_name in img_list:
        image = Image.open(f'data/labelIds/{img_name}')
        tfl_coordinates = np.where(np.array(image) == 19)

        if tfl_coordinates[0].any():
            origin_img = Image.open(f'data/leftImg8bit/{img_name[:-20]}_leftImg8bit.png')
            crop_image_by_coordinates(dir_name, origin_img, tfl_coordinates, 1)
            crop_image_by_coordinates(dir_name, origin_img, find_not_tfl(np.array(origin_img), tfl_coordinates), 0)


def set_data() -> None:
    prepare_dataset("train")
    prepare_dataset("val")
