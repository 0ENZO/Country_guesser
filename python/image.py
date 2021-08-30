import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from PIL import Image
import tensorflow.keras as keras
from mlp import *

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# r"path\path"
DATASET_FOLDER = "C:/Users/Enzo/Documents/Github/Country_guesser/dataset/"
TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "train")
TEST_FOLDER = os.path.join(DATASET_FOLDER, "test")

TRAIN_FIRST_FOLDER = os.path.join(TRAIN_FOLDER, "jordanie")
TRAIN_SECOND_FOLDER = os.path.join(TRAIN_FOLDER, "palestine")
TRAIN_THIRD_FOLDER = os.path.join(TRAIN_FOLDER, " soudan")

TEST_FIRST_FOLDER = os.path.join(TEST_FOLDER, "jordanie")
TEST_SECOND_FOLDER = os.path.join(TEST_FOLDER, "palestine")
TEST_THIRD_FOLDER = os.path.join(TEST_FOLDER, "soudan")


def fill_x_and_y_with_images_and_labels(folder, x_list, y_list, label):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        image = Image.open(file_path)
        image = image.resize((32, 32))

        if image.mode == "RGBA":
            rgba = np.array(image)
            rgba[rgba[..., -1] == 0] = [255, 255, 255, 0]
            image = Image.fromarray(rgba)

        image = image.convert("RGB")
        im_arr = np.array(image).flatten()
        im_arr = im_arr / 255.0
        x_list.append(im_arr)
        y_list.append(label)


def import_dataset():
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    fill_x_and_y_with_images_and_labels(TRAIN_FIRST_FOLDER, X_train, Y_train, 0)
    fill_x_and_y_with_images_and_labels(TRAIN_SECOND_FOLDER, X_train, Y_train, 1)
    fill_x_and_y_with_images_and_labels(TRAIN_THIRD_FOLDER, X_train, Y_train, 2)

    fill_x_and_y_with_images_and_labels(TEST_FIRST_FOLDER, X_test, Y_test, 0)
    fill_x_and_y_with_images_and_labels(TEST_SECOND_FOLDER, X_test, Y_test, 1)
    fill_x_and_y_with_images_and_labels(TEST_THIRD_FOLDER, X_test, Y_test, 2)

    return (np.array(X_train).astype(np.float), np.array(Y_train).astype(np.float)), \
           (np.array(X_test).astype(np.float), np.array(Y_test).astype(np.float))


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = import_dataset()
