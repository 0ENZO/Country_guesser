import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from PIL import Image
#import tensorflow.keras as keras
from mlp import *
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "train")
TEST_FOLDER = os.path.join(DATASET_FOLDER, "test")

TRAIN_FIRST_FOLDER = os.path.join(TRAIN_FOLDER, classes[0])
TRAIN_SECOND_FOLDER = os.path.join(TRAIN_FOLDER, classes[1])
TRAIN_THIRD_FOLDER = os.path.join(TRAIN_FOLDER, classes[1])

TEST_FIRST_FOLDER = os.path.join(TEST_FOLDER, classes[0])
TEST_SECOND_FOLDER = os.path.join(TEST_FOLDER, classes[1])
TEST_THIRD_FOLDER = os.path.join(TEST_FOLDER, classes[2])


def fill_x_and_y_with_images_and_labels(folder, x_list, y_list, label):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        image = Image.open(file_path)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        if image.mode == "RGBA":
            rgba = np.array(image)
            rgba[rgba[..., -1] == 0] = [255, 255, 255, 0]
            image = Image.fromarray(rgba)

        image = image.convert("RGB")
        im_arr = np.array(image).flatten()
        # print(im_arr.shape)
        im_arr = im_arr / 255.0
        x_list.append(im_arr)
        y_list.append(label)


def import_dataset():
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    fill_x_and_y_with_images_and_labels(TRAIN_FIRST_FOLDER, X_train, Y_train, [1, -1, -1])
    fill_x_and_y_with_images_and_labels(TRAIN_SECOND_FOLDER, X_train, Y_train, [-1, 1, -1])
    fill_x_and_y_with_images_and_labels(TRAIN_THIRD_FOLDER, X_train, Y_train, [-1, -1, 1])

    fill_x_and_y_with_images_and_labels(TEST_FIRST_FOLDER, X_test, Y_test, [1, -1, -1])
    fill_x_and_y_with_images_and_labels(TEST_SECOND_FOLDER, X_test, Y_test, [-1, 1, -1])
    fill_x_and_y_with_images_and_labels(TEST_THIRD_FOLDER, X_test, Y_test, [-1, -1, 1])

    return (np.array(X_train).astype(np.float), np.array(Y_train).astype(np.float)), \
           (np.array(X_test).astype(np.float), np.array(Y_test).astype(np.float))


def train():
    print("Dataset import en cours")
    (X_train, Y_train), (X_test, Y_test) = import_dataset()
    print("Dataset importé")

    np_arr = np.array([len(X_train[0]), 3, 3])
    npl = np.ctypeslib.as_ctypes(np_arr)
    model = create_mlp_model(npl)

    predicted_train_outputs_before_training = [predict_mlp_model_classification(model, x, 3) for x in X_train]
    cpt = 0
    for p, y in zip(predicted_train_outputs_before_training, Y_train):
        if np.argmax(p) == np.argmax(y):
            cpt += 1
    print("Dataset d'entraînement, avant entraînement : ", cpt / len(predicted_train_outputs_before_training) * 100, "%")

    train_classification_stochastic_backprop_mlp_model(model, X_train.flatten( ), Y_train.flatten(), alpha=0.01, epochs=2000000)

    predicted_train_outputs_after_training = [predict_mlp_model_classification(model, x, 3) for x in X_train]
    cpt = 0
    for p, y in zip(predicted_train_outputs_after_training, Y_train):
        if np.argmax(p) == np.argmax(y):
            cpt += 1
    print("Dataset d'entraînement, après entraînement : ", cpt / len(predicted_train_outputs_after_training) * 100, "%")

    predicted_test_outputs_after_training = [predict_mlp_model_classification(model, x, 3) for x in X_test]
    cpt = 0
    for p, y in zip(predicted_test_outputs_after_training, Y_test):
        if np.argmax(p) == np.argmax(y):
            cpt += 1
    print("Dataset de test, après entraînement : ", cpt / len(predicted_test_outputs_after_training) * 100, "%")

    save_mlp_model(model, "2m_32px")
    destroy_mlp_model(model)


def load():
    (X_train, Y_train), (X_test, Y_test) = import_dataset()
    model = load_mlp_model("2m_32px_03_09_14_40")

    predicted_train_outputs_after_training = [predict_mlp_model_classification(model, x, 3) for x in X_train]
    cpt = 0
    for p, y in zip(predicted_train_outputs_after_training, Y_train):
        if np.argmax(p) == np.argmax(y):
            cpt += 1
    print("Dataset d'entraînement, après entraînement : ", cpt / len(predicted_train_outputs_after_training) * 100, "%")

    predicted_test_outputs_after_training = [predict_mlp_model_classification(model, x, 3) for x in X_test]
    cpt = 0
    for p, y in zip(predicted_test_outputs_after_training, Y_test):
        if np.argmax(p) == np.argmax(y):
            cpt += 1
    print("Dataset de test, après entraînement : ", cpt / len(predicted_test_outputs_after_training) * 100, "%")

    destroy_mlp_model(model)

if __name__ == "__main__":
    # train()
    load()
