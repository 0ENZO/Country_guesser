import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from PIL import Image
#import tensorflow.keras as keras
from mlp import *
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
classes = ["jordanie", "palestine", "soudan"]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# r"path\path"
DATASET_FOLDER = "C:/Users/Erwan san/Desktop/Country_guesser/dataset"
TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "train")
TEST_FOLDER = os.path.join(DATASET_FOLDER, "test")

TRAIN_FIRST_FOLDER = os.path.join(TRAIN_FOLDER, "jordanie")
TRAIN_SECOND_FOLDER = os.path.join(TRAIN_FOLDER, "palestine")
TRAIN_THIRD_FOLDER = os.path.join(TRAIN_FOLDER, "soudan")

TEST_FIRST_FOLDER = os.path.join(TEST_FOLDER, "jordanie")
TEST_SECOND_FOLDER = os.path.join(TEST_FOLDER, "palestine")
TEST_THIRD_FOLDER = os.path.join(TEST_FOLDER, "soudan")


def fill_x_and_y_with_images_and_labels(folder, x_list, y_list, label):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        image = Image.open(file_path)
        image = image.resize((16, 16))

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


def run():

    # print(len(X_train[0]))
    # print(X_train[0])
    # print(len(X_train))
    # print(X_train)
    # print("Fin print dataset")
    # print(predict_mlp_model_classification(model, X_train[0], 3))
    # print(len(predicted_outputs))
    # print(predicted_outputs)
    # print(predicted_outputs[-1])

    print("Dataset import en cours")
    (X_train, Y_train), (X_test, Y_test) = import_dataset()
    print("Dataset importé")

    print("Création du modèle")
    # np_arr = np.array([2, 3, 3])
    np_arr = np.array([len(X_train[0]), 3, 3])
    npl = np.ctypeslib.as_ctypes(np_arr)
    model = create_mlp_model(npl)
    print("Modèle créé")

    print("Calcul des prédictions avant entraînement")
    predicted_outputs = [predict_mlp_model_classification(model, x, 3) for x in X_train]
    print("Fin des prédictions")

    cpt = 0
    for p, y in zip(predicted_outputs, Y_test):
        if np.argmax(p) == np.argmax(y):
            cpt += 1

    print("Avant entraînement : ", cpt / len(predicted_outputs) * 100, "%")

    print("Début entraînement")
    train_classification_stochastic_backprop_mlp_model(model, X_train.flatten( ), Y_train.flatten(), epochs=1000000)
    print("Fin entraînement")

    print("Calcul des prédictions avant entraînement")
    outputs_after_training = [predict_mlp_model_classification(model, x, 3) for x in X_train]
    print("Fin des prédictions")

    cpt = 0
    for p, y in zip(outputs_after_training, Y_test):
        if np.argmax(p) == np.argmax(y):
            cpt += 1

    print("Après entraînement : ", cpt / len(outputs_after_training) * 100, "%")

    destroy_mlp_model(model)
    exit(0)

    # flattened_dataset_inputs = []
    # for p in X_train:
    #     for x in p:
    #         flattened_dataset_inputs.append(x)
    #
    # flattened_dataset_outputs = []
    # for p in Y_train:
    #     for y in p:
    #         flattened_dataset_outputs.append(x)
    #
    # flattened_dataset_inputs = []
    # for x in dataset_input:
    #     flattened_dataset_inputs.append(x[0])
    #     flattened_dataset_inputs.append(x[1])
    #
    # train_classification_stochastic_backprop_mlp_model(model, flattened_dataset_inputs, flattened_dataset_outputs)

    # model = keras.models.Sequential()
    # model.add(keras.layers.Dense(3, activation=keras.activations.softmax))
    # model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.mse)
    # model.fit(X_train, Y_train, epochs=10)
    # print(model.predict(X_train))
    # print(model.predict(X_test))

    dataset_inputs = [
        [0, 0],
        [0.5, 0.5],
        [1, 0],
    ]

    dataset_expected_outputs = [
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]

    np_arr = np.array([2, 3, 3])
    npl = np.ctypeslib.as_ctypes(np_arr)
    model = create_mlp_model(npl)

    test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-10, 20) for x2 in range(-10, 20)]
    colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for output in dataset_expected_outputs]

    print(predict_mlp_model_classification(model, test_dataset[0], 3))
    predicted_outputs = [predict_mlp_model_classification(model, p, 3) for p in test_dataset]
    print(len(predicted_outputs))
    print(predicted_outputs)
    print(predicted_outputs[-1])
    # predicted_outputs_colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for output in predicted_outputs[-1][1:]]
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    # plt.show()
    #
    # flattened_dataset_inputs = []
    # for p in dataset_inputs:
    #     flattened_dataset_inputs.append(p[0])
    #     flattened_dataset_inputs.append(p[1])
    #
    # flattened_dataset_outputs = []
    # for p in dataset_expected_outputs:
    #     flattened_dataset_outputs.append(p[0])
    #     flattened_dataset_outputs.append(p[1])
    #     flattened_dataset_outputs.append(p[2])
    #
    # train_classification_stochastic_gradient_backpropagation_mlp_model(model, flattened_dataset_inputs, flattened_dataset_outputs)
    #
    # predicted_outputs = [predict_mlp_model_classification(model, p) for p in test_dataset]
    # predicted_outputs_colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for output in predicted_outputs]
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    # plt.show()

    destroy_mlp_model(model)

def ran():
    (X_train, y_train), (X_test, y_test) = import_dataset()
    dataset_inputs = np.array(X_train)
    print(X_train.shape)

    for p in dataset_inputs:
        print("Nouveau :")
        print(p[0])
        print(p[1])
        print("Fin \n")
    print (dataset_inputs)
    print()
    print(dataset_inputs.shape)
    exit(0)
    dataset_expected_outputs = np.array(y_train)

    np_arr = np.array([2, 3, 3])
    npl = np.ctypeslib.as_ctypes(np_arr)
    model = create_mlp_model(npl)

    test_dataset = X_test
    img_test = [X_test[0]]
    colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for output in dataset_expected_outputs]

    mylib.getXSize.argtypes = [c_void_p]
    mylib.getXSize.restype = c_int
    tmp_len = mylib.getXSize(model)

    flattened_dataset_inputs = []
    for p in dataset_inputs:
        flattened_dataset_inputs.append(p[0])
        flattened_dataset_inputs.append(p[1])

    arrsize_flat = len(flattened_dataset_inputs)
    arrtype_flat = c_float * arrsize_flat
    arr_flat = arrtype_flat(*flattened_dataset_inputs)

    arrsize_exp = len(flattened_dataset_inputs)
    arrtype_exp = c_float * arrsize_exp
    arr_exp = arrtype_exp(*flattened_dataset_inputs)

    predicted_outputs = []
    for p in img_test:
        arrsizeP = len(p)
        arrtypeP = c_float * arrsizeP
        arrP = arrtypeP(*p)
        mylib.predict_mlp_model_classification.argtypes = [c_void_p, arrtypeP]
        mylib.predict_mlp_model_classification.restype = POINTER(c_float)
        tmp = []

        tmp = mylib.predict_mlp_model_classification(model, arrP)
        np_arr = np.ctypeslib.as_array(tmp, (tmp_len,))
        predicted_outputs.append(np_arr)

    print(predicted_outputs)

    mylib.destroy_MLP(model)

if __name__ == "__main__":
    run()
    # ran()
    exit(0)
    (X_train, Y_train), (X_test, Y_test) = import_dataset()

    # np_arr = np.array([2, 3, 1])
    #np_arr = np.array([3072, 3, 3])
    npl = np.ctypeslib.as_ctypes(np_arr)
    model = create_mlp_model(npl)

    res = predict_mlp_model_classification(model, X_test[2])
    for r in res:
        print(r)

    destroy_mlp_model(model)
