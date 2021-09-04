import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from PIL import Image
from mlp import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ctypes import *


def transform_image(image, img_size):
    image = image.resize((img_size, img_size))
    if image.mode == "RGBA":
        rgba = np.array(image)
        rgba[rgba[..., -1] == 0] = [255, 255, 255, 0]
        image = Image.fromarray(rgba)

    image = image.convert("RGB")
    im_arr = np.array(image).flatten()
    # print(im_arr.shape)
    im_arr = im_arr / 255.0
    return im_arr


def fill_x_and_y_with_images_and_labels(folder, x_list, y_list, label, img_size):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        image = Image.open(file_path)
        im_arr = transform_image(image, img_size)
        x_list.append(im_arr)
        y_list.append(label)


def import_dataset(img_size=IMAGE_SIZE):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    fill_x_and_y_with_images_and_labels(TRAIN_FIRST_FOLDER, X_train, Y_train, [1, -1, -1], img_size)
    fill_x_and_y_with_images_and_labels(TRAIN_SECOND_FOLDER, X_train, Y_train, [-1, 1, -1], img_size)
    fill_x_and_y_with_images_and_labels(TRAIN_THIRD_FOLDER, X_train, Y_train, [-1, -1, 1], img_size)

    fill_x_and_y_with_images_and_labels(TEST_FIRST_FOLDER, X_test, Y_test, [1, -1, -1], img_size)
    fill_x_and_y_with_images_and_labels(TEST_SECOND_FOLDER, X_test, Y_test, [-1, 1, -1], img_size)
    fill_x_and_y_with_images_and_labels(TEST_THIRD_FOLDER, X_test, Y_test, [-1, -1, 1], img_size)

    return (np.array(X_train).astype(np.float), np.array(Y_train).astype(np.float)), \
           (np.array(X_test).astype(np.float), np.array(Y_test).astype(np.float))


def train(model, X_train, Y_train, X_test, Y_test, alpha, epochs):
    predicted_train_outputs_before_training = [predict_mlp_model_classification(model, x, 3) for x in X_train]
    cpt = 0
    for p, y in zip(predicted_train_outputs_before_training, Y_train):
        if np.argmax(p) == np.argmax(y):
            cpt += 1
    print("Dataset d'entraînement, avant entraînement : ", cpt / len(predicted_train_outputs_before_training) * 100, "%")
    # print(predicted_train_outputs_before_training)

    train_classification_stochastic_backprop_mlp_model(model, X_train.flatten(), Y_train.flatten(), alpha, epochs)
    test_train(model, X_train, Y_train, X_test, Y_test)


def test_train(model, X_train, Y_train, X_test, Y_test):
    predicted_train_outputs_after_training = [predict_mlp_model_classification(model, x, 3) for x in X_train]
    cpt = 0
    for p, y in zip(predicted_train_outputs_after_training, Y_train):
        if np.argmax(p) == np.argmax(y):
            cpt += 1
    print("Dataset d'entraînement, après entraînement : ", cpt / len(predicted_train_outputs_after_training) * 100, "%")
    # print(predicted_train_outputs_after_training)

    # print(X_test)
    predicted_test_outputs_after_training = [predict_mlp_model_classification(model, x, 3) for x in X_test]
    # print(predicted_test_outputs_after_training)
    # for p in predicted_test_outputs_after_training:
    #    print(p)

    cpt = 0
    for p, y in zip(predicted_test_outputs_after_training, Y_test):
        if np.argmax(p) == np.argmax(y):
            cpt += 1
    print("Dataset de test, après entraînement : ", cpt / len(predicted_test_outputs_after_training) * 100, "%")
    print(predicted_test_outputs_after_training)


def run():
    print("Dataset import en cours")
    (X_train, Y_train), (X_test, Y_test) = import_dataset()
    print("Dataset importé")

    np_arr = np.array([len(X_train[0]), 8, 3])
    npl = np.ctypeslib.as_ctypes(np_arr)
    model = create_mlp_model(npl)

    train(model, X_train, Y_train, X_test, Y_test, alpha=0.01, epochs=20000)

    # save_mlp_model(model, f"2M_10e3_{IMAGE_SIZE}px_{len(np_arr) - 2}hl_{np_arr[1]}n")
    destroy_mlp_model(model)


def show_graphs():
    loss, test_loss, acc, test_acc = []
    (X_train, Y_train), (X_test, Y_test) = import_dataset()
    model = load_mlp_model("2m_32px")

    test_train(model, X_train, Y_train, X_test, Y_test)
    destroy_mlp_model(model)


if __name__ == "__main__":
    model = load_mlp_model("models_by_hands/2M_80e4_32px_2hl_3n_63p_03_09_23_05")
    (X_train, Y_train), (X_test, Y_test) = import_dataset()

    predicted_test_outputs_after_training = [predict_mlp_model_classification(model, x, 3) for x in X_test]

    cpt = 0
    for p, y in zip(predicted_test_outputs_after_training, Y_test):
        if np.argmax(p) == np.argmax(y):
            cpt += 1
    print("Dataset de test, après entraînement : ", cpt / len(predicted_test_outputs_after_training) * 100, "%")
    print(predicted_test_outputs_after_training)

    print("-------------------------------")

    output_x0 = predict_mlp_model_classification(model, X_test[0], 3)
    print(output_x0)

    # image = Image.fromarray(X_test[0].astype('uint8'), 'RGB')
    # image = X_test[0] * 255
    # image = image.reshape(IMAGE_SIZE, IMAGE_SIZE , 3)
    # image = Image.fromarray(image, 'RGB')
    # image.show()

    # train(model, X_train, Y_train, X_test, Y_test, alpha=0.01, epochs=2000000)

    # print("ICI ------------------------------------------->")
    # print(X_test[20])
    # sample_inputs = np.array(X_test[20])
    # print(sample_inputs)
    # sample_inputs_type = c_float * len(sample_inputs)
    # print(sample_inputs_type)
    # final_sample_inputs = sample_inputs_type(*sample_inputs)
    # print(final_sample_inputs)

    # print("ICI ")
    # print(predict_mlp_model_classification_v2(model, X_test[20], 3))
    destroy_mlp_model(model)
    exit(0)

    output = np.argmax(predicted_test_outputs_after_training)
    label = CLASSES[output]
    prediction_score = predicted_test_outputs_after_training[output]
    print(predicted_test_outputs_after_training)
    print(output)
    print(label)
    print(prediction_score)




    
