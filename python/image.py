import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from PIL import Image
from mlp import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from ctypes import *
import matplotlib.pyplot as plt


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


def import_dataset(img_size=IMAGE_SIZE, dataset=DATASET_FOLDER2):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    classes = list(CLASSES)
    if dataset == DATASET_FOLDER3:
        classes = ["allemagne", "france", "italie"]
    print(classes)

    train_first_folder = os.path.join(dataset, f"train/{classes[0]}").replace(os.sep, '/')
    train_second_folder = os.path.join(dataset, f"train/{classes[1]}").replace(os.sep, '/')
    train_third_folder = os.path.join(dataset, f"train/{classes[2]}").replace(os.sep, '/')

    test_first_folder = os.path.join(dataset, f"test/{classes[0]}").replace(os.sep, '/')
    test_second_folder = os.path.join(dataset, f"test/{classes[1]}").replace(os.sep, '/')
    test_third_folder = os.path.join(dataset, f"test/{classes[2]}").replace(os.sep, '/')

    fill_x_and_y_with_images_and_labels(train_first_folder, X_train, Y_train, [1, -1, -1], img_size)
    fill_x_and_y_with_images_and_labels(train_second_folder, X_train, Y_train, [-1, 1, -1], img_size)
    fill_x_and_y_with_images_and_labels(train_third_folder, X_train, Y_train, [-1, -1, 1], img_size)

    fill_x_and_y_with_images_and_labels(test_first_folder, X_test, Y_test, [1, -1, -1], img_size)
    fill_x_and_y_with_images_and_labels(test_second_folder, X_test, Y_test, [-1, 1, -1], img_size)
    fill_x_and_y_with_images_and_labels(test_third_folder, X_test, Y_test, [-1, -1, 1], img_size)

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


def show_graphs(model):
    losses = test_losses = accs = test_accs = []
    (x_train, y_train), (x_test, y_test) = import_dataset()
    for epoch in range(100):
        print(f"Epoch numéro {epoch} en cours ..")
        train_classification_stochastic_backprop_mlp_model(model, x_train.flatten(), y_train.flatten(), epochs=len(x_train))

        train_predicted_outputs = [predict_mlp_model_classification(model, x, 3) for x in x_train]
        print(train_predicted_outputs)
        print(y_train)
        loss = mean_squared_error(y_train, train_predicted_outputs)
        losses.append(loss)

        test_predicted_outputs = [predict_mlp_model_classification(model, x, 3) for x in x_test]
        print(test_predicted_outputs)
        print(y_test)
        test_loss = mean_squared_error(y_test, test_predicted_outputs)
        test_losses.append(test_loss)

        # acc = 0
        # for p, y in zip(train_predicted_outputs, y_train):
        #     if np.argmax(p) == np.argmax(y):
        #         acc += 1
        acc = accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_predicted_outputs, axis=1))
        accs.append(acc)

        # test_acc = 0
        # for p, y in zip(test_predicted_outputs, y_test):
        #     if np.argmax(p) == np.argmax(y):
        #         acc += 1
        test_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_predicted_outputs, axis=1))
        test_accs.append(test_acc)

        plt.plot(losses)
        plt.plot(test_losses)
        plt.legend(['loss', 'test_loss'], loc='upper left')
        plt.title('Evolution of loss curve (MSE)')
        plt.xlabel('epochs')
        plt.ylabel('mean squared error')
        plt.show()

        plt.plot(accs)
        plt.plot(test_accs)
        plt.legend(['acc', 'test_acc'], loc='upper left')
        plt.title('Evolution of accuracy (MSE)')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.show()


def grid_search_mlp_model(name, npl, alphas, epochs, trainings_count, img_size, version=None, dataset=DATASET_FOLDER2, extraName=""):
    max_train = max_test = 25.0
    max_models = len(img_size) * len(alphas)
    current_model = 0
    for size in img_size:
        (x_train, y_train), (x_test, y_test) = import_dataset(size, dataset=dataset)
        for alpha in alphas:

            arr = [len(x_train[0])]
            for layer in npl:
                arr.append(layer)
            arr.append(3)

            model = create_mlp_model(np.ctypeslib.as_ctypes(np.array(arr)))
            saves = 0
            for _ in range(trainings_count):
                train_classification_stochastic_backprop_mlp_model(model, x_train.flatten(), y_train.flatten(), alpha=alpha, epochs=epochs)
                train_predicted_outputs = [predict_mlp_model_classification(model, x, len(CLASSES)) for x in x_train]
                test_predicted_outputs = [predict_mlp_model_classification(model, x, 3) for x in x_test]

                train_cpt = 0
                for p, y in zip(train_predicted_outputs, y_train):
                    if np.argmax(p) == np.argmax(y):
                        train_cpt += 1
                train_cpt = round(train_cpt / len(train_predicted_outputs) * 100, 2)

                test_cpt = 0
                for p, y in zip(test_predicted_outputs, y_train):
                    if np.argmax(p) == np.argmax(y):
                        test_cpt += 1
                test_cpt = round(test_cpt / len(test_predicted_outputs) * 100, 2)

                filename = f"{extraName}{name}_{size}px_{epochs}e_{alpha}a_{train_cpt}acc_{test_cpt}test_acc"
                print(f"Model : {extraName} {name} {size}px, {epochs}e, {alpha}a, {train_cpt}% acc, {test_cpt}% test_acc")

                if (train_cpt > max_train) & (test_cpt > max_test):
                    max_train = train_cpt
                    max_test = test_cpt
                    if version:
                        save_mlp_model(model, filename, f"{name}v{version}/")
                    else:
                        save_mlp_model(model, filename, f"{name}/")
                elif (((train_cpt + test_cpt) / 2) >= ((max_train + max_test) / 2)) & (train_cpt > max_train - (max_train/10.0)) & (test_cpt > max_test - (max_test/10.0)):
                    if saves == 0:
                        if version:
                            save_mlp_model(model, filename, f"{name}v{version}/")
                        else:
                            save_mlp_model(model, filename, f"{name}/")
                        saves += 1

            destroy_mlp_model(model)
            current_model += 1
            print(f"{current_model} modèle(s) entrainé(s) pour l'achitecture suivante : {name}")
    print(f"Parmi les {max_models} modèles {name} entrainés, le meilleur modèle a eu {max_train}% train_acc et {max_test}% test_acc")


def grid_search_MLP_0HNL_model(alphas, epochs, trainings_count, img_size=IMAGE_SIZE, version=None, dataset=DATASET_FOLDER2):
    if dataset == DATASET_FOLDER3:
        grid_search_mlp_model("MLP_0HNL", [], alphas, epochs, trainings_count, img_size, version=version, dataset=dataset, extraName="AFI")
    else:
        grid_search_mlp_model("MLP_0HNL", [], alphas, epochs, trainings_count, img_size, version=version, dataset=dataset)


def grid_search_train_MLP_1HNL_8N_model(alphas, epochs, trainings_count, img_size=IMAGE_SIZE, version=None, dataset=DATASET_FOLDER2):
    if dataset == DATASET_FOLDER3:
        grid_search_mlp_model("MLP_1HNL_8N", [8], alphas, epochs, trainings_count, img_size, version=version, dataset=dataset, extraName="AFI")
    else:
        grid_search_mlp_model("MLP_1HNL_8N", [8], alphas, epochs, trainings_count, img_size, version=version, dataset=dataset)


def grid_search_train_MLP_1HNL_32_model(alphas, epochs, trainings_count, img_size=IMAGE_SIZE, version=None, dataset=DATASET_FOLDER2):
    if dataset == DATASET_FOLDER3:
        grid_search_mlp_model("MLP_1HNL_32", [32], alphas, epochs, trainings_count, img_size, version=version, dataset=dataset, extraName="AFI")
    else:
        grid_search_mlp_model("MLP_1HNL_32", [32], alphas, epochs, trainings_count, img_size, version=version, dataset=dataset)


def grid_search_train_MLP_2HNL_32_model(alphas, epochs, trainings_count, img_size=IMAGE_SIZE, version=None, dataset=DATASET_FOLDER2):
    if dataset == DATASET_FOLDER3:
        grid_search_mlp_model("MLP_2HNL_32", [32, 32], alphas, epochs, trainings_count, img_size, version=version, dataset=dataset, extraName="AFI")
    else:
        grid_search_mlp_model("MLP_2HNL_32", [32, 32], alphas, epochs, trainings_count, img_size, version=version, dataset=dataset)


def run_train(choice, version=None, dataset=DATASET_FOLDER2):
    alphas = [0.001, 0.011]
    img_sizes = [8, 16, 32]
    trainings_count = 5
    epochs = 300000

    new_alphas = []
    for alpha in np.arange(alphas[0], alphas[1], alphas[0]):
        new_alphas.append(alpha)

    if choice == 1:
        grid_search_MLP_0HNL_model(new_alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
    elif choice == 2:
        grid_search_train_MLP_1HNL_8N_model(new_alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
    elif choice == 3:
        grid_search_train_MLP_1HNL_32_model(new_alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
    elif choice == 4:
        grid_search_train_MLP_2HNL_32_model(new_alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
        

def run_train2(choice, version=None, dataset=DATASET_FOLDER2):
    alphas = [0.008]
    img_sizes = [8, 12]
    trainings_count = 10
    epochs = 1000

    if choice == 1:
        grid_search_MLP_0HNL_model(alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
    elif choice == 2:
        grid_search_train_MLP_1HNL_8N_model(alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
    elif choice == 3:
        grid_search_train_MLP_1HNL_32_model(alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
    elif choice == 4:
        grid_search_train_MLP_2HNL_32_model(alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)


def run_train3(choice):
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05]
    img_sizes = [8]
    trainings_count = 10
    epochs = 300000

    if choice == 1:
        grid_search_mlp_model("MLP_0HNL", [], alphas, epochs, trainings_count, img_sizes, version=3)
    elif choice == 2:
        grid_search_mlp_model("MLP_1HNL_8N", [8], alphas, epochs, trainings_count, img_sizes, version=3)
    elif choice == 3:
        grid_search_mlp_model("MLP_1HNL_32", [32], alphas, epochs, trainings_count, img_sizes, version=3)
    elif choice == 4:
        grid_search_mlp_model("MLP_2HNL_32", [32, 32], alphas, epochs, trainings_count, img_sizes, version=3)


if __name__ == "__main__":
    # model = load_mlp_model("models_by_hands/2M_80e4_32px_2hl_3n_63p_03_09_23_05")
    # show_graphs(model)

    # exit(0)
    np_arr = np.array([3072, 8, 3])
    npl = np.ctypeslib.as_ctypes(np_arr)
    model = create_mlp_model(npl)
    name = "MLP_0HNL"
    save_mlp_model(model, "AFIMLP_OHNL_8px_1000e_0.008a_33.67acc_100.0test_acc_05_09_03_06.txt", f"{name}v2/")
    save_mlp_model(model, filename, f"{name}v{version}/")
    save_mlp_model(model, filename, f"{name}v{version}/")

    destroy_mlp_model(model)
    exit(0)

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





