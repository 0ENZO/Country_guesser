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
from tqdm import tqdm


def transform_image(img, img_size):
    img = img.resize((img_size, img_size))

    if img.mode == "RGBA":
        rgba = np.array(img)
        rgba[rgba[..., -1] == 0] = [255, 255, 255, 0]
        img = Image.fromarray(rgba)
    img = img.convert("RGB")

    return img
    # print(im_arr.shape)
    # return im_arr


def fill_x_and_y_with_images_and_labels(folder, x_list, y_list, label, img_size):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        image = Image.open(file_path)
        img = transform_image(image, img_size)

        hor_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        ver_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
        # rotate_45 = img.rotate(45)
        # rotate_90 = img.rotate(90)

        im_arr = np.array(img).flatten()
        hor_arr = np.array(hor_flip).flatten()
        ver_arr = np.array(ver_flip).flatten()
        # r45_arr = np.array(rotate_45).flatten()
        # r90_arr = np.array(rotate_90).flatten()

        im_arr = im_arr / 255.0
        hor_arr = hor_arr / 255.0
        ver_arr = ver_arr / 255.0
        # r45_arr = r45_arr / 255.0
        # r90_arr = r90_arr / 255.0

        x_list.append(im_arr)
        x_list.append(hor_arr)
        x_list.append(ver_arr)
        # x_list.append(r45_arr)
        # x_list.append(r90_arr)

        y_list.append(label)
        y_list.append(label)
        y_list.append(label)
        # y_list.append(label)
        # y_list.append(label)


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
    # print(predicted_test_outputs_after_training)


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
        # print(train_predicted_outputs)
        # print(y_train)
        loss = mean_squared_error(y_train, train_predicted_outputs)
        losses.append(loss)

        test_predicted_outputs = [predict_mlp_model_classification(model, x, 3) for x in x_test]
        # print(test_predicted_outputs)
        # print(y_test)
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
    max_train = max_test = 65.0
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
    trainings_count = 5
    epochs = 10000000

    if choice == 1:
        grid_search_MLP_0HNL_model(alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
    elif choice == 2:
        grid_search_train_MLP_1HNL_8N_model(alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
    elif choice == 3:
        grid_search_train_MLP_1HNL_32_model(alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
    elif choice == 4:
        grid_search_train_MLP_2HNL_32_model(alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)


def run_train3(choice, version=None, dataset=DATASET_FOLDER2):
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05]
    img_sizes = [8, 16]
    trainings_count = 5
    epochs = 300000

    if choice == 1:
        grid_search_MLP_0HNL_model(alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
    elif choice == 2:
        grid_search_train_MLP_1HNL_8N_model(alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
    elif choice == 3:
        grid_search_train_MLP_1HNL_32_model(alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)
    elif choice == 4:
        grid_search_train_MLP_2HNL_32_model(alphas, epochs, trainings_count, img_size=img_sizes, version=version, dataset=dataset)


def get_accuracy(predicted, expected, train=True):
    acc = 0
    for p, y in zip(predicted, expected):
        if np.argmax(p) == np.argmax(y):
            acc += 1
    acc = acc / len(predicted) * 100
    # if train:
    #     print("Dataset d'entraînement : ", acc, "%")
    # else:
    #     print("Dataset de test : ", acc, "%")
    return acc


def good_train(hnl, alpha, alpha_step, alpha_count, epochs=100):
    (x_train, y_train), (x_test, y_test) = import_dataset(img_size=IMAGE_SIZE)
    goal = 80.

    for _ in range(alpha_count):
        arr = [len(x_train[0])]
        for layer in hnl:
            arr.append(layer)
        arr.append(3)
        model = create_mlp_model(np.ctypeslib.as_ctypes(np.array(arr)))

        last_acc_save = goal
        last_test_acc_save = goal
        for ep in tqdm(range(epochs)):
            train_classification_stochastic_backprop_mlp_model(model, x_train.flatten(), y_train.flatten(), round(alpha, 4), len(x_train))

            predicted_train_outputs_after_training = [predict_mlp_model_classification(model, x, 3) for x in x_train]
            acc = get_accuracy(predicted_train_outputs_after_training, y_train, train=True)

            predicted_test_outputs_after_training = [predict_mlp_model_classification(model, x, 3) for x in x_test]
            test_acc = get_accuracy(predicted_test_outputs_after_training, y_test, train=False)

            if ((acc > last_acc_save) & (test_acc > last_test_acc_save)) & (acc - test_acc < 5.0):
                if len(hnl) > 0:
                    save_mlp_model(model, f"MLP_{ep}epochs_{len(x_train[0])}it_{round(alpha, 4)}a_{IMAGE_SIZE}px_{len(hnl)}hl_{hnl[0]}n_{round(acc, 2)}acc_{round(test_acc, 2)}test_acc")
                else:
                    save_mlp_model(model, f"MLP_{ep}epochs_{len(x_train[0])}it_{round(alpha, 4)}a_{IMAGE_SIZE}px_{len(hnl)}hl_{round(acc, 2)}acc_{round(test_acc, 2)}test_acc")
                last_acc_save = acc
                last_test_acc_save = test_acc

        destroy_mlp_model(model)
    alpha += alpha_step


if __name__ == "__main__":
    #good_train([], 0.002, 0.001)
    good_train([], 0.01, 0.005, 19)
    exit(0)
    (x_train, y_train), (x_test, y_test) = import_dataset(img_size=IMAGE_SIZE)
    # (x_train, y_train) = np.shuffle(x_train, y_train)
    # (x_test, y_test) = np.shuffle(x_test, y_test)
    alpha = 0.005
    epochs = 2000

    for i in range(5):
        alpha += 0.0001
        global_acc = 0
        global_test_acc = 0

        for j in range(1, 3):
            np_arr = np.array([len(x_train[0]), 3])
            npl = np.ctypeslib.as_ctypes(np_arr)
            model = create_mlp_model(npl)

            predicted_train_outputs_before_training = [predict_mlp_model_classification(model, x, 3) for x in x_train]
            cpt = 0
            for p, y in zip(predicted_train_outputs_before_training, y_train):
                if np.argmax(p) == np.argmax(y):
                    cpt += 1
            print("Dataset d'entraînement, avant entraînement : ", cpt / len(predicted_train_outputs_before_training) * 100, "%")

            train_classification_stochastic_backprop_mlp_model(model, x_train.flatten(), y_train.flatten(), round(alpha, 4), epochs)

            predicted_train_outputs_after_training = [predict_mlp_model_classification(model, x, 3) for x in x_train]
            acc = 0
            for p, y in zip(predicted_train_outputs_after_training, y_train):
                if np.argmax(p) == np.argmax(y):
                    acc += 1
            acc = acc / len(predicted_train_outputs_after_training) * 100
            print("Dataset d'entraînement, après entraînement : ", acc, "%")
            global_acc += acc

            predicted_test_outputs_after_training = [predict_mlp_model_classification(model, x, 3) for x in x_test]

            test_acc = 0
            for p, y in zip(predicted_test_outputs_after_training, y_test):
                if np.argmax(p) == np.argmax(y):
                    test_acc += 1
            test_acc = test_acc / len(predicted_test_outputs_after_training) * 100
            print("Dataset de test, après entraînement : ", test_acc, "%")

            print(f"Alpha : {alpha}, Nombre d'epochs : {epochs}, Global_acc : {global_acc}, Global_test_acc: {global_test_acc}")
            global_test_acc += test_acc

            if j == 2 and global_acc - global_test_acc < 11.0:
                if len(np_arr) - 2 == 0:
                    save_mlp_model(model, f"MLP_200K_{round(alpha, 4)}a_{IMAGE_SIZE}px_{len(np_arr) - 2}hl_{round(global_acc / 2, 2)}acc_{round(global_test_acc / 2, 2)}test_acc")
                else:
                    save_mlp_model(model, f"MLP_200K_{round(alpha, 4)}a_{IMAGE_SIZE}px_{len(np_arr) - 2}hl_{np_arr[1]}n_{round(global_acc / 2, 2)}acc_{round(global_test_acc / 2, 2)}test_acc")
            destroy_mlp_model(model)

    exit(0)
    (X_train, Y_train), (X_test, Y_test) = import_dataset(img_size=8, dataset=DATASET_FOLDER3)

    np_arr = np.array([len(X_train[0]), 3, 3])
    npl = np.ctypeslib.as_ctypes(np_arr)
    model = create_mlp_model(npl)

    train(model, X_train, Y_train, X_test, Y_test, alpha=0.01, epochs=1000000)

    save_mlp_model(model, f"AFI_1M_10e3_{8}px_{len(np_arr) - 2}hl_{np_arr[1]}n_99p")
    destroy_mlp_model(model)

    exit(0)

    # save_mlp_model(model, f"2M_10e3_{IMAGE_SIZE}px_{len(np_arr) - 2}hl_{np_arr[1]}n")
    destroy_mlp_model(model)
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

    destroy_mlp_model(model)
    exit(0)





