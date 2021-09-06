from conf import *
from image import *
import numpy as np

if __name__ == "__main__":

    # good_train([8], 0.002, 0.001)
    good_train([8], 0.01, 0.005, 19)
    exit(0)

    (x_train, y_train), (x_test, y_test) = import_dataset(img_size=IMAGE_SIZE)
    # (x_train, y_train) = np.shuffle(x_train, y_train)
    # (x_test, y_test) = np.shuffle(x_test, y_test)
    alpha = 0.01
    epochs = 200000

    for i in range(80):
        alpha += 0.001
        global_acc = 0
        global_test_acc = 0

        for j in range(1, 3):
            np_arr = np.array([len(x_train[0]), 8, 3])
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