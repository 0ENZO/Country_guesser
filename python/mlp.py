import numpy as np
from ctypes import *
import os
import time
from conf import *

mylib = cdll.LoadLibrary(PATH_TO_DLL)


def create_mlp_model(npl):
    # mylib.create_mlp_model.argtypes = [c_int * len(npl), c_int]
    mylib.create_mlp_model.argtypes = [POINTER(c_int), c_int]
    mylib.create_mlp_model.restype = c_void_p
    return mylib.create_mlp_model(npl, len(npl))


def as_C_array(dataset):
    arr_size = len(dataset)

    arr_type = c_float * arr_size
    arr = arr_type(*dataset)

    return arr, arr_type


def train_classification_stochastic_backprop_mlp_model(model, inputs, outputs, alpha=0.01, epochs=1000):
    # print("Nombre epochs :", epochs)
    np_inputs = np.array(inputs)
    inputs_type = c_float * len(inputs)
    final_inputs = inputs_type(*inputs)

    np_outputs = np.array(outputs)
    outputs_type = c_float * len(outputs)
    final_outputs = outputs_type(*outputs)

    # mylib.train_classification_stochastic_backprop_mlp_model.argtypes = [c_void_p, inputs_type, c_int, outputs_type, c_float, c_int]
    mylib.train_classification_stochastic_backprop_mlp_model.argtypes = [c_void_p, POINTER(c_float), c_int,
                                                                         POINTER(c_float), c_float, c_int]
    mylib.train_classification_stochastic_backprop_mlp_model.restype = None

    mylib.train_classification_stochastic_backprop_mlp_model(model,
                                                             final_inputs,
                                                             len(final_inputs),
                                                             final_outputs,
                                                             alpha,
                                                             epochs)


def train_regression_stochastic_backprop_mlp_model(model, inputs, outputs, alpha=0.001, epochs=1000):
    np_inputs = np.array(inputs)
    inputs_type = c_float * len(np_inputs)
    final_inputs = inputs_type(*inputs)

    np_outputs = np.array(outputs)
    outputs_type = c_float * len(np_outputs)
    final_outputs = outputs_type(*outputs)

    # mylib.train_regression_stochastic_backprop_mlp_model.argtypes = [c_void_p, inputs_type, c_int, outputs_type, c_float, c_int]
    mylib.train_regression_stochastic_backprop_mlp_model.argtypes = [c_void_p, POINTER(c_float), c_int,
                                                                     POINTER(c_float), c_float, c_int]
    mylib.train_regression_stochastic_backprop_mlp_model.restype = None

    mylib.train_regression_stochastic_backprop_mlp_model(model, final_inputs, len(final_inputs), final_outputs, alpha,
                                                         epochs)


def predict_mlp_model_classification(model, sample_inputs, layer=1):
    sample_inputs = np.array(sample_inputs)
    sample_inputs_type = c_float * len(sample_inputs)
    final_sample_inputs = sample_inputs_type(*sample_inputs)

    # mylib.predict_mlp_model_classification.argtypes = [c_void_p, sample_inputs_type]
    mylib.predict_mlp_model_classification.argtypes = [c_void_p, POINTER(c_float)]
    mylib.predict_mlp_model_classification.restype = POINTER(c_float)

    result = mylib.predict_mlp_model_classification(model, final_sample_inputs)
    # return list(result)
    return list(np.ctypeslib.as_array(result, (layer,)))


def predict_mlp_model_classification_v2(model, sample_inputs, layer=1):
    sample_inputs = np.array(sample_inputs)
    sample_inputs_type = c_float * len(sample_inputs)
    final_sample_inputs = sample_inputs_type(*sample_inputs)

    # mylib.predict_mlp_model_classification.argtypes = [c_void_p, sample_inputs_type]
    mylib.predict_mlp_model_classification.argtypes = [c_void_p, POINTER(c_float)]
    mylib.predict_mlp_model_classification.restype = POINTER(c_float)

    result = mylib.predict_mlp_model_classification(model, final_sample_inputs)

    return list(np.ctypeslib.as_array(result, (layer,)))


def predict_mlp_model_regression(model, sample_inputs):
    sample_inputs = np.array(sample_inputs)
    sample_inputs_type = c_float * len(sample_inputs)
    final_sample_inputs = sample_inputs_type(*sample_inputs)

    # mylib.predict_mlp_model_regression.argtypes = [c_void_p, sample_inputs_type]
    mylib.predict_mlp_model_regression.argtypes = [c_void_p, POINTER(c_float)]

    mylib.predict_mlp_model_regression.restype = POINTER(c_float)

    return mylib.predict_mlp_model_regression(model, final_sample_inputs)


def destroy_mlp_model(model):
    mylib.destroy_mlp_model.argtypes = [c_void_p]
    mylib.destroy_mlp_model.restype = None
    mylib.destroy_mlp_model(model)


def save_mlp_model(model, name, subFolder=None):
    date = time.strftime("_%d_%m_%H_%M")
    if subFolder:
        path = os.path.join(SAVE_FOLDER, subFolder) + name + date + ".txt"
    else:
        path = SAVE_FOLDER + name + date + ".txt"
    path = path.encode('utf-8')

    mylib.save_mlp_model.argtypes = [c_void_p, c_char_p]
    mylib.save_mlp_model.restype = None
    mylib.save_mlp_model(model, path)


def save_mlp_model_v2(model, name):
    date = time.strftime("_%d_%m_%H_%M")
    path = SAVE_FOLDER_V2 + name + date + ".txt"
    path = path.encode('utf-8')

    mylib.save_mlp_model.argtypes = [c_void_p, c_char_p]
    mylib.save_mlp_model.restype = None
    mylib.save_mlp_model(model, path)


def save_mlp_model_v3(model, name):
    date = time.strftime("_%d_%m_%H_%M")
    path = SAVE_FOLDER_V3 + name + date + ".txt"
    path = path.encode('utf-8')

    mylib.save_mlp_model.argtypes = [c_void_p, c_char_p]
    mylib.save_mlp_model.restype = None
    mylib.save_mlp_model(model, path)


def load_mlp_model(name):
    path = SAVE_FOLDER + name + ".txt"
    path = path.encode('utf-8')

    mylib.load_mlp_model.argtypes = [c_char_p]
    mylib.load_mlp_model.restype = c_void_p
    return mylib.load_mlp_model(path)


if __name__ == "__main__":

    flattened_inputs = [1, 1, 1, 2, 2, 1]
    flattened_outputs = [1, -1, -1]

    np_arr = np.array([2, 3, 1])
    npl = np.ctypeslib.as_ctypes(np_arr)
    model = create_mlp_model(npl)

    print("Avant entraînement : \n")
    for i in range(3):
        # np_flattened_inputs = np.array(flattened_inputs[i * 2:(i + 1) * 2])
        # dataset_flattened_inputs = np.ctypeslib.as_ctypes(np_flattened_inputs)
        # res = predict_mlp_model_classification(model, dataset_flattened_inputs)
        res = predict_mlp_model_classification(model, flattened_inputs[i * 2:(i + 1) * 2])
        print(res[0])

    train_classification_stochastic_backprop_mlp_model(model, flattened_inputs, flattened_outputs, epochs=10000)

    print("Après entraînement : \n")
    for i in range(3):
        # np_flattened_inputs = np.array(flattened_inputs[i * 2:(i + 1) * 2])
        # dataset_flattened_inputs = np.ctypeslib.as_ctypes(np_flattened_inputs)
        # res = predict_mlp_model_classification(model, dataset_flattened_inputs)
        res = predict_mlp_model_classification(model, flattened_inputs[i * 2:(i + 1) * 2])
        print(res[0])
        # print(type(res))
        # print(type(res[0]))

    destroy_mlp_model(model)
