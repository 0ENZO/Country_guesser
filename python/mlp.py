import numpy as np
from ctypes import *
import os
import time
from conf import *

mylib = cdll.LoadLibrary(path_to_dll)


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
    print("Nombre epochs :", epochs)
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

    # samples_count = len(inputs)
    #
    # X = np.array(inputs)
    # flattened_inputs = X.flatten()
    #
    # arr_size = len(flattened_inputs)
    # input_type = c_float * arr_size
    # input_dataset = input_type(*flattened_inputs)
    #
    # input_dataset, input_type = as_C_array(flattened_inputs)
    # output_dataset, output_type = as_C_array(outputs)
    #
    # mylib.train_classification_stochastic_backprop_mlp_model.argtypes = [c_void_p,
    #                                                                      input_type,
    #                                                                      c_int,
    #                                                                      output_type,
    #                                                                      c_float,
    #                                                                      c_int]
    # mylib.train_classification_stochastic_backprop_mlp_model.restype = None
    #
    # mylib.train_classification_stochastic_backprop_mlp_model(model,
    #                                                          input_dataset,
    #                                                          samples_count,
    #                                                          output_dataset,
    #                                                          alpha,
    #                                                          epochs)

    # print("------------------")
    # print(inputs)
    # print(np_inputs)
    # print(inputs_type)
    # print(final_inputs)
    # print(POINTER(c_float))
    # print("------------------")


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


def save_mlp_model(model, name):
    #date = time.strftime("_%d_%m_%H_%M")
    #path = SAVE_FOLDER + name + date + ".txt"
    print(f"\n{name} sauvegardé \n")
    path = name.encode('utf-8')

    mylib.save_mlp_model.argtypes = [c_void_p, c_char_p]
    mylib.save_mlp_model.restype = None
    mylib.save_mlp_model(model, path)


def load_mlp_model(name):
    #path = SAVE_FOLDER + name + ".txt"
    path = name.encode('utf-8')

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

    date = time.strftime("_%d_%m_%H_%M")
    path = SAVE_FOLDER + "test" + date + ".txt"

    destroy_mlp_model(model)
