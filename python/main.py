import numpy as np
from ctypes import *

path_to_dll = "C:/Users/Enzo/Desktop/mlp_library/cmake-build-debug/mlp_library.dll"
mylib = cdll.LoadLibrary(path_to_dll)


def create_mlp_model(npl):
    mylib.create_mlp_model.argtypes = [c_int * len(npl), c_int]
    mylib.create_mlp_model.restype = c_void_p
    return mylib.create_mlp_model(npl, len(npl))


def train_classification_stochastic_backprop_mlp_model(model, inputs, outputs, alpha=0.001, epochs=100000):
    np_inputs = np.array(inputs)
    np_outputs = np.array(outputs)

    inputs_type = c_float * len(np_inputs)
    final_inputs = inputs_type(*inputs)

    outputs_type = c_float * len(np_outputs)
    final_outputs = outputs_type(*outputs)

    mylib.train_classification_stochastic_backprop_mlp_model.argtypes = [c_void_p, inputs_type, c_int, outputs_type,
                                                                         c_float, c_int]
    mylib.train_classification_stochastic_backprop_mlp_model.restype = None

    mylib.train_classification_stochastic_backprop_mlp_model(model, final_inputs, len(final_inputs), final_outputs,
                                                             alpha, epochs)


def train_regression_stochastic_backprop_mlp_model(model, inputs, outputs, alpha=0.001, epochs=100000):
    np_inputs = np.array(inputs)
    np_outputs = np.array(outputs)

    inputs_type = c_float * len(np_inputs)
    final_inputs = inputs_type(*inputs)

    outputs_type = c_float * len(np_outputs)
    final_outputs = outputs_type(*outputs)

    mylib.train_regression_stochastic_backprop_mlp_model.argtypes = [c_void_p, inputs_type, c_int, outputs_type,
                                                                     c_float, c_int]
    mylib.train_regression_stochastic_backprop_mlp_model.restype = None

    mylib.train_regression_stochastic_backprop_mlp_model(model, final_inputs, len(final_inputs), final_outputs, alpha,
                                                         epochs)


def predict_mlp_model_classification(model, sample_inputs):
    sample_inputs = np.array(sample_inputs)
    sample_inputs_type = c_float * len(sample_inputs)
    final_sample_inputs = sample_inputs_type(*sample_inputs)

    mylib.predict_mlp_model_classification.argtypes = [c_void_p, sample_inputs_type]
    mylib.predict_mlp_model_classification.restype = POINTER(c_float)

    return mylib.predict_mlp_model_classification(model, final_sample_inputs)


def predict_mlp_model_regression(model, sample_inputs):
    sample_inputs = np.array(sample_inputs)
    sample_inputs_type = c_float * len(sample_inputs)
    final_sample_inputs = sample_inputs_type(*sample_inputs)

    mylib.predict_mlp_model_regression.argtypes = [c_void_p, sample_inputs_type]
    mylib.predict_mlp_model_regression.restype = POINTER(c_float)

    return mylib.predict_mlp_model_regression(model, final_sample_inputs)


def destroy_mlp_model(model):
    mylib.destroy_mlp_model.argtypes = [c_void_p]
    mylib.destroy_mlp_model.restype = None
    mylib.destroy_mlp_model(model)


if __name__ == "__main__":

    np_arr = np.array([2, 3, 1])
    npl = np.ctypeslib.as_ctypes(np_arr)
    model = create_mlp_model(npl)

    flattened_inputs = [1, 1, 1, 2, 2, 1]
    flattened_outputs = [1, 1, -1]

    print("Avant entraînement : \n")
    for i in range(3):
        # np_flattened_inputs = np.array(flattened_inputs[i * 2:(i + 1) * 2])
        # dataset_flattened_inputs = np.ctypeslib.as_ctypes(np_flattened_inputs)
        # res = predict_mlp_model_classification(model, dataset_flattened_inputs)
        res = predict_mlp_model_classification(model, flattened_inputs[i * 2:(i + 1) * 2])
        print(res[0])

    train_classification_stochastic_backprop_mlp_model(model, flattened_inputs, flattened_outputs)

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
