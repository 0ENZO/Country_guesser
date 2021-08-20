#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

#include "stdlib.h"
#include "stdio.h"
#include <iostream>
#include <cmath>

using namespace std;

class mlp {
public:
    float ***W;
    int *d;
    float **X;
    float **deltas;
    int npl_length;

    mlp(float ***W, int *d, float **X, float **deltas, int npl_length){
        this->W = W;
        this->d = d;
        this->X = X;
        this->deltas = deltas;
        this->npl_length = npl_length;
    }

    void forward_pass(mlp *mlp, float *sample_inputs, bool is_classification){
        for (int j = 1; j < mlp->d[0] + 1; j++)
            mlp->X[0][j] = sample_inputs[j - 1];

        for (int l = 1; l < mlp->npl_length + 1; l++){
            for (int j = 1; j < mlp->d[l] + 1; j++){
                float sum_result = 0.0;
                for (int i = 0; i < mlp->d[l - 1] + 1; i++)
                    sum_result += mlp->W[l][i][j] * mlp->X[l-1][i];
                mlp->X[l][j] = sum_result;
                if (l < mlp->npl_length - 1 || is_classification)
                    mlp->X[l][j] = tanhf(mlp->X[l][j]);
            }
        }
    }

    void train_stochastic_gradient_backpropagation(mlp *model,
                                                   float *flattened_dataset_inputs,
                                                   float *flattened_expected_outputs,
                                                   bool is_classification,
                                                   float alpha = 0.01,
                                                   int iterations_count = 1000
                                                   ){

        int input_dim = model->d[0];
        int output_dim = model->d[model->npl_length - 1];
        int samples_count = (sizeof(flattened_dataset_inputs)/sizeof(flattened_dataset_inputs[0])) / input_dim;

        for (int it = 0; it < iterations_count; it++){
            //int k = rand() % samples_count;
            int k = rand() % samples_count;
            float *sample_inputs = (float *)malloc(sizeof(float) * input_dim);
            for(int i = 0; i < input_dim; i++)
                sample_inputs[i] = flattened_dataset_inputs[k * input_dim + i];

            float *sample_expected_outputs = (float *)malloc(sizeof(float) * output_dim);
            for(int i = 0; i < output_dim; i++)
                sample_expected_outputs[i] = flattened_expected_outputs[k * output_dim + i];

            model->forward_pass(model, sample_inputs, is_classification);
            //On se sert jamais de l'indice 0, pointe vers le neuronne de biais
            for (int j = 1; j < model->npl_length; j++){
                model->deltas[npl_length - 1][j] = model->X[npl_length -1][j] - sample_expected_outputs[j - 1];
                if (is_classification)
                    model->deltas[npl_length - 1][j] *= (1 - model->X[npl_length -1][j] * model->X[npl_length -1][j]);
            }

            //for (int l  = model->npl_length; l > 1; l--){
            for (int l  = model->npl_length -1; l > 0; l--){
                for (int i = 0; i < model->d[l - 1] + 1; i++){
                    float sum_result = 0.0;
                    for (int j = 1; j < model->d[l] + 1; j++)
                        sum_result += model->W[l][i][j] * model->deltas[l][j];

                    model->deltas[l - 1][i] = (1 - model->X[l - 1][i] *  model->X[l - 1][i]) * sum_result;
                }
            }

            for (int l  = 1 ; l < model->npl_length; l++){
                for (int i = 0; i < model->d[l - 1] + 1; i++){
                    for (int j = 1; j < model->d[l] + 1; j++)
                        model->W[l][i][j] -= alpha * model->X[l-1][i] * model->deltas[l][j];
                }
            }
        }
    }
};

DLLEXPORT mlp* create_mlp_model(int *npl, int npl_length){
    int *d = (int *)malloc(sizeof(int) * npl_length);

    for (int i = 0; i < npl_length; i++)
        d[i] = npl[i];

    float ***W = (float ***)malloc(sizeof(float **) * npl_length);

    for (int l = 0; l < npl_length; l++){
        W[l] = (float **)malloc(sizeof(float*) * (npl[l - 1] + 1));
        if (l == 0) continue;
        for (int i = 0; i < npl[l - 1] + 1; i++){
            W[l][i] = (float *)malloc(sizeof(float) * (npl[l] + 1));
            for (int j = 0; j < npl[l] + 1; j++)
                W[l][i][j] = (((float)rand()/(float)(RAND_MAX)) * 2) - 1;
        }
    }

    float **X = (float **)malloc(sizeof(float *) * npl_length);
    for (int l = 0; l < npl_length; l++){
        X[l] = (float *)malloc(sizeof(float) * (npl[l] + 1));
        for (int j = 0; j < npl[l] + 1; j++){
            if (j == 0)
                X[l][j] = 1.0;
            else
                X[l][j] = 0.0;
        }
    }

    float **deltas = (float **)malloc(sizeof(float *) * npl_length);
    for (int l = 0; l < npl_length; l++){
        deltas[l] = (float *)malloc(sizeof(float) * (npl[l] + 1));
        for (int j = 0; j < npl[l] + 1; j++)
            deltas[l][j] = 0.0;
    }

    mlp *model = new mlp(W, d, X, deltas, npl_length);
    return model;
}

DLLEXPORT void train_classification_stochastic_backprop_mlp_model(mlp *model,
                                                                  float *flattened_dataset_inputs,
                                                                  float *flattened_expected_outputs,
                                                                  float alpha = 0.01,
                                                                  int iterations_count = 1000
                                                                  ){
    model->train_stochastic_gradient_backpropagation(model,
                                                     flattened_dataset_inputs,
                                                     flattened_expected_outputs,
                                                     true,
                                                     alpha,
                                                     iterations_count);
}

DLLEXPORT void train_regression_stochastic_backprop_mlp_model(mlp *model,
                                                              float *flattened_dataset_inputs,
                                                              float *flattened_expected_outputs,
                                                              float alpha = 0.01,
                                                              int iterations_count = 1000
){
    model->train_stochastic_gradient_backpropagation(model,
                                                     flattened_dataset_inputs,
                                                     flattened_expected_outputs,
                                                     false,
                                                     alpha,
                                                     iterations_count);
}

DLLEXPORT float* predict_mlp_model_classification(mlp *model, float *sample_inputs){
    model->forward_pass(model, sample_inputs, true);
    //return model.X[len(model.d) - 1][1:]

    float *result = (float *)malloc(sizeof(float) * model->d[model->npl_length - 1] - 1);
    for(int j = 1; j < model->d[model->npl_length - 1] + 1; j++)
        result[j - 1] = model->X[model->npl_length - 1][j];

    return result;
}

DLLEXPORT float* predict_mlp_model_regression(mlp *model, float *sample_inputs){
    model->forward_pass(model, sample_inputs, false);

    float *result = (float *)malloc(sizeof(float) * model->d[model->npl_length - 1] - 1);
    for(int j = 1; j < model->d[model->npl_length - 1] + 1; j++)
        result[j - 1] = model->X[model->npl_length - 1][j];

    return result;
}

DLLEXPORT void destroy_mlp_model(){
}

DLLEXPORT float* array_slice(float* array, int start, int end){
    float *result = (float *)malloc(sizeof(float) * (end-start));
    for (int i = start; i < end; i++)
        result[i - start] = array[i];
    return result;
}
int main(){
    int npl_lenght = 3;
    int npl[] = {2,3,1};
    /*
    int *npl = (int *)malloc(sizeof(int) * npl_lenght);
    npl[0] = 2;
    npl[1] = 3;
    npl[2] = 1;
    */
    mlp *model = create_mlp_model(npl, npl_lenght);
    printf("Les X : \n");
    printf("%f\n", model->X[0][0]);
    printf("%f\n", model->X[0][1]);
    printf("%f\n\n", model->X[0][2]);

    printf("%f\n", model->X[1][0]);
    printf("%f\n", model->X[1][1]);
    printf("%f\n", model->X[1][2]);
    printf("%f\n\n", model->X[1][3]);

    printf("%f\n", model->X[2][0]);
    printf("%f\n\n", model->X[2][1]);

    printf("Les deltas : \n");
    printf("%f\n", model->deltas[0][0]);
    printf("%f\n", model->deltas[0][1]);
    printf("%f\n\n", model->deltas[0][2]);

    printf("%f\n", model->deltas[1][0]);
    printf("%f\n", model->deltas[1][1]);
    printf("%f\n", model->deltas[1][2]);
    printf("%f\n\n", model->deltas[1][3]);

    printf("%f\n", model->deltas[2][0]);
    printf("%f\n\n", model->deltas[2][1]);

    printf("Les W : \n");
    for (int l = 0; l < 3; l++){
        if (l == 0) continue;
        for (int i = 0; i < npl[l - 1] + 1; i++){
            for (int j = 0; j < npl[l] + 1; j++)
                printf("%f\n", model->W[l][i][j]);
        }
    }

    printf("\nPrediction : \n");

    /*
    float sample_inputs[] = {42, 51};
    float *result_regression = predict_mlp_model_regression(model, sample_inputs);
    printf("%f\n", result_regression[0]);
    */

    float dataset_flattened_inputs[] = {1, 1, 1, 2, 2, 1};
    float dataset_flattened_outputs[] = {1,1, -1};

    printf("\nAvant Entrainement : \n");
    for (int i = 0; i < 3; i++){
        float *result_classification = predict_mlp_model_classification(model, array_slice(dataset_flattened_inputs, i*2, (i+1)*2));
        printf("%f\n", result_classification[0]);
    }

    train_classification_stochastic_backprop_mlp_model(model, dataset_flattened_inputs, dataset_flattened_outputs);

    printf("\nApres Entrainement : \n");
    for (int i = 0; i < 3; i++){
        float *result_classification = predict_mlp_model_classification(model, array_slice(dataset_flattened_inputs, i*2, (i+1)*2));
        printf("%f\n", result_classification[0]);
    }
}