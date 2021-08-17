#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

#include "mlp.h"
#include "stdlib.h"
#include "stdio.h"

using namespace std;

class mlp {
    float ***W;
    int *d;
    float **X;
    float **deltas;

    mlp(float ***W, int *d, float **X, float **deltas){
        this->W = W;
        this->d = d;
        this->X = X;
        this->deltas = deltas;
    }

    mlp* create_mlp_model(int * npl, int npl_length){
        int *d = (int *)malloc(sizeof(int) * npl_length);

        for (int i = 0; i < npl_length; i++){
            d[i] = npl[i];
        }

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
            for (int j = 0; j < npl[l] + 1; j++){
                deltas[l][j] = 0.0;
            }
        }

        mlp *model = new mlp(W, d, X, deltas);
        return model;
    }

    void train_classification_stochastic_backprop_mlp_model(){
    }

    void train_regression_stochastic_backprop_mlp_model(){
    }

    float predict_mlp_model_classification(){
        return 0;
    }

    float predict_mlp_model_regression(){
        return 0;
    }

    void destroy_mlp_model(){
    }
};