#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

#include "stdlib.h"
#include "stdio.h"

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
                    continue;
                    //mlp->X[l][j] = tanh(mlp->X[l][j]);
            }
        }
    }
};

DLLEXPORT mlp* create_mlp_model(int * npl, int npl_length){
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

DLLEXPORT void train_classification_stochastic_backprop_mlp_model(){
}

DLLEXPORT void train_regression_stochastic_backprop_mlp_model(){
}

DLLEXPORT float predict_mlp_model_classification(){
    return 0;
}

DLLEXPORT float predict_mlp_model_regression(){
    return 0;
}

DLLEXPORT void destroy_mlp_model(){
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
}