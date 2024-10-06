#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pca.h"
#include "grid_search.h"
#include "dataset.h"
#include "decision_tree.h"
#include "gradient_boosting.h"
#include "knn.h"
#include "logical_regression.h"
#include "random_forest.h"
#include "svm.h"
#include "preprocess.h"
#include "utils.h"
#include "ensemble_methods.h"
#include "cross_validation.c"

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    double *weights_input_hidden;
    double *weights_hidden_output;
    double learning_rate;
} NeuralNetwork;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void initialize_weights(double *weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = (double)rand() / RAND_MAX;
    }
}

NeuralNetwork* initialize_network(int input_size, int hidden_size, int output_size, double learning_rate) {
    NeuralNetwork *network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    network->input_size = input_size;
    network->hidden_size = hidden_size;
    network->output_size = output_size;
    network->learning_rate = learning_rate;

    network->weights_input_hidden = (double*)malloc(input_size * hidden_size * sizeof(double));
    network->weights_hidden_output = (double*)malloc(hidden_size * output_size * sizeof(double));

    initialize_weights(network->weights_input_hidden, input_size * hidden_size);
    initialize_weights(network->weights_hidden_output, hidden_size * output_size);

    return network;
}

void forward(NeuralNetwork *network, double *input, double *hidden, double *output) {
    for (int i = 0; i < network->hidden_size; i++) {
        hidden[i] = 0;
        for (int j = 0; j < network->input_size; j++) {
            hidden[i] += input[j] * network->weights_input_hidden[j + i * network->input_size];
        }
        hidden[i] = sigmoid(hidden[i]);
    }
    
    for (int i = 0; i < network->output_size; i++) {
        output[i] = 0;
        for (int j = 0; j < network->hidden_size; j++) {
            output[i] += hidden[j] * network->weights_hidden_output[j + i * network->hidden_size];
        }
        output[i] = sigmoid(output[i]);
    }
}

int main() {
    printf("Neural Network Example:\n");
    int input_size = 3, hidden_size = 4, output_size = 1;
    NeuralNetwork *nn = initialize_network(input_size, hidden_size, output_size, 0.01);

    double input[3] = {1.0, 0.5, 0.2};
    double hidden[4], output[1];
    forward(nn, input, hidden, output);
    printf("Neural Network Output: %f\n\n", output[0]);

    printf("PCA Example:\n");
    int n_samples = 5, n_features = 3;
    double **data = (double**)malloc(n_samples * sizeof(double*));
    for (int i = 0; i < n_samples; i++) {
        data[i] = (double*)malloc(n_features * sizeof(double));
        for (int j = 0; j < n_features; j++) {
            data[i][j] = (double)rand() / RAND_MAX;  // Random data
        }
    }

    int n_components = 2;
    PCA *pca = fit_pca(data, n_samples, n_features, n_components);
    printf("PCA Components:\n");
    for (int i = 0; i < n_components; i++) {
        for (int j = 0; j < n_features; j++) {
            printf("%f ", pca->components[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i < n_samples; i++) {
        free(data[i]);
    }
    free(data);
    
    for (int i = 0; i < n_components; i++) {
        free(pca->components[i]);
    }
    free(pca->components);
    free(pca);
    free(nn->weights_input_hidden);
    free(nn->weights_hidden_output);
    free(nn);

    return 0;
}
