#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Global variable needed by decision tree and other classifiers
int n_classes = 2;

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
#include "bagging.h"
#include "ensemble_methods.h"
#include "cross_validation.h"
#include "model.h"
#include "matrix.h"

const char *model_names[] = {"Decision Tree", "KNN", "SVM"};
const int k = 5;

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
    if (network == NULL) {
        printf("Error: unable to allocate memory for neural network\n");
        exit(1);
    }
    network->input_size = input_size;
    network->hidden_size = hidden_size;
    network->output_size = output_size;
    network->learning_rate = learning_rate;

    network->weights_input_hidden = (double*)malloc(input_size * hidden_size * sizeof(double));
    if (network->weights_input_hidden == NULL) {
        printf("Error: unable to allocate memory for weights_input_hidden\n");
        exit(1);
    }
    network->weights_hidden_output = (double*)malloc(hidden_size * output_size * sizeof(double));
    if (network->weights_hidden_output == NULL) {
        printf("Error: unable to allocate memory for weights_hidden_output\n");
        exit(1);
    }

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
    if (data == NULL) {
        printf("Error: unable to allocate memory for data\n");
        exit(1);
    }

    // Create test data and target values
    double *target = (double*)malloc(n_samples * sizeof(double));
    if (target == NULL) {
        printf("Error: unable to allocate memory for target\n");
        exit(1);
    }

    for (int i = 0; i < n_samples; i++) {
        data[i] = (double*)malloc(n_features * sizeof(double));
        if (data[i] == NULL) {
            printf("Error: unable to allocate memory for data[%d]\n", i);
            exit(1);
        }
        // Generate sample feature values
        for (int j = 0; j < n_features; j++) {
            data[i][j] = (double)rand() / RAND_MAX;
        }
        // Generate binary target values based on features
        target[i] = (data[i][0] + data[i][1] > 1.0) ? 1.0 : 0.0;
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

    Dataset dataset;
    dataset.X = data;
    dataset.y = target;
    dataset.n_samples = n_samples;
    dataset.n_features = n_features;

    // Create and train base models
    Model *decision_tree_model = create_decision_tree();
    Model *knn_model = create_knn();
    Model *svm_model = create_svm();

    // Train base models
    decision_tree_model->train(decision_tree_model, data, target, n_samples, n_features, 5, 2);
    knn_model->train(knn_model, data, target, n_samples, n_features, 5, 2);
    svm_model->train(svm_model, data, target, n_samples, n_features, 5, 2);

    Model *models[] = {decision_tree_model, knn_model, svm_model};
    int num_models = 3;

    for (int i = 0; i < num_models; i++) {
        printf("Performing %d-fold cross-validation on %s:\n", k, model_names[i]);
        cross_validation(models[i], &dataset, k);
    }

    printf("Applying Bagging Ensemble Method:\n");
    Model *bagging_model = bagging(models, &dataset, num_models);
    if (bagging_model != NULL) {
        // Train the bagging model
        bagging_model->train(bagging_model, data, target, n_samples, n_features, 5, 2);
        double bagging_accuracy = model_evaluate(bagging_model, data, target, n_samples, n_features);
        printf("Bagging Accuracy: %.2f\n", bagging_accuracy);
    }

    printf("Applying Stacking Ensemble Method:\n");
    Model *stacking_model = stacking(models, &dataset, num_models);
    if (stacking_model != NULL) {
        // Train the stacking model
        stacking_model->train(stacking_model, data, target, n_samples, n_features, 5, 2);
        double stacking_accuracy = model_evaluate(stacking_model, data, target, n_samples, n_features);
        printf("Stacking Accuracy: %.2f\n", stacking_accuracy);
    }

    // Cleanup
    for (int i = 0; i < n_samples; i++) {
        free(data[i]);
    }
    free(data);
    free(target);

    for (int i = 0; i < n_components; i++) {
        free(pca->components[i]);
    }
    free(pca->components);
    free(pca);

    free(nn->weights_input_hidden);
    free(nn->weights_hidden_output);
    free(nn);

    free_model(decision_tree_model);
    free_model(knn_model);
    free_model(svm_model);
    if (bagging_model) free_model(bagging_model);
    if (stacking_model) free_model(stacking_model);

    return 0;
}
