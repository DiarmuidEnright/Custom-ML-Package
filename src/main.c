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
#include "cross_validation.h"
#include "model.h"

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

void create_decision_tree() {
    DecisionTreeNode* root = create_decision_tree_node(0, 1.5, 1);
    printf("Created a decision tree with root feature index %d and threshold %f\n", root->feature_index, root->threshold);
}

typedef struct {
    double* features;
    int label;
} DataPoint;

void create_knn() {
    printf("KNN model created.\n");
    DataPoint dataset[3] = {
        {(double[]){1.0, 2.0}, 0},
        {(double[]){2.0, 1.0}, 0},
        {(double[]){3.0, 4.0}, 1}
    };
    for (int i = 0; i < 3; i++) {
        printf("DataPoint %d: (%f, %f) -> Class %d\n", i, dataset[i].features[0], dataset[i].features[1], dataset[i].label);
    }
}

void create_svm() {
    printf("SVM model created.\n");
    printf("SVM training would occur here.\n");
}

typedef struct {
    DecisionTreeNode* decision_tree;
} Model;

void free_model(Model* model) {
    if (model->decision_tree) {
        free(model->decision_tree);
    }
    free(model);
}

int n_classes = 2;

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
    for (int i = 0; i < n_samples; i++) {
        data[i] = (double*)malloc(n_features * sizeof(double));
        if (data[i] == NULL) {
            printf("Error: unable to allocate memory for data[%d]\n", i);
            exit(1);
        }
        for (int j = 0; j < n_features; j++) {
            data[i][j] = (double)rand() / RAND_MAX;
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

    Dataset dataset;
    dataset.X = data;
    dataset.n_samples = n_samples;
    dataset.n_features = n_features;
    dataset.y = NULL;

    Model *decision_tree_model = create_decision_tree();
    Model *knn_model = create_knn();
    Model *svm_model = create_svm();

    Model *models[] = {decision_tree_model, knn_model, svm_model};
    int num_models = 3;

    for (int i = 0; i < num_models; i++) {
        printf("Performing %d-fold cross-validation on %s:\n", k, model_names[i]);
        cross_validation(models[i], &dataset, k);
    }

    printf("Applying Bagging Ensemble Method:\n");
    Model *bagging_model = bagging(models, &dataset, num_models);
    double bagging_accuracy = model_evaluate(bagging_model, dataset.X, dataset.n_samples, dataset.n_features);
    printf("Bagging Accuracy: %.2f\n", bagging_accuracy);

    printf("Applying Stacking Ensemble Method:\n");
    Model *stacking_model = stacking(models, &dataset, num_models);
    double stacking_accuracy = model_evaluate(stacking_model, dataset.X, dataset.n_samples, dataset.n_features);
    printf("Stacking Accuracy: %.2f\n", stacking_accuracy);

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

    free_model(decision_tree_model);
    free_model(knn_model);
    free_model(svm_model);
    free_model(bagging_model);
    free_model(stacking_model);

    return 0;
}