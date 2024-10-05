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

    printf("\nGrid Search Example:\n");
    int param1_values[3] = {1, 2, 3};
    int param2_values[3] = {4, 5, 6};
    GridSearchResult best_result = grid_search(param1_values, param2_values, 3, 3);
    printf("Best Hyperparameters: param1 = %d, param2 = %d, score = %f\n", best_result.param1, best_result.param2, best_result.score);

    printf("\nDataset Management Example:\n");
    size_t n_samples_loaded, n_features_loaded;
    double **dataset = load_dataset("data.csv", &n_samples_loaded, &n_features_loaded);
    print_dataset(dataset, n_samples_loaded, n_features_loaded);
    free_dataset(dataset, n_samples_loaded);

    printf("\nDecision Tree Example:\n");
    DecisionTree *tree = create_decision_tree();
    train_decision_tree(tree, dataset);
    print_decision_tree(tree);
    free_decision_tree(tree);

    printf("\nGradient Boosting Example:\n");
    GradientBoosting *gb = create_gradient_boosting();
    train_gradient_boosting(gb, dataset);
    print_gradient_boosting(gb);
    free_gradient_boosting(gb);

    printf("\nk-Nearest Neighbors Example:\n");
    KNN *knn = create_knn(3);
    train_knn(knn, dataset);
    double knn_prediction = predict_knn(knn, input);
    printf("k-NN Prediction: %f\n", knn_prediction);
    free_knn(knn);

    printf("\nLogistic Regression Example:\n");
    LogisticRegression *log_reg = create_logistic_regression();
    train_logistic_regression(log_reg, dataset);
    double log_reg_prediction = predict_logistic_regression(log_reg, input);
    printf("Logistic Regression Prediction: %f\n", log_reg_prediction);
    free_logistic_regression(log_reg);

    printf("\nRandom Forest Example:\n");
    RandomForest *rf = create_random_forest(100);
    train_random_forest(rf, dataset);
    double rf_prediction = predict_random_forest(rf, input);
    printf("Random Forest Prediction: %f\n", rf_prediction);
    free_random_forest(rf);

    printf("\nSupport Vector Machine Example:\n");
    SVM *svm = create_svm();
    train_svm(svm, dataset);
    double svm_prediction = predict_svm(svm, input);
    printf("SVM Prediction: %f\n", svm_prediction);
    free_svm(svm);

    printf("\nPreprocessing Example:\n");
    normalize_dataset(dataset, n_samples_loaded, n_features_loaded);
    print_dataset(dataset, n_samples_loaded, n_features_loaded);

    printf("\nUtilities Example:\n");
    double mean = calculate_mean(input, input_size);
    printf("Mean of input: %f\n", mean);

    free(nn->weights_input_hidden);
    free(nn->weights_hidden_output);
    free(nn);

    return 0;
}
