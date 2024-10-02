#include "logistic_regression.h"
#include <stdlib.h>
#include <math.h>

static double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

LogisticRegression* logistic_regression_train(double **X, double *y, size_t n_samples, size_t n_features, double learning_rate, size_t max_iter) {
    LogisticRegression *model = (LogisticRegression *)malloc(sizeof(LogisticRegression));
    model->weights = (double *)calloc(n_features, sizeof(double));
    model->bias = 0;
    model->n_features = n_features;

    for (size_t iter = 0; iter < max_iter; iter++) {
        double gradient_bias = 0.0;
        double *gradient_weights = (double *)calloc(n_features, sizeof(double));

        for (size_t i = 0; i < n_samples; i++) {
            double z = model->bias;
            for (size_t j = 0; j < n_features; j++) {
                z += model->weights[j] * X[i][j];
            }
            double prediction = sigmoid(z);
            double error = prediction - y[i];

            gradient_bias += error;
            for (size_t j = 0; j < n_features; j++) {
                gradient_weights[j] += error * X[i][j];
            }
        }

        model->bias -= learning_rate * gradient_bias / n_samples;
        for (size_t j = 0; j < n_features; j++) {
            model->weights[j] -= learning_rate * gradient_weights[j] / n_samples;
        }

        free(gradient_weights);
    }

    return model;
}

double logistic_regression_predict(LogisticRegression *model, double *x) {
    double z = model->bias;
    for (size_t i = 0; i < model->n_features; i++) {
        z += model->weights[i] * x[i];
    }
    return sigmoid(z) >= 0.5 ? 1 : 0;
}

void logistic_regression_free(LogisticRegression *model) {
    free(model->weights);
    free(model);
}
