#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ensemble_methods.h"

extern int n_classes;

typedef struct {
    double *weights;
    double bias;
    Model **base_models;
    int num_models;
    int n_features;
} StackingModel;

static void stacking_train(Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split) {
    StackingModel *model = (StackingModel *)self->_internal;

    // Train base models
    for (int i = 0; i < model->num_models; i++) {
        model->base_models[i]->train(model->base_models[i], X, y, n_samples, n_features, max_depth, min_samples_split);
    }

    // Get predictions from base models to create meta-features
    double **meta_features = (double **)malloc(n_samples * sizeof(double *));
    for (int i = 0; i < n_samples; i++) {
        meta_features[i] = (double *)malloc(model->num_models * sizeof(double));
        for (int j = 0; j < model->num_models; j++) {
            meta_features[i][j] = model->base_models[j]->predict(model->base_models[j], X[i], n_features);
        }
    }

    // Train meta-model (logistic regression) using gradient descent
    double learning_rate = 0.01;
    int max_iterations = 1000;

    // Initialize weights and bias
    for (int i = 0; i < model->num_models; i++) {
        model->weights[i] = (double)rand() / RAND_MAX;
    }
    model->bias = (double)rand() / RAND_MAX;

    // Gradient descent
    for (int iter = 0; iter < max_iterations; iter++) {
        double total_error = 0.0;
        for (int i = 0; i < n_samples; i++) {
            // Forward pass
            double logit = model->bias;
            for (int j = 0; j < model->num_models; j++) {
                logit += model->weights[j] * meta_features[i][j];
            }
            double prediction = 1.0 / (1.0 + exp(-logit));

            // Compute error
            double error = y[i] - prediction;
            total_error += error * error;

            // Update weights and bias
            for (int j = 0; j < model->num_models; j++) {
                model->weights[j] += learning_rate * error * meta_features[i][j];
            }
            model->bias += learning_rate * error;
        }

        // Early stopping
        if (total_error < 0.01) break;
    }

    // Clean up
    for (int i = 0; i < n_samples; i++) {
        free(meta_features[i]);
    }
    free(meta_features);
}

static double stacking_predict(Model *self, double *x, int n_features) {
    StackingModel *model = (StackingModel *)self->_internal;

    // Get predictions from base models
    double *meta_features = (double *)malloc(model->num_models * sizeof(double));
    for (int i = 0; i < model->num_models; i++) {
        meta_features[i] = model->base_models[i]->predict(model->base_models[i], x, n_features);
    }

    // Meta-model prediction (logistic regression)
    double logit = model->bias;
    for (int i = 0; i < model->num_models; i++) {
        logit += model->weights[i] * meta_features[i];
    }
    double prediction = 1.0 / (1.0 + exp(-logit));

    free(meta_features);
    return prediction >= 0.5 ? 1.0 : 0.0;
}

static void stacking_free(Model *self) {
    StackingModel *model = (StackingModel *)self->_internal;
    if (model) {
        free(model->weights);
        // Note: We don't free base_models as they are managed by the caller
        free(model);
    }
    free(self);
}

Model* stacking(Model **models, Dataset *dataset, int num_models) {
    Model *model = (Model *)malloc(sizeof(Model));
    if (!model) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    StackingModel *stacking_model = (StackingModel *)malloc(sizeof(StackingModel));
    if (!stacking_model) {
        free(model);
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    stacking_model->weights = (double *)malloc(num_models * sizeof(double));
    if (!stacking_model->weights) {
        free(stacking_model);
        free(model);
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    stacking_model->base_models = models;
    stacking_model->num_models = num_models;
    stacking_model->n_features = dataset->n_features;

    model->train = stacking_train;
    model->predict = stacking_predict;
    model->free = stacking_free;
    model->current_tree = NULL;
    model->_internal = stacking_model;

    return model;
}
