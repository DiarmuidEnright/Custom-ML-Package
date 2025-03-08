#include <stdio.h>
#include <stdlib.h>
#include "bagging.h"

typedef struct {
    Model **models;
    int num_models;
} BaggingModel;

static void train_bagging(Model *self, double **data, double *target, int n_samples, int n_features, size_t max_depth, size_t min_samples_split) {
    BaggingModel *bagging_model = (BaggingModel *)self->_internal;
    if (!bagging_model) return;

    for (int i = 0; i < bagging_model->num_models; i++) {
        // Create bootstrap indices
        int *indices = (int *)malloc(n_samples * sizeof(int));
        if (!indices) {
            fprintf(stderr, "Memory allocation failed\n");
            return;
        }

        for (int j = 0; j < n_samples; j++) {
            indices[j] = rand() % n_samples;
        }
        
        // Create bootstrap sample
        double **bootstrap_sample = (double **)malloc(n_samples * sizeof(double *));
        if (!bootstrap_sample) {
            fprintf(stderr, "Memory allocation failed\n");
            free(indices);
            return;
        }

        double *bootstrap_target = (double *)malloc(n_samples * sizeof(double));
        if (!bootstrap_target) {
            fprintf(stderr, "Memory allocation failed\n");
            free(indices);
            free(bootstrap_sample);
            return;
        }

        // Fill bootstrap sample
        for (int j = 0; j < n_samples; j++) {
            bootstrap_sample[j] = data[indices[j]];
            bootstrap_target[j] = target[indices[j]];
        }

        // Train individual model
        bagging_model->models[i]->train(bagging_model->models[i], 
                                      bootstrap_sample, 
                                      bootstrap_target, 
                                      n_samples, 
                                      n_features, 
                                      max_depth, 
                                      min_samples_split);
        
        // Clean up
        free(bootstrap_sample);
        free(bootstrap_target);
        free(indices);
    }
}

static double predict_bagging(Model *self, double *data, int n_features) {
    BaggingModel *bagging_model = (BaggingModel *)self->_internal;
    if (!bagging_model) return -1;

    double *predictions = (double *)malloc(bagging_model->num_models * sizeof(double));
    if (!predictions) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }

    for (int i = 0; i < bagging_model->num_models; i++) {
        predictions[i] = bagging_model->models[i]->predict(bagging_model->models[i], data, n_features);
    }

    // Average predictions
    double final_prediction = 0.0;
    for (int i = 0; i < bagging_model->num_models; i++) {
        final_prediction += predictions[i];
    }
    free(predictions);

    return (final_prediction / bagging_model->num_models) >= 0.5 ? 1.0 : 0.0;
}

static void free_bagging(Model *self) {
    if (!self) return;

    BaggingModel *model = (BaggingModel *)self->_internal;
    if (!model) {
        free(self);
        return;
    }

    // Don't free individual models as they are managed by the caller
    free(model->models);
    free(model);
    free(self);
}

Model* bagging(Model **models, Dataset *data, int num_models) {
    // Create bagging model
    BaggingModel *bagging_model = (BaggingModel *)malloc(sizeof(BaggingModel));
    if (!bagging_model) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // Allocate and copy model pointers
    bagging_model->models = (Model **)malloc(num_models * sizeof(Model *));
    if (!bagging_model->models) {
        fprintf(stderr, "Memory allocation failed\n");
        free(bagging_model);
        return NULL;
    }

    bagging_model->num_models = num_models;
    for (int i = 0; i < num_models; i++) {
        bagging_model->models[i] = models[i];
    }

    // Create model interface
    Model *model = (Model *)malloc(sizeof(Model));
    if (!model) {
        fprintf(stderr, "Memory allocation failed\n");
        free(bagging_model->models);
        free(bagging_model);
        return NULL;
    }

    model->train = train_bagging;
    model->predict = predict_bagging;
    model->free = free_bagging;
    model->current_tree = NULL;
    model->_internal = bagging_model;

    return model;
}
