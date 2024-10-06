#include <stdio.h>
#include <stdlib.h>
#include "bagging.h"

typedef struct {
    Model **models;
    int num_models;
} BaggingModel;

void train_bagging(Model *self, double **data, double *target, int n_samples, int n_features, size_t sample_size, size_t num_features) {
    BaggingModel *bagging_model = (BaggingModel *)self;
    for (int i = 0; i < bagging_model->num_models; i++) {
        int *indices = (int *)malloc(n_samples * sizeof(int));
        for (int j = 0; j < n_samples; j++) {
            indices[j] = rand() % n_samples;
        }
        
        double **bootstrap_sample = (double **)malloc(n_samples * sizeof(double *));
        for (int j = 0; j < n_samples; j++) {
            bootstrap_sample[j] = data[indices[j]];
        }
        bagging_model->models[i]->train(bagging_model->models[i], bootstrap_sample, target, n_samples, n_features, sample_size, num_features);
        
        free(bootstrap_sample);
        free(indices);
    }
}

double predict_bagging(Model *self, double *data, int n_features) {
    BaggingModel *bagging_model = (BaggingModel *)self;
    double *predictions = (double *)malloc(bagging_model->num_models * sizeof(double));

    for (int i = 0; i < bagging_model->num_models; i++) {
        predictions[i] = bagging_model->models[i]->predict(bagging_model->models[i], data, n_features);
    }

    double final_prediction = 0.0;

    for (int i = 0; i < bagging_model->num_models; i++) {
        final_prediction += predictions[i];
    }
    final_prediction /= bagging_model->num_models;

    free(predictions);
    return final_prediction;
}

void free_bagging(BaggingModel *model) {
    for (int i = 0; i < model->num_models; i++) {
        model->models[i]->free(model->models[i]);
    }
    free(model->models);
    free(model);
}

Model* bagging(Model **models, Dataset *data, int num_models) {
    BaggingModel *bagging_model = (BaggingModel *)malloc(sizeof(BaggingModel));
    bagging_model->models = (Model **)malloc(num_models * sizeof(Model *));
    bagging_model->num_models = num_models;

    for (int i = 0; i < num_models; i++) {
        bagging_model->models[i] = models[i];
    }

    Model *model = (Model *)malloc(sizeof(Model));
    model->train = (void (*)(Model *, double **, double *, int, int, size_t, size_t))train_bagging;
    model->predict = (double (*)(Model *, double *, int))predict_bagging;
    model->free = (void (*)(Model *))free_bagging;

    return model;
}