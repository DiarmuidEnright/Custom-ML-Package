#include "model.h"
#include <stdio.h>
#include <stdlib.h>

void placeholder_train(Model *self, double **data, int n_samples, int n_features) {
    printf("Training a placeholder model...\n");
}

double placeholder_predict(Model *self, double *data, int n_features) {
    printf("Predicting with a placeholder model...\n");
    return 0.0;
}

void placeholder_free(Model *self) {
    printf("Freeing the placeholder model...\n");
    free(self);
}

double model_evaluate(Model *model, double **data, int n_samples, int n_features) {
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        double prediction = model->predict(model, data[i], n_features);
        if (prediction == data[i][n_features - 1]) {
            correct++;
        }
    }
    return (double)correct / n_samples;
}

Model* create_placeholder_model() {
    Model *model = (Model*)malloc(sizeof(Model));
    model->train = placeholder_train;
    model->predict = placeholder_predict;
    model->free = placeholder_free;
    return model;
}
