#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ml/svm.h"

extern int n_classes;  // Defined in main.c

typedef struct {
    double **support_vectors;
    double *alphas;
    double *y;
    double bias;
    int n_support_vectors;
    int n_features;
    double learning_rate;
    int max_iterations;
} SVM;

Model* create_svm() {
    Model *model = (Model *)malloc(sizeof(Model));
    if (!model) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    SVM *svm = (SVM *)malloc(sizeof(SVM));
    if (!svm) {
        free(model);
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    svm->learning_rate = 0.01;
    svm->max_iterations = 1000;
    model->train = svm_train;
    model->predict = svm_predict;
    model->free = svm_free;
    model->current_tree = NULL;
    model->_internal = svm;

    return model;
}

static double linear_kernel(double *x1, double *x2, int n_features) {
    double sum = 0.0;
    for (int i = 0; i < n_features; i++) {
        sum += x1[i] * x2[i];
    }
    return sum;
}

void svm_train(Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split) {
    SVM *svm = (SVM *)self->_internal;
    
    // Initialize parameters
    svm->n_features = n_features;
    svm->n_support_vectors = n_samples;  // Initially, all points are support vectors

    // Allocate memory for support vectors
    svm->support_vectors = (double **)malloc(n_samples * sizeof(double *));
    if (!svm->support_vectors) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }

    for (int i = 0; i < n_samples; i++) {
        svm->support_vectors[i] = (double *)malloc(n_features * sizeof(double));
        if (!svm->support_vectors[i]) {
            fprintf(stderr, "Memory allocation failed\n");
            for (int j = 0; j < i; j++) {
                free(svm->support_vectors[j]);
            }
            free(svm->support_vectors);
            return;
        }
        for (int j = 0; j < n_features; j++) {
            svm->support_vectors[i][j] = X[i][j];
        }
    }

    // Allocate memory for alphas and labels
    svm->alphas = (double *)calloc(n_samples, sizeof(double));
    svm->y = (double *)malloc(n_samples * sizeof(double));
    if (!svm->alphas || !svm->y) {
        fprintf(stderr, "Memory allocation failed\n");
        for (int i = 0; i < n_samples; i++) {
            free(svm->support_vectors[i]);
        }
        free(svm->support_vectors);
        free(svm->alphas);
        free(svm->y);
        return;
    }

    // Copy labels and transform to {-1, 1}
    for (int i = 0; i < n_samples; i++) {
        svm->y[i] = y[i] == 0 ? -1 : 1;
    }

    // Simple gradient descent optimization
    svm->bias = 0.0;
    for (int iter = 0; iter < svm->max_iterations; iter++) {
        double total_error = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double prediction = 0.0;
            for (int j = 0; j < n_samples; j++) {
                prediction += svm->alphas[j] * svm->y[j] * 
                            linear_kernel(svm->support_vectors[j], X[i], n_features);
            }
            prediction += svm->bias;

            // Update alphas and bias if prediction is wrong
            if (svm->y[i] * prediction < 1) {
                svm->alphas[i] += svm->learning_rate;
                svm->bias += svm->learning_rate * svm->y[i];
                total_error += 1 - svm->y[i] * prediction;
            }
        }
        
        // Early stopping if error is small enough
        if (total_error < 0.01) break;
    }
}

double svm_predict(Model *self, double *x, int n_features) {
    SVM *svm = (SVM *)self->_internal;
    double prediction = 0.0;

    for (int i = 0; i < svm->n_support_vectors; i++) {
        prediction += svm->alphas[i] * svm->y[i] * 
                     linear_kernel(svm->support_vectors[i], x, n_features);
    }
    prediction += svm->bias;

    return prediction > 0 ? 1.0 : 0.0;
}

void svm_free(Model *self) {
    SVM *svm = (SVM *)self->_internal;
    if (svm) {
        if (svm->support_vectors) {
            for (int i = 0; i < svm->n_support_vectors; i++) {
                free(svm->support_vectors[i]);
            }
            free(svm->support_vectors);
        }
        free(svm->alphas);
        free(svm->y);
        free(svm);
    }
    free(self);
}
