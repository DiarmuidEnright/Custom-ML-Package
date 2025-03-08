#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ml/knn.h"

extern int n_classes;  // Defined in main.c

typedef struct {
    double **X;
    double *y;
    int n_samples;
    int n_features;
    int k;
} KNN;

Model* create_knn() {
    Model *model = (Model *)malloc(sizeof(Model));
    if (!model) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    KNN *knn = (KNN *)malloc(sizeof(KNN));
    if (!knn) {
        free(model);
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    knn->k = 3;  // Default k value
    model->train = knn_train;
    model->predict = knn_predict;
    model->free = knn_free;
    model->current_tree = NULL;
    model->_internal = knn;

    return model;
}

static double euclidean_distance(double *a, double *b, int n_features) {
    double sum = 0.0;
    for (int i = 0; i < n_features; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void knn_train(Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split) {
    KNN *knn = (KNN *)self->_internal;
    
    knn->X = (double **)malloc(n_samples * sizeof(double *));
    if (!knn->X) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }

    for (int i = 0; i < n_samples; i++) {
        knn->X[i] = (double *)malloc(n_features * sizeof(double));
        if (!knn->X[i]) {
            fprintf(stderr, "Memory allocation failed\n");
            for (int j = 0; j < i; j++) {
                free(knn->X[j]);
            }
            free(knn->X);
            return;
        }
        for (int j = 0; j < n_features; j++) {
            knn->X[i][j] = X[i][j];
        }
    }

    knn->y = (double *)malloc(n_samples * sizeof(double));
    if (!knn->y) {
        fprintf(stderr, "Memory allocation failed\n");
        for (int i = 0; i < n_samples; i++) {
            free(knn->X[i]);
        }
        free(knn->X);
        return;
    }

    for (int i = 0; i < n_samples; i++) {
        knn->y[i] = y[i];
    }

    knn->n_samples = n_samples;
    knn->n_features = n_features;
}

double knn_predict(Model *self, double *x, int n_features) {
    KNN *knn = (KNN *)self->_internal;
    double *distances = (double *)malloc(knn->n_samples * sizeof(double));
    int *indices = (int *)malloc(knn->n_samples * sizeof(int));
    
    if (!distances || !indices) {
        fprintf(stderr, "Memory allocation failed\n");
        free(distances);
        free(indices);
        return -1;
    }

    // Calculate distances and store indices
    for (int i = 0; i < knn->n_samples; i++) {
        distances[i] = euclidean_distance(x, knn->X[i], n_features);
        indices[i] = i;
    }

    // Simple bubble sort for k nearest neighbors
    for (int i = 0; i < knn->k; i++) {
        for (int j = 0; j < knn->n_samples - 1; j++) {
            if (distances[j] > distances[j + 1]) {
                double temp_dist = distances[j];
                distances[j] = distances[j + 1];
                distances[j + 1] = temp_dist;

                int temp_idx = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp_idx;
            }
        }
    }

    // Majority vote among k nearest neighbors
    double *votes = (double *)calloc(n_classes, sizeof(double));
    if (!votes) {
        fprintf(stderr, "Memory allocation failed\n");
        free(distances);
        free(indices);
        return -1;
    }

    for (int i = 0; i < knn->k; i++) {
        votes[(int)knn->y[indices[i]]]++;
    }

    double max_votes = 0;
    int prediction = 0;
    for (int i = 0; i < n_classes; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            prediction = i;
        }
    }

    free(votes);
    free(distances);
    free(indices);
    return prediction;
}

void knn_free(Model *self) {
    KNN *knn = (KNN *)self->_internal;
    if (knn) {
        if (knn->X) {
            for (int i = 0; i < knn->n_samples; i++) {
                free(knn->X[i]);
            }
            free(knn->X);
        }
        free(knn->y);
        free(knn);
    }
    free(self);
}
