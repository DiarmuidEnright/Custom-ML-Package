#include <pthread.h>
#include "svm.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define MAX_THREADS 4

typedef struct {
    SVM *model;
    double **X;
    double *y;
    size_t start;
    size_t end;
    double C;
} svm_thread_data;

void update_weights(SVM *model, double *x_i, double y_i, double C) {
    double prediction = 0.0;
    for (size_t j = 0; j < model->n_features; j++) {
        prediction += model->weights[j] * x_i[j];
    }
    prediction += model->bias;

    double loss = fmax(0, 1 - y_i * prediction);
    for (size_t j = 0; j < model->n_features; j++) {
        if (loss > 0) {
            model->weights[j] += C * y_i * x_i[j];
        }
        model->weights[j] *= (1 - C);
    }
    model->bias += (loss > 0) ? C * y_i : 0;
}

void* svm_train_thread(void *arg) {
    svm_thread_data *data = (svm_thread_data*)arg;

    for (size_t i = data->start; i < data->end; i++) {
        update_weights(data->model, data->X[i], data->y[i], data->C);
    }

    return NULL;
}

SVM* svm_train(double **X, double *y, size_t n_samples, size_t n_features, double C, double tol, double max_iter) {
    SVM *model = (SVM *)malloc(sizeof(SVM));
    model->weights = (double *)calloc(n_features, sizeof(double));
    model->bias = 0;
    model->n_features = n_features;

    pthread_t threads[MAX_THREADS];
    svm_thread_data thread_data[MAX_THREADS];
    size_t chunk_size = n_samples / MAX_THREADS;

    for (size_t iter = 0; iter < max_iter; iter++) {
        for (size_t t = 0; t < MAX_THREADS; t++) {
            thread_data[t].model = model;
            thread_data[t].X = X;
            thread_data[t].y = y;
            thread_data[t].start = t * chunk_size;
            thread_data[t].end = (t == MAX_THREADS - 1) ? n_samples : (t + 1) * chunk_size;
            thread_data[t].C = C;
            pthread_create(&threads[t], NULL, svm_train_thread, &thread_data[t]);
        }

        for (size_t t = 0; t < MAX_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
    }

    return model;
}
