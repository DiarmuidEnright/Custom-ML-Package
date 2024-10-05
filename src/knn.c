#include <pthread.h>
#include "knn.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define MAX_THREADS 4

typedef struct {
    kNN *model;
    double *x;
    double *distances;
    size_t start;
    size_t end;
} knn_thread_data;

double euclidean_distance(double *a, double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

void* knn_distance_thread(void *arg) {
    knn_thread_data *data = (knn_thread_data*)arg;
    for (size_t i = data->start; i < data->end; i++) {
        data->distances[i] = euclidean_distance(data->model->X_train[i], data->x, data->model->n_features);
    }
    return NULL;
}

double knn_predict(kNN *model, double *x) {
    double *distances = (double*)malloc(model->n_samples * sizeof(double));
    if (!distances) {
        return -1;
    }
    
    pthread_t threads[MAX_THREADS];
    knn_thread_data thread_data[MAX_THREADS];

    size_t chunk_size = model->n_samples / MAX_THREADS;

    for (size_t t = 0; t < MAX_THREADS; t++) {
        thread_data[t].model = model;
        thread_data[t].x = x;
        thread_data[t].distances = distances;
        thread_data[t].start = t * chunk_size;
        thread_data[t].end = (t == MAX_THREADS - 1) ? model->n_samples : (t + 1) * chunk_size;
        pthread_create(&threads[t], NULL, knn_distance_thread, &thread_data[t]);
    }

    for (size_t t = 0; t < MAX_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    double min_distance = DBL_MAX;
    double predicted_label = 0;
    for (size_t i = 0; i < model->n_samples; i++) {
        if (distances[i] < min_distance) {
            min_distance = distances[i];
            predicted_label = model->y_train[i];
        }
    }

    free(distances);
    return predicted_label;
}

void knn_free(kNN *model) {
    if (model) {
        free(model->X_train);
        free(model->y_train);
        free(model);
    }
}
