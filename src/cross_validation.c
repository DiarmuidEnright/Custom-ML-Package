#include <stdio.h>
#include <stdlib.h>
#include "cross_validation.h"

void cross_validation(Model *model, Dataset *dataset, int k) {
    int fold_size = dataset->n_samples / k;
    double total_accuracy = 0.0;
    
    // Allocate memory for indices
    int *indices = (int *)malloc(dataset->n_samples * sizeof(int));
    if (!indices) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }

    // Initialize indices
    for (int i = 0; i < dataset->n_samples; i++) {
        indices[i] = i;
    }

    // Shuffle indices
    for (int i = dataset->n_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // Perform k-fold cross validation
    for (int fold = 0; fold < k; fold++) {
        int test_start = fold * fold_size;
        int test_end = (fold == k - 1) ? dataset->n_samples : (fold + 1) * fold_size;
        int test_size = test_end - test_start;
        int train_size = dataset->n_samples - test_size;

        // Create train and test sets
        double **X_train = (double **)malloc(train_size * sizeof(double *));
        double **X_test = (double **)malloc(test_size * sizeof(double *));
        double *y_train = (double *)malloc(train_size * sizeof(double));
        double *y_test = (double *)malloc(test_size * sizeof(double));

        if (!X_train || !X_test || !y_train || !y_test) {
            fprintf(stderr, "Memory allocation failed\n");
            free(indices);
            free(X_train);
            free(X_test);
            free(y_train);
            free(y_test);
            return;
        }

        // Split data into train and test sets
        int train_idx = 0;
        int test_idx = 0;
        for (int i = 0; i < dataset->n_samples; i++) {
            if (i >= test_start && i < test_end) {
                X_test[test_idx] = dataset->X[indices[i]];
                if (dataset->y) y_test[test_idx] = dataset->y[indices[i]];
                test_idx++;
            } else {
                X_train[train_idx] = dataset->X[indices[i]];
                if (dataset->y) y_train[train_idx] = dataset->y[indices[i]];
                train_idx++;
            }
        }

        // Train model
        if (dataset->y) {
            model->train(model, X_train, y_train, train_size, dataset->n_features, 10, 2);  // Default hyperparameters
        }

        // Evaluate model
        int correct = 0;
        if (dataset->y) {
            for (int i = 0; i < test_size; i++) {
                double prediction = model->predict(model, X_test[i], dataset->n_features);
                if (prediction == y_test[i]) {
                    correct++;
                }
            }
            double accuracy = (double)correct / test_size;
            total_accuracy += accuracy;
            printf("Fold %d accuracy: %.2f\n", fold + 1, accuracy);
        }

        // Free memory for this fold
        free(X_train);
        free(X_test);
        free(y_train);
        free(y_test);
    }

    if (dataset->y) {
        printf("Average accuracy: %.2f\n", total_accuracy / k);
    }

    free(indices);
}
