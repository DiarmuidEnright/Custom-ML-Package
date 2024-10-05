#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double **components;
    int n_components;
} PCA;

void standardize(double **data, int n_samples, int n_features) {
    for (int i = 0; i < n_features; i++) {
        double mean = 0.0;
        for (int j = 0; j < n_samples; j++) {
            mean += data[j][i];
        }
        mean /= n_samples;

        double std = 0.0;
        for (int j = 0; j < n_samples; j++) {
            data[j][i] -= mean;
            std += data[j][i] * data[j][i];
        }
        std = sqrt(std / n_samples);

        for (int j = 0; j < n_samples; j++) {
            data[j][i] /= std;
        }
    }
}

double** covariance_matrix(double **data, int n_samples, int n_features) {
    double **cov_matrix = (double**)malloc(n_features * sizeof(double*));
    for (int i = 0; i < n_features; i++) {
        cov_matrix[i] = (double*)malloc(n_features * sizeof(double));
    }

    for (int i = 0; i < n_features; i++) {
        for (int j = 0; j < n_features; j++) {
            double cov = 0.0;
            for (int k = 0; k < n_samples; k++) {
                cov += data[k][i] * data[k][j];
            }
            cov_matrix[i][j] = cov / (n_samples - 1);
        }
    }

    return cov_matrix;
}

void eigen_decomposition(double **cov_matrix, double **eigenvectors, double *eigenvalues, int n_features) {
    // random eigenvalues, fix later
    for (int i = 0; i < n_features; i++) {
        eigenvalues[i] = (double)rand() / RAND_MAX;
        for (int j = 0; j < n_features; j++) {
            eigenvectors[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

void swap_double(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

void swap_array(double *a, double *b, int length) {
    for (int i = 0; i < length; i++) {
        double temp = a[i];
        a[i] = b[i];
        b[i] = temp;
    }
}

void sort_eigenvalues(double *eigenvalues, double **eigenvectors, int n_features) {
    for (int i = 0; i < n_features - 1; i++) {
        for (int j = 0; j < n_features - i - 1; j++) {
            if (eigenvalues[j] < eigenvalues[j + 1]) {
                swap_double(&eigenvalues[j], &eigenvalues[j + 1]);
                swap_array(eigenvectors[j], eigenvectors[j + 1], n_features);
            }
        }
    }
}

PCA* fit_pca(double **data, int n_samples, int n_features, int n_components) {
    PCA *pca = (PCA*)malloc(sizeof(PCA));
    pca->n_components = n_components;
    pca->components = (double**)malloc(n_components * sizeof(double*));

    standardize(data, n_samples, n_features);

    double **cov_matrix = covariance_matrix(data, n_samples, n_features);

    double *eigenvalues = (double*)malloc(n_features * sizeof(double));
    double **eigenvectors = (double**)malloc(n_features * sizeof(double*));
    for (int i = 0; i < n_features; i++) {
        eigenvectors[i] = (double*)malloc(n_features * sizeof(double));
    }
    eigen_decomposition(cov_matrix, eigenvectors, eigenvalues, n_features);

    sort_eigenvalues(eigenvalues, eigenvectors, n_features);

    for (int i = 0; i < n_components; i++) {
        pca->components[i] = (double*)malloc(n_features * sizeof(double));
        for (int j = 0; j < n_features; j++) {
            pca->components[i][j] = eigenvectors[i][j];
        }
    }

    for (int i = 0; i < n_features; i++) {
        free(cov_matrix[i]);
        free(eigenvectors[i]);
    }
    free(cov_matrix);
    free(eigenvectors);
    free(eigenvalues);

    return pca;
}
