#ifndef PCA_H
#define PCA_H

typedef struct {
    double **components;
    int n_components;
} PCA;

PCA* fit_pca(double **data, int n_samples, int n_features, int n_components);

void free_pca(PCA *pca);

#endif
