#ifndef MODEL_H
#define MODEL_H

#include "decision_tree.h"

typedef struct {
    void (*train)(Model *self, double **data, double *target, int n_samples, int n_features);
    double (*predict)(Model *self, double *data, int n_features);
    void (*free)(Model *self);
} Model;

double model_evaluate(Model *model, double **data, int n_samples, int n_features);
Model* create_model();
Model* create_knn();
Model* create_svm();
Model* create_placeholder_model();
void free_model(Model *model);

#endif