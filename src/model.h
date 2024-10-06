#ifndef MODEL_H
#define MODEL_H

#include "decision_tree.h"

double model_evaluate(Model *model, double **data, int n_samples, int n_features);
Model* create_model();
Model* create_knn();
Model* create_svm();
Model* create_placeholder_model();
void free_model(Model *model);

#endif