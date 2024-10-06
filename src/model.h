#ifndef MODEL_H
#define MODEL_H

typedef struct Model {
    void (*train)(struct Model *self, double **data, int n_samples, int n_features);
    double (*predict)(struct Model *self, double *data, int n_features);
    void (*free)(struct Model *self);
} Model;

double model_evaluate(Model *model, double **data, int n_samples, int n_features);

Model* create_decision_tree();
Model* create_knn();
Model* create_svm();
Model* create_placeholder_model();
void free_model(Model *model);

#endif
