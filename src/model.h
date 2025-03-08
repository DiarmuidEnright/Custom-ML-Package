#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>

// Forward declaration for Dataset
struct Dataset;

// Forward declaration for DecisionTree
typedef struct DecisionTree DecisionTree;

typedef struct Model {
    void (*train)(struct Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split);
    double (*predict)(struct Model *self, double *x, int n_features);
    void (*free)(struct Model *self);
    DecisionTree *current_tree;
    void *_internal;  // For algorithm-specific data
} Model;

void free_model(Model *model);
double model_evaluate(Model *model, double **X, double *y, int n_samples, int n_features);
Model* create_placeholder_model(void);

#endif
