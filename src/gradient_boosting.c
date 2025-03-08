#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gradient_boosting.h"
#include "decision_tree.h"

static void compute_residuals(double *y, double *predictions, double *residuals, size_t n_samples) {
    for (size_t i = 0; i < n_samples; i++) {
        residuals[i] = y[i] - predictions[i];
    }
}

GradientBoosting* gradient_boosting_train(Model *model, double **X, double *y, size_t n_samples, size_t n_features, size_t n_trees, size_t max_depth, double learning_rate, size_t min_samples_split) {
    GradientBoosting *gb_model = (GradientBoosting *)malloc(sizeof(GradientBoosting));
    gb_model->n_trees = n_trees;
    gb_model->learning_rate = learning_rate;
    gb_model->trees = (DecisionTree **)malloc(n_trees * sizeof(DecisionTree *));

    double *predictions = (double *)calloc(n_samples, sizeof(double));
    double *residuals = (double *)malloc(n_samples * sizeof(double));

    for (size_t i = 0; i < n_trees; i++) {
        compute_residuals(y, predictions, residuals, n_samples);

        decision_tree_train(model, X, residuals, n_samples, n_features, max_depth, min_samples_split);
        gb_model->trees[i] = model->current_tree;
        model->current_tree = NULL;

        for (size_t j = 0; j < n_samples; j++) {
            double tree_prediction = tree_predict(gb_model->trees[i]->root, X[j]);
            predictions[j] += learning_rate * tree_prediction;
        }
    }

    free(predictions);
    free(residuals);

    return gb_model;
}

double gradient_boosting_predict(GradientBoosting *model, double *x) {
    double prediction = 0.0;

    for (size_t i = 0; i < model->n_trees; i++) {
        prediction += model->learning_rate * tree_predict(model->trees[i]->root, x);
    }

    return (prediction > 0.5) ? 1 : 0;
}

void gradient_boosting_free(GradientBoosting *model) {
    for (size_t i = 0; i < model->n_trees; i++) {
        tree_free(model->trees[i]->root);
        free(model->trees[i]);
    }
    free(model->trees);
    free(model);
}
