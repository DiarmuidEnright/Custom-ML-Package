#include "decision_tree.h"
#include "gradient_boosting.h"
#include <stdlib.h>
#include <math.h>

static void compute_residuals(double *y, double *predictions, double *residuals, size_t n_samples) {
    for (size_t i = 0; i < n_samples; i++) {
        residuals[i] = y[i] - predictions[i];
    }
}

GradientBoosting* gradient_boosting_train(double **X, double *y, size_t n_samples, size_t n_features, size_t n_trees, size_t max_depth, double learning_rate) {
    GradientBoosting *model = (GradientBoosting *)malloc(sizeof(GradientBoosting));
    model->n_trees = n_trees;
    model->learning_rate = learning_rate;
    model->trees = (DecisionTree **)malloc(n_trees * sizeof(DecisionTree *));

    double *predictions = (double *)calloc(n_samples, sizeof(double));
    double *residuals = (double *)malloc(n_samples * sizeof(double));

    for (size_t i = 0; i < n_trees; i++) {
        compute_residuals(y, predictions, residuals, n_samples);

        model->trees[i] = decision_tree_train(X, residuals, n_samples, n_features, max_depth, 2);

        for (size_t j = 0; j < n_samples; j++) {
            double tree_prediction = decision_tree_predict(model->trees[i]->root, X[j]);
            predictions[j] += learning_rate * tree_prediction;
        }
    }

    free(predictions);
    free(residuals);

    return model;
}

double gradient_boosting_predict(GradientBoosting *model, double *x) {
    double prediction = 0.0;

    for (size_t i = 0; i < model->n_trees; i++) {
        prediction += model->learning_rate * decision_tree_predict(model->trees[i]->root, x);
    }

    return (prediction > 0.5) ? 1 : 0;
}

void gradient_boosting_free(GradientBoosting *model) {
    for (size_t i = 0; i < model->n_trees; i++) {
        decision_tree_free(model->trees[i]);
    }
    free(model->trees);
    free(model);
}
